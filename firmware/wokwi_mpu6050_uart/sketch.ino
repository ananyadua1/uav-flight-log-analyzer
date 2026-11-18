/*
 * MPU6050 over I2C -> UART, bare-metal C on ATmega328P (Arduino Uno in Wokwi).
 *
 * No Wire, no Serial: the TWI (I2C) and USART0 peripherals are driven
 * directly through their registers, as you would on any small MCU.
 *
 *   MPU6050  SDA -> A4 (PC4/SDA)     SCL -> A5 (PC5/SCL)     VCC 5V, GND
 *   UART0 TX (PD1) -> Serial Monitor, 115200 baud, 8N1
 *
 * Output (one CSV line every 20 ms = 50 Hz):
 *   t_ms,ax_mg,ay_mg,az_mg,gx_ddps,gy_ddps,gz_ddps,temp_cC,flag
 *   (mg = milli-g, ddps = 0.1 deg/s, cC = 0.01 degC, flag = OK or VIB)
 *
 * flag=VIB when |a| deviates > 300 mg from 1 g for 5 samples in a row,
 * a tiny on-board version of the vibration check in the Python analyzer.
 * In Wokwi, click the MPU6050 and move its acceleration sliders to test.
 */
#include <avr/io.h>
#include <avr/interrupt.h>
#include <stdint.h>

#define MPU_ADDR      0x68
#define REG_WHO_AM_I  0x75
#define REG_PWR_MGMT  0x6B
#define REG_ACCEL_CFG 0x1C
#define REG_GYRO_CFG  0x1B
#define REG_DATA      0x3B          /* ACCEL_XOUT_H .. GYRO_ZOUT_L (14 bytes) */

#define TWI_START_OK  0x08
#define TWI_RSTART_OK 0x10
#define TWI_SLAW_ACK  0x18
#define TWI_DATA_ACK  0x28
#define TWI_SLAR_ACK  0x40

/* ------------------------------------------------------------- timebase */
static volatile uint32_t g_ms;
ISR(TIMER1_COMPA_vect) { g_ms++; }

static void timer1_init_1ms(void) {
  TCCR1A = 0;
  TCCR1B = (1 << WGM12) | (1 << CS11) | (1 << CS10);   /* CTC, clk/64 */
  OCR1A  = 249;                                        /* 16 MHz/64/250 = 1 kHz */
  TIMSK1 = (1 << OCIE1A);
}
static uint32_t millis32(void) {
  uint8_t s = SREG; cli();
  uint32_t v = g_ms;
  SREG = s;
  return v;
}

/* ----------------------------------------------------------------- UART */
static void uart_init(void) {
  UCSR0A = (1 << U2X0);                                /* double speed */
  UBRR0  = (uint16_t)((F_CPU / 8UL / 115200UL) - 1);   /* = 16 */
  UCSR0B = (1 << TXEN0);
  UCSR0C = (1 << UCSZ01) | (1 << UCSZ00);              /* 8N1 */
}
static void uart_putc(char c) {
  while (!(UCSR0A & (1 << UDRE0))) { }
  UDR0 = c;
}
static void uart_puts(const char *s) { while (*s) uart_putc(*s++); }
static void uart_put_int(int32_t v) {
  char buf[12]; uint8_t n = 0;
  if (v < 0) { uart_putc('-'); v = -v; }
  do { buf[n++] = '0' + (v % 10); v /= 10; } while (v);
  while (n) uart_putc(buf[--n]);
}

/* ------------------------------------------------------------ I2C (TWI) */
static void twi_init(void) {
  TWSR = 0;                                            /* prescaler 1 */
  TWBR = (uint8_t)(((F_CPU / 100000UL) - 16) / 2);     /* 100 kHz = 72 */
  TWCR = (1 << TWEN);
}
static uint8_t twi_wait(uint8_t expect) {
  uint16_t guard = 0xFFFF;
  while (!(TWCR & (1 << TWINT))) { if (!--guard) return 0; }
  return (TWSR & 0xF8) == expect;
}
static uint8_t twi_start(uint8_t expect) {
  TWCR = (1 << TWINT) | (1 << TWSTA) | (1 << TWEN);
  return twi_wait(expect);
}
static void twi_stop(void) {
  TWCR = (1 << TWINT) | (1 << TWSTO) | (1 << TWEN);
  while (TWCR & (1 << TWSTO)) { }
}
static uint8_t twi_write(uint8_t data, uint8_t expect) {
  TWDR = data;
  TWCR = (1 << TWINT) | (1 << TWEN);
  return twi_wait(expect);
}
static uint8_t twi_read(uint8_t ack) {
  TWCR = (1 << TWINT) | (1 << TWEN) | (ack ? (1 << TWEA) : 0);
  twi_wait(ack ? 0x50 : 0x58);
  return TWDR;
}

static uint8_t mpu_write(uint8_t reg, uint8_t val) {
  uint8_t ok = twi_start(TWI_START_OK)
            && twi_write((MPU_ADDR << 1) | 0, TWI_SLAW_ACK)
            && twi_write(reg, TWI_DATA_ACK)
            && twi_write(val, TWI_DATA_ACK);
  twi_stop();
  return ok;
}
static uint8_t mpu_read(uint8_t reg, uint8_t *buf, uint8_t len) {
  if (!(twi_start(TWI_START_OK)
     && twi_write((MPU_ADDR << 1) | 0, TWI_SLAW_ACK)
     && twi_write(reg, TWI_DATA_ACK)
     && twi_start(TWI_RSTART_OK)
     && twi_write((MPU_ADDR << 1) | 1, TWI_SLAR_ACK))) { twi_stop(); return 0; }
  for (uint8_t i = 0; i < len; i++) buf[i] = twi_read(i < len - 1);
  twi_stop();
  return 1;
}

/* ------------------------------------------------------------------ app */
static int16_t be16(const uint8_t *p) { return (int16_t)((p[0] << 8) | p[1]); }
static int32_t iabs32(int32_t v) { return v < 0 ? -v : v; }

static int32_t isqrt32(int32_t x) {                   /* integer sqrt, no libm */
  int32_t r = 0, b = 1L << 30;
  while (b > x) b >>= 2;
  while (b) { if (x >= r + b) { x -= r + b; r = (r >> 1) + b; } else r >>= 1; b >>= 2; }
  return r;
}

void setup(void) {
  uart_init();
  timer1_init_1ms();
  twi_init();
  sei();

  uint8_t who = 0;
  mpu_read(REG_WHO_AM_I, &who, 1);
  uart_puts("# MPU6050 WHO_AM_I=0x"); uart_putc("0123456789ABCDEF"[who >> 4]);
  uart_putc("0123456789ABCDEF"[who & 15]);
  if (who != 0x68) { uart_puts(" -> NOT FOUND, check wiring\r\n"); for (;;) { } }
  uart_puts(" -> OK\r\n");

  mpu_write(REG_PWR_MGMT, 0x00);                       /* wake up */
  mpu_write(REG_ACCEL_CFG, 0x00);                      /* +-2 g  : 16384 LSB/g */
  mpu_write(REG_GYRO_CFG, 0x00);                       /* +-250 dps: 131 LSB/dps */
  uart_puts("# t_ms,ax_mg,ay_mg,az_mg,gx_ddps,gy_ddps,gz_ddps,temp_cC,flag\r\n");
}

void loop(void) {
  static uint32_t next = 0;
  static uint8_t vib_count = 0;
  uint32_t now = millis32();
  if ((int32_t)(now - next) < 0) return;
  next = now + 20;                                     /* 50 Hz */

  uint8_t d[14];
  if (!mpu_read(REG_DATA, d, 14)) { uart_puts("# I2C error\r\n"); return; }

  int32_t ax = (int32_t)be16(d + 0)  * 1000 / 16384;   /* milli-g */
  int32_t ay = (int32_t)be16(d + 2)  * 1000 / 16384;
  int32_t az = (int32_t)be16(d + 4)  * 1000 / 16384;
  int32_t tc = (int32_t)be16(d + 6) * 100 / 340 + 3653;/* 0.01 degC */
  int32_t gx = (int32_t)be16(d + 8)  * 100 / 131;      /* 0.1 deg/s */
  int32_t gy = (int32_t)be16(d + 10) * 100 / 131;
  int32_t gz = (int32_t)be16(d + 12) * 100 / 131;

  int32_t mag = isqrt32(ax * ax + ay * ay + az * az);  /* mg, ~1000 at rest */
  vib_count = (iabs32(mag - 1000) > 300) ? (vib_count < 255 ? vib_count + 1 : 255) : 0;

  uart_put_int(now);  uart_putc(',');
  uart_put_int(ax);   uart_putc(',');
  uart_put_int(ay);   uart_putc(',');
  uart_put_int(az);   uart_putc(',');
  uart_put_int(gx);   uart_putc(',');
  uart_put_int(gy);   uart_putc(',');
  uart_put_int(gz);   uart_putc(',');
  uart_put_int(tc);   uart_putc(',');
  uart_puts(vib_count >= 5 ? "VIB" : "OK");
  uart_puts("\r\n");
}
