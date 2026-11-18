# Wokwi embedded-C demo: MPU6050 → I2C → UART

Bare-metal C (AVR registers, no Wire/Serial library) reading a simulated MPU6050 and streaming CSV over UART.

## Run it (2 minutes, no hardware)
1. Open https://wokwi.com/projects/new/arduino-uno
2. Replace the editor contents with `sketch.ino`; open the `diagram.json` tab and paste `diagram.json`.
3. Press ▶. Open the serial monitor: you get `t_ms,ax_mg,ay_mg,az_mg,...,flag` at 50 Hz.
4. Click the MPU6050 and move its acceleration sliders; shake past ±0.3 g and the flag turns `VIB`.
5. Use **Save → Share** and put that link in your resume / README.

## What it demonstrates
| Topic | Where |
|---|---|
| TWI (I2C) master: START / SLA+W / repeated START / burst read / STOP, status-code checks | `twi_*`, `mpu_read` |
| USART0 init (baud via UBRR + U2X), polled TX | `uart_*` |
| Timer1 CTC interrupt for a 1 ms tick (ISR) | `timer1_init_1ms` |
| Sensor bring-up: WHO_AM_I check, wake, full-scale config | `setup` |
| Fixed-point scaling (no floats), integer sqrt | `loop`, `isqrt32` |
| On-board vibration flag (links to the Python analyzer) | `vib_count` |

Pins: A4 = SDA, A5 = SCL, 5V, GND.  Baud 115200 8N1.
