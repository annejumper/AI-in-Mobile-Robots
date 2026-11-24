// Microcontroller side of the campus navigator (sampling unchanged from the
// StepCounter / GaitIdentifier).
//
// Reads an MPU-6050 at 100 Hz over I2C and keeps the raw samples in a ring
// buffer. The Linux side (python/main.py) collects them over the Bridge a
// batch at a time. If the sensor stops answering, the LED matrix shows an X
// and the sketch retries once a second, so unplugging and replugging the
// sensor recovers by itself.
//
// The LED matrix shows whatever icon Linux asks for: idle, recording, or a
// navigation instruction (straight, left, right, stairs, arrived).
//
// Wiring (GY-521, everything at the ankle): VCC -> 5V, GND -> GND,
// SDA -> SDA (D20), SCL -> SCL (D21). AD0 unconnected -> address 0x68.
// Mounted with X up the leg, Y forward (direction of travel), Z to the left.
//
// Bridge:
//   get_samples() -> "<ok>|<seq>|ax,ay,az,gx,gy,gz;ax,..." raw counts
//                    ok  = 1 if the sensor is answering, 0 if not
//                    seq = sample number of the first sample in the batch
//                          (a jump between batches = dropped samples)
//                    accel: 2048 counts per g (+-16 g)
//                    gyro: 16.4 counts per deg/s (+-2000 deg/s)
//   get_whoami()  -> the sensor's WHO_AM_I register (0x68 on a real
//                    MPU-6050; clones often report something else), -1 on error
//   set_matrix(n) -> show icon n: 0 idle, 1 recording, 2 straight, 3 left,
//                    4 right, 5 stairs, 6 arrived

#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>
#include <Wire.h>

const uint8_t MPU_ADDR = 0x68;
const unsigned long SAMPLE_US = 10000;       // 100 Hz
const int BUF_SIZE = 256;                    // 2.5 s of samples
const int MAX_BATCH = 50;                    // samples per get_samples() call
const int MATRIX_COLS = 13;
const int MATRIX_ROWS = 8;

// Icons are drawn upright for the board mounted sideways: 8 dots wide by
// 13 tall, '#' = lit. drawIcon() turns them a quarter turn onto the 13x8
// matrix. If they come out upside down, flip TURN_CLOCKWISE.
const bool TURN_CLOCKWISE = false;
const int ICON_COLS = MATRIX_ROWS;           // 8
const int ICON_ROWS = MATRIX_COLS;           // 13
const char *ICON_IDLE[ICON_ROWS] = {
  "........",
  "........",
  "........",
  ".##..##.",
  ".##..##.",
  ".##..##.",
  ".##..##.",
  ".##..##.",
  ".##..##.",
  ".##..##.",
  "........",
  "........",
  "........",
};
const char *ICON_REC[ICON_ROWS] = {
  "........",
  "........",
  "........",
  "........",
  "..####..",
  ".######.",
  ".######.",
  ".######.",
  "..####..",
  "........",
  "........",
  "........",
  "........",
};
const char *ICON_STRAIGHT[ICON_ROWS] = {
  "........",
  "...##...",
  "..####..",
  ".######.",
  "########",
  "...##...",
  "...##...",
  "...##...",
  "...##...",
  "...##...",
  "...##...",
  "...##...",
  "........",
};
const char *ICON_LEFT[ICON_ROWS] = {
  "........",
  "........",
  "........",
  "...#....",
  "..##....",
  ".#######",
  "########",
  ".#######",
  "..##....",
  "...#....",
  "........",
  "........",
  "........",
};
const char *ICON_RIGHT[ICON_ROWS] = {
  "........",
  "........",
  "........",
  "....#...",
  "....##..",
  "#######.",
  "########",
  "#######.",
  "....##..",
  "....#...",
  "........",
  "........",
  "........",
};
const char *ICON_STAIRS[ICON_ROWS] = {
  "........",
  "........",
  "......##",
  "......##",
  "....####",
  "....#...",
  "..###...",
  "..#.....",
  "###.....",
  "#.......",
  "#.......",
  "........",
  "........",
};
const char *ICON_ARRIVE[ICON_ROWS] = {
  "........",
  "........",
  "........",
  ".......#",
  "......##",
  ".....##.",
  "#...##..",
  "##.##...",
  ".###....",
  "..#.....",
  "........",
  "........",
  "........",
};
const char *ICON_ERROR[ICON_ROWS] = {
  "........",
  "........",
  "........",
  "#......#",
  ".#....#.",
  "..#..#..",
  "...##...",
  "...##...",
  "..#..#..",
  ".#....#.",
  "#......#",
  "........",
  "........",
};
const int NUM_ICONS = 7;
const char *const *ICONS[NUM_ICONS] = {
  ICON_IDLE, ICON_REC, ICON_STRAIGHT, ICON_LEFT, ICON_RIGHT, ICON_STAIRS, ICON_ARRIVE,
};

Arduino_LED_Matrix matrix;
uint8_t frame[MATRIX_ROWS * MATRIX_COLS];

// The ring buffer is written by loop() and read by the Bridge handler (which
// runs on another thread). loop() fills slot head % BUF_SIZE and only then
// bumps head, so the handler never reads a half-written sample.
int16_t samples[BUF_SIZE][6];
volatile uint32_t head = 0;                  // total samples ever written
uint32_t sent = 0;                           // total samples handed to Linux

volatile bool sensorOk = false;
volatile int8_t wantedIcon = 0;              // icon Linux asked for
int shownIcon = -1;                          // NUM_ICONS = error
unsigned long lastSampleUs = 0;
unsigned long lastInitTryMs = 0;
int failCount = 0;

bool writeReg(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

int readReg(uint8_t reg) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return -1;
  if (Wire.requestFrom((int)MPU_ADDR, 1) != 1) return -1;
  return Wire.read();
}

bool mpuInit() {
  bool ok = writeReg(0x6B, 0x01);            // PWR_MGMT_1: wake, PLL with gyro X clock
  delay(10);
  ok = ok && writeReg(0x1A, 0x03);           // CONFIG: DLPF ~44 Hz
  ok = ok && writeReg(0x19, 0x09);           // SMPLRT_DIV: 1 kHz / 10 = 100 Hz
  ok = ok && writeReg(0x1B, 0x18);           // GYRO_CONFIG: +-2000 deg/s
  ok = ok && writeReg(0x1C, 0x18);           // ACCEL_CONFIG: +-16 g
  return ok;
}

// Burst-read accel and gyro as raw counts into out[6]. False on an I2C error.
bool mpuRead(int16_t *out) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);                          // ACCEL_XOUT_H
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)MPU_ADDR, 14) != 14) return false;
  int16_t v[7];
  for (int i = 0; i < 7; i++) {
    uint8_t hi = Wire.read();
    uint8_t lo = Wire.read();
    v[i] = (int16_t)((hi << 8) | lo);
  }
  out[0] = v[0]; out[1] = v[1]; out[2] = v[2];   // accel X, Y, Z
  // v[3] is temperature
  out[3] = v[4]; out[4] = v[5]; out[5] = v[6];   // gyro X, Y, Z
  return true;
}

String get_samples() {
  uint32_t h = head;
  // If Linux fell behind, skip ahead to samples loop() can't be overwriting.
  if (h - sent > BUF_SIZE - 8) sent = h - (BUF_SIZE - 8);
  uint32_t n = h - sent;
  if (n > MAX_BATCH) n = MAX_BATCH;

  String out;
  out.reserve(16 + n * 46);
  out += sensorOk ? "1|" : "0|";
  out += String(sent);
  out += "|";
  for (uint32_t i = 0; i < n; i++) {
    const int16_t *s = samples[(sent + i) % BUF_SIZE];
    for (int k = 0; k < 6; k++) {
      out += String(s[k]);
      out += k < 5 ? "," : ";";
    }
  }
  sent += n;
  return out;
}

int get_whoami() {
  return readReg(0x75);                      // WHO_AM_I
}

void set_matrix(int n) {
  if (n >= 0 && n < NUM_ICONS) wantedIcon = n;
}

// Draw an upright ICON_COLS x ICON_ROWS icon turned a quarter turn onto the matrix.
void drawIcon(const char *const *icon) {
  for (int r = 0; r < ICON_ROWS; r++) {
    for (int c = 0; c < ICON_COLS; c++) {
      int row = TURN_CLOCKWISE ? c : MATRIX_ROWS - 1 - c;
      int col = TURN_CLOCKWISE ? MATRIX_COLS - 1 - r : r;
      frame[row * MATRIX_COLS + col] = icon[r][c] == '#' ? 7 : 0;   // brightness 0-7
    }
  }
  matrix.draw(frame);
}

void setup() {
  matrix.begin();
  matrix.setGrayscaleBits(3);
  matrix.clear();
  drawIcon(ICON_ERROR);                      // until the sensor answers

  // Bridge first, so Linux can still reach the sketch if the sensor hangs.
  Bridge.begin();
  Bridge.provide("get_samples", get_samples);
  Bridge.provide("get_whoami", get_whoami);
  Bridge.provide("set_matrix", set_matrix);

  Wire.begin();
  Wire.setClock(400000);                     // short wires, so fast mode is fine
  sensorOk = mpuInit();
  lastInitTryMs = millis();
  lastSampleUs = micros();
}

void loop() {
  int icon = sensorOk ? wantedIcon : NUM_ICONS;
  if (icon != shownIcon) {
    shownIcon = icon;
    drawIcon(icon < NUM_ICONS ? ICONS[icon] : ICON_ERROR);
  }

  if (!sensorOk) {
    // Not answering (unplugged?): retry once a second.
    if (millis() - lastInitTryMs > 1000) {
      lastInitTryMs = millis();
      sensorOk = mpuInit();
      failCount = 0;
    }
    delay(5);
    return;
  }

  unsigned long now = micros();
  if (now - lastSampleUs < SAMPLE_US) return;
  lastSampleUs += SAMPLE_US;
  if (now - lastSampleUs > SAMPLE_US) lastSampleUs = now;   // fell behind: don't burst

  if (!mpuRead(samples[head % BUF_SIZE])) {
    if (++failCount > 10) sensorOk = false;
    return;
  }
  failCount = 0;
  head = head + 1;
}
