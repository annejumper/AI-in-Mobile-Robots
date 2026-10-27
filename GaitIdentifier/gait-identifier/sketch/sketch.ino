// Microcontroller side of the GaitIdentifier wearable (step 2: labeled recording).
//
// Reads an MPU-6050 at 100 Hz over I2C and keeps the raw samples in a ring
// buffer. The Linux side (python/main.py) collects them over the Bridge a
// batch at a time. If the sensor stops answering, the LED matrix shows an X
// and the sketch retries once a second, so unplugging and replugging the
// sensor recovers by itself.
//
// The activity being recorded is picked on the web page (python/main.py calls
// set_label) or, optionally, with a pushbutton between D2 and GND that steps
// PAUSE -> STAND -> WALK -> RUN -> BIKE -> PAUSE. The matrix shows the current
// one (|| for pause, else S, W, R or B). Every sample is tagged with the
// activity selected when it was taken.
//
// Wiring (GY-521, everything at the ankle): VCC -> 5V, GND -> GND,
// SDA -> SDA (D20), SCL -> SCL (D21). AD0 unconnected -> address 0x68.
// Mounted with X up the leg, Y forward (direction of travel), Z to the left.
//
// Bridge:
//   get_samples() -> "<ok>|<seq>|lbl,ax,ay,az,gx,gy,gz;lbl,ax,..." raw counts
//                    ok  = 1 if the sensor is answering, 0 if not
//                    seq = sample number of the first sample in the batch
//                          (a jump between batches = dropped samples)
//                    lbl = 0 pause, 1 stand, 2 walk, 3 run, 4 bike
//                    accel: 2048 counts per g (+-16 g)
//                    gyro: 16.4 counts per deg/s (+-2000 deg/s)
//   get_whoami()  -> the sensor's WHO_AM_I register (0x68 on a real
//                    MPU-6050; clones often report something else), -1 on error
//   set_label(n)  -> switch to activity n (same numbering as lbl)
//   set_matrix(n) -> show activity n's icon (the live prediction) instead of
//                    the recording label; -1 goes back to showing the label

#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>
#include <Wire.h>

const uint8_t MPU_ADDR = 0x68;
const unsigned long SAMPLE_US = 10000;       // 100 Hz
const int BUF_SIZE = 256;                    // 2.5 s of samples
const int MAX_BATCH = 50;                    // samples per get_samples() call
const int MATRIX_COLS = 13;
const int MATRIX_ROWS = 8;
const int BUTTON_PIN = 2;                    // pushbutton to GND
const unsigned long DEBOUNCE_MS = 40;
const int NUM_LABELS = 5;                    // pause, stand, walk, run, bike

// Icons are drawn upright for the board mounted sideways: 8 dots wide by
// 13 tall, '#' = lit. drawIcon() turns them a quarter turn onto the 13x8
// matrix. If they come out upside down, flip TURN_CLOCKWISE.
const bool TURN_CLOCKWISE = false;
const int ICON_COLS = MATRIX_ROWS;           // 8
const int ICON_ROWS = MATRIX_COLS;           // 13
const char *ICON_PAUSE[ICON_ROWS] = {
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
const char *ICON_STAND[ICON_ROWS] = {
  "........",
  "........",
  "..####..",
  ".#....#.",
  ".#......",
  ".#......",
  "..####..",
  "......#.",
  "......#.",
  ".#....#.",
  "..####..",
  "........",
  "........",
};
const char *ICON_WALK[ICON_ROWS] = {
  "........",
  "........",
  "#......#",
  "#......#",
  "#......#",
  "#......#",
  "#......#",
  "#..##..#",
  "#.#..#.#",
  "##....##",
  "#......#",
  "........",
  "........",
};
const char *ICON_RUN[ICON_ROWS] = {
  "........",
  "........",
  ".#####..",
  ".#....#.",
  ".#....#.",
  ".#....#.",
  ".#####..",
  ".#..#...",
  ".#...#..",
  ".#....#.",
  ".#....#.",
  "........",
  "........",
};
const char *ICON_BIKE[ICON_ROWS] = {
  "........",
  "........",
  ".#####..",
  ".#....#.",
  ".#....#.",
  ".#####..",
  ".#....#.",
  ".#....#.",
  ".#....#.",
  ".#....#.",
  ".#####..",
  "........",
  "........",
};
const char *const *LABEL_ICONS[NUM_LABELS] = {
  ICON_PAUSE, ICON_STAND, ICON_WALK, ICON_RUN, ICON_BIKE,
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

Arduino_LED_Matrix matrix;
uint8_t frame[MATRIX_ROWS * MATRIX_COLS];

// The ring buffer is written by loop() and read by the Bridge handler (which
// runs on another thread). loop() fills slot head % BUF_SIZE and only then
// bumps head, so the handler never reads a half-written sample.
int16_t samples[BUF_SIZE][6];
uint8_t sampleLabels[BUF_SIZE];
volatile uint32_t head = 0;                  // total samples ever written
uint32_t sent = 0;                           // total samples handed to Linux

volatile bool sensorOk = false;
volatile uint8_t label = 0;                  // activity being recorded
volatile int8_t matrixOverride = -1;         // icon to show instead of label
int shownIcon = -1;                          // label index, NUM_LABELS = error
bool buttonDown = false;
unsigned long buttonChangeMs = 0;
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
  out.reserve(16 + n * 48);
  out += sensorOk ? "1|" : "0|";
  out += String(sent);
  out += "|";
  for (uint32_t i = 0; i < n; i++) {
    const int16_t *s = samples[(sent + i) % BUF_SIZE];
    out += String(sampleLabels[(sent + i) % BUF_SIZE]);
    out += ",";
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

void set_label(int n) {
  if (n >= 0 && n < NUM_LABELS) label = n;
}

void set_matrix(int n) {
  if (n >= -1 && n < NUM_LABELS) matrixOverride = n;
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

// Step to the next activity on each press (acts on the press, not release).
void pollButton() {
  bool down = digitalRead(BUTTON_PIN) == LOW;
  if (down == buttonDown || millis() - buttonChangeMs < DEBOUNCE_MS) return;
  buttonDown = down;
  buttonChangeMs = millis();
  if (down) label = (label + 1) % NUM_LABELS;
}

void setup() {
  matrix.begin();
  matrix.setGrayscaleBits(3);
  matrix.clear();
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  drawIcon(ICON_ERROR);                      // until the sensor answers

  // Bridge first, so Linux can still reach the sketch if the sensor hangs.
  Bridge.begin();
  Bridge.provide("get_samples", get_samples);
  Bridge.provide("get_whoami", get_whoami);
  Bridge.provide("set_label", set_label);
  Bridge.provide("set_matrix", set_matrix);

  Wire.begin();
  Wire.setClock(400000);                     // short wires, so fast mode is fine
  sensorOk = mpuInit();
  lastInitTryMs = millis();
  lastSampleUs = micros();
}

void loop() {
  pollButton();
  int icon = !sensorOk ? NUM_LABELS : matrixOverride >= 0 ? matrixOverride : label;
  if (icon != shownIcon) {
    shownIcon = icon;
    drawIcon(icon < NUM_LABELS ? LABEL_ICONS[icon] : ICON_ERROR);
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
  sampleLabels[head % BUF_SIZE] = label;
  failCount = 0;
  head = head + 1;
}
