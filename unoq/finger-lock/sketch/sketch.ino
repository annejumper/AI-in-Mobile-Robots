// Microcontroller side of the finger-code safe. The Linux side (python/main.py) does the
// camera, the neural network and the lock policy; this sketch is its hands and eyes:
// it reads the start button, drives the servo latch and draws on the LED matrix.
//
// Wiring:
//   Button: D2 <-> GND           (internal pull-up: pressed = LOW)
//   Servo:  signal -> D9, power from its own 5-6 V supply, grounds shared with the UNO Q
//
// Bridge (called from python/main.py):
//   get_presses()      -> how many times the button has been pressed since start-up
//   set_locked(1/0)    -> servo to LOCKED_ANGLE / UNLOCKED_ANGLE
//   set_angle(n)       -> servo to n degrees (tuning only)
//   show_digits(p)     -> 3 slots packed as p = s1*256 + s2*16 + s3,
//                         each slot 0 = blank, 1-6 = digit 0-5, 7 = "_",
//                         8-13 = digit 0-5 dimmed (the live preview, not entered yet)
//                         (only s3 set -> one centred digit)
//   show_icon(n)       -> 0 = X, 1 = check, 2 = all LEDs on, 3 = blank
//
// No Servo library in the UNO Q core, so the 50 Hz servo pulse is made here.

#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>

const int BUTTON_PIN = 2;
const int SERVO_PIN = 9;
const int UNLOCKED_ANGLE = 90;    // upright
const int LOCKED_ANGLE = 180;     // 90 degrees from upright
const int SERVO_MIN_US = 544;
const int SERVO_MAX_US = 2400;
const unsigned long SERVO_PERIOD_MS = 20;
const unsigned long DEBOUNCE_MS = 40;
const unsigned long FLASH_MS = 250;   // all LEDs on when the button is pressed

const int COLS = 13, ROWS = 8;
Arduino_LED_Matrix matrix;
uint8_t frame[ROWS * COLS];

volatile int presses = 0;
volatile int servoAngle = LOCKED_ANGLE;
volatile int requestAngle = -1;
volatile long requestDigits = -1;
volatile int requestIcon = -1;
unsigned long lastPulseMs = 0;
unsigned long flashUntil = 0;
long shownDigits = -1;
int shownIcon = 0;               // what to put back after a flash
bool showingIcon = true;

// ── Pictures ─────────────────────────────────────────────────────────────

const char *DIGITS[10][5] = {
  {"###", "#.#", "#.#", "#.#", "###"},
  {".#.", "##.", ".#.", ".#.", "###"},
  {"###", "..#", "###", "#..", "###"},
  {"###", "..#", "###", "..#", "###"},
  {"#.#", "#.#", "###", "..#", "..#"},
  {"###", "#..", "###", "..#", "###"},
  {"###", "#..", "###", "#.#", "###"},
  {"###", "..#", "..#", "..#", "..#"},
  {"###", "#.#", "###", "#.#", "###"},
  {"###", "#.#", "###", "..#", "###"},
};

const char *CHECK[ROWS] = {
  ".............",
  "..........#..",
  ".........##..",
  "........##...",
  "..#....##....",
  "..##..##.....",
  "...####......",
  "....##.......",
};

const char *CROSS[ROWS] = {
  "..#......#...",
  "...#....#....",
  "....#..#.....",
  ".....##......",
  ".....##......",
  "....#..#.....",
  "...#....#....",
  "..#......#...",
};

void drawPicture(const char *rows[ROWS]) {
  memset(frame, 0, sizeof(frame));
  for (int r = 0; r < ROWS; r++)
    for (int c = 0; c < COLS; c++)
      if (rows[r][c] == '#') frame[r * COLS + c] = 7;
  matrix.draw(frame);
}

void drawSlot(int slot, int x0) {
  if ((slot >= 1 && slot <= 6) || (slot >= 8 && slot <= 13)) {
    int digit = slot <= 6 ? slot - 1 : slot - 8;
    uint8_t level = slot <= 6 ? 7 : 2;          // preview digits are dim
    for (int r = 0; r < 5; r++)
      for (int c = 0; c < 3; c++)
        if (DIGITS[digit][r][c] == '#') frame[(r + 1) * COLS + x0 + c] = level;
  } else if (slot == 7) {
    for (int c = 0; c < 3; c++) frame[6 * COLS + x0 + c] = 7;   // "_"
  }
}

void drawDigits(long packed) {
  memset(frame, 0, sizeof(frame));
  int s1 = (packed >> 8) & 0xF, s2 = (packed >> 4) & 0xF, s3 = packed & 0xF;
  if (s1 == 0 && s2 == 0) {
    drawSlot(s3, 5);                 // one centred digit
  } else {
    drawSlot(s1, 0);
    drawSlot(s2, 5);
    drawSlot(s3, 10);
  }
  matrix.draw(frame);
}

void drawIcon(int n) {
  if (n == 0) drawPicture(CROSS);
  else if (n == 1) drawPicture(CHECK);
  else if (n == 2) { memset(frame, 7, sizeof(frame)); matrix.draw(frame); }
  else { memset(frame, 0, sizeof(frame)); matrix.draw(frame); }
}

void redraw() {
  if (showingIcon) drawIcon(shownIcon);
  else drawDigits(shownDigits);
}

// ── Servo ────────────────────────────────────────────────────────────────

void servoTick() {
  unsigned long now = millis();
  if (now - lastPulseMs < SERVO_PERIOD_MS) return;
  lastPulseMs = now;
  int us = map(servoAngle, 0, 180, SERVO_MIN_US, SERVO_MAX_US);
  digitalWrite(SERVO_PIN, HIGH);
  delayMicroseconds(us);
  digitalWrite(SERVO_PIN, LOW);
}

// ── Bridge handlers: only record the request; loop() acts on it ─────────

int get_presses() { return presses; }
void set_locked(int lock) { requestAngle = lock ? LOCKED_ANGLE : UNLOCKED_ANGLE; }
void set_angle(int angle) { requestAngle = constrain(angle, 0, 180); }
void show_digits(int packed) { requestDigits = packed; }
void show_icon(int n) { requestIcon = n; }

void setup() {
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(SERVO_PIN, OUTPUT);
  digitalWrite(SERVO_PIN, LOW);
  matrix.begin();
  matrix.setGrayscaleBits(3);
  drawIcon(0);

  Bridge.begin();
  Bridge.provide("get_presses", get_presses);
  Bridge.provide("set_locked", set_locked);
  Bridge.provide("set_angle", set_angle);
  Bridge.provide("show_digits", show_digits);
  Bridge.provide("show_icon", show_icon);
}

void loop() {
  static bool lastReading = HIGH, stable = HIGH;
  static unsigned long changedMs = 0;

  servoTick();

  if (requestAngle >= 0) { servoAngle = requestAngle; requestAngle = -1; }
  if (requestDigits >= 0) { shownDigits = requestDigits; showingIcon = false; requestDigits = -1; if (!flashUntil) redraw(); }
  if (requestIcon >= 0) { shownIcon = requestIcon; showingIcon = true; requestIcon = -1; if (!flashUntil) redraw(); }

  bool reading = digitalRead(BUTTON_PIN);
  if (reading != lastReading) { lastReading = reading; changedMs = millis(); }
  if (millis() - changedMs > DEBOUNCE_MS && reading != stable) {
    stable = reading;
    if (stable == LOW) {             // pressed: count it and light every LED for a moment
      presses++;
      drawIcon(2);
      flashUntil = millis() + FLASH_MS;
    }
  }
  if (flashUntil && (long)(millis() - flashUntil) >= 0) {
    flashUntil = 0;
    redraw();
  }
  delay(1);
}
