// Hardware test for the safe: LED matrix, start button and servo latch.
//
// On start-up (self test):
//   1. the matrix shows 1, 2, 3, 4, 5, then "123", a check mark and an X
//   2. the servo moves to UNLOCKED, waits, then to LOCKED
// Then, every press of the button:
//   - while the button is held, every LED is lit (shows any dead LEDs)
//   - on release the latch toggles: UNLOCKED shows a check, LOCKED shows an X
//
// Wiring:
//   Button: one leg -> D2, other leg -> GND     (internal pull-up: pressed = LOW)
//   Servo:  signal (orange/yellow) -> D9
//           red -> + of the separate 5-6 V supply
//           brown/black -> - of that supply AND a UNO Q GND
//
// There is no Servo library in the UNO Q core, so the servo pulse is made here:
// a HIGH pulse of 544-2400 us (0-180 degrees) every 20 ms.
//
// Bridge:
//   get_status()    -> "<presses>,<locked 1/0>,<button down 1/0>,<servo angle>"
//   set_locked(1/0) -> lock / unlock (servo + check/X)
//   set_angle(n)    -> move the servo to n degrees (0-180), for tuning the two angles
// From the laptop:  python3 tools/latch.py lock | unlock | angle 120 | status

#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>

const int BUTTON_PIN = 2;
const int SERVO_PIN = 9;

// Tune these two so the latch fully opens and fully closes without straining.
const int UNLOCKED_ANGLE = 90;   // upright
const int LOCKED_ANGLE = 180;    // 90 degrees from upright (0 turned the wrong way; 175 didn't fully lock)

const int SERVO_MIN_US = 544;    // pulse width at 0 degrees
const int SERVO_MAX_US = 2400;   // pulse width at 180 degrees
const unsigned long SERVO_PERIOD_MS = 20;
const unsigned long DEBOUNCE_MS = 40;

const int COLS = 13, ROWS = 8;
Arduino_LED_Matrix matrix;
uint8_t frame[ROWS * COLS];

volatile int presses = 0;
volatile bool locked = true;
volatile bool buttonDown = false;
volatile int servoAngle = LOCKED_ANGLE;
unsigned long lastPulseMs = 0;

// ── LED matrix ────────────────────────────────────────────────────────────

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

void clearFrame() { memset(frame, 0, sizeof(frame)); }

void showPicture(const char *rows[ROWS]) {
  clearFrame();
  for (int r = 0; r < ROWS; r++)
    for (int c = 0; c < COLS; c++)
      if (rows[r][c] == '#') frame[r * COLS + c] = 7;
  matrix.draw(frame);
}

// Up to 3 digits, 3x5 font, in columns 0-2, 5-7 and 10-12 (single digits are centred).
void showDigits(const int *digits, int n) {
  clearFrame();
  for (int i = 0; i < n && i < 3; i++) {
    int x0 = (n == 1) ? 5 : i * 5;
    for (int r = 0; r < 5; r++)
      for (int c = 0; c < 3; c++)
        if (DIGITS[digits[i]][r][c] == '#') frame[(r + 1) * COLS + x0 + c] = 7;
  }
  matrix.draw(frame);
}

void showAll() {
  memset(frame, 7, sizeof(frame));
  matrix.draw(frame);
}

// ── Servo ─────────────────────────────────────────────────────────────────

// Called often from loop(); sends one pulse every 20 ms.
void servoTick() {
  unsigned long now = millis();
  if (now - lastPulseMs < SERVO_PERIOD_MS) return;
  lastPulseMs = now;
  int us = map(servoAngle, 0, 180, SERVO_MIN_US, SERVO_MAX_US);
  digitalWrite(SERVO_PIN, HIGH);
  delayMicroseconds(us);
  digitalWrite(SERVO_PIN, LOW);
}

// Like delay(), but keeps the servo pulses going so it holds its position.
void wait(unsigned long ms) {
  unsigned long start = millis();
  while (millis() - start < ms) {
    servoTick();
    delay(1);
  }
}

void setLocked(bool lock) {
  locked = lock;
  servoAngle = lock ? LOCKED_ANGLE : UNLOCKED_ANGLE;
  showPicture(lock ? CROSS : CHECK);
}

// ── Bridge ────────────────────────────────────────────────────────────────

String get_status() {
  return String(presses) + "," + (locked ? "1" : "0") + "," + (buttonDown ? "1" : "0") + "," + String(servoAngle);
}

// Requests from Python. The handlers only record them; loop() acts on them.
volatile int requestLock = -1;    // 1 lock, 0 unlock, -1 nothing
volatile int requestAngle = -1;   // 0-180 to move the servo directly (for tuning)

void set_locked(int lock) { requestLock = lock ? 1 : 0; }
void set_angle(int angle) { requestAngle = constrain(angle, 0, 180); }

// ── Main ──────────────────────────────────────────────────────────────────

void setup() {
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(SERVO_PIN, OUTPUT);
  digitalWrite(SERVO_PIN, LOW);
  matrix.begin();
  matrix.setGrayscaleBits(3);   // brightness values 0-7

  Bridge.begin();
  Bridge.provide("get_status", get_status);
  Bridge.provide("set_locked", set_locked);
  Bridge.provide("set_angle", set_angle);

  // Self test
  for (int d = 1; d <= 5; d++) {
    showDigits(&d, 1);
    wait(500);
  }
  int code[3] = {1, 2, 3};
  showDigits(code, 3);
  wait(1000);
  setLocked(false);   // check + servo to UNLOCKED
  wait(1500);
  setLocked(true);    // X + servo to LOCKED
}

void loop() {
  static bool lastReading = HIGH, stable = HIGH;
  static unsigned long changedMs = 0;

  servoTick();

  if (requestLock >= 0) {
    setLocked(requestLock == 1);
    requestLock = -1;
  }
  if (requestAngle >= 0) {
    servoAngle = requestAngle;
    requestAngle = -1;
  }

  bool reading = digitalRead(BUTTON_PIN);
  if (reading != lastReading) {
    lastReading = reading;
    changedMs = millis();
  }
  if (millis() - changedMs > DEBOUNCE_MS && reading != stable) {
    stable = reading;
    if (stable == LOW) {          // pressed
      buttonDown = true;
      presses++;
      showAll();
    } else {                      // released: toggle the latch
      buttonDown = false;
      setLocked(!locked);
    }
  }
  delay(1);
}
