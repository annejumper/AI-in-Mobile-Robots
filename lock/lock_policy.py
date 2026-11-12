"""
The lock's decision rules, kept separate from the camera and the hardware so the exact same
file runs on the computer and on the UNO Q.

    policy = LockPolicy([3, 1, 5])
    policy.update(fingers, confidence, button_pressed, time.time())   # call every frame
    policy.state      -> LOCKED / ENTERING / UNLOCKED / DENIED / LOCKOUT
    policy.entered    -> digits accepted so far
    policy.candidate  -> the digit the network is steadily seeing right now (or None)

Two ways to enter digits (CONFIRM_WITH_BUTTON):
  True   show a digit, adjust your hand until the preview is right, press the button to accept it.
         A press only counts if the network has seen the same 1-5 confidently in at least
         STEADY_FRAMES of the last RECENT_FRAMES frames, so a press mid-movement is ignored.
  False  hold a digit still for HOLD_FRAMES frames, then take your hand out before the next one.
"""
from collections import Counter, deque


class LockPolicy:
    CONFIRM_WITH_BUTTON = True
    MIN_CONF = 0.80         # how sure the network must be for a frame to count
    RECENT_FRAMES = 6       # button mode: frames looked at when the button is pressed (~0.4 s)
    STEADY_FRAMES = 4       # button mode: how many of them must agree
    HOLD_FRAMES = 12        # hold mode: frames in a row the same digit must be seen
    CLEAR_FRAMES = 6        # hold mode: frames of "0" (fist / no hand) needed between digits
    DIGIT_TIMEOUT = 30.0    # seconds allowed per digit before giving up
    RESULT_TIME = 3.0       # how long DENIED is shown
    MAX_FAILS = 3           # wrong codes in a row before a lockout
    LOCKOUT_TIME = 30.0

    def __init__(self, code):
        self.code = list(code)
        self.state = "LOCKED"
        self.entered = []
        self.fails = 0
        self.ignored_press = False
        self._reset_digit(0.0)
        self.t_state = 0.0

    def _reset_digit(self, now):
        self.candidate, self.hold, self.clear, self.need_clear = None, 0, 0, False
        self.recent = deque(maxlen=self.RECENT_FRAMES)
        self.t_digit = now

    def _go(self, state, now):
        self.state, self.t_state = state, now

    def _accept(self, digit, now):
        self.entered.append(digit)
        self._reset_digit(now)
        if len(self.entered) == 3:
            if self.entered == self.code:
                self.fails = 0
                self._go("UNLOCKED", now)
            else:
                self.fails += 1
                self._go("DENIED", now)

    def update(self, fingers, conf, button, now):
        if self.state == "LOCKOUT":
            if now - self.t_state > self.LOCKOUT_TIME:
                self.fails = 0
                self._go("LOCKED", now)
            return

        if self.state == "LOCKED":
            if button:
                self.entered = []
                self._reset_digit(now)
                self.need_clear = not self.CONFIRM_WITH_BUTTON
                self.ignored_press = False
                self._go("ENTERING", now)
            return

        if self.state == "UNLOCKED":
            if button:
                self.entered = []
                self._go("LOCKED", now)
            return

        if self.state == "DENIED":
            if now - self.t_state > self.RESULT_TIME:
                self.entered = []
                self._go("LOCKOUT" if self.fails >= self.MAX_FAILS else "LOCKED", now)
            return

        # ENTERING
        if now - self.t_digit > self.DIGIT_TIMEOUT:
            self.entered = []
            self._go("LOCKED", now)
            return

        if self.CONFIRM_WITH_BUTTON:
            self.recent.append(fingers if (conf >= self.MIN_CONF and fingers != 0) else None)
            top, n = Counter(self.recent).most_common(1)[0]
            self.candidate = top if (top is not None and n >= self.STEADY_FRAMES) else None
            if button:
                if self.candidate is not None:
                    self.ignored_press = False
                    self._accept(self.candidate, now)
                else:
                    self.ignored_press = True     # not sure what's being shown: press ignored
            return

        # hold mode
        if self.need_clear:                  # wait for the hand to leave between digits
            self.clear = self.clear + 1 if (fingers == 0 and conf >= self.MIN_CONF) else 0
            if self.clear >= self.CLEAR_FRAMES:
                self.need_clear, self.clear = False, 0
            return
        if fingers == 0 or conf < self.MIN_CONF:
            self.candidate, self.hold = None, 0
            return
        if fingers == self.candidate:
            self.hold += 1
        else:
            self.candidate, self.hold = fingers, 1
        if self.hold >= self.HOLD_FRAMES:
            self._accept(self.candidate, now)
            self.need_clear = self.state == "ENTERING"

    def message(self):
        n = len(self.entered) + 1
        if self.state == "ENTERING":
            if self.CONFIRM_WITH_BUTTON:
                if self.ignored_press:
                    return f"not sure what you're showing - steady your hand, then press for digit {n}"
                return (f"seeing {self.candidate}: press the button to enter it as digit {n}"
                        if self.candidate else f"show digit {n} of 3")
            return ("take your hand out of the square" if self.need_clear
                    else f"show digit {n} of 3 and hold it")
        return {
            "LOCKED": "press the button to enter a code",
            "UNLOCKED": "open!  press the button to lock",
            "DENIED": "wrong code",
            "LOCKOUT": f"too many tries - wait {self.LOCKOUT_TIME:.0f} s",
        }[self.state]
