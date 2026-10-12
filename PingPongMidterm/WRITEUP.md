# Virtual Tennis: ME193 Midterm

**Anne Jumper**

> [VIDEO / GIF OF GAMEPLAY HERE]

I play tennis, so I turned virtual ping-pong into **virtual tennis**. The racket is my LEGO
Education Double Motor (card 6065). I rally against a virtual opponent on **clay, grass or hard
court**. Each surface changes how the ball bounces, which changes *where my racket has to be*.
How I swing decides the ball's **spin**, and spin changes whether the ball lands in.
My record rally (most continuous hits) is posted live to `ME193/Rogers/AnneJumper`.

## How it uses everything from class

| Requirement | How Virtual Tennis uses it |
|---|---|
| **Pose** (paddle in the right location) | MediaPipe Pose tracks my racket hand (wrist, extended a little past the wrist toward where the hub sits). The ball's height and side are mapped onto *my own body*: hip height is 0.95 m and my torso is the ruler. To hit, my hand has to be within the level's hit radius of the ball. |
| **IMU** (am I swinging) | The double motor's gyro **pitch** rate. A burst above threshold is a swing, and the peak rate is the **power**. The **yaw** twist during the swing is the **spin**. |
| **AprilTags** (level and start) | Tag 0 = START. Tags 1–3 = Club / Varsity / Pro (13 / 18 / 25 m/s balls, smaller hit zone, tighter timing). Tags 4–6 = Clay / Grass / Hard. |
| **MQTT score** | The record number of continuous hits, as a float (e.g. `7.0`), published to `ME193/Rogers/AnneJumper` every time a ball lands in. |
| **Something new** | Haptics, court surfaces and spin physics (below). |

### The new, cool stuff

1. **Haptic feedback through the motors.** The hub's two motors jerk forward and back, so you feel
   the reaction torque in your hand. Each event has its own "texture":
   - a light *tap* when the incoming ball bounces (a timing cue you can feel)
   - one solid *thump* for a flat hit
   - a fast *brrrt* triple buzz for topspin
   - a long smooth push for slice
   - an extra kick for a PERFECT hit
   - a rumble plus a red light when you miss

   The hub light shows the court color, and beeps confirm hits, balls landing in, and new records.
2. **Three court surfaces with real bounce physics.** Each surface has a coefficient of restitution
   (bounce height), a friction factor (does the ball skid or slow down) and a spin grip. Grass comes
   in at about knee height (around 0.6 m), clay at chest height (around 1.3 m), and hard court is in
   between. So **the racket position you need really changes with the surface**. When you miss, the
   game coaches you: "racket 40 cm too high, grass stays low!"
3. **Spin from torque, like a real racket.** Swinging about the pitch axis hits the ball. Applying
   a twisting torque about the **yaw** axis during the swing spins it. One direction gives
   **topspin**, the other gives **slice**, and no twist gives a clean **flat** ball. A torque
   applied for an instant changes angular velocity (angular impulse: ∫τ dt = IΔω). So the peak
   yaw rate during the swing measures the twisting torque I applied, and its sign gives the spin
   type. A dead zone ignores the small wobble every swing has, so flat shots are really flat.
4. **Spin matters, like real tennis.** The ball flies with gravity plus a Magnus force: topspin
   dives and slice floats. Every shot is aimed to clear the net. A big *flat* swing at a low ball
   therefore has to go up and sails **LONG**, while topspin brings it back down into the court.
   Grass punishes flat power and clay rewards it, as in real tennis.
5. **A 3D broadcast view (Three.js).** The main screen is a 3D tennis stadium drawn in the
   browser with Three.js. Each surface has its own look: clay with dust puffs and ball marks, grass
   with mowing stripes, blue hard court. The racket follows my hand, and the ball has a shadow, a
   trail and synthesized hit and bounce sounds. My webcam and pose skeleton sit in the corner.
   Python still does all the sensing and decisions and streams the game state to the page over a
   local websocket.
6. **Train my own model (optional).** `train_strokes.py` records my **forehand and backhand** swings,
   each flat, topspin and slice (6 classes). It trains a k-nearest-neighbors classifier on IMU
   features and prints a confusion table showing which strokes it mixes up. The game then uses the
   model to decide the stroke and the spin type. This matters because the twist for backhand
   topspin can go the opposite way from forehand topspin, and the model learns that from my own
   swings. If I hit a backhand on my forehand side, the game notices.

## Describe the policy: how does it make decisions?

The game is a state machine (MENU → COUNTDOWN → INCOMING → OUTGOING → … → OVER). The key
decision is **"was that swing a hit?"**, and it is made from the swings the IMU reports:

1. **Timing (IMU).** Find the instant of peak pitch rate in the swing, then look up where the ball
   was at *that* instant.
   - The ball was still more than *z_early* meters away: "Too early!" This is a free practice
     swing, so you can still hit the ball.
   - The ball was already past you: "Too late!" The rally ends.
   - Inside the window, every swing becomes a **candidate**, and the **strongest one wins**. A
     windup or reset is slower than the real swing, so it never counts as the shot. The game waits
     about 0.08 s after a swing in case a stronger one follows, then judges.
2. **Position (pose).** At that same instant, how far was my racket hand from the ball, in meters
   (using my torso as the scale)? If it's farther than the level's hit radius (45 / 35 / 27 cm),
   it's a miss, and the game reports how far off I was.
3. **Hit.** Power comes from peak pitch rate. Spin comes from the signed yaw twist: positive is
   topspin, negative is slice, and inside the dead zone it's flat. If the KNN is trained, it picks
   forehand/backhand and the spin type.
   - Within 0.6 m of perfect contact counts as **PERFECT**.
4. **Shot outcome (physics).** The return is simulated. Landing past the far baseline means LONG
   and the rally ends. Landing in adds +1 to the streak, and MQTT gets the record.
5. **No swing at all** by the time the ball is past me means the rally ends.

The **opponent** follows a simple policy. It always returns the ball, at the level's speed. It
aims at a random forehand or backhand spot within reach, picks a random spin up to the level's
maximum, and gets about 3% faster per hit as the rally builds (capped at +35%).

## Potential limitations

- **One 2D camera.** Pose gives where my hand is up/down and left/right, but not how far
  forward it is. Timing (from the IMU) stands in for depth.
- **Body-size assumptions.** Ball heights are mapped using an average hip height (0.95 m) and
  torso length (0.5 m), so very tall or short players see slightly shifted targets.
- **Pose tracking breaks** if my racket hand crosses in front of my body (backhands), it's dark,
  or I'm too close to see my hips.
- **The IMU depends on how you hold it.** "Yaw" is the gyro z axis only if the hub is held a
  particular way (`--swing-axis` / `--spin-axis` fix this), and the thresholds are in raw gyro units that need tuning
  for each person (`imu_monitor.py`).
- **Latency.** Bluetooth plus camera delay is about 50–150 ms. I judge the swing at the
  timestamp of its peak instead of when the event arrives, which helps but isn't perfect.
- **Simplified physics.** No air drag, and the spin force is constant between bounces. The
  opponent never misses, so the rally can only end on my mistake.
- **Haptics are motor kicks, not a true vibration motor.** An off-center LEGO piece on the shaft
  helps.
- **Public MQTT broker** (test.mosquitto.org): anyone could publish to my topic.

## AI / ML algorithms used

- **MediaPipe Pose Landmarker (BlazePose), a neural network.** A detector network finds the person.
  Then a convolutional neural network regresses 33 body landmarks, tracking them frame to frame.
- **AprilTag detection, classical computer vision.** The image is thresholded to find dark
  four-sided shapes, and the grid of black/white bits inside each one is read. The bits are matched
  to the 36h11 codebook, which can correct a few wrong bits, giving the tag's ID.
- **k-Nearest Neighbors (my trained stroke model, 6 classes).** Every training swing is stored as a feature
  vector (pitch rate, yaw twist, the third gyro axis, acceleration, duration). A new swing gets the
  majority label of the 5 closest training swings after standardizing the features.
- **Rule-based policy + physics simulation (the game "AI").** A state machine with hand-designed
  rules (above) decides hits, misses and the opponent's shots. A projectile simulation with gravity
  and Magnus force decides where the ball goes.
- **Signal processing on the IMU.** Thresholding with hysteresis detects a swing (it starts above
  250 and ends below 100). A dead zone on the yaw twist separates flat shots from spin.

## Reflection

**What I'm proud of:** *[Anne: edit in your own words.]* Getting real tennis strategy to come out
of the physics. On grass, big flat swings fly long, so you have to hit topspin. On clay, the high
bounce lets you swing big. Twisting the racket for topspin or slice works the way a tennis player expects.

**What was hard:** *[Anne: fill in from testing. For example, tuning the swing thresholds,
getting the pose landmarks to line up with the virtual ball, or BLE lag.]*

**What I'd do next:** Use the hub's pitch and roll to read the racket-face angle. Add a second
player over MQTT so two people can rally against each other. Render it in Unity.

**What I learned:** *[Anne: fill in.]*

*Anne Jumper, ME193*
