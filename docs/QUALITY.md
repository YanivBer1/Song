# v6: 3D hand-mesh readings (WiLoR)

The weak cases of v5 were all hands whose depth cannot be read from 2D landmarks: palms
edge-on to the camera, fingers pointing at the lens, a fist resting on a palm. v6 adds a
3D hand-mesh model, WiLoR, as an optional tracking stage (`track_wilor.py`):

- RTMW-x whole-body pose finds each hand box and ties it to the correct arm.
- WiLoR reconstructs the hand in 3D from that crop (about 2.2 s per hand per frame on CPU).
- A frame is used only when WiLoR's projected hand agrees with an independent 2D detector.
- On those frames the motion carries x, y and depth from one consistent 3D hand; the
  avatar fit trusts that depth and no longer guesses between a hand and its depth mirror.
- Hands that overlap in the signer's image are now brought into contact along their
  actual separation direction (the fist in HELP rests on the palm; it used to float 35 mm
  above it).

Measured on the posed avatar, same phrase with and without the 3D readings:

| phrase | metric | v5 | v6 |
|---|---|---|---|
| HE HELP SMALL | handshape on signing frames | 72.9 | **99.5** |
| | jerk-free frames inside signs | 81.9 | **99.2** |
| | hand placement | 87.6 | **100.0** |
| | no hand through hand | 91.3 | **100.0** |
| HI MY NAME IS JOHN | handshape on signing frames | 97.0 | **98.6** |
| | jerk-free frames inside signs | 99.3 | 98.7 |

Side-by-side frames against the source video confirm it: SMALL now has both palms facing
each other (v5 turned them to the camera) and HELP has the thumb-up fist on a flat,
palm-up base hand (v5 had the base hand vertical).

24 random phrases (6,731 frames; only 5 of the 99 signs have 3D readings), v5 → v6:
handshape on signing frames 89.3 → 92.2, hand placement 98.3 → 98.8, no hand/forearm
inside body 99.3 → 99.9, no hand through hand 98.9 → 99.3, joint limits and junction
continuity 100.

Also in v6:

- **Face:** the head is taken relative to each sign's average head pose (the recordings
  carry a constant 15-20° nose-down tilt that made the avatar look sad), the resting mouth
  is neutral instead of downturned, mouth movement is subtle (an open mouth read as a
  grimace), and the expression is carried across frames where a hand covers the face.
- **Render time:** `render_fast.py` renders the backdrop once per clip and then only the
  character's bounding box per frame, with a cheaper denoiser prefilter: about 2.9 s per
  1280x720 frame on 4 idle CPU cores against 11.5 s before, at the same look (38.7 dB vs
  39.8 dB PSNR on hand and face crops against a 128-sample reference).
- **Contact-aware placement:** near the face the hand is positioned by the fingertip that
  touches the face in the source, not by the hand's centroid (finger proportions differ
  between signer and avatar). Fingertip-to-face error on DEAF, HOME, THINK, MOTHER: 0.8-1.5 cm
  → 0.3-0.4 cm. Touching hands keep the offset between the touching points: NAME 2.8 → 2.0 cm,
  MORE 3.6 → 2.3 cm (the rest is the gap needed so fingers do not pass through each other).
- **Tracking filters:** a hand read twice (as both sides) and detections that teleport for a
  few frames are removed (`track_rtm.drop_ghosts`, `drop_jumps`).
- **All word orders** (`audit_pairs.py`, 9,801 ordered pairs): 9,800 clean junctions
  (99.99%); no freezes, velocity jumps or path loops; one brief handshape spike
  (THANK-YOU → NICE). Median transition 633 ms.
- **Visual review of every sign:** [`docs/QA_V6.md`](QA_V6.md). 26 signs match the source,
  48 have minor deviations, 25 are wrong, mostly from the depth problem above.
- **Finger anatomy:** the last finger joint is coupled to the middle one (DIP ≈ 2/3 PIP) in
  the hand fit. Against the 3D readings this lowers the finger-joint error of the 2D-only
  path from 17.5° to 16.3°; handshape match and smoothness on three test phrases are
  unchanged. A fingers-together prior gave nothing on top and is off. (The v6 videos were
  fitted just before this was switched on.)
- **A test that did not help:** tuning the 2D-only fit's priors (wrist comfort, depth
  weight, regularisation) against the 3D readings. On the five signs with 3D data the
  2D-only fit stays more than 45° off in hand orientation on 28% of frames whatever the
  setting. The depth has to be measured; it cannot be guessed.

**Coverage:** WiLoR results exist for 5 of the 99 signs (JOHN, SMALL, IS, HE, HELP), stored
in `asl_v2/data/wilor/`. The other 94 signs still use the v5 2D path until
`track_wilor.py` is run on them (`scripts/setup_wilor.sh`).

---

# v5 update (superseded for signs that have 3D readings, see above)

The earlier numbers in this file compared the avatar with an intermediate 3D hand
reconstruction, not with the tracked image, and overstated accuracy. A frame-by-frame
review of JOHN against the source video showed open hands where the signer has a fist.

Changes since:

- **Fit target:** the avatar's hand skeleton is fitted directly to the tracked 2D
  landmarks (`handfit.py`), under finger, wrist and forearm range-of-motion limits.
- **Two detectors:** MediaPipe and RTMPose-m hand are fused per frame by temporal
  consistency (`track_rtm.py`), because MediaPipe flips its reading on some frames.
- **Smoothing:** One-Euro de-jitter of the input, then a second solving pass that pulls
  each frame toward the low-passed trajectory.
- **Motion:** whole-sentence bake with smoothed corrections; transitions use
  minimum-jerk hand paths with IK and a hinge elbow.
- **Look and pace:** calmer symmetric face, darker outfit, warmer skin, recorded-speed
  signs, longer transitions.

There is no generative video model in this pipeline: tracking, fitting, a MakeHuman
character and Blender's Cycles renderer produce every frame.

## Results, 24 random phrases (6,356 frames), measured against the tracked 2D landmarks

| metric | mean | worst phrase |
|---|---|---|
| handshape match on signing frames | 89.3 | 78.5 |
| hand placement | 98.3 | 83.2 |
| finger joints within human range | 100.0 | 100.0 |
| no hand/forearm inside body | 99.3 | 96.5 |
| no hand through hand | 98.9 | 89.1 |
| jerk-free frames inside signs | 93.4 | 82.6 |
| jerk-free frames in transitions | 99.6 | 97.5 |
| (reference) signer's own tracked motion, jerk-free | 81.9 | 65.7 |
| junction continuity / no freeze | 100.0 | 100.0 |

"Hi, my name is John" alone: handshape 96.9, jerk-free in signs 99.3.

Known weak cases: hands edge-on or pointing at the camera (SMALL), fist-on-palm contact
signs (HELP), and frames where both detectors misread the hand. A PCA hand-pose prior
trained on 6,975 fitted poses gave no measurable gain and is off by default. These
scores are relative to the trackers; visual checks against the source video
(`outputs/audit`, review sheets) remain necessary.

---

# v3 pipeline: method, audits and measured quality

These are research notes for the realistic ASL avatar pipeline in `asl_v2/`. Every
number below is produced by scripts in the repository and can be regenerated.

## Pipeline

1. **Tracking, frame by frame** (`track.py`)
   - MediaPipe Holistic gives body, face and coarse hands.
   - Each hand then gets a second pass: MediaPipe HandLandmarker runs on a crop around
     the wrist, upscaled to 256 px. This gives metric 3D finger landmarks.
   - If the hand isn't found, the crop is retried rotated 90°, 180° and 270°.
     Downward-pointing hands are often only found this way.
   - A per-frame identity check stops a crop from locking onto the other hand.
2. **Anatomical hand solver** (`kinematics.py`, built with Fable 5.2)
   - Constant bone lengths and published range-of-motion limits.
   - Smoothing and gap filling are done in joint-angle space, with handshapes held
     steady inside a sign.
   - Two-hand symmetry, plus a PCA prior trained on 22.5k ASL Citizen hand poses.
3. **Motion per sign** (`motion.py`)
   - Normalisation and a resting non-signing hand.
   - Hampel filtering removes one-frame glitches in wrist, elbow and hand shape.
   - **Hold, don't drop:** if a hand's tracking is lost while the wrist stays in
     signing space, the hand keeps its last shape.
   - **Battison's Symmetry Condition:** a hand that is never tracked but is clearly
     signing takes the mirror of the other hand's shape.
4. **Phrase assembly** (`phrase.py`)
   - Each sign is segmented into preparation, stroke and retraction. Signing space is
     defined anatomically (wrist above the belt line).
   - Citation holds are shortened, and the slow approach and departure of moving signs
     are trimmed (target undershoot).
   - Clips cut off mid-movement get a short settle with decaying velocity.
   - Each sign continues from exactly where the previous stroke ended, along a quintic
     Hermite curve that matches both position and velocity.
   - The next handshape forms during the movement (anticipatory), and the hand rotates
     with slerp rather than blending points.
   - An idle weak hand hovers in a ready position between two signs that use it.
   - A transition depends only on the boundary states of the two signs, so any word
     order works.
5. **Retargeting onto a CC0 MakeHuman character** (`mh_avatar.py`)
   - **Calibration:** MediaPipe is run on a render of the avatar, and a body-relative
     warp maps the signer's landmarks onto the avatar's (`calibrate.py`).
   - **Arms:** two-bone IK with depth recovered from bone lengths. A closed-loop
     end-effector correction moves each hand to its target.
   - **Shoulders:** scapulohumeral rhythm (the collarbone rises with arm elevation).
   - **Fingers:** the first bone of each finger is aimed by position, the other two by
     direction. Range of motion is clamped hard.
   - **Depth ambiguity** (a hand versus its depth mirror) is resolved by
     analysis-by-synthesis, then median-filtered over the whole sequence.
   - **Collisions:** fingers and forearms are pushed out of the torso and head. Hands
     that should touch are brought into contact, and hands that interpenetrate are
     separated.
   - **Face:** FACS-style expression units, driven by the signer's face.
   - **Ambient life:** breathing and postural sway.
6. **Rendering**: Blender Cycles, 1280×720, 50 mm interpreter-style camera
   (`render_human.py`).

## Audits

- **All word orders** (`audit_pairs.py`): every ordered pair of the 99 tracked signs
  (9,801 junctions) is assembled exactly as inside a sentence.
- **Avatar suite** (`evaluate.py`, `eval_suite.py`): 24 random phrases (2 to 6 signs)
  are posed on the real rig and measured. Includes body-mesh and hand capsule
  collision tests.
- **Visual QA**: three Fable 5.2 reviewers compared source and avatar side by side on
  396 frames (4 per sign, 99 signs). Their findings drove the fixes listed above.

## Results (latest run)

### All 9,801 word orders

| | |
|---|---|
| Clean junctions | **99.93%** |
| Freezes | 0 |
| Velocity jumps | 0 |
| Path loops | 0 |
| Brief handshape spikes | 7 (0.07%) |
| Median transition | 300 ms |

### Avatar suite

Avatar suite `final_v6`: 24 phrases, 4,227 frames.

| metric | mean |
|---|---|
| junction velocity continuity | **100.0** |
| junction without freeze | **100.0** |
| hand placement | **100.0** |
| coverage (both hands always present) | **100.0** |
| finger joints within human range | **100.0** |
| no finger/forearm inside body | **99.6** |
| no hand-through-hand | 96.6 |
| smooth (no velocity spikes) | 94.3 |
| elbow direction | 93.2 |
| handshape, size-normalised (all frames) | 88.1 |
| **handshape on signing frames** (15 visible joints) | **83.6** |

## What limits the handshape score

The remaining error is 3D hand reconstruction from hands only about 50 px tall in the
source video:

- **Fists:** S, A and E fists render too loose. Applying measured angles directly to
  the hinge joints over-curled, so that mode is off by default.
- **Palm-up base hands:** a palm seen edge-on is ambiguous in depth, so it can flip.
- **Fingers pointing at the camera:** the tracked shape collapses when a finger points
  straight at the lens.
- **Most promising next step:** WiLoR or HaMeR (transformer hand-mesh models, used for
  research only) on the hardest frames. Installing them was blocked by the session's
  permission policy.
