# Visual QA, v6: every sign against its source video

Each of the 99 signs was built as a one-word phrase through the full pipeline and rendered
in the source framing; six moments per sign were compared side by side with the source
frames (`asl_v2/qa_sheets.py`, sheets in `outputs/audit/qa/sheets/`). Three independent
reviewers judged handshape, palm orientation, finger spread, placement, contact and
two-hand arrangement, strictly. Full notes per sign: `docs/analysis/qa_v6.json`.

| verdict | signs |
|---|---|
| match (reads as the same hand configuration) | 26 |
| minor (small deviation, still readable) | 48 |
| wrong (orientation or handshape clearly different in 2+ moments) | 25 |

The sheets are not stored in the repository (about 50 MB); regenerate them with the three
commands at the top of `asl_v2/qa_sheets.py`.

Fixes that came out of this review (made after the verdicts above were given):

- **Face:** head upright, neutral resting mouth, subtle mouth movement (11 face notes).
- **A second hand popping up in one-handed signs** (THANK-YOU, NOT, also FINISH): the
  detectors read the signing hand a second time as the other hand for a few frames.
  `track_rtm.drop_ghosts` removes such duplicates; re-rendered THANK-YOU shows one hand.
- **A hand teleporting for a few frames** (HAVE: to the shoulder and back, which also made
  the stroke segmentation pick the wrong part of the clip): `track_rtm.drop_jumps` drops
  detections more than 2.5 hand sizes off the running median of the hand's path, and
  `motion.process_clip` no longer trusts a fallback wrist that leaves that path. Of the 99
  signs only HAVE is affected; it now plays the downward movement of the sign.
- **Fist floating above the palm** (HELP): contact is closed along the true separation
  direction.

Signs with 3D hand readings (WiLoR): HE match, HELP match, IS minor, JOHN wrong, SMALL minor.

## Issue types (count of notes)

| type | notes |
|---|---|
| handshape | 54 |
| placement | 26 |
| contact | 22 |
| orientation | 17 |
| finger_spread | 15 |
| face | 11 |
| other | 5 |
| penetration | 3 |
| missing_hand | 1 |

## Recurring patterns

1. **Closed handshapes do not close.** Fists and the letters M, N, E, S come out as loose or open bent hands; the non-index fingers of a pointing hand hang half open.
2. **Flat hands splay.** Fingers-together base hands come out spread; flat-O and C shapes become claws.
3. **Depth-ambiguous hands get rotated.** Fingers pointing at the camera or a palm-up hand seen edge-on end up pointing down or standing upright.
4. **Contacts fall short.** Fingertip-to-face and hand-to-hand touches stop a quarter to half a hand length away.
5. **Weak hand errors.** Wrong stacking order, a second hand entering a one-handed sign, early separation.

Patterns 1-3 are the depth problem that the 3D hand readings fix (see `docs/QUALITY.md`, v6).
Patterns 4-5 are partly placement (avatar proportions differ from the signer) and partly tracking.

## Wrong

- **AM**: Source shows a closed M (fingers curled down into the palm over the tucked thumb); avatar shows an open bent hand with straight fingers folded sideways at the knuckles and the palm fully exposed, thumb not tucked under the fingers. Source fingers are still mostly curled in transition; avatar hand is already an open bent hand with a large gap between fingers and palm.
- **BILLION**: Source shows a closed fist-like letter with fingers folded down over the thumb, knuckles forward; avatar shows an open bent hand with straight fingers extended horizontally to image-right and the palm exposed. Source B-hand has its palm square to the camera; avatar B-hand is yawed roughly 30-40 degrees so it reads narrower and partly edge-on.
- **BOOK**: Source palms are pressed together with fingers pointing straight at the camera (seen end-on); avatar hands are pressed together with fingers pointing down, the full finger length visible like an inverted prayer. Source hands are still closed palm-to-palm seen end-on; avatar hands are crossed with the image-left hand's fingers passing through the image-right hand.
- **BROTHER**: Source lower hand is a fist with thumb up and index forward directly under the upper hand; avatar lower hand is an open flat hand with fingers pointing straight down and thumb out to the side. Source lower hand is fully visible supporting the upper hand; avatar lower hand is mostly hidden below/behind the upper hand and cannot be read as a matching index hand.
- **BUY**: Source base hand stays flat, palm up, fingers pointing horizontally to image-left; avatar base hand is rotated upright with fingers pointing up and the palm facing sideways toward the other hand. Source base hand is palm-up seen edge-on; avatar base hand is tilted about 45 degrees toward the camera with the thumb standing up.
- **FRIEND**: Source has two hooked index fingers (X-hands) locked together; avatar image-left index is straight and points up-right while the image-right hand is a loose curl above it with no hook. Source index fingers are interlocked; avatar hands are separated, the index tip only pointing toward the other hand.
- **GOOD**: Source moving hand lies palm-up inside the base palm; avatar moving hand stays to the image-left of the base hand, partly out of frame, and never rests in the palm. Source moving hand is a flat hand with fingers together; avatar shows a single digit pointing toward the base hand with the other fingers curled.
- **HAVE**: Source arm hangs straight down with the hand at hip level; avatar arm is bent with the hand raised to the shoulder/neck. Source hand hangs at the side as a downward-pointing fist; avatar forearm is raised to horizontal with the hand bent down at the wrist.
- **HOME**: Source holds a flattened-O (all fingertips bunched onto the thumb); avatar shows a bent open hand with the thumb hanging apart from the fingers (pair 5 is an open C-like hand). Source fingertips press on the cheek (near mouth, then near ear); avatar fingertips stop a third to half a hand short of the face.
- **JOHN**: Source N has index and middle folded down over the thumb; avatar keeps index and middle extended straight up-right (H/U-like), so the last letter is not formed. Source fist and O are tightly closed; avatar fingers are loosely curled with a visible gap to the thumb.
- **LANGUAGE**: Source L-hand (thumb up, index pointing inward); avatar hand is a closed fist or thumb-only with the index not extended. Source has index pointing inward with the other fingers closed; avatar index points inward but the remaining fingers hang open pointing down.
- **LATER**: Source L-hand tips forward (thumb up, index toward camera, other fingers closed); avatar keeps index up with the other fingers extended and splayed sideways, hand not tipped forward. Source hand is beside/overlapping the cheek; avatar hand floats half to one hand-width out from the face.
- **LIKE**: Source fist is on the mouth; avatar fist is beside the cheek (pair 1) or over the nose/eyes (pair 2), a full fist-width off the mouth. Source forearm comes up vertically from below with knuckles to camera; avatar elbow is raised and the forearm crosses the face horizontally, hiding it.
- **LOOK**: Source is a closed fist; avatar has the index finger sticking straight up (pair 6 also thumb out, an L-like hand). Source fist rests on the eye; avatar hand is on the forehead/temple above the brow with fingers pressed into the brow.
- **LOVE**: Source closes to a tight fist under the chin; avatar hand stays loosely curled exactly as in pairs 1-3 and never closes. Source fist sits under the chin; avatar fingertips stay pressed on/into the lower lip.
- **MEET**: Source front hand is a fist with only the index up; avatar front hand has two or three fingers loosely extended upward. Source front hand keeps its index upright level with the rear hand; avatar front hand has the index curled into the fist and sits half a hand below the rear hand.
- **MORNING**: Source dominant hand is loosely closed/cupped palm-up; avatar hand is open with fingers splayed wide. Source non-dominant hand stays in the crook of the dominant elbow; avatar non-dominant hand has left the arm and floats with fingers pointing up-left.
- **NOT**: Source is one-handed with the other hand out of frame; avatar raises a second hand into the signing space (open hand in 3, fist in 4, thumb-up in 5). Avatar second hand overlaps and interlocks with the dominant thumb-up hand.
- **PAY**: Source base hand is flat with fingers together; avatar base hand has fingers splayed wide and angled down toward the camera. Source base hand is palm up; avatar base hand shows the back of the hand (palm down/toward body) at the belt.
- **PM**: Source M is a closed fist with three fingers folded over the thumb; avatar is an open claw with only the fingertips bent and the thumb out to the side. Source P has the index extended straight down; avatar index is pulled in, barely protruding from the curled fingers.
- **SCHOOL**: Source top hand is flat and nearly horizontal, palm down, hovering parallel above the lower palm; avatar top hand is pitched up 35-50 degrees with its palm turned toward the camera and (pair 2) already resting its heel on the lower hand. Source top hand is still a flat open hand lying on the lower palm; avatar top hand is curled into a fist/claw sitting on the lower palm.
- **SISTER**: Source hand has its thumb on the chin/jaw line; avatar hand sits lower at neck/collar height, about half a hand length below the chin and not touching the face. Source ends with the image-left hand stacked directly on top of the image-right hand; avatar has the image-left hand below (pair 4) or beside and lower than (pairs 5, 6) the image-right hand, so the hands sit side by side instead of one on top.
- **TEACHER**: Source hands at the temples are flat-O shapes with fingertips closed onto the thumb; avatar hands are open curved hands/claws with fingers extended upward and the thumb clearly apart from them. Source image-left hand is visible beside the forehead; avatar image-left hand is out of the frame above/left of the head.
- **THANK_YOU**: Source shows only the signing hand with nothing else in front of the torso; avatar has a second hand raised palm-up in front of the belly just below the signing hand, making it look two-handed. Source hand is a palm-up flat hand with fingers together pointing at the camera; avatar hand is palm-up but cupped with all five fingers splayed apart and curled upward like a claw.
- **WATER**: Source is a W with index, middle and ring straight and fanned wide in the image plane; avatar has the index up but middle and ring appear as short hooked stubs bunched close together, so the three-finger W is not readable. Source hand is seen side-on with the fan of fingers spread across the image; avatar hand is turned so the fingers are foreshortened toward the camera and the hand is a narrow strip against the neck.

## Minor

AGAIN, ARE, BAD, CAN, CAR, COME, DAY, DEAF, DOLLARS, DRINK, EAT, FATHER, FINE, FINISH, HEARING, HELLO, HI, HOUSE, HOW, HUNGRY, IS, KNOW, LEARN, MORE, MOTHER, NAME, NEED, NIGHT, SAD, SEE, SEVEN, SHE, SIGN, SMALL, THEY, THINK, TIME, TIRED, TODAY, TOMORROW, UNDERSTAND, WHAT, WHEN, WIN, WORK, YES, YESTERDAY, YOU

## Match

BIG, FAMILY, FEEL, FOOD, GO, HAPPY, HE, HELP, I, ME, MONEY, MY, NICE, NO, NOW, OUR, PLEASE, SORRY, STUDENT, WANT, WE, WHERE, WHO, WHY, WILL, YOUR
