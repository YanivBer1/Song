# Better ASL video generation: research notes and what was built

Date: November 2026. Hardware here: 4 CPU cores, 15 GB RAM, no GPU.

## 1. What the original Space actually does

`deenasun/ai-sl-api` does not generate video with a neural model. The pipeline is:

1. English words are matched to the nearest sign in a dictionary of ~11.4k
   recorded clips (word2vec cosine similarity).
2. MediaPipe **Pose** and **Hands**, run separately, produce keypoints.
3. Keypoints are drawn as a 2D cartoon, with 15 linearly interpolated
   frames between signs.

### Why hands "appear out of the blue" (root causes, verified on the clips)

| Cause | Evidence |
|---|---|
| The clips (Signing Savvy recordings) are cropped at the waist. A hand that isn't signing hangs **below the frame**, so it is never detected, and the Space draws no hand at all. | Holistic extraction: the left hand was detected in 0 frames for *hi, my, is, john, he, pm, seven, billion*, and in every frame for *name, win, dollars*. In *john* the left wrist sits at y = 1.00 (the bottom edge) with visibility 0.15. |
| Separate Hands model with "Left"/"Right" labels. Labels flip on fast or mirrored hands, so a hand can jump sides. | `video_gen.py` assigns hands by `handedness.label`. |
| No normalization between clips. Different signers and framings make the body jump at every join. | Each clip is rendered in its own image coordinates. |
| Frames without a pose are dropped, and any missing hand is all zeros. | `extract_keypoints_from_video` skips frames with `continue` and fills missing hands with `np.zeros`. |

## 2. Survey of open-source alternatives (verified Nov 2026)

**No end-to-end neural English→ASL video/pose model with released weights runs on CPU.**

| System | Status for us |
|---|---|
| ZurichNLP `spoken-to-signed-translation` (MIT) | The same retrieval idea, with better cropping, connection-point search, interpolation and Butterworth smoothing. The downloadable lexicon is Swiss only. **Ideas adopted** (see §3). |
| sign.mt API (`spoken_text_to_signed_pose`, `signed=ase`) | Works and returns ASL `.pose`, but its lexicon is private and it hides idle hands. Useful as a baseline only. |
| SignSparK (ECCV 2026) | Weights released (non-commercial; 3 × 5.6 GB). Needs CUDA. Free-text retrieval supports only CSL/PHOENIX. Not runnable here. |
| SOKE, wSignGen, SignAvatar, Spoken2Sign, Progressive Transformers, Sign-IDD, FGDM, signvip | Some combination of: missing generator weights, GPU-only, or non-ASL (PHOENIX/CSL). |
| SignLLM, SignDiff, Neural Sign Actors, T2S-GPT, SignAligner | No code or weights released. |
| HamNoSys/SiGML avatars (CWASA) | No sizeable ASL SiGML lexicon. |
| ASL Citizen poses (HF `SorensenAI/asl-citizen-poses`), ASLCitizen-3D (SMPL-X) | Good data for a bigger lexicon later. Check licenses: ASL Citizen does not allow redistribution, and SMPL-X needs registration. |

Realistic rendering options:

| Option | Verdict |
|---|---|
| **Rigged CC0 MakeHuman character in Blender (Cycles, CPU)** | **Chosen.** Realistic skin, clothes, hair and eyes, with a full finger rig. Takes ~4–6 s/frame at 960×540. |
| sign.mt pix2pix pose-to-video | 256 px, blurry, one fixed signer, CC BY-NC code. |
| ControlNet / AnimateDiff / MimicMotion / AnimateAnyone | Minutes per frame on CPU, so hours to days per clip. Not feasible without a GPU. |
| SMPL-X / MANO hand meshes | Registration-gated licenses; can't be fetched by script. |

## 3. What v2 changes (`asl_v2/`)

1. **MediaPipe Holistic** tracks body, hands and the 478-point face together.
   Hands are cropped from the body's wrist, so left and right can't swap.
   Face tracking keeps eyebrows and mouth (ASL grammar).
2. **Normalization.** Every clip is moved and scaled so the shoulders sit in
   the same place at the same width.
3. **Resting hands.** A hand that isn't signing is placed **relaxed at the
   side from frame 1**: a natural hand-shape template on a softly bent arm,
   with a 13-frame ease between rest and signing. Result: both hands are
   present in 100% of frames.
4. **Gap filling.** Hand-tracking dropouts of up to 12 frames are
   interpolated in wrist-relative coordinates, so the hand always stays
   attached to the wrist.
5. **Transitions** (ideas from ZurichNLP):
   - signs play 1.2× faster (citation forms are slow);
   - each join uses the closest pair of frames within 0.3 s of the boundary;
   - eased (cosine) in-betweens, sized by how far the hands travel;
   - lead-in from and lead-out to the rest pose;
   - light temporal smoothing.
6. **Renderers:**
   - `render2d.py`: an illustrated person with filled, outlined hands,
     knuckles and nails, and a face built from the face mesh (brows, eyes
     with irises, lips).
   - `render_human.py` + `mh_avatar.py`: **realistic** output. A CC0
     MakeHuman body, clothes, hair, eyes and brows, assembled from
     MakeHuman's data files with our own loader (no MakeHuman/MPFB code is
     executed). It has a 53-bone game-engine rig with all finger bones:
     - **Hands:** finger bones are aimed at the tracked finger landmarks, and
       the palm plane is matched so the hand twists correctly; the forearm
       takes half of the twist.
     - **Arms:** solved with **two-bone IK**. MediaPipe's monocular wrist
       depth was off by 4–5 shoulder widths, so depth is re-derived from the
       avatar's bone lengths. Hands are kept just in front of the chest and
       face, so contact signs like MY land on the body.
     - **Head:** turned to match the tracked face (split between neck and head).
   - `render3d.py`: a procedural "mannequin" made with Blender's Skin
     modifier. An intermediate experiment, kept for reference.

## 4. Known limitations / next steps

- **Lexicon and glossing.** Words are still looked up one by one in English
  word order. A real English→ASL gloss step (e.g. Claude with the Space's
  `asl_gloss.py` prompt), plus fingerspelling for unknown names and numbers,
  would help most with meaning. For example `$7` currently maps to the sign
  for "pm".
- **Facial expression on the realistic avatar.** The head turns, but the face
  stays neutral. MakeHuman ships expression morph targets
  (`data/targets/expression`) that could be driven from the tracked face mesh.
- **Clip licensing.** The sign videos come from Signing Savvy (copyrighted,
  watermarked). For anything public, rebuild the lexicon from licensed data.
- **Speed.** Realistic rendering takes ~25–30 min per 10 s on 4 CPU cores. A
  GPU would bring it to near real time; EEVEE would also work, with Mesa EGL
  installed.
