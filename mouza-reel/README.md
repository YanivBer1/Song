# Mouza: "11 Attempts". 100% animated viral Reel (Remotion)

27 s · 810 frames · 1080×1920 · 30 fps · Hebrew RTL · real human VO (yours).

Yaniv, the animated founder, confesses he's animated **because the real him was too scared to film this.
Just like you.** Then: the freeze, one sentence in Mouza, retake only the word you messed up,
before/after, the finished video, and a send line that names one specific friend. It ends mid-laugh,
on the frame-0 framing, so it loops.

---

## 1. Run it

```bash
cd mouza-reel
npm install
npm run sfx        # generates music + SFX from code into public/ (already committed, re-run any time)
npm run dev        # opens Remotion Studio → http://localhost:3000
```

In Studio, pick a composition in the left sidebar:

| Composition | What |
|---|---|
| `Mouza11Attempts` | **Main Reel** |
| `MouzaHookA` … `MouzaHookE` | A/B Trial Reels: same body, different 0–2.8 s hook |
| `MouzaCover` | Cover (export frame 24) |

Scrub the timeline; each scene is a named `<Sequence>` ("1 HOOK", "2 PAIN", …) so you can see the cut points.
Drop your phone VO into `public/vo/` (see **VO_SCRIPT.md**) and Studio picks it up on reload. Before you record,
Yaniv's mouth flaps on the scripted timing so you can preview.

## 2. Render

Main video (exact command from the brief; bitrate/AAC settings come from `remotion.config.ts`):

```bash
npx remotion render Mouza11Attempts out/mouza.mp4 --codec h264 --image-format jpeg --scale 1
```

Each hook variant:

```bash
npx remotion render MouzaHookA out/MouzaHookA.mp4 --codec h264 --image-format jpeg --scale 1
npx remotion render MouzaHookB out/MouzaHookB.mp4 --codec h264 --image-format jpeg --scale 1
npx remotion render MouzaHookC out/MouzaHookC.mp4 --codec h264 --image-format jpeg --scale 1
npx remotion render MouzaHookD out/MouzaHookD.mp4 --codec h264 --image-format jpeg --scale 1
npx remotion render MouzaHookE out/MouzaHookE.mp4 --codec h264 --image-format jpeg --scale 1
```

Cover still:

```bash
npx remotion still MouzaCover out/cover.png --frame=24
```

All 6 videos and the cover, plus loudness normalization, in one go:

```bash
npm run render:hooks          # → out/final/*.mp4 (−14 LUFS, AAC 128k, 48 kHz) + out/cover.png
```

**Export settings** (`remotion.config.ts` + `scripts/loudnorm.sh`): H.264 MP4, 1080×1920, 30 fps, 10 Mbps
video (inside the 8–12 Mbps target), yuv420p, AAC 128 kbps, 48 kHz, two-pass EBU R128 loudnorm to **−14 LUFS**,
true peak −1.5 dBTP. Upload the files from `out/final/`.

**No internet / behind a proxy?** Fonts come from `@remotion/google-fonts` by default. Prefix any command with
`REMOTION_LOCAL_FONTS=1` to use the bundled Heebo files in `public/fonts/` instead.

## 3. Project layout

```
remotion.config.ts            codec, jpeg, 10M video, 128k audio
src/index.ts                  registerRoot
src/Root.tsx                  Mouza11Attempts + MouzaHookA–E + MouzaCover
src/MouzaVideo.tsx            the 7 scenes as <Sequence>s + soundtrack
src/timing.ts                 ★ VO timing map: scene frames, VO offsets, talk windows, music automation
src/hooks.ts                  ★ the 5 hook variants (text, word cues, VO, counter, jump-cut list)
src/voice.tsx                 plays your VO + drives Yaniv's mouth from its loudness
src/Soundtrack.tsx            room tone, music bed automation, all SFX hits
src/theme.ts                  palette, Heebo, safe zone, spring presets
src/scenes/Scene1Hook.tsx … Scene7Loop.tsx
src/scenes/shared.tsx         YanivShot camera, B-roll, finished-edit, raw-take, thought cloud, grain
src/components/
  CharacterYaniv.tsx          the brand character (10 expressions, point / phone / slump poses, line boil)
  Room.tsx                    his messy room (cobwebbed ring light, "מחר" sticky note)
  PhoneMockup.tsx             phone drawn in code
  KineticText.tsx             word-by-word headline (yellow / white / gray-40%)
  CaptionWordByWord.tsx       auto-caption style inside the finished edit
  Counter.tsx                 "ניסיון 1 → 7 → 11" sticker slams (+ variant icons)
  SplitBeforeAfter.tsx        before/after split
  Timer.tsx                   25:00 → 00:00 in 2 s
  CoverFrame.tsx              cover still
  DoodleArrow.tsx, Doodles.tsx   hand-drawn arrows, circles, underlines, X, bursts
  HandheldShake.tsx           2–4 px phone wobble + 2.5D parallax layers
  MouzaLogo.tsx               mark + wordmark
scripts/make-audio.mjs        synthesizes music + SFX (no samples)
scripts/render-all.sh         render all + loudnorm
scripts/loudnorm.sh           −14 LUFS two-pass
VO_SCRIPT.md                  what to record, with timing and delivery notes
POSTING.md                    caption, hashtags, first comment
```

## 4. Timing checklist (frames @ 30 fps)

| Scene | Brief | Built | ✔ What's on screen |
|---|---|---|---|
| 1 HOOK | 0.0–2.8 s · 0–84 | `SCENES.hook` 0–84 | Hard open, no intro/logo. Frame 0 already shows Yaniv frozen, the "ניסיון 1" sticker and the headline. Counter slams **ניסיון 1 @0 → ניסיון 7 @20 → ניסיון 11 @40** (all inside 2 s), each with a spring overshoot, punch zoom, white flash and pop SFX. Jump cuts @0/9/20/30/40/51/63/74 (every 0.3–0.4 s). Text "אני לא אמיתי. אני מצויר." word-by-word, "מצויר." circled. No music, room tone only. |
| 2 PAIN | 2.8–5.5 s · 84–165 | 84–165 | Same Yaniv; shoulders drop (spring @86). Thought cloud pops @92: phone with red X, then the brain drains to a dashed blank on "המוח ריק" (@142). Text "כל בעל עסק מכיר את זה." Piano bed enters @84–96 at low level. |
| 3 SOLUTION | 5.5–9.0 s · 165–270 | 165–270 | Over-the-shoulder phone, Mouza app drawn in code. Finger types one sentence @173–207 (tap SFX), taps "כתוב לי תסריט" @215, script pops line by line @223/232/241. Text "כותבים משפט אחד." |
| 4 RETAKE | 9.0–13.0 s · 270–390 | 270–390 | Teleprompter. Yaniv trips on "הידיים" @292 → red wobbly underline, wince. Timeline segments @298, doodle arrow to the button @310, tap "צלם שוב רק את הקטע הזה" @332 (ding). **Only segment 3** re-records @336–370 → ✓. Text "טעית? מצלמים רק את הקטע הזה." Music lifts @270–282. |
| 5 BEFORE/AFTER | 13.0–18.0 s · 390–540 | 390–540 | Split: left "לפני" gray, frozen, "…" bubble, flat waveform. Right "אחרי" color, word-by-word captions, B-roll (hands @430–454, shop @482–506), music notes. Timer 25:00→00:00 @410–470 then stops, "25 דקות." stamp. "=" doodle @448 ("same person, same phone"). Music to full @390–410. |
| 6 PAYOFF | 18.0–23.0 s · 540–690 | 540–690 | The "after" half expands to full screen; the finished edit plays (captions word-by-word, B-roll @590–616 and @636–660). PiP of Yaniv in his room watching it on his phone: smile → surprised → proud. Text "הסרטון הזה. בלי עורך. בלי צלם." Music full, ducks under VO @590–600. |
| 7 LOOP + SEND | 23.0–27.0 s · 690–810 | 690–810 | Same close-up as scene 1. Points at camera, paper-plane "send" doodle, jump-cut in on "אתם יודעים מי", laugh @778. Music cuts on the laugh; soft pop @805. Last frame = frame-0 framing (`LOOP_FRAMING`). Text "שלחו את זה למי שתמיד אומר 'מחר אני מצלם.'" Small Mouza wordmark bottom-center of the safe area @760. |

Total: **810 frames = 27.0 s.** Variants change only frames 0–83 (text, VO file, counter, cut list); 84–810 are identical.

**Safe zone:** all text and stickers sit between y 269 (top 14%) and y 1248 (bottom 35%), x 65–1015 (6% sides).
Only drawing (Yaniv's shirt, the phone body, B-roll scenery) goes under the Instagram UI.

**Hebrew/RTL:** every text container has `dir="rtl"`; words are separate `unicode-bidi: isolate` spans inside an
RTL flex row, so the first word sits on the right and `?`, `.`, `:` and `'…'` stay attached to their own word on the
correct side. Checked visually on rendered frames, including "טעית?", "על מה הסרטון?", "עליי:", "בעל עסק?", "'מחר אני מצלם.'" and "800 ₪".

## 5. Confirmations

- **100% code-generated visuals.** Every pixel is SVG/HTML/CSS drawn in this repo: Yaniv, the room, phones, app
  screens, B-roll (hands, shop), doodles, logo. No footage, no presenter, no webcam, no stock, no photos, no
  images of any kind, no AI-generated humans. Yaniv is a flat doodle cartoon, deliberately not photorealistic.
- **Lo-fi on purpose:** hand-drawn line boil re-rolls every 4 frames, marker doodles re-jitter, 2–4 px handheld
  shake, phone-cam grain, mouth animated on twos, crooked stickers. It shouldn't look like a corporate explainer.
- **Audio:** music and SFX are synthesized by `scripts/make-audio.mjs` (no samples, no stock). **The voice is designed
  for a real human phone recording, not AI TTS.** There is no TTS anywhere in the project. Your files go in
  `public/vo/`, and the mouth sync reads your actual recording, stutters, laughs and pauses included.

## 6. Tuning after you record

- Word lands early or late → edit the `cues` arrays (headline highlights) in the scene file or `src/hooks.ts`.
- Whole line late or early → `offset` in `src/timing.ts` (`VO`) or `src/hooks.ts` (`vo.offset`).
- Laugh moment → `LOOP_LAUGH_AT` in `src/timing.ts` (music cut follows automatically).
- Music levels → `MUSIC_KEYS` in `src/timing.ts`.
- Want "before" on the right (Hebrew reading order)? `beforeSide="right"` in `src/scenes/Scene5Proof.tsx`.
