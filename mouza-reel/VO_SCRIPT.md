# Mouza Reel: VO recording script (real human voice, phone only)

**No AI voice. No TTS. No studio.** Record on your phone in the same room you'd film in.
Keep the stutters, the breath, the half-laugh. The animation follows your voice:
Yaniv's mouth is driven by the loudness of your recording, frame by frame.

## How to record

1. Phone voice-memo app, phone ~20 cm from your mouth, a normal room (not a closet, not the car).
2. Record **each line as its own file**. Do 3–5 takes per line and keep the most human one, not the cleanest.
3. Trim the silence at the start so the first sound is at 0.0 s. Leave the end alone.
4. Convert to WAV and drop into `public/vo/` with the exact name below:
   ```bash
   npx remotion ffmpeg -i "my-take.m4a" -ac 1 -ar 48000 public/vo/scene2-pain.wav
   ```
5. Open `npm run dev`, scrub, and adjust `offset` / word `cues` in `src/timing.ts` and `src/hooks.ts` if a word lands early or late.

Every line plays inside its scene and **is cut hard at the scene's end**. That keeps the edit tight.
If a take runs long, speak faster or trim a pause. Don't stretch the scene.

## Main video: Mouza11Attempts

| # | Window (s) | Frames | Start offset | File | Line (copy exactly) | Delivery |
|---|---|---|---|---|---|---|
| 1 | 0.0–2.8 | 0–84 | 0 f | `hook-A.wav` | "כן, אני מצויר. כי האמיתי פחד לצלם. כמוך." | Fast, embarrassed, laughing at yourself. Like you just got caught. The "כמוך." is a little jab with a smile. |
| 2 | 2.8–5.5 | 84–165 | 0 f | `scene2-pain.wav` | "אתם מכירים את זה, נכון? יושבים לצלם, ופתאום המוח ריק." | Slower. Eye contact. One friend, not an audience. Small pause after "נכון?". Let "ריק" go flat. |
| 3 | 5.5–9.0 | 165–270 | 4 f (0.13 s) | `scene3-solution.wav` | "אז עשיתי משהו אחר. כתבתי משפט אחד במוזה." | Calm, conspiratorial, "come here, let me show you." Slight lean in. |
| 4 | 9.0–13.0 | 270–390 | 2 f | `scene4-retake.wav` | "הנה, פה טעיתי. לא מצלמים הכול מחדש. רק את הקטע הזה." | Real-time, like you're pointing at your own screen. A bit surprised it works. A tiny laugh on "טעיתי" is gold. |
| 5 | 13.0–18.0 | 390–540 | 8 f (0.27 s) | `scene5-proof.wav` | "וזה מה שיצא. אותו בן אדם, אותו טלפון." | Proud but not salesy. Half-smile. A beat between the two halves. |
| 6 | 18.0–23.0 | 540–690 | 54 f (1.8 s) | `scene6-payoff.wav` | "אתם מצלמים. מוזה עורכת." | Quiet, confident. A fact, not a slogan. Low energy on purpose. The music ducks under you here. |
| 7 | 23.0–27.0 | 690–810 | 0 f | `scene7-loop.wav` | "שלחו את זה למי שתמיד אומר מחר אני מצלם. אתם יודעים מי." | Teasing, like you're pointing at someone. After "אתם יודעים מי." **laugh for real**, around 2.9 s in. The video cuts mid-laugh (that's the loop). |

Stutters are welcome. If you trip on a word in one take, try leaving it in. It's literally the product's story.

## Hook variants: scene 1 only (0–84 frames, 2.8 s)

| Composition | File | Line (copy exactly) | Delivery |
|---|---|---|---|
| `MouzaHookA` (confession) | `hook-A.wav` | "כן, אני מצויר. כי האמיתי פחד לצלם. כמוך." | Same file as the main video. |
| `MouzaHookB` (call-out) | `hook-B.wav` | "בעל עסק? תעצור רגע. זה יחסוך לך 19 יום של התכתבויות עם עורך." | Rapid-fire, like stopping someone in the street. **⚠ 13 words in 2.8 s ≈ 4.6 words/s, very tight.** Practice it; anything past 2.8 s is cut. |
| `MouzaHookC` (shame) | `hook-C.wav` | "אני שונא להצטלם. תראו מה יצא לי בכל זאת." | Grimace in the voice on "שונא", turn to proud-surprised on "בכל זאת". |
| `MouzaHookD` (anti-ad) | `hook-D.wav` | "אל תשלחו את זה לעורך שלכם. באמת, אל תשלחו." | Mock-serious, conspiratorial. A smile breaks through on the second "אל תשלחו." |
| `MouzaHookE` (question) | `hook-E.wav` | "כמה פעמים כבר אמרת 'מחר אני מצלם' ולא צילמת?" | Teasing, a raised eyebrow. Imitate the excuse voice on "מחר אני מצלם". |

## Sound bed (already generated, no stock)

- 0–2.8 s: **no music**, phone room tone only, plus a pop on every counter slam.
- 2.8 s: low warm piano enters under the voice. 9.0 s: lifts slightly.
- 13.0 s: drums and bass come in, music at full (clean, no "ad drop").
- 18.0 s: the finished video's music, then it ducks under "אתם מצלמים" at 20 s.
- On your final laugh (≈25.9 s) the music cuts to silence, then one soft pop at 26.8 s to loop.

Regenerate with `npm run sfx`, or replace any file in `public/music` / `public/sfx` with your own (same names).
