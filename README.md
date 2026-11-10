# autooverlay

Puts a logo or photo on your video at the moment you say a company's or person's name. It also writes Persian captions. Everything runs on your own computer, free.

```
your video ──► Whisper (open weights) ──► words + timestamps
                                              │
entities.yaml (names + Persian spellings) ───►│ matching
                                              ▼
                                         cues.csv  ◄── edit if you want
                                              │
                                    ffmpeg ──►  video_final.mp4  (+ captions.srt)
```

## One-time setup (Mac)

```bash
brew install ffmpeg            # needs Homebrew: https://brew.sh
cd autooverlay
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads the Whisper model (about 3 GB for `large-v3`). After that, it works offline.

## Every video

1. Put the logos you need in `assets/` (PNG with a transparent background looks best).
2. Make sure each name is in `entities.yaml`, with its usual Persian spellings as aliases.
3. Run:

```bash
python autooverlay.py run ~/Movies/oura.mov
```

This creates:

| File | What it is |
| --- | --- |
| `oura_final.mp4` | The video with cards, ready to post (or to trim in CapCut) |
| `oura_overlay/captions.srt` | Persian captions; import into CapCut, or add `--burn-captions` to bake them in |
| `oura_overlay/cues.csv` | The list of what appears when |
| `oura_overlay/transcript.json` | Saved transcript, so later runs skip transcription |

The terminal prints every card and the words it heard, for example:

```
  1.15-  3.05s  Oura          heard: اورا
  3.05-  5.55s  Apple Watch   heard: اپل‌واچ
```

## Fixing things (optional, quick)

| Problem | Fix |
| --- | --- |
| A wrong card appeared | Delete its row in `cues.csv`, then `python autooverlay.py render VIDEO` |
| A name was missed | Add its spelling (copy it from the "heard" text or `captions.srt`) to `aliases`, then `python autooverlay.py match VIDEO` and `render VIDEO` |
| Not sure which names Whisper heard | `python autooverlay.py suggest VIDEO` lists English-script words that have no entity yet |
| Too many false matches | Raise `threshold` in `entities.yaml` (e.g. 90) |
| Card covers your face | Change `card_y` (e.g. 0.05 for higher) |

Only `render` needs re-running after edits. Transcription is saved and doesn't run again unless you pass `--retranscribe`.

## How the matching works

- Persian text is normalized first: ي/ی and ك/ک are unified, diacritics and half-spaces are removed, and spacing is ignored. So `اپل واچ`, `اپل‌واچ` and `Apple Watch` all match.
- Common endings still match: `اپله`, `گوگلی`, `اوراشون`. ("ای" is left out on purpose, since `اپلای` means "apply".)
- Small spelling mistakes match on longer names (5+ letters), e.g. `استنفرود` → Stanford. Short names must match exactly, to avoid false hits.
- The longer name wins: "Apple Watch" beats "Apple".
- Each card stays 2.5–4 seconds, only one card shows at a time, and the same card waits 20 seconds before it can appear again.
- Your entity names are passed to Whisper as a hint, so it tends to spell them the way you wrote them.

## Notes

- **Speed:** `large-v3` gives the best Persian, and on a recent Mac it takes a few minutes for a 90-second clip. Use `--model small` for quick tests. On a machine with an NVIDIA GPU it picks it up automatically and is much faster.
- **iPhone HDR:** iPhones record HDR by default, which can look washed out after processing. Turn off HDR video in Settings → Camera → Record Video.
- **Images:** prefer official logos from companies' press or media kits, and for people use press photos rather than random search results.
- **Font:** `fonts/Vazirmatn-Bold.ttf` is the free Vazirmatn font (SIL Open Font License, see `fonts/OFL.txt`), used for Persian labels and captions.
