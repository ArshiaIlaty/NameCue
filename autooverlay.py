#!/usr/bin/env python3
"""
autooverlay: show a logo or photo on your video whenever you say a name.

Pipeline (everything runs locally and free):
  1. transcribe  Whisper (open weights, via faster-whisper) turns speech into
                 words with start/end times. Persian works.
  2. match       Each name in entities.yaml (with Persian and English spellings)
                 is looked up in the transcript, tolerant of spelling variants.
  3. cues.csv    A plain list of "show X from second A to B". Edit or delete
                 rows if you want; nothing else needs touching.
  4. render      ffmpeg places a card (logo/photo + optional label) on the video
                 at those times, with a short fade. Also writes captions.srt.

Commands:
  run      VIDEO   transcribe (if needed) + match + render, all in one go
  match    VIDEO   redo matching only (after you add names); overwrites cues.csv
  render   VIDEO   re-render from your edited cues.csv
  suggest  VIDEO   list English-script words in the transcript you have no entity for
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("Missing package: pip install pyyaml")

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Text normalization (Persian + English)
# --------------------------------------------------------------------------
_CHAR_MAP = {
    "ي": "ی", "ى": "ی", "ئ": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه",
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ؤ": "و",
}
for _i, _d in enumerate("۰۱۲۳۴۵۶۷۸۹"):
    _CHAR_MAP[_d] = str(_i)
for _i, _d in enumerate("٠١٢٣٤٥٦٧٨٩"):
    _CHAR_MAP[_d] = str(_i)
_CHAR_MAP = str.maketrans(_CHAR_MAP)
_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
_NONWORD = re.compile(r"[\W_]+", re.UNICODE)
_ARABIC_SCRIPT = re.compile(r"[\u0600-\u06FF]")

# Endings a Persian speaker glues onto a name: "اپله", "گوگلی", "اورارو"...
# ("ای" is left out on purpose: "اپل" + "ای" is "اپلای" = "apply".)
FA_SUFFIXES = {"ه", "ی", "یه", "ها", "های", "هایی", "ش", "شون", "مون",
               "تون", "م", "ت", "رو", "و", "هم"}
EN_SUFFIXES = {"s"}


def norm(text: str) -> str:
    """Lowercase, unify Arabic/Persian letters, drop diacritics, spaces, ZWNJ, punctuation."""
    t = unicodedata.normalize("NFKC", text).translate(_CHAR_MAP)
    t = _DIACRITICS.sub("", t)
    return _NONWORD.sub("", t.lower())


def is_arabic_script(text: str) -> bool:
    return bool(_ARABIC_SCRIPT.search(text))


# --------------------------------------------------------------------------
# Entities
# --------------------------------------------------------------------------
@dataclass
class Entity:
    name: str
    image: Path | None
    label: str | None
    aliases: list[str]
    norm_aliases: list[tuple[str, int, bool]] = field(default_factory=list)  # (norm, n_words, persian?)


DEFAULT_SETTINGS = {
    "language": "fa",           # Whisper language code ("en" for English videos)
    "model": "large-v3",        # Whisper size: tiny/base/small/medium/large-v3
    "threshold": 86,            # fuzzy match strictness, 0-100 (higher = stricter)
    "cooldown": 20.0,           # seconds before the same card can appear again
    "min_duration": 2.5,        # seconds a card stays on screen, at least
    "max_duration": 4.0,        # ...and at most
    "lead": 0.15,               # show the card this many seconds before the word
    "card_width": 0.42,         # max card width as a fraction of video width
    "card_y": 0.09,             # card's top edge as a fraction of video height
    "font": None,               # path to a .ttf for labels (e.g. Vazirmatn)
}


def load_entities(path: Path) -> tuple[list[Entity], dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    settings = {**DEFAULT_SETTINGS, **(data.get("settings") or {})}
    entities = []
    for raw in data.get("entities") or []:
        name = str(raw["name"]).strip()
        img = raw.get("image")
        img_path = (path.parent / img).resolve() if img else None
        aliases = [name] + [str(a) for a in (raw.get("aliases") or [])]
        e = Entity(name=name, image=img_path, label=raw.get("label"), aliases=aliases)
        seen = set()
        for a in aliases:
            n = norm(a)
            if n and n not in seen:
                seen.add(n)
                e.norm_aliases.append((n, max(1, len(a.split())), is_arabic_script(a)))
        entities.append(e)
    if not entities:
        sys.exit(f"No entities found in {path}")
    return entities, settings


# --------------------------------------------------------------------------
# 1. Transcribe
# --------------------------------------------------------------------------
def load_audio(video: Path):
    """Decode the soundtrack with ffmpeg (16 kHz mono float), as Whisper expects."""
    import numpy as np
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found. Mac: brew install ffmpeg")
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(video), "-vn",
                        "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                       capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.int16).astype(np.float32) / 32768.0

def transcribe(video: Path, out_json: Path, model_name: str, language: str,
               entities: list[Entity], device: str = "auto") -> dict:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("Missing package: pip install faster-whisper")

    compute = "int8"
    if device in ("auto", "cuda"):
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() > 0:
                device, compute = "cuda", "float16"
            else:
                device = "cpu"
        except Exception:
            device = "cpu"

    print(f"[transcribe] Whisper {model_name} on {device} ({compute}); "
          "the first run downloads the model, later runs are offline.")
    model = WhisperModel(model_name, device=device, compute_type=compute)
    # A hint listing your names nudges Whisper to spell them the way you wrote them.
    names = "، ".join(e.name for e in entities[:40])
    prompt = f"در این ویدیو درباره‌ی {names} صحبت می‌کنم." if language == "fa" else \
             f"In this video I talk about {names}."
    segments, info = model.transcribe(
        load_audio(video), language=language, word_timestamps=True, vad_filter=True,
        beam_size=5, initial_prompt=prompt,
    )
    words, segs = [], []
    for s in segments:
        segs.append({"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip()})
        for w in s.words or []:
            words.append({"w": w.word.strip(), "start": round(w.start, 3), "end": round(w.end, 3)})
        print(f"  {s.start:6.1f}s  {s.text.strip()}")
    result = {"language": info.language, "duration": round(info.duration, 3),
              "segments": segs, "words": words}
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[transcribe] {len(words)} words -> {out_json}")
    return result


# --------------------------------------------------------------------------
# 2. Match names in the transcript
# --------------------------------------------------------------------------
@dataclass
class Match:
    entity: Entity
    i: int          # first word index
    n: int          # number of words
    score: float
    alias_len: int


def _score(cand: str, alias: str, persian: bool, threshold: float) -> float:
    if not cand:
        return 0.0
    if cand == alias:
        return 100.0
    if cand.startswith(alias) and cand[len(alias):] in (FA_SUFFIXES if persian else EN_SUFFIXES):
        return 98.0
    if len(alias) >= 5 and abs(len(cand) - len(alias)) <= 3:  # no fuzzy for short names
        from rapidfuzz import fuzz
        s = fuzz.ratio(cand, alias)
        return s if s >= threshold else 0.0
    return 0.0


def find_matches(words: list[dict], entities: list[Entity], threshold: float) -> list[Match]:
    max_n = max(n for e in entities for _, n, _ in e.norm_aliases) + 1
    tokens = [norm(w["w"]) for w in words]
    found: list[Match] = []
    for i in range(len(words)):
        if not tokens[i]:
            continue
        for n in range(1, max_n + 1):
            if i + n > len(words):
                break
            cand = "".join(tokens[i:i + n])
            head = "".join(tokens[i:i + n - 1])      # all but the last word
            tail = "".join(tokens[i + 1:i + n])      # all but the first word
            for e in entities:
                for alias, a_words, persian in e.norm_aliases:
                    if abs(n - a_words) > 1:
                        continue
                    if n > 1 and max(len(head), len(tail)) >= len(alias) - 1:
                        continue  # an extra word (e.g. "اورا" + "و") is not part of the name
                    s = _score(cand, alias, persian, threshold)
                    if s:
                        found.append(Match(e, i, n, s, len(alias)))
    # Keep the best non-overlapping matches: the one covering more words wins
    # ("Apple Watch" beats "Apple"), then the closer spelling.
    found.sort(key=lambda m: (-m.n, -m.score, -m.alias_len, m.i))
    used: set[int] = set()
    kept = []
    for m in found:
        span = set(range(m.i, m.i + m.n))
        if span & used:
            continue
        used |= span
        kept.append(m)
    kept.sort(key=lambda m: m.i)
    return kept


@dataclass
class Cue:
    start: float
    end: float
    entity: str
    heard: str = ""
    score: float = 100.0


def build_cues(matches: list[Match], words: list[dict], s: dict, duration: float) -> list[Cue]:
    cues: list[Cue] = []
    last_shown: dict[str, float] = {}
    for m in matches:
        t0 = max(0.0, words[m.i]["start"] - s["lead"])
        t_word_end = words[m.i + m.n - 1]["end"]
        name = m.entity.name
        if name in last_shown and t0 - last_shown[name] < s["cooldown"]:
            continue
        t1 = min(t0 + s["max_duration"], max(t0 + s["min_duration"], t_word_end + 1.0))
        if duration:
            t1 = min(t1, duration)
        # One card at a time: shorten the previous card if this one starts first.
        if cues and cues[-1].end > t0:
            if t0 - cues[-1].start < 1.0:
                continue  # previous card barely started; keep it, skip this one
            cues[-1].end = t0
        heard = " ".join(w["w"] for w in words[m.i:m.i + m.n])
        cues.append(Cue(round(t0, 2), round(t1, 2), name, heard, round(m.score)))
        last_shown[name] = t0
    return cues


def write_cues(cues: list[Cue], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:  # -sig: Excel shows Persian correctly
        w = csv.writer(f)
        w.writerow(["start", "end", "entity", "heard", "score"])
        for c in cues:
            w.writerow([f"{c.start:.2f}", f"{c.end:.2f}", c.entity, c.heard, int(c.score)])


def read_cues(path: Path) -> list[Cue]:
    cues = []
    with path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            ent = (row.get("entity") or "").strip()
            if not ent or ent.startswith("#"):
                continue
            try:
                cues.append(Cue(float(row["start"]), float(row["end"]), ent, row.get("heard", "")))
            except (TypeError, ValueError):
                print(f"[render] skipping unreadable row: {row}")
    cues.sort(key=lambda c: c.start)
    return cues


# --------------------------------------------------------------------------
# Captions (.srt) from the word timings
# --------------------------------------------------------------------------
def _ts(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def write_srt(words: list[dict], path: Path, max_words: int = 7, max_secs: float = 3.0) -> None:
    lines, cur = [], []
    for w in words:
        if cur and (len(cur) >= max_words or w["end"] - cur[0]["start"] > max_secs):
            lines.append(cur)
            cur = []
        cur.append(w)
        if len(cur) >= 3 and w["w"].endswith((".", "؟", "?", "!", "،", ",")):
            lines.append(cur)
            cur = []
    if cur:
        lines.append(cur)
    out = []
    for k, ln in enumerate(lines, 1):
        out.append(f"{k}\n{_ts(ln[0]['start'])} --> {_ts(ln[-1]['end'])}\n"
                   f"{' '.join(x['w'] for x in ln)}\n")
    path.write_text("\n".join(out), encoding="utf-8")


# --------------------------------------------------------------------------
# Cards (the picture that appears on screen)
# --------------------------------------------------------------------------
_FONT_CANDIDATES = [
    HERE / "fonts" / "Vazirmatn-Bold.ttf",
    HERE / "fonts" / "Vazirmatn-Regular.ttf",
    Path("/Library/Fonts/Arial Unicode.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/System/Library/Fonts/Supplemental/GeezaPro.ttc"),
    Path("C:/Windows/Fonts/tahoma.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
]


def _load_font(size: int, font_path: str | None):
    from PIL import ImageFont
    paths = ([Path(font_path)] if font_path else []) + _FONT_CANDIDATES
    for p in paths:
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except OSError:
                continue
    return ImageFont.load_default(size=size)


def _shape(text: str) -> str:
    """Right-to-left shaping for Persian labels when Pillow lacks libraqm."""
    from PIL import features
    if not is_arabic_script(text) or features.check("raqm"):
        return text
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        return get_display(arabic_reshaper.reshape(text))
    except ImportError:
        print("[cards] Persian label may look broken: pip install arabic-reshaper python-bidi")
        return text


def _open_image(path: Path):
    from PIL import Image
    if path.suffix.lower() == ".svg":
        try:
            import cairosvg
        except ImportError:
            print(f"[cards] {path.name} is SVG; pip install cairosvg, or export it as PNG")
            return None
        import io
        png = cairosvg.svg2png(url=str(path), output_width=1200)
        return Image.open(io.BytesIO(png)).convert("RGBA")
    return Image.open(path).convert("RGBA")


def make_card(e: Entity, out: Path, W: int, H: int, s: dict) -> Path:
    from PIL import Image, ImageDraw, ImageFilter
    pad = int(0.025 * W)
    radius = int(0.03 * W)
    max_w = int(s["card_width"] * W) - 2 * pad
    max_h = int(0.13 * H)

    img = None
    if e.image:
        if e.image.exists():
            img = _open_image(e.image)
        else:
            print(f"[cards] image not found for {e.name}: {e.image} (using a text card)")
    label = e.label if e.label is not None else (None if img else e.name)

    # Bigger text when the card is only a name, smaller when it sits under a picture.
    font = _load_font(int((0.024 if img else 0.034) * H), s.get("font"))
    text = _shape(label) if label else None
    if text:
        tb = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
    else:
        tw = th = 0

    if img:
        img.thumbnail((max_w, max_h), Image.LANCZOS)
        if img.width < max_w // 2 and img.height < max_h // 2:  # upscale tiny logos a bit
            scale = min(max_w / img.width, max_h / img.height) * 0.6
            img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    iw, ih = (img.size if img else (0, 0))
    gap = int(0.012 * H) if (img and text) else 0
    cw = max(iw, tw) + 2 * pad
    ch = ih + gap + th + 2 * pad + (int(th * 0.3) if text else 0)

    shadow_off = int(0.006 * H)
    canvas = Image.new("RGBA", (cw + 4 * shadow_off, ch + 4 * shadow_off), (0, 0, 0, 0))
    sh = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle(
        (2 * shadow_off, 2 * shadow_off + shadow_off, 2 * shadow_off + cw, 3 * shadow_off + ch),
        radius, fill=(0, 0, 0, 90))
    canvas = Image.alpha_composite(canvas, sh.filter(ImageFilter.GaussianBlur(shadow_off)))
    d = ImageDraw.Draw(canvas)
    x0, y0 = 2 * shadow_off, 2 * shadow_off
    d.rounded_rectangle((x0, y0, x0 + cw, y0 + ch), radius, fill=(255, 255, 255, 240))
    if img:
        canvas.alpha_composite(img, (x0 + (cw - iw) // 2, y0 + pad))
    if text:
        ty = y0 + pad + ih + gap
        d.text((x0 + (cw - tw) // 2 - tb[0], ty - tb[1]), text, font=font, fill=(20, 20, 20, 255))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


# --------------------------------------------------------------------------
# 3. Render with ffmpeg
# --------------------------------------------------------------------------
def probe(video: Path) -> tuple[int, int, str]:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,color_transfer:stream_tags=rotate:stream_side_data=rotation",
         "-of", "json", str(video)], capture_output=True, text=True, check=True)
    st = json.loads(r.stdout)["streams"][0]
    W, H = int(st["width"]), int(st["height"])
    rot = 0
    for sd in st.get("side_data_list") or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    rot = rot or int((st.get("tags") or {}).get("rotate", 0))
    if abs(rot) % 180 == 90:
        W, H = H, W
    return W, H, st.get("color_transfer", "")


def render(video: Path, cues: list[Cue], entities: list[Entity], s: dict,
           work: Path, out: Path, srt: Path | None, burn: bool) -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found. Mac: brew install ffmpeg")
    W, H, transfer = probe(video)
    if transfer in ("arib-std-b67", "smpte2084"):
        print("[render] Note: this is HDR video; colors may look washed out. "
              "Turning off HDR video in your phone's camera settings avoids this.")
    by_name = {e.name: e for e in entities}
    cards: dict[str, Path] = {}
    inputs = ["-i", str(video.resolve())]
    filters, last, k = [], "[0:v]", 0
    y = int(s["card_y"] * H)
    for c in cues:
        e = by_name.get(c.entity)
        if not e:
            print(f"[render] '{c.entity}' is not in entities.yaml; skipping")
            continue
        if c.entity not in cards:
            safe = re.sub(r"[^\w-]+", "_", c.entity)
            cards[c.entity] = make_card(e, work / "cards" / f"{safe}.png", W, H, s)
        dur = round(c.end - c.start, 3)
        if dur <= 0.2:
            continue
        k += 1
        fade = round(min(0.25, dur / 4), 3)
        inputs += ["-loop", "1", "-framerate", "30", "-t", f"{dur}", "-i", str(cards[c.entity].resolve())]
        filters.append(
            f"[{k}:v]format=rgba,fade=t=in:st=0:d={fade}:alpha=1,"
            f"fade=t=out:st={dur - fade:.3f}:d={fade}:alpha=1,setpts=PTS+{c.start}/TB[c{k}]")
        filters.append(
            f"{last}[c{k}]overlay=x=(W-w)/2:y={y}:eof_action=pass:"
            f"enable='between(t,{c.start},{c.end})'[v{k}]")
        last = f"[v{k}]"

    if burn and srt and srt.exists():
        tmp = work / "burn.srt"
        shutil.copyfile(srt, tmp)
        fontsdir = HERE / "fonts"
        fd = f":fontsdir={fontsdir.as_posix()}" if fontsdir.exists() else ""
        style = ("FontName=Vazirmatn,FontSize=15,Bold=1,PrimaryColour=&H00FFFFFF,"
                 "OutlineColour=&H80000000,BorderStyle=1,Outline=2,Shadow=0,MarginV=40")
        filters.append(f"{last}subtitles=filename=burn.srt{fd}:force_style='{style}'[vsub]")
        last = "[vsub]"
    if not filters:
        filters.append("[0:v]null[vout]")
        last = "[vout]"

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-stats", *inputs,
           "-filter_complex", ";".join(filters), "-map", last, "-map", "0:a?",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out.resolve())]
    print(f"[render] {k} card(s) -> {out}")
    subprocess.run(cmd, check=True, cwd=work)
    print(f"[render] done: {out}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def paths_for(video: Path) -> dict[str, Path]:
    work = video.parent / f"{video.stem}_overlay"
    work.mkdir(exist_ok=True)
    return {"work": work, "transcript": work / "transcript.json", "cues": work / "cues.csv",
            "srt": work / "captions.srt", "out": video.parent / f"{video.stem}_final.mp4"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "match", "render", "suggest"])
    ap.add_argument("video", type=Path)
    ap.add_argument("--entities", type=Path, default=HERE / "entities.yaml")
    ap.add_argument("--model", help="Whisper model (default from entities.yaml settings)")
    ap.add_argument("--lang", help="language code, e.g. fa or en")
    ap.add_argument("--device", default="auto", help="auto, cpu or cuda")
    ap.add_argument("--retranscribe", action="store_true", help="ignore the saved transcript")
    ap.add_argument("--burn-captions", action="store_true", help="also burn Persian captions into the video")
    a = ap.parse_args()

    if not a.video.exists():
        sys.exit(f"Video not found: {a.video}")
    entities, s = load_entities(a.entities)
    p = paths_for(a.video)

    if a.command in ("run", "match", "suggest"):
        if a.retranscribe or not p["transcript"].exists():
            if a.command == "suggest":
                sys.exit("No transcript yet; run: python autooverlay.py run VIDEO")
            tr = transcribe(a.video, p["transcript"], a.model or s["model"],
                            a.lang or s["language"], entities, a.device)
        else:
            tr = json.loads(p["transcript"].read_text(encoding="utf-8"))
            print(f"[transcribe] using saved {p['transcript']} (--retranscribe to redo)")
        words = tr["words"]

        if a.command == "suggest":
            known = {al for e in entities for al, _, _ in e.norm_aliases}
            counts: dict[str, int] = {}
            for w in words:
                t = w["w"].strip(".,!?؟،:;\"'()")
                if re.search(r"[A-Za-z]", t) and norm(t) not in known:
                    counts[t] = counts.get(t, 0) + 1
            print("English-script words with no entity yet (add the ones you want to entities.yaml):")
            for t, n in sorted(counts.items(), key=lambda x: -x[1]):
                print(f"  {t}  ×{n}")
            return

        write_srt(words, p["srt"])
        matches = find_matches(words, entities, s["threshold"])
        cues = build_cues(matches, words, s, tr.get("duration") or 0)
        write_cues(cues, p["cues"])
        print(f"[match] {len(cues)} card(s) -> {p['cues']}")
        for c in cues:
            print(f"  {c.start:6.2f}-{c.end:6.2f}s  {c.entity:<20} heard: {c.heard}  ({int(c.score)})")
        if a.command == "match":
            print("Edit cues.csv if you like, then: python autooverlay.py render VIDEO")
            return

    if not p["cues"].exists():
        sys.exit("No cues.csv yet; run: python autooverlay.py run VIDEO")
    render(a.video, read_cues(p["cues"]), entities, s, p["work"], p["out"],
           p["srt"], a.burn_captions)


if __name__ == "__main__":
    main()
