"""showcase.py – visual demo of sprite-rig capabilities.

Demonstrates morphing, mutations, blend sweeps, and feature-channel
variations.  Run with::

    python showcase.py

Outputs
-------
sprite_showcase.png
    A labelled contact sheet (4 rows × N columns).

sprite_showcase.gif
    An animated GIF cycling through the full parameter space.
"""

import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from demo import build_demo_rig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SIZE = 96          # sprite canvas (pixels)
SCALE = 4          # upscale for display
COLS = 9           # sprites per row in the contact sheet
GIF_FRAMES = 36    # frames in the animated GIF
GIF_DELAY = 80     # ms between GIF frames

LABEL_H = 20       # pixel height reserved for row labels
THUMB = SIZE * SCALE

# Background colour for the contact sheet
SHEET_BG = (30, 30, 40, 255)
# Transparent background replaces the black canvas background for display
SPRITE_BG = (0, 0, 0, 0)      # transparent
SPRITE_FG_NONE = (30, 30, 40, 255)   # matches sheet background (hidden)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _render_thumb(rig, part_state, feature_state, scale=SCALE) -> Image.Image:
    """Render one sprite and return an upscaled Pillow RGBA image."""
    arr = rig.render(part_state=part_state, feature_state=feature_state)
    img = Image.fromarray(arr, mode="RGBA")

    # Make the "empty" (zero-alpha) pixels transparent so the sheet BG shows
    r, g, b, a = img.split()
    # Pixels that are pure black (not painted) → transparent
    black = np.array(img)
    mask = (black[:, :, 0] == 0) & (black[:, :, 1] == 0) & (black[:, :, 2] == 0)
    alpha = np.array(a)
    alpha[mask] = 0
    img.putalpha(Image.fromarray(alpha.astype(np.uint8)))

    return img.resize((THUMB, THUMB), Image.NEAREST)


def _label(draw, x, y, text, color=(220, 220, 220)):
    """Draw small text, falling back gracefully if no truetype available."""
    try:
        font = ImageFont.load_default(size=12)
    except TypeError:
        font = ImageFont.load_default()
    draw.text((x, y), text, fill=color, font=font)


# ---------------------------------------------------------------------------
# Build rows
# ---------------------------------------------------------------------------

def _sweep(rig, vary_key, fixed_state, feature_state, n=COLS):
    """Return *n* (label, thumb) pairs sweeping vary_key from 0 → 1."""
    ts = [i / (n - 1) for i in range(n)]
    results = []
    for t in ts:
        ps = dict(fixed_state)
        ps[vary_key] = t
        thumb = _render_thumb(rig, ps, feature_state)
        results.append((f"{vary_key}={t:.2f}", thumb))
    return results


def _feature_sweep(rig, part_state, channel_key, n=COLS):
    """Return *n* (label, thumb) pairs sweeping a feature-channel strength."""
    ts = [i / (n - 1) for i in range(n)]
    results = []
    for t in ts:
        fs = {channel_key: t}
        thumb = _render_thumb(rig, part_state, fs)
        results.append((f"f0={t:.2f}", thumb))
    return results


def _random_mutations(rig, seed=42, n=COLS):
    """Return *n* (label, thumb) pairs with random parameter combos."""
    rng = random.Random(seed)
    results = []
    for i in range(n):
        p0 = round(rng.random(), 2)
        p1 = round(rng.random(), 2)
        f0 = round(rng.random(), 2)
        ps = {"p0": p0, "p1": p1}
        fs = {"f0": f0}
        thumb = _render_thumb(rig, ps, fs)
        results.append((f"#{i+1}", thumb))
    return results


# ---------------------------------------------------------------------------
# Assemble contact sheet
# ---------------------------------------------------------------------------

def build_contact_sheet(rig) -> Image.Image:
    rows = [
        ("Body morph  (p0: rect → ellipse, head fixed at 0.5, dots on)",
         _sweep(rig, "p0", {"p1": 0.5}, {"f0": 1.0})),

        ("Head morph  (p1: small → large circle, body fixed at 0.5, dots on)",
         _sweep(rig, "p1", {"p0": 0.5}, {"f0": 1.0})),

        ("Dot strength sweep  (f0: 0 → 1, p0=0.5, p1=0.5)",
         _feature_sweep(rig, {"p0": 0.5, "p1": 0.5}, "f0")),

        ("Random freaks & mutations  (p0, p1, f0 all randomised)",
         _random_mutations(rig)),
    ]

    row_h = LABEL_H + THUMB
    total_h = len(rows) * row_h
    total_w = COLS * THUMB

    sheet = Image.new("RGBA", (total_w, total_h), SHEET_BG)
    draw = ImageDraw.Draw(sheet)

    for r_idx, (title, thumbs) in enumerate(rows):
        y_top = r_idx * row_h
        # Row label
        _label(draw, 4, y_top + 3, title, color=(255, 220, 80))
        # Sprites
        for c_idx, (lbl, thumb) in enumerate(thumbs):
            x = c_idx * THUMB
            y = y_top + LABEL_H
            sheet.paste(thumb, (x, y), mask=thumb)
            _label(draw, x + 2, y + THUMB - LABEL_H, lbl)

    return sheet


# ---------------------------------------------------------------------------
# Animated GIF
# ---------------------------------------------------------------------------

def build_gif(rig, n_frames=GIF_FRAMES) -> list:
    """Return a list of P-mode PIL frames for the animated GIF.

    The animation cycles through the full (p0, p1, f0) parameter space in
    a continuous loop, creating an organic morphing effect.
    """
    frames = []
    for i in range(n_frames):
        phase = i / n_frames  # 0 → 1

        # Three independent sinusoidal sweeps so the animation feels organic
        p0  = (math.sin(2 * math.pi * phase * 1.0 + 0.00) + 1) / 2
        p1  = (math.sin(2 * math.pi * phase * 1.5 + 1.00) + 1) / 2
        f0  = (math.sin(2 * math.pi * phase * 2.0 + 2.50) + 1) / 2

        arr = rig.render(
            part_state={"p0": p0, "p1": p1},
            feature_state={"f0": f0},
        )
        img = Image.fromarray(arr, mode="RGBA").resize(
            (THUMB, THUMB), Image.NEAREST
        )

        # Convert to palette mode (required for GIF)
        frames.append(img.convert("RGBA"))

    return frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Building demo rig…")
    rig = build_demo_rig(size=SIZE)

    print(f"Rendering contact sheet ({COLS} × 4 sprites)…")
    sheet = build_contact_sheet(rig)
    sheet.save("sprite_showcase.png")
    print(f"  → sprite_showcase.png  ({sheet.width}×{sheet.height}px)")

    print(f"Rendering animated GIF ({GIF_FRAMES} frames)…")
    frames = build_gif(rig)
    frames[0].save(
        "sprite_showcase.gif",
        save_all=True,
        append_images=frames[1:],
        duration=GIF_DELAY,
        loop=0,
        optimize=False,
    )
    print(f"  → sprite_showcase.gif  ({len(frames)} frames @ {GIF_DELAY}ms each)")

    print("\nDone!  Open sprite_showcase.png and sprite_showcase.gif to see the show.")


if __name__ == "__main__":
    main()
