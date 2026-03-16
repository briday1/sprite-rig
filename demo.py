"""demo.py – builds a minimal demo rig with two parts and one feature channel.

Usage
-----
    from demo import build_demo_rig
    rig = build_demo_rig()
    img = rig.render(part_state={"p0": 0.4, "p1": 0.6}, feature_state={"f0": 1.0})
"""

import numpy as np
from sprite_rig import Rig, Part, ShapePrototype, Attachment, FeatureChannel


# ---------------------------------------------------------------------------
# Helper: simple shape factories
# ---------------------------------------------------------------------------

def _ellipse_mask(h: int, w: int, ry: float, rx: float) -> np.ndarray:
    """Boolean mask: filled ellipse centred in an (h, w) canvas."""
    Y, X = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    return ((Y - cy) / ry) ** 2 + ((X - cx) / rx) ** 2 <= 1.0


def _rect_mask(h: int, w: int, margin_y: int = 0, margin_x: int = 0) -> np.ndarray:
    """Boolean mask: filled rectangle with optional inset."""
    m = np.zeros((h, w), dtype=bool)
    m[margin_y: h - margin_y, margin_x: w - margin_x] = True
    return m


def _circle_mask(size: int, radius: int) -> np.ndarray:
    """Boolean mask: filled circle of given radius centred in a square canvas."""
    Y, X = np.ogrid[:size, :size]
    cy = cx = size / 2.0
    return (Y - cy) ** 2 + (X - cx) ** 2 <= radius ** 2


# ---------------------------------------------------------------------------
# Feature-channel generator
# ---------------------------------------------------------------------------

def _dot_generator(strength: float, anchor_name: str, image_size: int) -> np.ndarray:
    """Generate a small filled circle whose radius scales with *strength*."""
    radius = max(1, int(strength * image_size * 0.08))
    return _circle_mask(image_size, radius)


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_demo_rig(size: int = 96) -> Rig:
    """Return a minimal demo :class:`~sprite_rig.Rig`.

    The rig contains:

    * **p0** – a body part that morphs between a tall rectangle and a wide
      ellipse.  It exposes a ``"socket"`` anchor (top-centre) and two
      feature anchors ``"a0"`` and ``"a1"`` on its left and right sides.
    * **p1** – a head part that morphs between a circle and a slightly
      smaller circle.  It exposes an ``"attach"`` anchor at its bottom
      centre.  An :class:`~sprite_rig.Attachment` connects ``p1.attach``
      to ``p0.socket``.
    * **f0** – a :class:`~sprite_rig.FeatureChannel` that places a small
      dot at each feature anchor on *p0*, scaled by the channel strength.
    """
    # ------------------------------------------------------------------
    # p0 – body (96 × 96 canvas, anchors in (x, y) = (col, row) order)
    # ------------------------------------------------------------------
    h, w = size, size

    # Prototype A: tall rectangle occupying centre two-thirds
    p0a_mask = _rect_mask(h, w, margin_y=h // 6, margin_x=w // 4)
    p0a = ShapePrototype(
        mask=p0a_mask,
        anchors={
            "socket": (w // 2, h // 6),           # top-centre
            "a0":     (w // 4, h * 2 // 3),       # left side, lower
            "a1":     (w * 3 // 4, h * 2 // 3),   # right side, lower
        },
    )

    # Prototype B: wide ellipse
    p0b_mask = _ellipse_mask(h, w, ry=h * 0.35, rx=w * 0.45)
    p0b = ShapePrototype(
        mask=p0b_mask,
        anchors={
            "socket": (w // 2, int(h * 0.15)),
            "a0":     (int(w * 0.18), int(h * 0.65)),
            "a1":     (int(w * 0.82), int(h * 0.65)),
        },
    )

    body = Part(name="p0", proto_a=p0a, proto_b=p0b, color=(120, 200, 120, 255))

    # ------------------------------------------------------------------
    # p1 – head  (same canvas size; will be shifted by attachment)
    # ------------------------------------------------------------------
    head_r_a = h // 7
    p1a_mask = _circle_mask(h, head_r_a)
    p1a = ShapePrototype(
        mask=p1a_mask,
        anchors={
            "attach": (w // 2, h // 2 + head_r_a),   # bottom of circle
        },
    )

    head_r_b = int(h * 0.18)
    p1b_mask = _circle_mask(h, head_r_b)
    p1b = ShapePrototype(
        mask=p1b_mask,
        anchors={
            "attach": (w // 2, h // 2 + head_r_b),
        },
    )

    head = Part(name="p1", proto_a=p1a, proto_b=p1b, color=(220, 160, 100, 255))

    # ------------------------------------------------------------------
    # Attachment: p1.attach → p0.socket
    # ------------------------------------------------------------------
    att = Attachment(
        child_part="p1",
        child_anchor="attach",
        parent_part="p0",
        parent_anchor="socket",
    )

    # ------------------------------------------------------------------
    # f0 – feature channel: dots at p0's side anchors
    # ------------------------------------------------------------------
    # local_anchor_map: the generator puts a circle centered at (w//2, h//2)
    center: tuple = (w // 2, h // 2)
    f0 = FeatureChannel(
        name="f0",
        owner_part="p0",
        owner_anchors=["a0", "a1"],
        generator=_dot_generator,
        local_anchor_map={"a0": center, "a1": center},
        color=(200, 80, 80, 255),
    )

    # ------------------------------------------------------------------
    # Assemble
    # ------------------------------------------------------------------
    rig = Rig(size=size, hi=8)
    rig.add_part(body)
    rig.add_part(head)
    rig.add_attachment(att)
    rig.add_feature_channel(f0)

    return rig
