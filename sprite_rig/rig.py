import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Callable, List, Optional
from scipy import ndimage

Point = Tuple[float, float]


def signed_distance(mask: np.ndarray) -> np.ndarray:
    """Return a signed distance field for a boolean mask.

    Positive values are inside the shape, negative outside.
    """
    inside = ndimage.distance_transform_edt(mask)
    outside = ndimage.distance_transform_edt(~mask)
    return inside - outside


def morph_mask(mask_a: np.ndarray, mask_b: np.ndarray, t: float, hi: int = 8) -> np.ndarray:
    """Interpolate between two boolean masks using SDF blending.

    Parameters
    ----------
    mask_a, mask_b : bool ndarray of shape (H, W)
    t              : blend parameter in [0, 1]
    hi             : oversampling factor for sub-pixel accuracy
    """
    if t <= 0:
        return mask_a.copy()
    if t >= 1:
        return mask_b.copy()

    h, w = mask_a.shape

    a_hi = np.repeat(np.repeat(mask_a.astype(np.uint8), hi, 0), hi, 1).astype(bool)
    b_hi = np.repeat(np.repeat(mask_b.astype(np.uint8), hi, 0), hi, 1).astype(bool)

    da = signed_distance(a_hi)
    db = signed_distance(b_hi)

    d = (1 - t) * da + t * db
    m_hi = d >= 0

    return m_hi.reshape(h, hi, w, hi).any(axis=(1, 3))


def shift_mask(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Translate a boolean mask by (dx, dy) pixels, clipping at image boundaries."""
    h, w = mask.shape
    out = np.zeros_like(mask)
    ys, xs = np.where(mask)

    xs2 = np.round(xs + dx).astype(int)
    ys2 = np.round(ys + dy).astype(int)

    good = (xs2 >= 0) & (xs2 < w) & (ys2 >= 0) & (ys2 < h)

    out[ys2[good], xs2[good]] = True
    return out


def lerp(a: Point, b: Point, t: float) -> Point:
    """Linearly interpolate between two 2-D points."""
    return ((1 - t) * a[0] + t * b[0], (1 - t) * a[1] + t * b[1])


@dataclass
class ShapePrototype:
    """A single prototype shape: a boolean mask plus named anchor points."""

    mask: np.ndarray
    anchors: Dict[str, Point]


@dataclass
class Part:
    """A morphable shape that blends between two ShapePrototypes."""

    name: str
    proto_a: ShapePrototype
    proto_b: ShapePrototype
    color: Tuple[int, int, int, int] = (120, 200, 120, 255)

    def mask(self, t: float, hi: int = 8) -> np.ndarray:
        """Return the morphed boolean mask for blend parameter *t* in [0, 1]."""
        return morph_mask(self.proto_a.mask, self.proto_b.mask, t, hi)

    def anchor(self, name: str, t: float) -> Point:
        """Return the interpolated anchor position for blend parameter *t*."""
        return lerp(self.proto_a.anchors[name], self.proto_b.anchors[name], t)


@dataclass
class Attachment:
    """Geometric constraint that aligns a child anchor to a parent anchor."""

    child_part: str
    child_anchor: str
    parent_part: str
    parent_anchor: str


@dataclass
class FeatureChannel:
    """Additive geometry layer driven by a generator function.

    The *generator* callable has the signature::

        generator(strength, anchor_name, image_size) -> bool ndarray (H, W)

    The returned mask is expressed in local coordinates.  It is then
    translated so that *local_anchor_map[anchor_name]* aligns with the
    owner part's anchor position (after the owner's own offset is applied).
    """

    name: str
    owner_part: str
    owner_anchors: List[str]
    generator: Callable
    local_anchor_map: Dict[str, Point]
    color: Tuple[int, int, int, int] = (100, 120, 220, 255)


@dataclass
class Rig:
    """Top-level object that manages parts, attachments, and feature channels.

    Parameters
    ----------
    size : int
        Output image dimension (pixels).  The image will be *size × size*.
    hi   : int
        Oversampling factor used during SDF-based mask morphing.
    """

    size: int = 96
    hi: int = 8
    parts: Dict[str, Part] = field(default_factory=dict)
    attachments: List[Attachment] = field(default_factory=list)
    channels: Dict[str, FeatureChannel] = field(default_factory=dict)

    def add_part(self, p: Part) -> None:
        self.parts[p.name] = p

    def add_attachment(self, a: Attachment) -> None:
        self.attachments.append(a)

    def add_feature_channel(self, c: FeatureChannel) -> None:
        self.channels[c.name] = c

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_part_offsets(self, part_state: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
        """Resolve attachment constraints and return (dx, dy) for every part."""
        offsets: Dict[str, Tuple[float, float]] = {name: (0.0, 0.0) for name in self.parts}

        for att in self.attachments:
            child = self.parts[att.child_part]
            parent = self.parts[att.parent_part]

            t_child = part_state.get(att.child_part, 0.0)
            t_parent = part_state.get(att.parent_part, 0.0)

            child_anch = child.anchor(att.child_anchor, t_child)
            parent_anch = parent.anchor(att.parent_anchor, t_parent)
            parent_offset = offsets[att.parent_part]

            # Solve: child_anch + child_offset = parent_anch + parent_offset
            dx = parent_anch[0] + parent_offset[0] - child_anch[0]
            dy = parent_anch[1] + parent_offset[1] - child_anch[1]
            offsets[att.child_part] = (dx, dy)

        return offsets

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(
        self,
        part_state: Optional[Dict[str, float]] = None,
        feature_state: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Generate a sprite and return an RGBA numpy array of shape (H, W, 4).

        Parameters
        ----------
        part_state    : mapping of part name → blend parameter t ∈ [0, 1]
        feature_state : mapping of feature channel name → strength ∈ [0, 1]
        """
        if part_state is None:
            part_state = {}
        if feature_state is None:
            feature_state = {}

        img = np.zeros((self.size, self.size, 4), dtype=np.uint8)

        # Resolve attachment offsets once
        offsets = self._compute_part_offsets(part_state)

        # --- Feature channels (rendered first / behind parts) ---
        for fname, channel in self.channels.items():
            strength = feature_state.get(fname, 0.0)
            if strength <= 0:
                continue

            owner = self.parts[channel.owner_part]
            t_owner = part_state.get(channel.owner_part, 0.0)
            owner_offset = offsets[channel.owner_part]

            for anchor_name in channel.owner_anchors:
                owner_anch = owner.anchor(anchor_name, t_owner)
                local_anch = channel.local_anchor_map.get(anchor_name, (0.0, 0.0))

                # Generate mask in local (unshifted) coordinates
                feat_mask = channel.generator(strength, anchor_name, self.size)

                # Translate so local_anch aligns with owner_anch + owner_offset
                dx = owner_anch[0] + owner_offset[0] - local_anch[0]
                dy = owner_anch[1] + owner_offset[1] - local_anch[1]

                shifted = shift_mask(feat_mask, dx, dy)
                img[shifted] = channel.color

        # --- Parts (rendered on top of feature channels) ---
        for pname, part in self.parts.items():
            t = part_state.get(pname, 0.0)
            mask = part.mask(t, self.hi)
            offset = offsets.get(pname, (0.0, 0.0))
            shifted = shift_mask(mask, offset[0], offset[1])
            img[shifted] = part.color

        return img
