"""sprite_rig – procedural 2D sprite generator.

Public API re-exported here for convenience:

    from sprite_rig import Rig, Part, ShapePrototype, Attachment, FeatureChannel
"""

from .rig import (
    ShapePrototype,
    Part,
    Attachment,
    FeatureChannel,
    Rig,
    morph_mask,
    shift_mask,
    signed_distance,
    lerp,
)

__all__ = [
    "ShapePrototype",
    "Part",
    "Attachment",
    "FeatureChannel",
    "Rig",
    "morph_mask",
    "shift_mask",
    "signed_distance",
    "lerp",
]
