"""example.py – generate a sprite with the demo rig and save it as a PNG.

Run with::

    python example.py

This will write ``sprite_output.png`` in the current directory.
"""

from PIL import Image
from demo import build_demo_rig

def main() -> None:
    rig = build_demo_rig(size=96)

    img_array = rig.render(
        part_state={"p0": 0.4, "p1": 0.7},
        feature_state={"f0": 1.0},
    )

    img = Image.fromarray(img_array, mode="RGBA")
    img.save("sprite_output.png")
    print(f"Saved sprite_output.png  ({img.width}×{img.height} RGBA)")


if __name__ == "__main__":
    main()
