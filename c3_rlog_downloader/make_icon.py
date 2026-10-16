"""Generate the standalone downloader's multi-resolution Windows icon."""

from pathlib import Path

from PIL import Image, ImageDraw

SOURCE_DIR = Path(__file__).resolve().parent
CANVAS_SIZE = 1024


def draw_icon() -> Image.Image:
    scale = CANVAS_SIZE / 256
    image = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    def box(values):
        return tuple(round(value * scale) for value in values)

    def points(values):
        return [(round(x * scale), round(y * scale)) for x, y in values]

    navy = "#16324f"
    blue = "#2f7de1"
    pale = "#dce8f5"
    paper = "#ffffff"

    draw.rounded_rectangle(box((8, 8, 248, 248)), radius=round(54 * scale), fill=navy)
    draw.rounded_rectangle(
        box((55, 32, 197, 200)), radius=round(12 * scale), fill=paper
    )
    draw.polygon(points(((153, 32), (197, 76), (153, 76))), fill=pale)

    draw.rounded_rectangle(box((77, 91, 118, 99)), radius=round(4 * scale), fill=blue)
    draw.rounded_rectangle(box((77, 111, 157, 119)), radius=round(4 * scale), fill=pale)
    waveform = points(
        (
            (78, 151),
            (94, 151),
            (104, 135),
            (115, 168),
            (127, 143),
            (138, 151),
            (162, 151),
        )
    )
    draw.line(waveform, fill=blue, width=round(7 * scale), joint="curve")
    for x, y in waveform:
        radius = round(3.5 * scale)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=blue)

    draw.ellipse(box((151, 151, 235, 235)), fill=blue)
    draw.rounded_rectangle(
        box((188, 168, 198, 202)), radius=round(5 * scale), fill=paper
    )
    draw.polygon(points(((174, 193), (193, 213), (212, 193))), fill=paper)
    draw.rounded_rectangle(
        box((174, 216, 212, 222)), radius=round(3 * scale), fill=paper
    )
    return image


def main() -> None:
    icon = draw_icon()
    icon.save(
        SOURCE_DIR / "C3_Rlog_Downloader.ico",
        format="ICO",
        sizes=(
            (16, 16),
            (24, 24),
            (32, 32),
            (48, 48),
            (64, 64),
            (128, 128),
            (256, 256),
        ),
    )
    icon.resize((512, 512), Image.Resampling.LANCZOS).save(
        SOURCE_DIR / "C3_Rlog_Downloader_icon.png"
    )


if __name__ == "__main__":
    main()
