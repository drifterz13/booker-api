from pathlib import Path

import pymupdf


def render_thumbnail(source: Path) -> bytes:
    """Render the first page as a PNG with its longest edge at 240 pixels."""
    with pymupdf.open(source) as document:
        page = document[0]
        scale = 240 / max(page.rect.width, page.rect.height)
        image = page.get_pixmap(
            matrix=pymupdf.Matrix(scale, scale),
            colorspace=pymupdf.csRGB,
            alpha=False,
        )
        return image.tobytes("png")
