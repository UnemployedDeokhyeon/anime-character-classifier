"""Build metadata CSV from data/processed/ folder structure.

Expected folder name format:  {캐릭터} - {애니메이션}
Example:                       Ichigo - 블리치
"""
import unicodedata
from pathlib import Path

import polars as pl


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def build_metadata(data_dir: str | Path = "data/processed") -> pl.DataFrame:
    """Scan the processed dataset folders and return a metadata table."""
    data_dir = Path(data_dir)

    rows: list[dict] = []
    for folder in sorted(data_dir.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue

        # macOS stores filenames as NFD — normalize to NFC for correct Korean rendering
        nfc_name = unicodedata.normalize("NFC", folder.name)
        parts = nfc_name.split(" - ", maxsplit=1)
        character = parts[0].strip()
        anime = parts[1].strip() if len(parts) == 2 else ""

        for img in sorted(folder.iterdir()):
            if img.suffix.lower() in IMAGE_EXTS:
                rows.append({
                    "캐릭터": character,
                    "애니메이션": anime,
                    "사진_경로": str(img),
                })

    df = (
        pl.DataFrame(rows)
        .with_row_index(name="id", offset=0)
        .select(["id", "애니메이션", "캐릭터", "사진_경로"])
    )
    return df


def save_metadata(
    data_dir: str | Path = "data/processed",
    out_path: str | Path = "data/processed/metadata.csv",
) -> pl.DataFrame:
    """Build metadata and persist it as a CSV file."""
    df = build_metadata(data_dir)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out_path)
    print(f"Saved {len(df)} rows → {out_path}")
    print(df.head(5))
    print(f"\n캐릭터 수: {df['캐릭터'].n_unique()}  |  애니메이션 수: {df['애니메이션'].n_unique()}")
    return df


if __name__ == "__main__":
    save_metadata()
