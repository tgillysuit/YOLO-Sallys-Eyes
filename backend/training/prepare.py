"""
Prepare a YOLO dataset from a Label Studio "YOLO with Images" export.

Usage:
    python backend/training/prepare.py <path-to-label-studio-export-dir>

The export directory should contain:
    images/          - all labeled images
    labels/          - matching .txt YOLO label files
    classes.txt      - one class name per line

Outputs to backend/training/dataset/:
    images/train/    images/val/
    labels/train/    labels/val/
    dataset.yaml
"""
import random
import shutil
import sys
from pathlib import Path


SEED = 42
TRAIN_RATIO = 0.80


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python prepare.py <label-studio-export-dir>")
        sys.exit(1)

    export_dir = Path(sys.argv[1]).resolve()
    if not export_dir.exists():
        print(f"ERROR: export directory not found: {export_dir}")
        sys.exit(1)

    images_src = export_dir / "images"
    labels_src = export_dir / "labels"
    classes_file = export_dir / "classes.txt"

    for p, name in [(images_src, "images/"), (labels_src, "labels/"), (classes_file, "classes.txt")]:
        if not p.exists():
            print(f"ERROR: expected {name} inside {export_dir} — not found")
            sys.exit(1)

    # Read class names
    classes = [line.strip() for line in classes_file.read_text().splitlines() if line.strip()]
    print(f"Classes: {classes}")

    # Collect labeled images (only those that have a matching label file)
    image_exts = {".jpg", ".jpeg", ".png"}
    images = sorted(
        p for p in images_src.iterdir()
        if p.suffix.lower() in image_exts and (labels_src / (p.stem + ".txt")).exists()
    )

    unlabeled = [p for p in images_src.iterdir() if p.suffix.lower() in image_exts
                 and not (labels_src / (p.stem + ".txt")).exists()]
    if unlabeled:
        print(f"WARNING: {len(unlabeled)} image(s) have no label file and will be skipped:")
        for u in unlabeled[:5]:
            print(f"  {u.name}")
        if len(unlabeled) > 5:
            print(f"  ... and {len(unlabeled) - 5} more")

    if not images:
        print("ERROR: no labeled images found")
        sys.exit(1)

    # Shuffle + split
    random.seed(SEED)
    random.shuffle(images)
    split_idx = max(1, int(len(images) * TRAIN_RATIO))
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # Output directories
    dataset_dir = Path(__file__).parent / "dataset"
    splits = {"train": train_images, "val": val_images}

    for split, imgs in splits.items():
        img_out = dataset_dir / "images" / split
        lbl_out = dataset_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in imgs:
            shutil.copy2(img_path, img_out / img_path.name)
            lbl_path = labels_src / (img_path.stem + ".txt")
            shutil.copy2(lbl_path, lbl_out / (img_path.stem + ".txt"))

    # Write dataset.yaml
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(classes))
    yaml_content = (
        f"path: {dataset_dir.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(classes)}\n"
        f"names:\n{names_block}\n"
    )
    yaml_path = dataset_dir / "dataset.yaml"
    yaml_path.write_text(yaml_content)

    print(f"\nDataset written to: {dataset_dir}")
    print(f"  train: {len(train_images)} images")
    print(f"  val:   {len(val_images)} images")
    print(f"  yaml:  {yaml_path}")


if __name__ == "__main__":
    main()
