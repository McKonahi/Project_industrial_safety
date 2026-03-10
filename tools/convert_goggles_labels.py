from pathlib import Path

DATASET_DIR = Path(r"E:\Project\industrial_safety_cv\datasets\goggles_yolo")
TARGET_CLASS_ID = 2  # Goggles in your PPE

for split in ["train", "valid", "test"]:
    labels_dir = DATASET_DIR / split / "labels"
    for p in labels_dir.glob("*.txt"):
        lines = p.read_text(encoding="utf-8").splitlines()
        out = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.split()
            parts[0] = str(TARGET_CLASS_ID)
            out.append(" ".join(parts))
        p.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")

print("OK: goggles labels remapped to class 2")