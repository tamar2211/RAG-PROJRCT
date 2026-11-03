from pathlib import Path
import yaml
from llama_index.core import Document

DATA_DIR = Path(__file__).parent / "data"


def parse_page(path):
    """מפרידה דף אחד לכותרת YAML ולגוף, ומחזירה Document שהגוף שלו הוא הטקסט והשדות הם ה-metadata."""
    raw = path.read_text(encoding="utf-8")
    _, header, body = raw.split("---", 2)
    meta = yaml.safe_load(header)
    meta = {k: str(v) for k, v in meta.items() if v is not None}  # בלי null, והכל כמחרוזת
    meta["file_name"] = path.name
    return Document(text=body.strip(), metadata=meta, id_=meta["id"])


def load_documents(data_dir=DATA_DIR):
    """טוענת את כל דפי ה-md מתיקיית data ומחזירה רשימת Documents."""
    docs = [parse_page(p) for p in sorted(data_dir.glob("*.md"))]
    print(f"[loader] נטענו {len(docs)} דפים")
    return docs


if __name__ == "__main__":
    docs = load_documents()
    print(docs[0].metadata)
    print(docs[0].text[:200])
