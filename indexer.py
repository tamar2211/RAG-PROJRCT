import hashlib
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import config
from loader import load_documents, DATA_DIR
from chunker import split_documents

ROOT = Path(__file__).parent
CHROMA_DIR = ROOT / "storage" / "chroma"


def get_client():
    """פותחת את מסד Chroma ששמור בדיסק (בלי שליחת נתוני שימוש החוצה)."""
    return chromadb.PersistentClient(path=str(CHROMA_DIR),
                                     settings=ChromaSettings(anonymized_telemetry=False))


def fingerprint(strategy):
    """מחשבת חתימה מהנתונים, מקוד הטעינה והחלוקה, מהאסטרטגיה וממודל ה-embedding. כל שינוי באחד מהם משנה את החתימה."""
    h = hashlib.sha256(f"{strategy}|{config.EMBED_MODEL}".encode())
    for path in sorted(DATA_DIR.glob("*.md")) + [ROOT / "loader.py", ROOT / "chunker.py"]:
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def build_index(strategy, fp):
    """מוחקת אוסף ישן אם יש, מחשבת embedding לכל הצ'אנקים, ושומרת אותם ב-Chroma יחד עם החתימה."""
    client = get_client()
    name = f"galil_{strategy}"
    if name in [c.name for c in client.list_collections()]:
        client.delete_collection(name)
    collection = client.create_collection(name, metadata={"hnsw:space": "cosine", "fingerprint": fp})
    storage = StorageContext.from_defaults(vector_store=ChromaVectorStore(chroma_collection=collection))
    nodes = split_documents(load_documents(), strategy)
    index = VectorStoreIndex(nodes, storage_context=storage, show_progress=True)
    print(f"[indexer] נבנה ונשמר ב-Chroma: {name} ({collection.count()} וקטורים)")
    return index


def get_index(strategy="section"):
    """טוענת את האינדקס מ-Chroma אם החתימה תואמת לנתונים הנוכחיים; אחרת בונה מחדש אוטומטית."""
    fp = fingerprint(strategy)
    name = f"galil_{strategy}"
    client = get_client()
    if name in [c.name for c in client.list_collections()]:
        collection = client.get_collection(name)
        if (collection.metadata or {}).get("fingerprint") == fp and collection.count() > 0:
            print(f"[indexer] נטען מ-Chroma: {name} ({collection.count()} וקטורים)")
            return VectorStoreIndex.from_vector_store(ChromaVectorStore(chroma_collection=collection))
        print(f"[indexer] הנתונים או הקוד השתנו -> בונה מחדש את {name}")
    return build_index(strategy, fp)


if __name__ == "__main__":
    config.setup(llm=False)
    for strategy in ("section", "page"):
        get_index(strategy)
