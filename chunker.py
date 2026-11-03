from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter

EMBED_KEYS = {"name", "sub_area", "site_type"}  # מה נכנס ל-embedding מתוך ה-metadata


def split_by_section(documents):
    """צ'אנק לכל כותרת ##, בלי הצ'אנק הריק של כותרת הדף (#)."""
    nodes = MarkdownNodeParser().get_nodes_from_documents(documents)
    return [n for n in nodes if n.text.startswith("##")]


def split_by_page(documents):
    """צ'אנק אחד לכל דף (גודל מקסימלי שגדול מכל דף, כך שאף דף לא נחתך)."""
    return SentenceSplitter(chunk_size=2048, chunk_overlap=0).get_nodes_from_documents(documents)


def split_documents(documents, strategy="section"):
    """מחלקת לפי האסטרטגיה שנבחרה, וקובעת: ל-embedding רק שם/אזור/סוג, ול-LLM שום metadata (נצרף אותו פעם אחת לאתר בשלב התשובה)."""
    nodes = split_by_section(documents) if strategy == "section" else split_by_page(documents)
    for node in nodes:
        keys = list(node.metadata)
        node.excluded_embed_metadata_keys = [k for k in keys if k not in EMBED_KEYS]
        node.excluded_llm_metadata_keys = keys
    print(f"[chunker] {strategy}: {len(documents)} דפים -> {len(nodes)} צ'אנקים")
    return nodes


if __name__ == "__main__":
    from loader import load_documents
    docs = load_documents()
    split_documents(docs, "page")
    nodes = split_documents(docs, "section")
    print("--- מה נכנס ל-embedding בצ'אנק הראשון ---")
    print(nodes[0].get_content(metadata_mode="embed"))
