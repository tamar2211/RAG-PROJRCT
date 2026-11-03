import re
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.retrievers.bm25 import BM25Retriever
from indexer import get_index
from loader import load_documents
from chunker import split_documents
from query_filters import extract_filters

HEBREW_STOPWORDS = ["של", "על", "את", "עם", "או", "גם", "כי", "אם", "מה", "מי", "איפה", "איך", "יש",
                    "אין", "זה", "זו", "הוא", "היא", "אני", "אפשר", "כדאי", "לא", "כן", "אל", "עד",
                    "בין", "כל", "רק", "הם", "הן"]
TOKEN_PATTERN = r"(?u)\w[\w'׳״]*\w"  # מילה = 2 תווים ומעלה, כולל גרש ("ג'יפים" נשארת מילה אחת)
CANDIDATES = 5  # כמה מועמדים כל חיפוש מביא לפני האיחוד

_cache = {}


def _index():
    """אינדקס הסעיפים ב-Chroma (נטען פעם אחת)."""
    if "index" not in _cache:
        _cache["index"] = get_index("section")
    return _cache["index"]


def _nodes():
    """כל 60 הסעיפים, עבור BM25 (נוצרים פעם אחת, מקומית, בלי קריאות ל-Google)."""
    if "nodes" not in _cache:
        _cache["nodes"] = split_documents(load_documents(), "section")
    return _cache["nodes"]


def _norm(text):
    """מסירה גרשים ומרכאות, כדי ש'ג'יפים' ו'גיפים' ייחשבו אותה מילה ב-BM25."""
    return re.sub(r"['׳״\"]", "", text)


def _bm25_copy(node):
    """עותק של צ'אנק עם טקסט מנורמל, לחיפוש BM25 בלבד (באותו מזהה, כדי שנוכל להחזיר את המקורי)."""
    return TextNode(text=_norm(node.text), metadata=node.metadata, id_=node.node_id,
                    excluded_embed_metadata_keys=node.excluded_embed_metadata_keys,
                    excluded_llm_metadata_keys=node.excluded_llm_metadata_keys)


def _rrf(result_lists, top_k, k=60):
    """מאחדת כמה רשימות תוצאות לפי המיקום בכל רשימה (Reciprocal Rank Fusion), בלי כפילויות."""
    scores, nodes = {}, {}
    for results in result_lists:
        for rank, r in enumerate(results):
            scores[r.node.text] = scores.get(r.node.text, 0) + 1 / (k + rank + 1)
            nodes.setdefault(r.node.text, r.node)
    best = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [NodeWithScore(node=nodes[t], score=scores[t]) for t in best]


def retrieve(question, conditions=None, top_k=3):
    """חיפוש היברידי (סמנטי + BM25) רק בסעיפים שעומדים בתנאים, ואיחוד בשיטת RRF."""
    conditions = conditions or {}
    nodes = [n for n in _nodes() if all(n.metadata.get(f) in v for f, v in conditions.items())]
    if not nodes:
        print("[retriever] אף אתר לא עומד בכל התנאים -> חיפוש בלי סינון")
        conditions, nodes = {}, _nodes()
    filters = MetadataFilters(filters=[MetadataFilter(key=f, value=v, operator=FilterOperator.IN)
                                       for f, v in conditions.items()]) if conditions else None
    semantic = _index().as_retriever(similarity_top_k=CANDIDATES, filters=filters).retrieve(question)
    originals = {n.node_id: n for n in nodes}
    keyword = BM25Retriever.from_defaults(
        nodes=[_bm25_copy(n) for n in nodes], similarity_top_k=min(CANDIDATES, len(nodes)),
        skip_stemming=True, language=HEBREW_STOPWORDS, token_pattern=TOKEN_PATTERN).retrieve(_norm(question))
    keyword = [NodeWithScore(node=originals[r.node.node_id], score=r.score) for r in keyword]
    return _rrf([semantic, keyword], top_k)


def search(question, top_k=3):
    """הצינור המלא של האחזור: חילוץ תנאים מהשאלה, ואז חיפוש היברידי. מחזירה (תנאים, תוצאות)."""
    conditions = extract_filters(question)
    return conditions, retrieve(question, conditions, top_k)
