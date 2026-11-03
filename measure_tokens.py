import config  # noqa: F401  (truststore + מפתח מ-.env)
from statistics import mean
from google import genai
from llama_index.core.node_parser import MarkdownNodeParser
from loader import load_documents

COUNT_MODEL = "gemini-3.6-flash"  # רק לספירת טוקנים, לא נשלחת שאלה למודל
LLM_KEYS = ["name", "length_km", "duration", "best_season",
            "difficulty", "circular", "water", "family_friendly", "accessible", "entry_fee"]

client = genai.Client()


def count(text):
    """מחזירה כמה טוקנים יש בטקסט לפי ה-tokenizer של Gemini."""
    return client.models.count_tokens(model=COUNT_MODEL, contents=text).total_tokens


def llm_metadata_text(meta):
    """בונה את מחרוזת ה-metadata שהייתה נשלחת ל-LLM (השדות השימושיים + ההערות שלהם)."""
    keys = [k for base in LLM_KEYS for k in (base, base + "_note") if k in meta]
    return "\n".join(f"{k}: {meta[k]}" for k in keys)


def summarize(label, values):
    """מדפיסה ממוצע, מינימום ומקסימום של רשימת מספרים."""
    print(f"{label:<28} ממוצע {mean(values):>5.0f} | מינ' {min(values):>4} | מקס' {max(values):>4}")


def measure():
    """סופרת טוקנים לסעיפים, לדפים שלמים ול-metadata, ומשווה עלות משוערת לשאלה בשתי האסטרטגיות."""
    docs = load_documents()
    sections = [n for n in MarkdownNodeParser().get_nodes_from_documents(docs) if n.text.startswith("##")]

    page_tok = [count(d.text) for d in docs]
    meta_tok = [count(llm_metadata_text(d.metadata)) for d in docs]
    sec_tok = {}
    for n in sections:
        title = n.text.split("\n", 1)[0].lstrip("#").strip()
        sec_tok.setdefault(title, []).append(count(n.text))

    print("\n=== סעיפים ===")
    for title, vals in sec_tok.items():
        summarize(title, vals)
    all_sec = [v for vals in sec_tok.values() for v in vals]
    summarize("כל הסעיפים", all_sec)
    print("\n=== דפים ו-metadata ===")
    summarize("דף שלם", page_tok)
    summarize("metadata ל-LLM", meta_tok)

    m = mean(meta_tok)
    sec_cost = 5 * (mean(all_sec) + m)
    page_cost = 3 * (mean(page_tok) + m)
    print("\n=== עלות הקשר משוערת לשאלה ===")
    print(f"סעיפים, top_k=5: {sec_cost:.0f} טוקנים (מהם metadata: {5*m:.0f})")
    print(f"דפים,   top_k=3: {page_cost:.0f} טוקנים (מהם metadata: {3*m:.0f})")


if __name__ == "__main__":
    measure()
