import re
from collections import OrderedDict
from llama_index.core.llms import ChatMessage
import config
from retriever import search

SYSTEM_PROMPT = """אתה מדריך טיולים לגליל העליון. ענה בעברית, בקצרה ולעניין, רק על סמך המקורות שבהודעה.
אם התשובה לא מופיעה במקורות, כתוב בדיוק: "אין לי מידע על כך במאגר." אל תשתמש בידע כללי ואל תנחש.
ציין בסוגריים את שם האתר שממנו לקחת כל פרט. "לא אומת" פירושו שאין מידע מאומת."""

NO_INFO = "אין לי מידע על כך במאגר"
PREFIXES = ("שמורת טבע ", "שמורת ", "גן לאומי ", "מצפה ")

LLM_FIELDS = ["sub_area", "length_km", "duration", "difficulty", "circular", "water",
              "family_friendly", "accessible", "entry_fee", "best_season"]


def build_context(results):
    """מקבצת את הסעיפים שנשלפו לפי אתר, ומצרפת את ה-metadata של כל אתר פעם אחת בלבד."""
    sites = OrderedDict()
    for r in results:
        sites.setdefault(r.node.metadata["id"], []).append(r.node)
    blocks = []
    for nodes in sites.values():
        meta = nodes[0].metadata
        lines = [f"{k}: {meta[k]}" for f in LLM_FIELDS for k in (f, f + "_note") if k in meta]
        blocks.append(f"### {meta['name']}\n" + "\n".join(lines) + "\n\n" + "\n\n".join(n.text for n in nodes))
    return "\n\n".join(blocks)


def _name_forms(name):
    """צורות שבהן שם אתר עשוי להופיע בתשובה: מלא, בלי סוגריים, ובלי 'שמורת' / 'גן לאומי' (אם נשארות 2 מילים ומעלה)."""
    base = re.sub(r"\s*\(.*?\)", "", name).split(" – ")[0].strip()
    forms = {name, base}
    for p in PREFIXES:
        if base.startswith(p) and " " in base[len(p):]:
            forms.add(base[len(p):])
    return forms


def used_sources(answer_text, sources):
    """משאירה רק מקורות של אתרים שהתשובה מזכירה. תשובת 'אין מידע' -> אין מקורות. לא זוהה אף שם -> כל המקורות."""
    used = [s for s in sources if any(f in answer_text for f in _name_forms(s["site"]))]
    if used:
        return used
    return [] if NO_INFO in answer_text else sources


def answer(question):
    """הצינור המלא: אחזור, בניית הקשר, תשובה מה-LLM. מחזירה תשובה, מקורות ותנאי הסינון."""
    conditions, results = search(question)
    messages = [ChatMessage(role="system", content=SYSTEM_PROMPT),
                ChatMessage(role="user", content=f"מקורות:\n{build_context(results)}\n\nשאלה: {question}")]
    sources = [{"site": r.node.metadata["name"], "section": r.node.text.split("\n", 1)[0].lstrip("# "),
                "score": r.score, "text": r.node.text} for r in results]
    text, model = config.chat(messages)
    return {"answer": text, "sources": used_sources(text, sources), "conditions": list(conditions), "model": model}


if __name__ == "__main__":
    config.setup()
    res = answer("איפה אפשר לראות עגורים?")
    print(res["answer"])
    for s in res["sources"]:
        print(f"- {s['site']} | {s['section']}")
