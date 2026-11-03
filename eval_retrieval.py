from collections import Counter
import config
from retriever import search

# (שאלה, האתרים שאמורים להישלף). סט ריק = אין תשובה במאגר
QUESTIONS = [
    ("איפה אפשר לראות עגורים?", {"agamon_hahula"}),
    ("איפה אפשר ללכת בתוך המים בקיץ?", {"nahal_snir"}),
    ("אני מחפש מסלול מוצל ליום חם", {"nahal_amud_upper", "tel_dan"}),
    ("איך מגיעים לתל חצור?", {"tel_hatzor"}),
    ("מה הגובה של מפל התנור?", {"nahal_iyon"}),
    ("איפה שמור ספר התורה המיוחס לרבי יצחק אבוהב?", {"safed_old_city"}),
    ("איזה מסלול עם מים מתאים לכיסא גלגלים?", {"nahal_iyon", "nahal_snir", "hula_reserve", "agamon_hahula"}),
    ("האם נחל דישון מתאים לילדים, ומתי כדאי ללכת?", {"nahal_dishon"}),
    ("מה ההבדל בין אגמון החולה לשמורת החולה?", {"hula_reserve", "agamon_hahula"}),
    ("מה שעות הפתיחה של תל דן?", set()),
    ("מה כדאי לראות בעיר דוד?", set()),
    # שאלות קשות: ניסוח רחוק מהמסמך
    ("איפה אפשר לישון בשטח עם אוהל?", {"nahal_amud_upper"}),
    ("איפה אפשר לראות באפלו?", {"agamon_hahula", "hula_reserve"}),
    ("איפה נמצאת העיר שמלכה יבין נלחם בדבורה וברק?", {"tel_hatzor"}),
    ("יש מקום עם פעילות מיוחדת לילדים קטנים?", {"nahal_snir"}),
    ("איפה אפשר לראות שלג בגליל?", {"har_meron_summit"}),
    ("איפה כדאי לראות זריחה?", {"ramot_naftali_lookout"}),
    ("איפה אפשר לטייל בלי לשלם על כניסה?", {"har_meron_summit", "biriya_forest", "safed_old_city", "ramot_naftali_lookout"}),
    ("איפה עושים טיול ג'יפים?", {"nahal_dishon"}),
]


def evaluate(rows):
    """מדפיסה לכל שאלה: תנאים שחולצו, אתרים שנשלפו ופגיעה; ומסכמת פגיעה וכיסוי."""
    hits, coverage = [], []
    for i, ((q, expected), (conditions, results)) in enumerate(zip(QUESTIONS, rows), 1):
        sites = Counter(r.node.metadata["id"] for r in results)
        found = " ".join(f"{s}×{n}" for s, n in sites.most_common())
        mark = "—"
        if expected:
            hits.append(bool(expected & sites.keys()))
            mark = "✓" if hits[-1] else "✗"
            if len(expected) > 1:
                coverage.append(len(expected & sites.keys()) / len(expected))
        cond = ", ".join(conditions) or "-"
        print(f"{i:>2} {mark} תנאים: {cond:<22} | {found}")
    print(f"סיכום: פגיעה {sum(hits)}/{len(hits)} | כיסוי בשאלות מרובות-אתרים {sum(coverage)/len(coverage):.0%}")


if __name__ == "__main__":
    config.setup(llm=False)  # האחזור לא צריך LLM
    rows = [search(q) for q, _ in QUESTIONS]
    evaluate(rows)
