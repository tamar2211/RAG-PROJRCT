import time
import truststore
truststore.inject_into_ssl()  # אישורי האבטחה של Windows (נדרש ברשת עם סינון)

from dotenv import load_dotenv
from google.genai import types
from llama_index.core import Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

load_dotenv()

LLM_CHAIN = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash",
             "gemini-3.5-flash", "gemini-3.1-flash-lite"]  # מהחזק לחלש; לכל מודל מכסה נפרדת
EMBED_MODEL = "gemini-embedding-2"
RETRY_STRONGER_AFTER = 600  # אחרי 10 דקות מנסים שוב מהמודל החזק ביותר

_state = {"index": 0, "since": time.time()}  # המודל האחרון שעבד


def make_llm(name):
    """יוצרת LLM של Gemini עם הגבלת זמן וניסיון אחד, כדי שמודל עמוס ייכשל מהר ונעבור לבא."""
    try:
        return GoogleGenAI(model=name, temperature=0.1, max_retries=1,
                           http_options=types.HttpOptions(timeout=20_000,
                                                          retry_options=types.HttpRetryOptions(attempts=1)))
    except TypeError:  # גרסת ספרייה שלא מקבלת את הפרמטרים האלה
        return GoogleGenAI(model=name, temperature=0.1)


def chat(messages):
    """שולחת ל-LLM, החל מהמודל האחרון שעבד; אם הוא עמוס או חסום, עוברת לבא בשרשרת. מחזירה (טקסט, שם המודל)."""
    if time.time() - _state["since"] > RETRY_STRONGER_AFTER:
        _state.update(index=0, since=time.time())
    for i in range(_state["index"], len(LLM_CHAIN)):
        name = LLM_CHAIN[i]
        try:
            text = make_llm(name).chat(messages).message.content
            if i != _state["index"]:
                _state.update(index=i, since=time.time())
            return text, name
        except Exception as e:
            print(f"[config] {name} לא זמין: {str(e)[:60]}")
    _state.update(index=0, since=time.time())
    raise RuntimeError("אף מודל בשרשרת לא זמין כרגע.")


def setup(llm=None):
    """מגדירה את מודל ה-embedding. ה-LLM נבחר רק בזמן שאלה (chat), בלי בדיקה מראש שצורכת מכסה."""
    Settings.embed_model = GoogleGenAIEmbedding(model_name=EMBED_MODEL, timeout=30, retries=2)
    print(f"[config] Embeddings: {EMBED_MODEL} | LLM: נבחר בזמן שאלה מתוך {len(LLM_CHAIN)} מודלים")


if __name__ == "__main__":
    from llama_index.core.llms import ChatMessage
    setup()
    print(chat([ChatMessage(role="user", content="ענה במילה אחת: מה בירת ישראל?")]))
