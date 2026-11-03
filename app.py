import streamlit as st
import config
from answer import answer
from loader import load_documents

NO_INFO = "אין לי מידע על כך במאגר"
FILTER_LABELS = {"accessible": "♿ נגיש", "water": "💧 מים", "circular": "🔁 מעגלי",
                 "entry_fee": "🎟️ כניסה חינם", "difficulty": "🥾 מסלול קל"}
EXAMPLES = ["איפה אפשר לראות עגורים?", "מסלול עם מים שמתאים לכיסא גלגלים",
            "איפה מטיילים בלי לשלם על כניסה?", "איפה כדאי לראות זריחה?"]

st.set_page_config(page_title="המדריך לגליל העליון", page_icon="🧭", layout="centered")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;600;700&display=swap');
:root {--accent: #10b981; --accent-dark: #059669; --text: #0f172a; --muted: #64748b; --line: #e9ecef; --soft: #f5f7f6;}
html, body, .stApp, p, li, h1, h2, h3, h4, label, input, textarea, button, summary, .chip, .site, .label, .foot {font-family: 'Heebo', sans-serif !important;}
[data-testid="stHeaderActionElements"] {display: none !important;}
.stApp {direction: rtl; text-align: right; color: var(--text); background: #ffffff;}
header[data-testid="stHeader"], [data-testid="stToolbar"], footer {display: none !important;}
.block-container {max-width: 720px; padding-top: 5rem; padding-bottom: 4rem;}

.brand {text-align: center; margin-bottom: 2.2rem;}
.brand .tag {display: inline-block; font-size: .8rem; font-weight: 500; color: var(--accent-dark);
             background: #ecfdf5; border-radius: 999px; padding: 4px 12px; margin-bottom: 14px;}
.brand h1 {font-size: 2.6rem; font-weight: 700; margin: 0 0 .5rem; letter-spacing: -.03em; color: var(--text);}
.brand p {color: var(--muted); font-size: 1.05rem; font-weight: 300; margin: 0;}

[data-testid="stForm"] {border: none; padding: 0;}
[data-testid="stTextInputRootElement"] {border-radius: 999px !important; border: 1px solid var(--line) !important;
        background: #fff !important; transition: border-color .15s, box-shadow .15s;}
[data-testid="stTextInputRootElement"]:focus-within {border-color: var(--accent) !important; box-shadow: 0 0 0 4px rgba(16,185,129,.12);}
[data-testid="stTextInputRootElement"] > div {background: transparent !important;}
[data-testid="stTextInputField"] {padding: 14px 22px !important; font-size: 1.05rem !important; direction: rtl; text-align: right;}
[data-testid="stFormSubmitButton"] button {border-radius: 999px; height: 50px; width: 100%; border: none;
        background: var(--accent); color: #fff; font-weight: 500; font-size: 1rem;}
[data-testid="stFormSubmitButton"] button:hover {background: var(--accent-dark); color: #fff;}

.stButton button {border-radius: 999px; border: 1px solid transparent; background: var(--soft);
                  color: var(--text); font-size: .92rem; font-weight: 400; padding: 8px 16px;}
.stButton button:hover {border-color: var(--accent); background: #fff; color: var(--accent-dark);}

.label {color: var(--muted); font-size: .8rem; font-weight: 500; margin: 2rem 0 .6rem;}
.chip {display: inline-block; border-radius: 999px; padding: 4px 12px; font-size: .82rem; margin: 0 0 8px 6px;
       background: #ecfdf5; color: var(--accent-dark);}
.site {display: inline-block; font-size: .85rem; color: var(--muted); margin: 0 0 6px 14px;}
.site::before {content: "•"; color: var(--accent); margin-left: 6px;}

.st-key-answer {border-radius: 18px !important; border: 1px solid var(--line) !important; background: #fff !important;
        padding: 22px 26px !important; margin-top: .6rem;}
.st-key-answer p, .st-key-answer li {font-size: 1.05rem; line-height: 1.9; color: var(--text);}
[data-testid="stExpander"] details {border-radius: 12px !important; border: 1px solid var(--line) !important; background: #fff;}
[data-testid="stExpander"] summary:hover {color: var(--accent-dark);}
.foot {text-align: center; color: #a0a9b4; font-size: .78rem; margin-top: 3rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init():
    """מגדירה את מודל ה-embedding פעם אחת (בלי קריאות ל-Gemini; ה-LLM נבחר בזמן שאלה)."""
    config.setup()


@st.cache_data(show_spinner=False)
def site_names():
    """שמות 12 האתרים שבמאגר."""
    return [d.metadata["name"] for d in load_documents()]


@st.cache_data(show_spinner=False)
def cached_answer(question):
    """תשובה לשאלה; שאלה שכבר נשאלה לא נשלחת שוב ל-LLM (חיסכון בטוקנים)."""
    return answer(question)


def use_example(text):
    """לחיצה על שאלה לדוגמה: ממלאת את תיבת החיפוש ושולחת."""
    st.session_state.query = text
    st.session_state.run = True


def show_result(res):
    """מציגה תגיות סינון, כרטיס תשובה, ורק את המקורות שהתשובה השתמשה בהם."""
    if res["conditions"]:
        st.markdown("".join(f'<span class="chip">{FILTER_LABELS.get(c, c)}</span>' for c in res["conditions"]),
                    unsafe_allow_html=True)
    with st.container(border=True, key="answer"):
        st.markdown(res["answer"])
    if res["sources"]:
        st.markdown(f'<div class="label">📍 מקורות ({len(res["sources"])})</div>', unsafe_allow_html=True)
        for s in res["sources"]:
            with st.expander(f"{s['site']} · {s['section']}"):
                st.caption(s["text"])
    elif NO_INFO in res["answer"]:
        st.caption("לא נמצאו מקורות רלוונטיים במאגר.")


init()
st.session_state.setdefault("query", "")

st.markdown('<div class="brand"><div class="tag">🧭 מדריך טיולים · מבוסס RAG</div><h1>המדריך לגליל העליון</h1>'
            '<p>שאלו שאלה על אחד מ-12 האתרים. כל שאלה נענית בנפרד, מתוך המאגר בלבד.</p></div>',
            unsafe_allow_html=True)

with st.form("ask"):
    col_input, col_button = st.columns([5, 1], vertical_alignment="bottom")
    col_input.text_input("שאלה", key="query", label_visibility="collapsed",
                         placeholder="למשל: איפה אפשר להירטב בקיץ?")
    submitted = col_button.form_submit_button("שאל")

question = st.session_state.query.strip()
if (submitted or st.session_state.pop("run", False)) and question:
    with st.spinner("מחפש במאגר…"):
        try:
            st.session_state.result = cached_answer(question)
            st.session_state.model = st.session_state.result["model"]
        except Exception:
            st.session_state.result = None
            st.warning("🌧️ השירות עמוס כרגע. נסו שוב בעוד דקה.")

result = st.session_state.get("result")
if result:
    show_result(result)
else:
    st.markdown('<div class="label">נסו לשאול</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        cols[i % 2].button(example, key=f"ex{i}", use_container_width=True, on_click=use_example, args=(example,))
    st.markdown('<div class="label">האתרים במדריך</div>' +
                "".join(f'<span class="site">{n}</span>' for n in site_names()), unsafe_allow_html=True)

model = st.session_state.get("model")
st.markdown(f'<div class="foot">RAG · LlamaIndex · Chroma · Gemini{" (" + model + ")" if model else ""}</div>',
            unsafe_allow_html=True)
