"""
Toxic Comment Detection — Streamlit Demo
Run with: streamlit run app.py

"""

# ── ALL imports at the top — no conditional imports anywhere ──────────────────
import os
import string
import warnings
import joblib                                          # always imported here
import numpy as np
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.multioutput import ClassifierChain

warnings.filterwarnings("ignore")

# ── Page config (must be the very first Streamlit call) ───────────────────────
st.set_page_config(
    page_title="Toxic Comment Detector",
    page_icon="🚫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── NLTK downloads ────────────────────────────────────────────────────────────
@st.cache_resource
def download_nltk():
    for r in ["stopwords", "wordnet", "omw-1.4"]:
        nltk.download(r, quiet=True)

download_nltk()

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
  html, body, [class*="css"]  { font-family: 'DM Sans', sans-serif; }
  h1, h2, h3                  { font-family: 'Space Mono', monospace !important; }
  .stApp                      { background: #0d0d0d; color: #f0ede8; }
  .hero-title   { font-family:'Space Mono',monospace; font-size:3rem; font-weight:700;
                  color:#f0ede8; line-height:1.1; margin-bottom:.25rem; }
  .hero-accent  { color:#ff4d4d; }
  .hero-sub     { color:#888; font-size:1rem; margin-bottom:2rem; }
  .label-pill          { display:inline-block; padding:4px 12px; border-radius:999px;
                         font-size:.78rem; font-weight:600; margin:3px; letter-spacing:.03em; }
  .pill-toxic          { background:#ff4d4d22; color:#ff4d4d; border:1px solid #ff4d4d55; }
  .pill-severe_toxic   { background:#ff000033; color:#ff6666; border:1px solid #ff000055; }
  .pill-obscene        { background:#ff8c0022; color:#ff8c00; border:1px solid #ff8c0055; }
  .pill-threat         { background:#ff006622; color:#ff4488; border:1px solid #ff006655; }
  .pill-insult         { background:#aa00ff22; color:#cc77ff; border:1px solid #aa00ff55; }
  .pill-identity_hate  { background:#0088ff22; color:#44aaff; border:1px solid #0088ff55; }
  .stTextArea textarea  { background:#1a1a1a !important; border:1px solid #333 !important;
                          border-radius:10px !important; color:#f0ede8 !important;
                          font-size:1rem !important; }
  .stButton > button    { background:#ff4d4d !important; color:white !important;
                          border:none !important; border-radius:8px !important;
                          font-weight:600 !important; font-family:'Space Mono',monospace !important;
                          letter-spacing:.04em !important; width:100%; font-size:.9rem !important; }
  .stButton > button:hover { background:#ff3333 !important; }
  div[data-testid="metric-container"] { background:#1a1a1a; border:1px solid #2a2a2a;
                                         border-radius:10px; padding:.8rem 1rem; }
  .sidebar-header { font-family:'Space Mono',monospace; font-size:.8rem; color:#666;
                    text-transform:uppercase; letter-spacing:.08em; margin-bottom:.5rem; }
  [data-testid="stSidebar"] { background:#111 !important; border-right:1px solid #222; }
  .stMarkdown p { color:#ccc; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
LABELS        = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
LABEL_DISPLAY = ["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"]
LABEL_COLORS  = ["#ff4d4d", "#ff6666", "#ff8c00", "#ff4488", "#cc77ff", "#44aaff"]

EXAMPLES = {
    "Clean comment": "This is a really well-written article. Thank you for sharing!",
    "Mild insult":   "You're being completely ridiculous and childish right now.",
    "Threat":        "I know where you live and I will make you regret this.",
    "Identity hate": "People like you don't belong in this country.",
    "Multi-label":   "You stupid idiot, go die. I hate people like you.",
}

# ── NLP tools ─────────────────────────────────────────────────────────────────
@st.cache_resource
def build_nlp_tools():
    punct   = string.punctuation.replace("'", "") + "0123456789"
    trantab = str.maketrans(punct, " " * len(punct))
    sw      = (set(stopwords.words("english"))
               | {""} | {chr(c) for c in range(ord("b"), ord("z") + 1)})
    return trantab, sw, WordNetLemmatizer(), PorterStemmer()

def clean_text(text, trantab, sw, lem, stm):
    text   = text.lower().translate(trantab)
    tokens = [stm.stem(lem.lemmatize(w, pos="v"))
              for w in text.split() if w not in sw]
    return " ".join(tokens)

# ── Demo training data ────────────────────────────────────────────────────────
DEMO_DATA = [
    ("you are such an idiot i cannot believe it",            [1,0,0,0,1,0]),
    ("this is absolutely terrible and i hate it",            [1,0,0,0,0,0]),
    ("what a waste of space you are",                        [1,0,0,0,1,0]),
    ("go away nobody wants you here loser",                  [1,0,0,0,1,0]),
    ("you are the dumbest person i have ever met",           [1,0,0,0,1,0]),
    ("i cannot stand people like you at all",                [1,0,0,0,0,0]),
    ("your opinion is garbage and so are you",               [1,0,0,0,1,0]),
    ("shut your mouth you have nothing useful to say",       [1,0,0,0,1,0]),
    ("you disgust me with your stupidity",                   [1,0,0,0,1,0]),
    ("this is pathetic and so is everything you do",         [1,0,0,0,0,0]),
    ("go kill yourself you worthless piece of trash",        [1,1,0,0,1,0]),
    ("i hope you die in the most painful way possible",      [1,1,0,1,0,0]),
    ("you should not exist drop dead immediately",           [1,1,0,0,0,0]),
    ("nobody would miss you if you were gone forever",       [1,1,0,0,1,0]),
    ("you deserve the worst fate imaginable you scum",       [1,1,0,0,1,0]),
    ("what the hell is wrong with you you damn fool",        [1,0,1,0,1,0]),
    ("this is absolute crap and you know it",                [1,0,1,0,0,0]),
    ("holy crap this is the worst thing i have read",        [1,0,1,0,0,0]),
    ("you are a complete ass and everyone knows it",         [1,0,1,0,1,0]),
    ("what the heck were you thinking you idiot",            [1,0,1,0,1,0]),
    ("i know where you live and i will find you",            [1,0,0,1,0,0]),
    ("you will regret saying that i promise you",            [1,0,0,1,0,0]),
    ("watch your back because i am coming for you",          [1,0,0,1,0,0]),
    ("i will make sure you pay for what you did",            [1,0,0,1,0,0]),
    ("keep talking and see what happens to you next",        [1,0,0,1,0,0]),
    ("you are a complete moron with no intelligence",        [1,0,0,0,1,0]),
    ("what a stupid ignorant fool you are",                  [1,0,0,0,1,0]),
    ("you have the brain of a rock seriously",               [1,0,0,0,1,0]),
    ("only a complete imbecile would think that",            [1,0,0,0,1,0]),
    ("you are embarrassingly dumb and everyone sees it",     [1,0,0,0,1,0]),
    ("people like you do not belong in this country",        [1,0,0,0,0,1]),
    ("your kind is ruining everything for everyone",         [1,0,0,0,0,1]),
    ("go back to where you came from we do not want you",    [1,0,0,0,0,1]),
    ("your religion is a plague on our society",             [1,0,0,0,0,1]),
    ("people of your race are always causing problems",      [1,0,0,0,0,1]),
    ("you stupid idiot go die i hate people like you",       [1,1,1,0,1,0]),
    ("i will destroy you and everyone you love you scum",    [1,1,0,1,1,0]),
    ("your kind should be eliminated you disgusting fool",   [1,1,0,1,1,1]),
    ("you belong in a ditch you worthless immigrant",        [1,1,0,0,1,1]),
    ("I really enjoyed reading this article thank you",      [0,0,0,0,0,0]),
    ("great work on the project well done everyone",         [0,0,0,0,0,0]),
    ("could you please clarify what you mean here",          [0,0,0,0,0,0]),
    ("this is a very interesting and thoughtful perspective",[0,0,0,0,0,0]),
    ("I disagree but I respect your opinion completely",     [0,0,0,0,0,0]),
    ("thanks so much for sharing this information today",    [0,0,0,0,0,0]),
    ("the weather today is quite pleasant and sunny",        [0,0,0,0,0,0]),
    ("I would like to learn more about this topic",          [0,0,0,0,0,0]),
    ("excellent contribution to the discussion today",       [0,0,0,0,0,0]),
    ("your analysis seems thorough and well researched",     [0,0,0,0,0,0]),
    ("I appreciate your thoughtful and detailed response",   [0,0,0,0,0,0]),
    ("looking forward to collaborating with you on this",    [0,0,0,0,0,0]),
    ("this needs more context before I can comment on it",   [0,0,0,0,0,0]),
    ("happy to help if you have any further questions",      [0,0,0,0,0,0]),
    ("the results look very promising based on the data",    [0,0,0,0,0,0]),
    ("what a wonderful day to be outside and enjoy life",    [0,0,0,0,0,0]),
    ("I found your explanation very clear and helpful",      [0,0,0,0,0,0]),
    ("the team did an outstanding job on this project",      [0,0,0,0,0,0]),
    ("could we schedule a meeting to discuss this further",  [0,0,0,0,0,0]),
    ("I think we should approach this problem differently",  [0,0,0,0,0,0]),
]

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():

    trantab, sw, lem, stm = build_nlp_tools()

    # Paths resolved relative to this script so it works from any working directory
    here       = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(here, "model.pkl")
    vec_path   = os.path.join(here, "vectoriser.pkl")

    if os.path.isfile(model_path) and os.path.isfile(vec_path):
        vec   = joblib.load(vec_path)
        model = joblib.load(model_path)
        mode  = "real"
    else:
        texts   = [row[0] for row in DEMO_DATA]
        Y       = np.array([row[1] for row in DEMO_DATA], dtype=int)
        cleaned = [clean_text(t, trantab, sw, lem, stm) for t in texts]

        vec   = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
        X     = vec.fit_transform(cleaned).toarray()
        base  = LinearSVC(C=1.0, class_weight="balanced", max_iter=2000, random_state=42)
        model = ClassifierChain(base, order="random", random_state=42)
        model.fit(X, Y)
        mode  = "demo"

    return vec, model, trantab, sw, lem, stm, mode

vec, model, trantab, sw, lem, stm, model_mode = load_model()

# ── Prediction ────────────────────────────────────────────────────────────────
def predict(text: str):
    cleaned    = clean_text(text, trantab, sw, lem, stm)
    X          = vec.transform([cleaned]).toarray()
    pred       = model.predict(X)[0].astype(int)
    scores_raw = []
    for clf in model.estimators_:
        try:
            scores_raw.append(float(clf.decision_function(X)[0]))
        except Exception:
            scores_raw.append(0.0)
    ordered = [0.0] * 6
    for chain_ix, label_ix in enumerate(model.order_):
        ordered[label_ix] = scores_raw[chain_ix]
    confidences = [round(float(1 / (1 + np.exp(-s))), 3) for s in ordered]
    return pred, confidences


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<p class="sidebar-header">Status</p>', unsafe_allow_html=True)
    if model_mode == "real":
        st.success("Using your trained model (model.pkl) ✓")
    else:
        st.info("Demo mode — add model.pkl + vectoriser.pkl for full accuracy.")

    st.divider()
    st.markdown('<p class="sidebar-header">About</p>', unsafe_allow_html=True)
    st.markdown(
        "Multi-label classifier detecting **6 types** of harmful language. "
        "Built with NLP + Machine Learning on the Jigsaw dataset."
    )
    st.divider()
    st.markdown('<p class="sidebar-header">Model</p>', unsafe_allow_html=True)
    st.markdown("""
- **Architecture:** Classifier Chains
- **Vectoriser:** TF-IDF (1–2 grams)
- **Classifier:** Linear SVM
- **Imbalance:** class_weight=balanced
    """)
    st.divider()
    st.markdown('<p class="sidebar-header">Pipeline</p>', unsafe_allow_html=True)
    st.markdown(
        "`text` → lowercase → strip punct  \n"
        "→ lemmatise → stem → stop words  \n"
        "→ TF-IDF → Classifier Chain → labels"
    )
  
    


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="margin-bottom:2rem;">
  <p class="hero-title">🚫 Toxic Comment<br><span class="hero-accent">Detector</span></p>
  <p class="hero-sub">Multi-label NLP classifier · 6 toxicity categories · Jigsaw dataset</p>
</div>
""", unsafe_allow_html=True)

# Example buttons
st.markdown("**Try an example or type your own:**")
cols     = st.columns(len(EXAMPLES))
selected = None
for col, (label, text) in zip(cols, EXAMPLES.items()):
    if col.button(label, key=f"ex_{label}", use_container_width=True):
        selected = text

# Text input
user_input = st.text_area(
    "Comment",
    value=selected or "",
    height=120,
    placeholder="Type or paste a comment here...",
    label_visibility="collapsed",
)

_, btn_col, _ = st.columns([2, 1, 2])
with btn_col:
    run = st.button("ANALYSE →", use_container_width=True)

# Results
if run and user_input.strip():
    preds, confidences = predict(user_input.strip())
    st.markdown("---")

    detected_keys  = [LABELS[i]        for i, p in enumerate(preds) if p == 1]
    detected_names = [LABEL_DISPLAY[i] for i, p in enumerate(preds) if p == 1]

    if detected_names:
        pills = " ".join(
            f'<span class="label-pill pill-{k}">{n}</span>'
            for k, n in zip(detected_keys, detected_names)
        )
        st.markdown(f"""
        <div style="background:#1a0a0a;border:1px solid #ff4d4d44;
                    border-radius:12px;padding:1rem 1.25rem;margin-bottom:1.5rem;">
          <p style="color:#ff4d4d;font-weight:600;margin:0 0 8px;
                    font-family:'Space Mono',monospace;">⚠ Toxic content detected</p>
          <div>{pills}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#0a1a0f;border:1px solid #00cc6644;
                    border-radius:12px;padding:1rem 1.25rem;margin-bottom:1.5rem;">
          <p style="color:#00cc66;font-weight:600;margin:0;
                    font-family:'Space Mono',monospace;">✓ No toxic content detected</p>
        </div>""", unsafe_allow_html=True)

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Confidence per label**")
        for i, (lbl, conf) in enumerate(zip(LABEL_DISPLAY, confidences)):
            active      = preds[i] == 1
            color       = LABEL_COLORS[i] if active else "#333"
            pct         = int(conf * 100)
            name_color  = "#f0ede8" if active else "#666"
            score_color = LABEL_COLORS[i] if active else "#555"
            st.markdown(f"""
            <div style="margin-bottom:10px;">
              <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                <span style="font-size:.85rem;color:{name_color};">{lbl}</span>
                <span style="font-size:.85rem;font-family:'Space Mono',monospace;
                             color:{score_color};">{pct}%</span>
              </div>
              <div style="background:#1a1a1a;border-radius:4px;height:6px;overflow:hidden;">
                <div style="width:{pct}%;height:100%;background:{color};border-radius:4px;"></div>
              </div>
            </div>""", unsafe_allow_html=True)

    with right:
        st.markdown("**Summary**")
        n_det    = int(preds.sum())
        dominant = LABEL_DISPLAY[int(np.argmax(confidences))] if n_det else "—"
        st.metric("Labels detected", f"{n_det} / 6")
        st.metric("Max confidence",  f"{max(confidences)*100:.0f}%")
        st.metric("Dominant label",  dominant)

    st.markdown("---")
    st.markdown("**Preprocessed tokens**")
    trantab_, sw_, lem_, stm_ = build_nlp_tools()
    tokens = clean_text(user_input.strip(), trantab_, sw_, lem_, stm_).split()
    if tokens:
        token_html = " ".join(
            f'<span style="background:#1a1a1a;border:1px solid #333;border-radius:6px;'
            f'padding:3px 9px;font-size:.8rem;font-family:monospace;color:#aaa;'
            f'margin:2px;display:inline-block;">{t}</span>'
            for t in tokens[:40]
        )
        st.markdown(f'<div style="line-height:2;">{token_html}</div>',
                    unsafe_allow_html=True)
        if len(tokens) > 40:
            st.caption(f"... and {len(tokens) - 40} more tokens")
    else:
        st.caption("No tokens remaining after preprocessing.")

elif run:
    st.warning("Please enter a comment first.")





