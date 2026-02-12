import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from collections import Counter
import re
from typing import Optional, Dict, Tuple, Any

# ============================================================
# i18n (DE/EN)
# ============================================================
def init_lang():
    if "lang" not in st.session_state:
        st.session_state["lang"] = "de"

init_lang()

I18N = {
    "de": {
        # UI
        "lang_toggle": "EN Version",
        "page_title": "ONE Redirector",
        "title": "ONE Redirector – finde die passenden Redirect-Ziele 🔀",
        "help_expander": "ℹ️ Was macht das Tool? (Erklärung & Tipps)",

        "step_upload": "1. Dateien hochladen",
        "upload_old": "Datei mit den URLs, die weitergeleitet werden sollen (CSV oder Excel)",
        "upload_new": "Datei mit den Ziel-URLs (CSV oder Excel)",
        "need_address": "Beide Dateien müssen eine 'Address'-Spalte enthalten.",

        "step_method": "2. Matching-Methode wählen",
        "method_label": "Wie möchtest du matchen?",
        "method_exact": "Exact Match",
        "method_faiss": "Semantisches Matching mit FAISS (Schneller, für große Datenmengen geeignet)",
        "method_sklearn": "Semantisches Matching mit sklearn (Arbeitet gründlicher, aber langsamer)",

        "step_embed_source": "3. Embedding-Quelle",
        "embed_source_label": "Stellst du die Embeddings bereits zur Verfügung oder sollen sie erstellt werden?",
        "embed_create": "Embeddings müssen basierend auf meinen Input-Dateien erst noch erstellt werden",
        "embed_existing": "Embeddings sind bereits generiert und in Input-Dateien vorhanden",

        # Intuitive model dropdown
        "model_dropdown_label": "Welche Embedding-Qualität brauchst du?",
        "model_fast": "Schnell (MiniLM-L6) – für große Projekte, sehr flott",
        "model_balanced": "Balanced (MiniLM-L12) – besser als L6, noch schnell",
        "model_modern": "Modern Balanced (GTE-base) – sehr gute Semantik, CPU-tauglich",
        "model_quality": "Gründlich (MPNet) – beste Qualität, langsamer",

        "step_cols": "4. Spaltenauswahl",
        "exact_cols": "Spalten für Exact Match auswählen",

        "embed_col_header": "#### Embedding-Spaltenauswahl",
        "no_embed_col": "Keine Embedding-Spalte gefunden. Benenne deine Spalten z. B. 'Embeddings'.",
        "embed_col_old": "Embedding-Spalte (OLD)",
        "embed_col_new": "Embedding-Spalte (NEW)",

        # Simplified existing-embeddings UX
        "auto_dim_found": "Erkannte Embedding-Dimension (häufigster Wert): **{dim}**",
        "advanced_settings": "⚙️ Erweiterte Einstellungen (optional)",
        "override_dim": "Embedding-Dimension manuell überschreiben",
        "dim_input": "Expected Embedding Dimension",
        "dim_help": "Normalerweise automatisch erkannt. Überschreibe nur, wenn du sicher bist.",
        "padding_label": "Fehlende Werte mit 0 auffüllen (Padding)",
        "padlimit_label": "Max. Anteil fehlender Werte pro Zeile, der gepaddet werden darf",
        "padlimit_help": "Beispiel: 0.1 = bis zu 10% Padding pro Zeile.",
        "dims_diag": "Embedding-Dimensionen anzeigen (Diagnose)",
        "dims_old": "OLD dims:",
        "dims_new": "NEW dims:",

        "similarity_cols": "Spalten für semantisches Matching auswählen – auf Basis dieser Inhalte werden die Embeddings erstellt und verglichen",
        "need_similarity_cols": "Bitte wähle mindestens eine Spalte für das semantische Matching aus.",

        # FAISS (basic + expert)
        "faiss_settings": "#### FAISS Einstellungen",
        "faiss_ivf": "Schnellmodus aktivieren (IVF Flat) – empfohlen ab ~2.000 Ziel-URLs",
        "faiss_ivf_help": "IVF Flat nutzt Clustering (Approximate Search). Deutlich schneller bei großen Korpora, minimal ungenauer möglich.",
        "faiss_expert": "⚙️ Expertenmodus (optional)",
        "faiss_expert_desc": "Nur anfassen, wenn du weißt, was du tust. Standard ist automatische Optimierung.",
        "faiss_auto_on": "Automatische Optimierung aktiv (empfohlen).",
        "faiss_nlist": "nlist (Anzahl Cluster)",
        "faiss_nlist_help": "Mehr Cluster = feinere Vorselektion, mehr Speicher/Trainingszeit.",
        "faiss_nprobe": "nprobe (Cluster pro Suche)",
        "faiss_nprobe_help": "Mehr nprobe = höherer Recall, aber langsamer.",
        "faiss_use_custom": "Ich möchte nlist/nprobe manuell setzen",

        "step_threshold": "5. Cosine Similarity Schwelle",
        "threshold_label": "Minimaler Score für semantisches Matching – Empfehlung: mind. 0.75",

        "go": "Let's Go",

        # Better loading UX
        "loading_model": "Modell wird geladen: **{model}**",
        "building_texts": "Texte werden vorbereitet …",
        "encoding_embeddings": "Embeddings werden berechnet … (das kann je nach Datenmenge dauern)",
        "encoding_done": "Embeddings fertig berechnet.",

        "model_dim": "Embedding-Dimension des gewählten Modells: **{dim}**",

        "need_embed_selection": "Bitte oben die Embedding-Spalten auswählen.",
        "embed_parse_failed": "Embeddings konnten nicht zuverlässig geparst werden.",
        "ivf_fail_fallback": "IVF-Training fehlgeschlagen oder Datensatz zu klein. Fallback auf Flat-IP. Details: {err}",
        "faiss_ivf_active": "FAISS IVF Flat aktiv • nlist={nlist} • nprobe={nprobe} • Korpus={n}",
        "faiss_flat_active": "FAISS Flat (exakt) aktiv • Korpus={n}",

        # user-friendly parsing summary (Variante 1)
        "parse_summary_ok": "**{used} von {total} URLs konnten für das semantische Matching verwendet werden.**",
        "parse_summary_drop": "{dropped} URLs wurden wegen fehlerhafter oder unvollständiger Embeddings ignoriert.",

        "results_header": "🔽 Ergebnisse",
        "download": "📥 Ergebnisse als CSV herunterladen",
        "download_name": "redirect_mapping_result.csv",

        # OUTPUT HEADERS (DE)
        "out_old_url": "Alte URL",
        "out_match_type": "Match-Typ",
        "out_match_basis": "Matching-Basis",
        "out_matched_url": "Ziel-URL {rank}",
        "out_score": "Cosine-Score {rank}",
        "out_note": "Hinweis",
        "out_no_match": "Kein Match",
        "out_exact_prefix": "Exact Match ({col})",
        "out_similarity": "Similarity ({engine})",

        # Notes
        "note_dropped_old": "Embedding ungültig/inkonsistent (dim={dim}) → URL wurde im semantischen Matching ignoriert.",
        "note_bad_new": "Einige Ziel-URLs wurden ignoriert (ungültige Embeddings).",
        "note_below_threshold": "Kein Treffer über dem Threshold.",
        "note_no_semantic_run": "Semantisches Matching konnte nicht ausgeführt werden (fehlende Spalten/Embeddings).",
        "warn_batch_fallback": "Zu wenig Speicher für batch_size=64 – ich nutze batch_size=32.",
    },
    "en": {
        # UI
        "lang_toggle": "English",
        "page_title": "ONE Redirector",
        "title": "ONE Redirector – find the best redirect targets 🔀",
        "help_expander": "ℹ️ What does this tool do? (Explanation & tips)",

        "step_upload": "1. Upload files",
        "upload_old": "File with URLs to redirect (CSV or Excel)",
        "upload_new": "File with target URLs (CSV or Excel)",
        "need_address": "Both files must contain an 'Address' column.",

        "step_method": "2. Choose matching method",
        "method_label": "How would you like to match?",
        "method_exact": "Exact Match",
        "method_faiss": "Semantic matching with FAISS (faster, best for large datasets)",
        "method_sklearn": "Semantic matching with sklearn (more thorough, but slower)",

        "step_embed_source": "3. Embedding source",
        "embed_source_label": "Are embeddings already available, or should the tool create them?",
        "embed_create": "Embeddings should be created from my input files",
        "embed_existing": "Embeddings already exist inside my input files",

        # Intuitive model dropdown
        "model_dropdown_label": "Which embedding quality do you need?",
        "model_fast": "Fast (MiniLM-L6) – very quick for big projects",
        "model_balanced": "Balanced (MiniLM-L12) – better than L6, still fast",
        "model_modern": "Modern Balanced (GTE-base) – strong semantics, CPU-friendly",
        "model_quality": "High Quality (MPNet) – best quality, slower",

        "step_cols": "4. Column selection",
        "exact_cols": "Select columns for Exact Match",

        "embed_col_header": "#### Embedding column selection",
        "no_embed_col": "No embedding column found. Please name your column e.g. 'Embeddings'.",
        "embed_col_old": "Embedding column (OLD)",
        "embed_col_new": "Embedding column (NEW)",

        # Simplified existing-embeddings UX
        "auto_dim_found": "Detected embedding dimension (most common): **{dim}**",
        "advanced_settings": "⚙️ Advanced settings (optional)",
        "override_dim": "Override embedding dimension manually",
        "dim_input": "Expected embedding dimension",
        "dim_help": "Usually auto-detected. Override only if you're sure.",
        "padding_label": "Pad missing values with zeros",
        "padlimit_label": "Max share of missing values per row that may be padded",
        "padlimit_help": "Example: 0.1 = up to 10% padding per row.",
        "dims_diag": "Show embedding dimensions (diagnostics)",
        "dims_old": "OLD dims:",
        "dims_new": "NEW dims:",

        "similarity_cols": "Select columns for semantic matching – embeddings will be created from these fields and compared",
        "need_similarity_cols": "Please select at least one column for semantic matching.",

        # FAISS (basic + expert)
        "faiss_settings": "#### FAISS settings",
        "faiss_ivf": "Enable fast mode (IVF Flat) – recommended above ~2,000 target URLs",
        "faiss_ivf_help": "IVF Flat uses clustering (approximate search). Much faster on large corpora, potentially slightly less exact.",
        "faiss_expert": "⚙️ Expert mode (optional)",
        "faiss_expert_desc": "Only change if you know what you're doing. Default is automatic optimization.",
        "faiss_auto_on": "Automatic optimization is enabled (recommended).",
        "faiss_nlist": "nlist (number of clusters)",
        "faiss_nlist_help": "More clusters = finer preselection, more memory/training time.",
        "faiss_nprobe": "nprobe (clusters per query)",
        "faiss_nprobe_help": "Higher nprobe = better recall, but slower.",
        "faiss_use_custom": "I want to set nlist/nprobe manually",

        "step_threshold": "5. Cosine similarity threshold",
        "threshold_label": "Minimum score for semantic matching – recommendation: at least 0.75",

        "go": "Let's Go",

        # Better loading UX
        "loading_model": "Loading model: **{model}**",
        "building_texts": "Preparing texts …",
        "encoding_embeddings": "Computing embeddings … (this may take a while depending on dataset size)",
        "encoding_done": "Embeddings computed.",

        "model_dim": "Embedding dimension of the selected model: **{dim}**",

        "need_embed_selection": "Please select embedding columns above.",
        "embed_parse_failed": "Embeddings could not be parsed reliably.",
        "ivf_fail_fallback": "IVF training failed or dataset too small. Falling back to Flat-IP. Details: {err}",
        "faiss_ivf_active": "FAISS IVF Flat active • nlist={nlist} • nprobe={nprobe} • corpus={n}",
        "faiss_flat_active": "FAISS Flat (exact) active • corpus={n}",

        # user-friendly parsing summary (Variant 1)
        "parse_summary_ok": "**{used} of {total} URLs could be used for semantic matching.**",
        "parse_summary_drop": "{dropped} URLs were ignored due to invalid or incomplete embeddings.",

        "results_header": "🔽 Results",
        "download": "📥 Download results as CSV",
        "download_name": "redirect_mapping_result.csv",

        # OUTPUT HEADERS (EN)
        "out_old_url": "Old URL",
        "out_match_type": "Match Type",
        "out_match_basis": "Match Basis",
        "out_matched_url": "Matched URL {rank}",
        "out_score": "Cosine Similarity Score {rank}",
        "out_note": "Note",
        "out_no_match": "No Match",
        "out_exact_prefix": "Exact Match ({col})",
        "out_similarity": "Similarity ({engine})",

        # Notes
        "note_dropped_old": "Invalid/inconsistent embedding (dim={dim}) → URL was ignored in semantic matching.",
        "note_bad_new": "Some target URLs were ignored (invalid embeddings).",
        "note_below_threshold": "No matches above the threshold.",
        "note_no_semantic_run": "Semantic matching could not run (missing columns/embeddings).",
        "warn_batch_fallback": "Not enough memory for batch_size=64 – using batch_size=32.",
    },
}

HELP_MD = {
    "de": """
**Ziel:** Dieses Tool hilft dir dabei, bei **Relaunches** oder **Domain-Migrationen** passende Redirect-Ziele auf Knopfdruck zu finden.

**Du hast die Wahl zwischen:**
- **Exact Match**: 1:1-Abgleich über identische Werte in Spalten, z. B. die H1 bleibt vor und nach dem Relaunch gleich
- **Semantisches Matching**: Zuordnung über inhaltliche Ähnlichkeit (Embeddings)

**Hinweis:** Du kannst entweder schon Embeddings mitbringen oder du lässt sie vom Tool auf Basis ausgewählter Spalten deiner Input-Dateien erstellen.
""",
    "en": """
**Goal:** This tool helps you quickly find good redirect targets for **relaunches** or **domain migrations**.

**The choice is yours:**
- **Exact Match**: 1:1 matching via identical values in selected columns, i.e. H1 stays the same
- **Semantic matching**: mapping via content similarity (embeddings)

**Note:** You can either provide precomputed embeddings or have the tool generate them based on selected columns from your input files.
""",
}

def t(key: str, **kwargs) -> str:
    lang = st.session_state.get("lang", "de")
    s = I18N[lang].get(key, key)
    return s.format(**kwargs)

# ============================================================
# Model mapping (intuitive labels -> HF model name)
# ============================================================
def model_options():
    return [
        (t("model_fast"), "all-MiniLM-L6-v2"),
        (t("model_balanced"), "all-MiniLM-L12-v2"),
        (t("model_modern"), "thenlper/gte-base"),
        (t("model_quality"), "all-mpnet-base-v2"),
    ]

# ============================================================
# Parsing / Utility
# ============================================================
float_re = re.compile(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?')

def parse_series_to_matrix(
    series: pd.Series,
    expected_dim: int,
    allow_padding: bool = True,
    pad_limit_ratio: float = 0.1,
    label: str = ""
) -> Tuple[Optional[np.ndarray], Optional[list], Dict[Any, Dict[str, Any]]]:
    vecs, idxs = [], []
    dropped_info: Dict[Any, Dict[str, Any]] = {}
    dropped_parse = dropped_too_short = dropped_pad_limit = 0

    for idx, val in series.items():
        if pd.isna(val):
            dropped_parse += 1
            dropped_info[idx] = {"reason": "parse", "dim": 0}
            continue

        nums = float_re.findall(str(val))
        if not nums:
            dropped_parse += 1
            dropped_info[idx] = {"reason": "parse", "dim": 0}
            continue

        try:
            arr = np.array([float(x) for x in nums], dtype="float32")
        except ValueError:
            dropped_parse += 1
            dropped_info[idx] = {"reason": "parse", "dim": 0}
            continue

        L = len(arr)
        if L == expected_dim:
            vecs.append(arr); idxs.append(idx)
        elif L > expected_dim:
            vecs.append(arr[:expected_dim]); idxs.append(idx)
        else:
            missing = expected_dim - L
            missing_ratio = missing / expected_dim
            if allow_padding and missing_ratio <= pad_limit_ratio:
                arr = np.pad(arr, (0, missing), 'constant', constant_values=0.0)
                vecs.append(arr); idxs.append(idx)
            else:
                if missing_ratio > pad_limit_ratio:
                    dropped_pad_limit += 1
                    dropped_info[idx] = {"reason": "pad_limit", "dim": L}
                else:
                    dropped_too_short += 1
                    dropped_info[idx] = {"reason": "too_short", "dim": L}

    used = len(vecs)
    total = len(series)
    if used == 0:
        return None, None, dropped_info

    dropped_total = total - used
    st.success(t("parse_summary_ok", used=used, total=total))
    if dropped_total > 0:
        st.warning(t("parse_summary_drop", dropped=dropped_total))

    return np.vstack(vecs), list(idxs), dropped_info

def count_dims(series: pd.Series):
    dims = []
    for val in series.dropna():
        nums = float_re.findall(str(val))
        dims.append(len(nums))
    return Counter(dims)

def infer_expected_dim(*series_list: pd.Series) -> Optional[int]:
    combined = Counter()
    for s in series_list:
        combined.update(count_dims(s))
    if 0 in combined:
        del combined[0]
    if not combined:
        return None
    mode_count = max(combined.values())
    candidates = [d for d, c in combined.items() if c == mode_count]
    return max(candidates)

# ============================================================
# Streamlit Layout
# ============================================================
st.set_page_config(page_title=t("page_title"), layout="wide")

# Language toggle
colA, colB = st.columns([1, 6])
with colA:
    is_en = st.toggle(t("lang_toggle"), value=(st.session_state["lang"] == "en"))
    st.session_state["lang"] = "en" if is_en else "de"

# Logo
st.markdown("""
<div style="display: flex; align-items: center; gap: 20px;">
  <img src="https://onebeyondsearch.com/img/ONE_beyond_search%C3%94%C3%87%C3%B4gradient%20%282%29.png" alt="ONE Beyond Search" style="height: 60px;">
</div>
""", unsafe_allow_html=True)

st.title(t("title"))

with st.expander(t("help_expander"), expanded=False):
    st.markdown(HELP_MD[st.session_state["lang"]])

# ============================================================
# URL detection / normalization
# ============================================================
URL_COL_SYNONYMS = [
    "address", "url", "page", "landing page", "seiten-url", "seite",
    "ziel-url", "ziel", "canonical", "canonical url"
]

def _looks_like_url_series(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(50)
    hits = 0
    for v in sample:
        v = v.strip()
        if v.startswith("http://") or v.startswith("https://") or ("/" in v and "." in v):
            hits += 1
    return hits >= max(3, int(len(sample) * 0.2))

def detect_url_column(df: pd.DataFrame) -> Optional[str]:
    cols = [c.strip() for c in df.columns]
    lower_map = {c.lower(): c for c in cols}
    for key in URL_COL_SYNONYMS:
        if key in lower_map and _looks_like_url_series(df[lower_map[key]]):
            return lower_map[key]
    for c in cols:
        if any(k in c.lower() for k in URL_COL_SYNONYMS) and _looks_like_url_series(df[c]):
            return c
    for c in cols:
        if _looks_like_url_series(df[c]):
            return c
    return None

def normalize_url_column(df: pd.DataFrame) -> pd.DataFrame:
    url_col = detect_url_column(df)
    if url_col and url_col != "Address" and "Address" not in df.columns:
        df = df.rename(columns={url_col: "Address"})
    return df

def load_file(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
    else:
        df = pd.read_excel(uploaded_file)
    df.columns = [c.strip() for c in df.columns]
    df = normalize_url_column(df)
    return df

# ============================================================
# File upload
# ============================================================
st.subheader(t("step_upload"))
uploaded_old = st.file_uploader(t("upload_old"), type=["csv", "xlsx"], key="old")
uploaded_new = st.file_uploader(t("upload_new"), type=["csv", "xlsx"], key="new")

if uploaded_old and uploaded_new:
    df_old = load_file(uploaded_old)
    df_new = load_file(uploaded_new)

    if 'Address' not in df_old.columns or 'Address' not in df_new.columns:
        st.error(t("need_address"))
        st.stop()

    # ============================================================
    # Matching method
    # ============================================================
    st.subheader(t("step_method"))
    matching_method = st.selectbox(
        t("method_label"),
        [t("method_exact"), t("method_faiss"), t("method_sklearn")]
    )

    # ============================================================
    # Embedding source (semantic only)
    # ============================================================
    if matching_method != t("method_exact"):
        st.subheader(t("step_embed_source"))
        embedding_choice = st.radio(
            t("embed_source_label"),
            [t("embed_create"), t("embed_existing")]
        )

        if embedding_choice == t("embed_create"):
            opts = model_options()
            labels = [x[0] for x in opts]
            label_to_model = {lab: mod for lab, mod in opts}
            chosen_label = st.selectbox(t("model_dropdown_label"), labels, index=1)  
            model_name = label_to_model[chosen_label]
        else:
            model_name = None
    else:
        embedding_choice = None
        model_name = None

    # ============================================================
    # Column selection
    # ============================================================
    st.subheader(t("step_cols"))
    common_cols = sorted(list(set(df_old.columns) & set(df_new.columns)))

    # Exact columns only when method_exact selected
    if matching_method == t("method_exact"):
        exact_cols = st.multiselect(t("exact_cols"), common_cols)
    else:
        exact_cols = []

    # Embedding columns (existing embeddings)
    if matching_method != t("method_exact") and embedding_choice == t("embed_existing"):
        st.markdown(t("embed_col_header"))
        cand_old = [c for c in df_old.columns if 'embed' in c.lower()]
        cand_new = [c for c in df_new.columns if 'embed' in c.lower()]

        if not cand_old or not cand_new:
            st.error(t("no_embed_col"))
            st.stop()

        emb_col_old = st.selectbox(t("embed_col_old"), cand_old, index=0)
        emb_col_new = st.selectbox(t("embed_col_new"), cand_new, index=0)

        suggested_dim = infer_expected_dim(df_old[emb_col_old], df_new[emb_col_new]) or 768
        st.caption(t("auto_dim_found", dim=int(suggested_dim)))

        # Defaults (simple)
        expected_dim = int(suggested_dim)
        allow_padding = True
        pad_limit_ratio = 0.1

        with st.expander(t("advanced_settings"), expanded=False):
            override = st.checkbox(t("override_dim"), value=False)
            if override:
                expected_dim = st.number_input(
                    t("dim_input"),
                    min_value=8, max_value=4096, value=int(suggested_dim), step=8,
                    help=t("dim_help")
                )
            allow_padding = st.checkbox(t("padding_label"), value=True)
            pad_limit_ratio = st.slider(
                t("padlimit_label"),
                min_value=0.0, max_value=0.9, value=float(pad_limit_ratio), step=0.05,
                help=t("padlimit_help")
            )
            with st.expander(t("dims_diag")):
                st.write(t("dims_old"), dict(count_dims(df_old[emb_col_old])))
                st.write(t("dims_new"), dict(count_dims(df_new[emb_col_new])))

    else:
        emb_col_old = None
        emb_col_new = None
        expected_dim = None
        allow_padding = True
        pad_limit_ratio = 0.1

    # Text columns for on-the-fly embeddings
    if matching_method != t("method_exact") and embedding_choice == t("embed_create"):
        similarity_cols = st.multiselect(t("similarity_cols"), common_cols)
    else:
        similarity_cols = []

    # ============================================================
    # FAISS settings (basic + expert mode in expander)
    # ============================================================
    use_faiss_ivf = False
    faiss_custom = False
    faiss_nlist = None
    faiss_nprobe = None

    if matching_method == t("method_faiss"):
        st.markdown(t("faiss_settings"))

        use_faiss_ivf = st.checkbox(
            t("faiss_ivf"),
            value=True,
            help=t("faiss_ivf_help")
        )

        # Expert mode: only if IVF is enabled
        if use_faiss_ivf:
            with st.expander(t("faiss_expert"), expanded=False):
                st.caption(t("faiss_expert_desc"))
                faiss_custom = st.checkbox(t("faiss_use_custom"), value=False)

                if not faiss_custom:
                    st.info(t("faiss_auto_on"))
                else:
                    # Suggested defaults based on dataset size (NEW targets)
                    est_nlist = int(np.clip(int(np.sqrt(max(1, len(df_new))) * 2), 100, 16384))
                    default_nprobe = int(np.clip(max(1, est_nlist // 10), 1, 64))

                    faiss_nlist = st.number_input(
                        t("faiss_nlist"),
                        min_value=1, max_value=16384, value=int(est_nlist), step=1,
                        help=t("faiss_nlist_help")
                    )
                    faiss_nprobe = st.number_input(
                        t("faiss_nprobe"),
                        min_value=1, max_value=max(1, int(faiss_nlist)), value=int(default_nprobe), step=1,
                        help=t("faiss_nprobe_help")
                    )

    # ============================================================
    # Threshold (semantic only)  ✅ default 0.75
    # ============================================================
    if matching_method != t("method_exact"):
        st.subheader(t("step_threshold"))
        threshold = st.slider(t("threshold_label"), 0.0, 1.0, 0.75, 0.01)
    else:
        threshold = 0.75  # not used for exact, but keep consistent

    # ============================================================
    # Start
    # ============================================================
    if st.button(t("go"), type="primary"):
        results = []
        matched_old = set()

        # Output column names (language-specific)
        COL_OLD = t("out_old_url")
        COL_TYPE = t("out_match_type")
        COL_BASIS = t("out_match_basis")
        COL_NOTE = t("out_note")

        def col_matched(rank: int) -> str:
            return t("out_matched_url", rank=rank)

        def col_score(rank: int) -> str:
            return t("out_score", rank=rank)

        old_notes: Dict[str, str] = {}

        # 1) Exact matching
        if matching_method == t("method_exact") and exact_cols:
            for col in exact_cols:
                exact_matches = pd.merge(
                    df_old[["Address", col]],
                    df_new[["Address", col]],
                    on=col,
                    how="inner"
                )
                for _, row in exact_matches.iterrows():
                    old_url = row["Address_x"]
                    results.append({
                        COL_OLD: old_url,
                        col_matched(1): row["Address_y"],
                        COL_TYPE: t("out_exact_prefix", col=col),
                        col_score(1): 1.0,
                        COL_BASIS: f"{col}: {row[col]}",
                    })
                    matched_old.add(old_url)

        df_remaining = df_old[~df_old['Address'].isin(matched_old)].reset_index(drop=True)

        semantic_ran = False
        had_invalid_new = False

        # 2) Semantic matching
        if matching_method != t("method_exact") and df_remaining.shape[0] > 0:
            emb_old_mat = None
            emb_new_mat = None
            df_remaining_used = df_remaining.copy()
            df_new_used = df_new.copy()

            # On-the-fly embeddings
            if embedding_choice == t("embed_create"):
                if not similarity_cols:
                    for u in df_remaining["Address"].tolist():
                        old_notes[u] = t("note_no_semantic_run")
                else:
                    status = st.status(t("loading_model", model=model_name), expanded=False)
                    with status:
                        st.write(t("building_texts"))
                        model = SentenceTransformer(model_name)
                        expected_dim_gen = model.get_sentence_embedding_dimension()
                        st.caption(t("model_dim", dim=expected_dim_gen))

                        df_remaining_used['text'] = df_remaining_used[similarity_cols].fillna('').agg(' '.join, axis=1)
                        df_new_used['text'] = df_new_used[similarity_cols].fillna('').agg(' '.join, axis=1)

                        st.write(t("encoding_embeddings"))
                        try:
                            emb_old_mat = model.encode(
                                df_remaining_used["text"].tolist(),
                                show_progress_bar=False,
                                batch_size=64
                            )
                            emb_new_mat = model.encode(
                                df_new_used["text"].tolist(),
                                show_progress_bar=False,
                                batch_size=64
                            )
                        except RuntimeError:
                            st.warning(t("warn_batch_fallback"))
                            emb_old_mat = model.encode(
                                df_remaining_used["text"].tolist(),
                                show_progress_bar=False,
                                batch_size=32
                            )
                            emb_new_mat = model.encode(
                                df_new_used["text"].tolist(),
                                show_progress_bar=False,
                                batch_size=32
                            )

                        st.success(t("encoding_done"))
                    status.update(state="complete")
                    semantic_ran = True

            # Existing embeddings
            elif embedding_choice == t("embed_existing"):
                if not emb_col_old or not emb_col_new:
                    st.error(t("need_embed_selection"))
                    st.stop()

                emb_old_mat, rows_old, dropped_old = parse_series_to_matrix(
                    df_remaining[emb_col_old],
                    expected_dim=int(expected_dim),
                    allow_padding=allow_padding,
                    pad_limit_ratio=float(pad_limit_ratio),
                    label="OLD"
                )
                emb_new_mat, rows_new, dropped_new = parse_series_to_matrix(
                    df_new[emb_col_new],
                    expected_dim=int(expected_dim),
                    allow_padding=allow_padding,
                    pad_limit_ratio=float(pad_limit_ratio),
                    label="NEW"
                )

                if emb_old_mat is None or emb_new_mat is None or rows_old is None or rows_new is None:
                    st.error(t("embed_parse_failed"))
                    st.stop()

                for idx, info in dropped_old.items():
                    try:
                        old_url = df_remaining.loc[idx, "Address"]
                        dim = int(info.get("dim", 0))
                        old_notes[old_url] = t("note_dropped_old", dim=dim)
                    except Exception:
                        pass

                if len(dropped_new) > 0:
                    had_invalid_new = True

                df_remaining_used = df_remaining.loc[rows_old].reset_index(drop=True)
                df_new_used = df_new.loc[rows_new].reset_index(drop=True)
                semantic_ran = True

            # Similarity computation
            if semantic_ran and emb_old_mat is not None and emb_new_mat is not None and len(df_new_used) > 0 and len(df_remaining_used) > 0:
                if matching_method == t("method_sklearn"):
                    sim_matrix = cosine_similarity(emb_old_mat, emb_new_mat)

                    def get_topk(i):
                        row_scores = sim_matrix[i]
                        top_indices = np.argsort(row_scores)[::-1][:5]
                        return top_indices, row_scores[top_indices]

                    engine_for_type = "sklearn"
                else:
                    def _l2norm(x, eps=1e-12):
                        return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)

                    emb_new_norm = _l2norm(np.asarray(emb_new_mat)).astype("float32")
                    emb_old_norm = _l2norm(np.asarray(emb_old_mat)).astype("float32")
                    dim = emb_new_norm.shape[1]
                    k = min(5, len(df_new_used))

                    index = None
                    used_ivf = False

                    # ---- AUTO: compute nlist/nprobe unless expert overrides ----
                    N = len(df_new_used)
                    nlist_auto = int(np.clip(int(np.sqrt(max(1, N)) * 2), 100, 16384))
                    nprobe_auto = int(np.clip(max(1, nlist_auto // 10), 1, 64))

                    nlist_eff = int(faiss_nlist) if (faiss_custom and faiss_nlist is not None) else nlist_auto
                    nprobe_eff = int(faiss_nprobe) if (faiss_custom and faiss_nprobe is not None) else nprobe_auto

                    # IVF only if enabled + enough data to train well
                    if use_faiss_ivf and N >= max(1000, int(nlist_eff) * 5):
                        quantizer = faiss.IndexFlatIP(dim)
                        index = faiss.IndexIVFFlat(quantizer, dim, int(nlist_eff), faiss.METRIC_INNER_PRODUCT)
                        try:
                            index.train(emb_new_norm)
                            index.add(emb_new_norm)
                            index.nprobe = int(min(max(1, int(nprobe_eff)), int(nlist_eff)))
                            used_ivf = True
                        except Exception as e:
                            st.warning(t("ivf_fail_fallback", err=e))
                            index = None

                    if index is None:
                        index = faiss.IndexFlatIP(dim)
                        index.add(emb_new_norm)

                    faiss_scores, faiss_indices = index.search(emb_old_norm, k=k)

                    def get_topk(i):
                        return faiss_indices[i], faiss_scores[i]

                    if used_ivf:
                        st.info(t("faiss_ivf_active", nlist=int(nlist_eff), nprobe=int(min(max(1, int(nprobe_eff)), int(nlist_eff))), n=N))
                        engine_for_type = "faiss-ivf"
                    else:
                        st.info(t("faiss_flat_active", n=N))
                        engine_for_type = "faiss-flat"

                for i in range(len(df_remaining_used)):
                    old_url = df_remaining_used['Address'].iloc[i]
                    row_result = {COL_OLD: old_url}
                    top_indices, top_scores = get_topk(i)

                    rank = 1
                    for idx, score in zip(top_indices, top_scores):
                        if idx >= len(df_new_used):
                            continue
                        score = float(score)
                        if score < threshold:
                            continue

                        row_result[col_matched(rank)] = df_new_used['Address'].iloc[idx]
                        row_result[col_score(rank)] = round(score, 4)
                        if rank == 1:
                            row_result[COL_TYPE] = t("out_similarity", engine=engine_for_type)
                        rank += 1

                    if rank > 1:
                        results.append(row_result)
                    else:
                        if old_url not in old_notes:
                            old_notes[old_url] = t("note_below_threshold")

        # 3) Add unmatched old URLs
        matched_urls_final = set(r.get(COL_OLD) for r in results if r.get(COL_OLD) is not None)
        unmatched = df_old[~df_old['Address'].isin(matched_urls_final)]

        for _, row in unmatched.iterrows():
            old_url = row['Address']
            note = old_notes.get(old_url, "")

            if matching_method != t("method_exact") and not note:
                note = t("note_no_semantic_run")

            if matching_method != t("method_exact") and had_invalid_new:
                note = (note + " " + t("note_bad_new")).strip()

            out_row = {COL_OLD: old_url, COL_TYPE: t("out_no_match")}
            if note:
                out_row[COL_NOTE] = note
            results.append(out_row)

        # 4) Build dataframe
        df_result = pd.DataFrame(results)

        # Drop "Match Basis" if empty
        if COL_BASIS in df_result.columns:
            ser = df_result[COL_BASIS].astype(str).str.strip().replace("nan", "")
            if not ser.ne("").any():
                df_result = df_result.drop(columns=[COL_BASIS])

        # Drop Note column if empty
        if COL_NOTE in df_result.columns:
            ser = df_result[COL_NOTE].astype(str).str.strip().replace("nan", "")
            if not ser.ne("").any():
                df_result = df_result.drop(columns=[COL_NOTE])

        # Drop empty matched/score columns globally (keep tidy)
        for rnk in range(5, 1, -1):
            mu = col_matched(rnk)
            sc = col_score(rnk)
            if mu in df_result.columns:
                s = df_result[mu].astype(str).str.strip().replace("nan", "")
                if not s.ne("").any():
                    df_result = df_result.drop(columns=[mu])
            if sc in df_result.columns:
                s = df_result[sc].astype(str).str.strip().replace("nan", "")
                if not s.ne("").any():
                    df_result = df_result.drop(columns=[sc])

        # 5) Display + download
        st.subheader(t("results_header"))
        st.dataframe(df_result)

        csv = df_result.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label=t("download"),
            data=csv,
            file_name=t("download_name"),
            mime='text/csv'
        )
