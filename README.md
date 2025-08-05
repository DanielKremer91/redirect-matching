# 🔀 Semantic Redirect Mapping

Ein interaktives Tool zur Unterstützung bei Website-Relaunches: Es hilft dir dabei, Weiterleitungen (Redirects) zwischen ALT- und ZIEL-URLs intelligent vorzuschlagen – entweder per **Exact Match** (z. B. identische H1) oder über **semantisches Similarity Matching** mithilfe von **Embeddings**.

## 🚀 Live ausprobieren

👉 [Jetzt ausprobieren auf streamlit.app](https://dein-username.streamlit.app/redirect-mapping)  
*(Hinweis: Du kannst große CSV- oder Excel-Dateien hochladen.)*

---

## ✅ Was das Tool kann

- 🔁 Zwei Dateien hochladen (ALT- & NEU-Domain)
- ✅ Exakte Matches identifizieren (z. B. über gleiche H1)
- 🤖 Semantisches Matching via:
  - `sklearn.metrics.pairwise.cosine_similarity` (für maximale Präzision)
  - `FAISS` (für Geschwindigkeit bei großen Datenmengen)
- 📎 Optional: Verwende eigene Embeddings – oder lasse sie automatisch aus Spalteninhalten berechnen
- ⬇ Ergebnisse als CSV exportieren

---

## 🧪 Beispiel-Dateien (Struktur)

Beide Dateien sollten mindestens folgende Spalte enthalten:

```text
Address
