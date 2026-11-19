# Skill Retrieval Benchmark and Naming Optimizer — Working Demo

## Run the demo
1. Install Python 3.10 or newer.
2. Extract this ZIP.
3. Open a terminal in the extracted `skill_retrieval_demo` folder.
4. Run:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
5. Open the local URL shown in the terminal (usually http://localhost:8501).

## Included
- Skill catalogue with stable IDs, names, descriptions, aliases, tags and categories.
- Live retrieval using BM25, TF-IDF cosine similarity as an offline dense-retrieval proxy, or hybrid Reciprocal Rank Fusion (RRF, k=60).
- Candidate skill-name suggestions with human approval warning.
- Recall@5, MRR@10 and nDCG@10 benchmark calculations.
- CSV upload for your own skills and relevance-labelled queries.
- CSV downloads for sample data.

## CSV format
Skills CSV: required `skill_id`, `canonical_name`, `description`; optional `aliases`, `tags`, `category`.
Queries CSV: `query`, `relevant_ids` (semicolon-separated skill IDs).

## Accuracy / research note
The bundled skill records and relevance labels are synthetic demo data. Displayed metrics are computed from those examples only and must not be reported as experimental findings. Use independently reviewed relevance labels and a held-out test set for the actual research. Name suggestions are previews only; do not automatically rename production skills. Keep `skill_id` stable.
