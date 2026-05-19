# Stochastic Dominance Engine

Compares ETF return distributions using first‑ (FSD), second‑ (SSD), and third‑order (TSD) stochastic dominance. For each ETF, we count how many other ETFs it dominates (weighted by order). Higher dominance score → more attractive asset from a risk‑averse perspective. Model‑free, no distributional assumptions.

- **Orders:** 1,2,3 (configurable)
- **Algorithm:** Empirical CDF comparison on quantile grid
- **Windows:** 63, 252, 504, 1008, 2016 days (best per ETF)
- **Output:** top 3 ETFs per universe by dominance score

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
