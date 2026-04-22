# Degradation & Nitrosamine Predictor

Minimal Streamlit MVP for:
- pharmaceutical degradation product prediction
- API–excipient alerting
- nitrosamine risk triage

## Files in this repo
- `app.py` → main Streamlit app
- `requirements.txt` → Python dependencies
- `packages.txt` → Linux system package needed for RDKit drawing support on Streamlit Cloud
- `.streamlit/config.toml` → optional Streamlit config

## Local run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## GitHub + Streamlit Community Cloud
1. Create a GitHub repository.
2. Put `app.py`, `requirements.txt`, `packages.txt`, and the `.streamlit` folder in the repo root.
3. Push to GitHub.
4. In Streamlit Community Cloud, deploy the repo and choose `app.py` as the entrypoint.
5. In **Advanced settings**, select **Python 3.12**.

## Notes
- Keep only one Python dependency file. This repo uses `requirements.txt`.
- If you remove RDKit molecule drawing, `packages.txt` may become unnecessary.
- If deployment fails, reboot or redeploy the app after changing dependencies.
