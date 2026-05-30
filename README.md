# Handbook-RAG
cd Downloads 
cd handbook
vi .env
cat .env
OPENAI_API_KEY=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
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
