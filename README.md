1.Create a virtual environment
python -m venv.venv
source .venv/bin/activate
2.Install dependencies
pip install -r requirements.txt
3.Create env file
vi .env
cat .env
4.Add OPENAI_API_KEY in .env
5.Start the web app
uvicorn main:app --reload
6.Open the browser using url
http://127.0.0.1:8000
7.Upload PDF from the ui
8.Index PDF'S
python ingest.py
9.Ask questions in chat box   








