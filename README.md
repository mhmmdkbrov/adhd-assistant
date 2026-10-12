# ADHD Assistant

ADHD olan insanlar üçün tapşırıq, fokus və dəstək köməkçisi. Kurs layihəsi.

## Nələr var
- **Tapşırıqlar:** brain dump (sərbəst yaz, avtomatik kateqoriya/prioritet), günün saatlarına bölgü, silmə, progress
- **Fokus riski:** RandomForest modeli (real ADHD klinik datası) ilə proqnoz
- **Assistent:** RAG (ChromaDB + Ollama) ilə ADHD məsləhətləri və Akinator tipli "vəziyyətini tap"
- **Proqres:** həftəlik/aylıq hesabat, ümumi vəziyyət

## Texnologiyalar
Python, FastAPI, SQLite/SQLAlchemy, scikit-learn, ChromaDB, Ollama, vanilla HTML/JS

## Quraşdırma
1. `python -m venv venv` və `venv\Scripts\activate`
2. `python -m pip install -r requirements.txt`
3. Ollama quraşdır, sonra: `ollama pull nomic-embed-text`, `ollama pull llama3.1:8b`, `ollama pull llama3.2`
4. `python train_model.py`
5. `python ingest.py`
6. `python -m uvicorn main:app --reload`
7. `index.html` faylını brauzerdə aç

> Bu proqnoz klinik diaqnoz deyil, təhsil məqsədli demodur.
