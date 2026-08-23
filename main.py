from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import models
from database import engine, SessionLocal
import joblib
import json
import chromadb
import ollama as ollama_client

chroma_client = chromadb.PersistentClient(path="./chroma_db")
knowledge_collection = chroma_client.get_or_create_collection(name="adhd_knowledge")

model = joblib.load("model.pkl")
with open("features.json") as f:
    feature_list = json.load(f)
models.Base.metadata.create_all(bind=engine)
app = FastAPI(title="ADHD Assistant")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def log_action(db: Session, user_id: int, action: str, task_id: int = None):
    log_entry = models.Log(user_id=user_id, task_id=task_id, action=action)
    db.add(log_entry)
    db.commit()


class TaskCreate(BaseModel):
    user_id: int
    title: str
    category: Optional[str] = None
    priority: Optional[str] = None

class UserCreate(BaseModel):
    name: str

@app.post("/users")
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = models.User(name=user.name)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.get("/tasks")
def get_tasks(db: Session = Depends(get_db)):
    return db.query(models.Task).all()

@app.post("/tasks")
def create_task(task: TaskCreate, db: Session = Depends(get_db)):
    db_task = models.Task(**task.dict())
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    log_action(db, task.user_id, "task_created", db_task.id)
    return db_task


@app.put("/tasks/{task_id}")
def update_task(task_id: int, status: str, db: Session = Depends(get_db)):
    task = db.query(models.Task).filter(models.Task.id == task_id).first()
    if not task:
        raise HTTPException(404, "Task tapılmadı")
    task.status = status
    db.commit()
    log_action(db, task.user_id, f"task_status_changed_to_{status}", task.id)
    return task

@app.delete("/tasks/{task_id}")
def delete_task(task_id: int, db: Session = Depends(get_db)):
    task = db.query(models.Task).filter(models.Task.id == task_id).first()
    if not task:
        raise HTTPException(404, "Task tapılmadı")
    db.delete(task)
    db.commit()
    return {"deleted": True}
class PredictInput(BaseModel):
    age: float
    sex: int
    bdi_total: float
    bai_total: float
    audit_total: float
    matric_mark: float
    nbt_ave: float

@app.post("/predict")
def predict_risk(data: PredictInput, db: Session = Depends(get_db)):
    input_dict = {
        "age": data.age,
        "sex": data.sex,
        "bdi1_total": data.bdi_total,
        "bai1_total": data.bai_total,
        "audit1_total": data.audit_total,
        "matric_mark": data.matric_mark,
        "nbt_ave": data.nbt_ave
    }
    import pandas as pd
    X = pd.DataFrame([input_dict])[feature_list]
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    confidence = max(probabilities)
    log_action(db, 1, "prediction_requested")
    return {
        "risk_level": prediction,
        "confidence": round(float(confidence), 2),
        "note": "Bu proqnoz klinik diaqnoz deyil, təhsil məqsədli demo modelidir."
    }
class AskInput(BaseModel):
    question: str

@app.post("/ask")
def ask_question(data: AskInput, db: Session = Depends(get_db)):
    # 1. Sualı embed et
    q_embedding = ollama_client.embeddings(model="nomic-embed-text", prompt=data.question)["embedding"]

    # 2. Bilik bazasından ən uyğun 3 hissəni tap
    results = knowledge_collection.query(query_embeddings=[q_embedding], n_results=3)
    context = "\n\n".join(results["documents"][0])

    # 3. Ollama-ya sual + kontekst göndər
    prompt = f"""Sən ADHD sahəsində ixtisaslaşmış, elmi məsləhət verən bir mütəxəssissən. Aydın, peşəkar və strukturlu şəkildə danış.

Qaydalar:
- YALNIZ verilən məlumata əsaslan, başqa mövzu əlavə etmə
- Əvvəlcə əsas tövsiyəni bir cümlə ilə ver, sonra niyə işlədiyini qısa izah et
- Elmi, aydın dildə danış, jarqon işlətmə
- Cavabın 2-3 cümlədən uzun olmasın

Nümunə format:
Sual: "Necə yadda saxlaya bilərəm?"
Cavab: "Xatırlatmanı gündəlik rutininlə birləşdir — məsələn, dərmanı diş fırçasının yanına qoy. Bu, ayrıca yadda saxlamağa ehtiyacı azaldır, çünki xatırlatma artıq mühitin təbii hissəsinə çevrilir."

İndi bu qaydada cavab ver:

Məlumat:
{context}

Sual: {data.question}

Cavab:"""

    response = ollama_client.generate(
        model="llama3.1:8b",
        prompt=prompt,
        options={"temperature": 0.1}
    )
    return {
        "answer": response["response"],
        "sources": [m["source"] for m in results["metadatas"][0]]
    }
import json as json_lib
import re

class BrainDumpInput(BaseModel):
    user_id: int
    text: str

def split_into_candidates(text: str):
    parts = re.split(r',|\n|;|\bvə\b', text)
    parts = [p.strip().rstrip('.') for p in parts if p.strip()]
    return parts


@app.post("/brain-dump")
def brain_dump(data: BrainDumpInput, db: Session = Depends(get_db)):
    candidates = split_into_candidates(data.text)

    numbered_list = "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidates)])

    prompt = f"""Aşağıda nömrələnmiş tapşırıqlar var. HƏR BİRİ üçün, EYNİ SIRADA, bir category (work/study/personal) və bir priority (high/medium/low) təyin et.

VACIB: Sayı DƏYİŞMƏ. Yeni tapşırıq ƏLAVƏ ETMƏ. Yalnız {len(candidates)} element üçün cavab ver.

Tapşırıqlar:
{numbered_list}

Cavabı YALNIZ bu formatda JSON array kimi ver, başqa heç nə yazma:
[
  {{"category": "work", "priority": "high"}},
  {{"category": "study", "priority": "medium"}}
]"""

    response = ollama_client.generate(model="llama3.2", prompt=prompt)
    raw_text = response["response"]

    match = re.search(r'\[.*\]', raw_text, re.DOTALL)
    if not match:
        raise HTTPException(500, "Model düzgün format qaytarmadı, yenidən sına")

    labels = json_lib.loads(match.group())

    created_tasks = []
    for i, title in enumerate(candidates):
        label = labels[i] if i < len(labels) else {"category": "personal", "priority": "medium"}
        db_task = models.Task(
            user_id=data.user_id,
            title=title,
            category=label.get("category"),
            priority=label.get("priority")
        )
        db.add(db_task)
        db.commit()
        db.refresh(db_task)
        created_tasks.append(db_task)

    return {"created_tasks": [{"id": t.id, "title": t.title, "category": t.category, "priority": t.priority} for t in created_tasks]}

@app.get("/logs")
def get_logs(db: Session = Depends(get_db)):
    logs = db.query(models.Log).order_by(models.Log.timestamp.desc()).limit(50).all()
    return [{"action": l.action, "timestamp": l.timestamp, "task_id": l.task_id} for l in logs]

@app.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    total_users = db.query(models.User).count()
    total_tasks = db.query(models.Task).count()
    completed_tasks = db.query(models.Task).filter(models.Task.status == "done").count()
    pending_tasks = total_tasks - completed_tasks
    predictions = db.query(models.Log).filter(models.Log.action == "prediction_requested").count()
    questions = db.query(models.Log).filter(models.Log.action == "question_asked").count()

    return {
        "users": total_users,
        "tasks": total_tasks,
        "completed": completed_tasks,
        "pending": pending_tasks,
        "predictions": predictions,
        "questions": questions,
        "completion_rate": round(completed_tasks / total_tasks, 2) if total_tasks > 0 else 0
    }

class TalkInput(BaseModel):
    message: str
    history: list = []

@app.post("/talk")
def talk(data: TalkInput, db: Session = Depends(get_db)):
    history_text = "\n".join([f"{'İstifadəçi' if h['role']=='user' else 'Assistent'}: {h['content']}" for h in data.history])

    prompt = f"""Sən 'Assistent' adlı, isti, dinləyən və dəstəkləyici bir köməkçisən. İstifadəçi öz hisslərini, gününü, çətinliklərini bölüşür.

Qaydalar:
- Empatiya göstər, mühakimə etmə
- Sən klinik terapevt deyilsən — diaqnoz qoyma, dərman tövsiyə etmə
- Əgər istifadəçi özünə zərər vermək fikrindən bəhs edərsə, mütləq peşəkar dəstək (psixoloq, kризis xətti) almağı tövsiyə et
- Qısa (2-3 cümlə), səmimi Azərbaycan dilində cavab ver
- Sual verməklə söhbəti davam etdirə bilərsən, amma məcburi deyil

Əvvəlki söhbət:
{history_text}

İstifadəçi: {data.message}

Assistent:"""

    response = ollama_client.generate(
        model="llama3.1:8b",
        prompt=prompt,
        options={"temperature": 0.4}
    )

    log_action(db, 1, "talk_message_sent")

    return {"reply": response["response"].strip()}

    from sqlalchemy import func

@app.get("/weekly-report")
def weekly_report(db: Session = Depends(get_db)):
    # Hər log-u həftəsinə görə qruplaşdırırıq (məs. "2026-W27")
    rows = (
        db.query(
            func.strftime("%Y-W%W", models.Log.timestamp).label("period"),
            models.Log.action,
            func.count(models.Log.id).label("count")
        )
        .group_by("period", models.Log.action)
        .order_by("period")
        .all()
    )
    return group_rows_by_period(rows)

@app.get("/monthly-report")
def monthly_report(db: Session = Depends(get_db)):
    # Hər log-u ayına görə qruplaşdırırıq (məs. "2026-07")
    rows = (
        db.query(
            func.strftime("%Y-%m", models.Log.timestamp).label("period"),
            models.Log.action,
            func.count(models.Log.id).label("count")
        )
        .group_by("period", models.Log.action)
        .order_by("period")
        .all()
    )
    return group_rows_by_period(rows)

def group_rows_by_period(rows):
    # Nəticəni {period: {action: count}} formasına çeviririk
    report = {}
    for period, action, count in rows:
        if period not in report:
            report[period] = {}
        report[period][action] = count
    return report