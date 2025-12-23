import os, json, re, sqlite3, smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
from openai import OpenAI

import requests
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# =========================
# ENV / CONFIG
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=OPENAI_API_KEY)
APP_TOKEN   = os.getenv("APP_TOKEN", "changeme")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "changeme")

DB_PATH      = os.getenv("DB_PATH", "chat.db")
OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")

SMTP_HOST  = os.getenv("SMTP_HOST", "")
SMTP_PORT  = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER  = os.getenv("SMTP_USER", "")
SMTP_PASS  = os.getenv("SMTP_PASS", "")
ALERT_TO   = os.getenv("ALERT_TO", "")
ALERT_FROM = os.getenv("ALERT_FROM", SMTP_USER or "noreply@example.com")

BUSINESS = {
    "name": "Empower Pilates",
    "address": "2084 Lakewood Rd, Unit C2, Toms River, NJ 08755",
    "phone": "908-382-0211",
    "email": "info@empowerpilatesus.com",
    "hours": [
        "Sunday: 9:00 AM–12:00 PM",
        "Mon–Thu: 9:00 AM–2:00 PM and 6:30 PM–9:30 PM",
        "Friday: 9:30 AM–12:30 PM",
        "Saturday: Closed",
    ],
    "offers": [
        "Free Trial Class for newcomers: 25-minute intro session",
        "Intro Offer: 5 classes for $95",
    ],
    "classes": ["Pilates"],  # single type
}

# =========================
# RATE LIMIT (simple in-memory)
# =========================
_rate = {}  # session_id -> list[timestamps]
RATE_WINDOW_SEC = 12
RATE_MAX = 8

def rate_limit_ok(session_id: str) -> bool:
    now = datetime.utcnow().timestamp()
    arr = _rate.get(session_id, [])
    arr = [t for t in arr if now - t < RATE_WINDOW_SEC]
    if len(arr) >= RATE_MAX:
        _rate[session_id] = arr
        return False
    arr.append(now)
    _rate[session_id] = arr
    return True

# =========================
# DB
# =========================
def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    con = db()
    cur = con.cursor()
    cur.execute("""
      CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        role TEXT,
        content TEXT,
        created_at TEXT
      )
    """)
    cur.execute("""
      CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        state TEXT,
        is_new TEXT,
        class_type TEXT,
        preferred_day TEXT,
        preferred_time TEXT,
        name TEXT,
        phone TEXT,
        updated_at TEXT
      )
    """)
    cur.execute("""
      CREATE TABLE IF NOT EXISTS leads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT UNIQUE,
        name TEXT,
        phone TEXT,
        created_at TEXT
      )
    """)
    cur.execute("""
      CREATE TABLE IF NOT EXISTS booking_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        is_new TEXT,
        class_type TEXT,
        preferred_day TEXT,
        preferred_time TEXT,
        name TEXT,
        phone TEXT,
        status TEXT,
        created_at TEXT
      )
    """)
    con.commit()
    con.close()

def utc_now():
    return datetime.utcnow().isoformat()

def add_message(session_id: str, role: str, content: str):
    con = db()
    con.execute(
        "INSERT INTO messages(session_id, role, content, created_at) VALUES (?,?,?,?)",
        (session_id, role, content, utc_now()),
    )
    con.commit()
    con.close()

def get_session(session_id: str) -> dict:
    con = db()
    row = con.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
    con.close()
    if row:
        return dict(row)
    return {
        "session_id": session_id,
        "state": "idle",
        "is_new": "",
        "class_type": "",
        "preferred_day": "",
        "preferred_time": "",
        "name": "",
        "phone": "",
    }

def update_session(session_id: str, **fields):
    con = db()
    cur = con.cursor()
    existing = cur.execute("SELECT session_id FROM sessions WHERE session_id=?", (session_id,)).fetchone()
    fields["updated_at"] = utc_now()

    if existing:
        sets = ", ".join([f"{k}=?" for k in fields.keys()])
        cur.execute(f"UPDATE sessions SET {sets} WHERE session_id=?", (*fields.values(), session_id))
    else:
        data = get_session(session_id)
        data.update(fields)
        cur.execute("""
          INSERT INTO sessions(session_id, state, is_new, class_type, preferred_day, preferred_time, name, phone, updated_at)
          VALUES(?,?,?,?,?,?,?,?,?)
        """, (
            session_id,
            data.get("state","idle"),
            data.get("is_new",""),
            data.get("class_type",""),
            data.get("preferred_day",""),
            data.get("preferred_time",""),
            data.get("name",""),
            data.get("phone",""),
            data.get("updated_at"),
        ))
    con.commit()
    con.close()

def upsert_lead(session_id: str, name: str, phone: str):
    con = db()
    cur = con.cursor()
    row = cur.execute("SELECT id FROM leads WHERE session_id=?", (session_id,)).fetchone()
    if row:
        cur.execute("UPDATE leads SET name=?, phone=? WHERE session_id=?", (name, phone, session_id))
    else:
        cur.execute(
            "INSERT INTO leads(session_id,name,phone,created_at) VALUES(?,?,?,?)",
            (session_id, name, phone, utc_now()),
        )
    con.commit()
    con.close()

def create_booking(session_id: str, s: dict):
    con = db()
    con.execute("""
      INSERT INTO booking_requests(session_id,is_new,class_type,preferred_day,preferred_time,name,phone,status,created_at)
      VALUES(?,?,?,?,?,?,?,?,?)
    """, (
        session_id,
        s.get("is_new",""),
        s.get("class_type",""),
        s.get("preferred_day",""),
        s.get("preferred_time",""),
        s.get("name",""),
        s.get("phone",""),
        "new",
        utc_now(),
    ))
    con.commit()
    con.close()

def fetch_bookings(limit: int = 200):
    con = db()
    rows = con.execute("""
      SELECT id, session_id, is_new, class_type, preferred_day, preferred_time, name, phone, status, created_at
      FROM booking_requests
      ORDER BY id DESC
      LIMIT ?
    """, (limit,)).fetchall()
    con.close()
    return [dict(r) for r in rows]

def fetch_recent_messages(session_id: str, limit: int = 16):
    con = db()
    rows = con.execute("""
      SELECT role, content FROM messages
      WHERE session_id=? ORDER BY id DESC LIMIT ?
    """, (session_id, limit)).fetchall()
    con.close()
    rows = list(rows)[::-1]
    return [{"role": r["role"], "content": r["content"]} for r in rows]

def reset_session(session_id: str):
    con = db()
    con.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
    con.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
    con.commit()
    con.close()

# =========================
# EMAIL
# =========================
def email_is_configured() -> bool:
    return bool(SMTP_HOST and SMTP_USER and SMTP_PASS and ALERT_TO)

def send_email(subject: str, body: str):
    if not email_is_configured():
        return
    msg = MIMEMultipart()
    msg["From"] = ALERT_FROM
    msg["To"] = ALERT_TO
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
    except Exception:
        pass

# =========================
# INTENTS / PARSING
# =========================
def normalize_phone(text: str) -> str:
    digits = re.sub(r"\D", "", text or "")
    return digits[-10:] if len(digits) >= 10 else ""

def extract_name(text: str) -> str:
    # simple but solid
    m = re.search(r"\b(my name is|i am|this is)\s+([A-Za-z][A-Za-z\s'-]{1,40})\b", text or "", re.IGNORECASE)
    return m.group(2).strip() if m else ""

def looks_like_new(text: str) -> bool:
    return bool(re.search(r"\b(new|first time|first-time|i'm new|yes i'm new)\b", text or "", re.IGNORECASE))

def looks_like_returning(text: str) -> bool:
    return bool(re.search(r"\b(returning|been before|existing|not new)\b", text or "", re.IGNORECASE))

def is_booking_intent(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ["book", "booking", "schedule", "appointment", "reserve", "sign up", "class"])

def is_hours_intent(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ["hours", "open", "close", "opening", "when are you open"])

def is_location_intent(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ["where", "location", "address", "directions"])

def is_free_trial_intent(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ["free trial", "trial"])

def build_suggestions(state: str) -> list[str]:
    # smart quick replies
    if state == "idle":
        return ["Book a class", "Free trial", "What are your hours?", "Where are you located?"]
    if state == "collect_is_new":
        return ["I'm a new client", "I'm returning"]
    if state == "collect_day_time":
        return ["Tuesday evening", "Thursday morning", "Weekend morning"]
    if state == "collect_name_phone":
        return ["My name is John Doe, 9085551234"]
    return ["Book a class"]

# =========================
# LLM (human-like)
# =========================
def call_openai_json(system: str, user: str, timeout: int = 45) -> dict:
    if not OPENAI_API_KEY:
        return {"reply": "AI is not configured yet. Please try again later."}

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        text={"format": {"type": "json_object"}},
        timeout=timeout,
    )

    text = (resp.output_text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        return {"reply": text[:900] if text else "Sorry — something went wrong."}
def call_ollama(prompt: str) -> str:
    r = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=180,
    )
    r.raise_for_status()
    return (r.json().get("response") or "").strip()

def qa_reply(session_id: str, user_msg: str, state: str = "idle") -> str:
    history = fetch_recent_messages(session_id, limit=16)
    transcript = "\n".join(
        [f"{'USER' if m['role']=='user' else 'ASSISTANT'}: {m['content']}" for m in history]
    )

    system = f"""
You are a friendly, human-sounding assistant for {BUSINESS["name"]}.
ALWAYS reply in English.

Hard rules:
- Never invent user details (name, phone, day, time). If missing, ask.
- Ask only ONE question at the end.
- Do not invent pricing or availability beyond the facts below.

Business facts:
- Address: {BUSINESS["address"]}
- Phone: {BUSINESS["phone"]}
- Email: {BUSINESS["email"]}
- Hours: {", ".join(BUSINESS["hours"])}
- Offers: {", ".join(BUSINESS["offers"])}

Current booking state: {state}

Return ONLY JSON: {{"reply":"..."}}
""".strip()

    user = f"""CHAT HISTORY:
{transcript}

USER MESSAGE:
{user_msg}
""".strip()

    data = call_openai_json(system, user)
    return (data.get("reply") or "").strip()[:900] or "Sorry — can you rephrase that?"


    prompt = f"{system}\n\nCHAT HISTORY:\n{transcript}\n\nUSER: {user_msg}\nASSISTANT:"
    raw = call_ollama(prompt)
    try:
        data = json.loads(raw)
        return (data.get("reply") or "").strip() or raw[:900]
    except Exception:
        return raw[:900]

# =========================
# FASTAPI
# =========================
app = FastAPI()
init_db()
app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatIn(BaseModel):
    session_id: str
    message: str

class ResetIn(BaseModel):
    session_id: str

class StatusIn(BaseModel):
    id: int
    status: str

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.get("/favicon.ico")
def favicon():
    # optional: put static/favicon.ico if you want
    path = "static/favicon.ico"
    if os.path.exists(path):
        return FileResponse(path)
    return Response(status_code=204)

@app.get("/health")
def health():
    db_ok = True
    try:
        con = db(); con.execute("SELECT 1"); con.close()
    except Exception:
        db_ok = False
    ollama_ok = True
    try:
        requests.post(f"{OLLAMA_HOST}/api/tags", timeout=5)
    except Exception:
        ollama_ok = False
    return JSONResponse({"ok": db_ok and ollama_ok, "db_ok": db_ok, "ollama_ok": ollama_ok, "model": OLLAMA_MODEL})

@app.post("/api/reset")
def api_reset(payload: ResetIn, x_app_token: str | None = Header(default=None)):
    if x_app_token != APP_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    sid = (payload.session_id or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="session_id required")
    reset_session(sid)
    return {"ok": True}

@app.post("/api/chat")
def chat(payload: ChatIn, x_app_token: str | None = Header(default=None)):
    if x_app_token != APP_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    session_id = (payload.session_id or "").strip()
    msg = (payload.message or "").strip()
    if not session_id or not msg:
        raise HTTPException(status_code=400, detail="session_id and message are required")

    if not rate_limit_ok(session_id):
        return {"reply": "One sec 🙂 You’re sending messages too fast. Please try again in a moment.", "suggestions": build_suggestions(get_session(session_id).get("state","idle"))}

    add_message(session_id, "user", msg)
    s = get_session(session_id)

    # Fast intents (no LLM needed)
    if is_hours_intent(msg):
        reply = f"Our hours are:\n- " + "\n- ".join(BUSINESS["hours"])
        add_message(session_id, "assistant", reply)
        return {"reply": reply, "suggestions": build_suggestions(s.get("state","idle"))}

    if is_location_intent(msg):
        reply = f"We’re located at {BUSINESS['address']}. If you’d like, you can call {BUSINESS['phone']} for directions."
        add_message(session_id, "assistant", reply)
        return {"reply": reply, "suggestions": build_suggestions(s.get("state","idle"))}

    if is_free_trial_intent(msg):
        reply = "Yes — we offer a free trial for newcomers: a 25-minute intro session. Would you like to book it? If so, are you new or returning?"
        add_message(session_id, "assistant", reply)
        # push booking flow
        if s["state"] == "idle":
            update_session(session_id, state="collect_is_new", class_type="Pilates")
        return {"reply": reply, "suggestions": build_suggestions(get_session(session_id)["state"])}

    # Start booking flow
    if s["state"] == "idle" and is_booking_intent(msg):
        update_session(session_id, state="collect_is_new", class_type="Pilates")
        reply = "Absolutely — I can help get you scheduled. Are you a new client or returning?"
        add_message(session_id, "assistant", reply)
        return {"reply": reply, "suggestions": build_suggestions("collect_is_new")}

    # If user says "yes" in idle — be smart
    if s["state"] == "idle" and re.search(r"\b(yes|yeah|yep)\b", msg.lower()):
        reply = "Got it 🙂 Are you looking to book a Pilates class, or did you want our hours/location?"
        add_message(session_id, "assistant", reply)
        return {"reply": reply, "suggestions": build_suggestions("idle")}

    # Booking flow (with AI improvise)
    if s["state"] == "collect_is_new":
        is_new = ""
        if looks_like_new(msg): is_new = "new"
        elif looks_like_returning(msg): is_new = "returning"

        if not is_new:
            ai = qa_reply(session_id, msg, state="collect_is_new")
            reply = ai.rstrip() + "\n\nJust to confirm — are you a new client or returning?"
            add_message(session_id, "assistant", reply)
            return {"reply": reply, "suggestions": build_suggestions("collect_is_new")}

        update_session(session_id, is_new=is_new, state="collect_day_time")
        reply = "Great. What day works best, and what time window do you prefer?"
        add_message(session_id, "assistant", reply)
        return {"reply": reply, "suggestions": build_suggestions("collect_day_time")}

    if s["state"] == "collect_day_time":
        pref = msg
        update_session(session_id, preferred_day=pref, preferred_time=pref, state="collect_name_phone")
        reply = "Perfect. What’s your full name and best phone number for confirmation?"
        add_message(session_id, "assistant", reply)
        return {"reply": reply, "suggestions": build_suggestions("collect_name_phone")}

    if s["state"] == "collect_name_phone":
        name = extract_name(msg) or s.get("name","")
        phone = normalize_phone(msg) or s.get("phone","")
        update_session(session_id, name=name, phone=phone)

        if not name or not phone:
            ai = qa_reply(session_id, msg, state="collect_name_phone")
            reply = ai.rstrip() + "\n\nPlease share your full name and a 10-digit phone number."
            add_message(session_id, "assistant", reply)
            return {"reply": reply, "suggestions": build_suggestions("collect_name_phone")}

        update_session(session_id, state="idle")
        s2 = get_session(session_id)
        upsert_lead(session_id, s2["name"], s2["phone"])
        create_booking(session_id, s2)

        subject = f"New Booking Request — {s2['name']} ({s2['phone']})"
        body = (
            f"New booking request:\n\n"
            f"Name: {s2['name']}\n"
            f"Phone: {s2['phone']}\n"
            f"New/Returning: {s2.get('is_new','')}\n"
            f"Class: {s2.get('class_type','')}\n"
            f"Preference: {s2.get('preferred_day','')}\n"
            f"Created: {utc_now()} UTC\n"
        )
        send_email(subject, body)

        reply = (
            "Perfect — request received ✅\n"
            f"• Client: {s2['name']} ({s2['phone']})\n"
            f"• Class: {s2['class_type']}\n"
            f"• Preference: {s2['preferred_day']}\n\n"
            f"Our team will confirm the exact time shortly. If you’d like, you can also call {BUSINESS['phone']}."
        )
        add_message(session_id, "assistant", reply)
        return {"reply": reply, "suggestions": build_suggestions("idle")}

    # Normal Q&A
    reply = qa_reply(session_id, msg, state="idle")
    add_message(session_id, "assistant", reply)
    return {"reply": reply, "suggestions": build_suggestions("idle")}

# =========================
# ADMIN (HTML + CSV + STATUS UPDATE)
# =========================
@app.get("/admin", response_class=HTMLResponse)
def admin(token: str = ""):
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized. Add ?token=...")

    rows = fetch_bookings(200)

    def esc(s: str) -> str:
        return (s or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

    html = f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Admin — Booking Requests</title>
<style>
body{{font-family:system-ui,Segoe UI,Arial;margin:24px;background:#0b141a;color:#e9edef}}
h1{{margin:0 0 6px 0}} .sub{{color:#8696a0;margin:0 0 18px 0}}
.card{{background:#111b21;border:1px solid rgba(255,255,255,.08);border-radius:14px;padding:14px;overflow:auto}}
table{{width:100%;border-collapse:collapse;font-size:14px}}
th,td{{padding:10px 8px;border-bottom:1px solid rgba(255,255,255,.08);vertical-align:top}}
th{{color:#aebac1;text-align:left;font-weight:600}}
.pill{{display:inline-block;padding:3px 8px;border-radius:999px;border:1px solid rgba(255,255,255,.12);color:#aebac1;font-size:12px}}
.muted{{color:#8696a0}}
select{{background:#0b141a;color:#e9edef;border:1px solid rgba(255,255,255,.12);border-radius:10px;padding:6px 8px}}
a{{color:#00a884;text-decoration:none}} a:hover{{text-decoration:underline}}
</style></head><body>
<h1>Booking Requests</h1>
<p class="sub">
  <span class="pill">DB: {esc(DB_PATH)}</span>
  &nbsp; | &nbsp;
  <a href="/admin/export.csv?token={esc(token)}">Download CSV</a>
</p>
<div class="card"><table><thead><tr>
<th>ID</th><th>Created</th><th>Client</th><th>Phone</th><th>New/Returning</th><th>Preference</th><th>Status</th>
</tr></thead><tbody>
"""

    if not rows:
        html += '<tr><td colspan="7" class="muted">No booking requests yet.</td></tr>'
    else:
        for r in rows:
            rid = r.get("id")
            status = esc(r.get("status","new"))
            html += f"""
<tr>
<td>{rid}</td>
<td class="muted">{esc(r.get("created_at",""))}</td>
<td>{esc(r.get("name",""))}</td>
<td>{esc(r.get("phone",""))}</td>
<td>{esc(r.get("is_new",""))}</td>
<td>{esc(r.get("preferred_day",""))}</td>
<td>
  <select onchange="updateStatus({rid}, this.value)">
    <option value="new" {"selected" if status=="new" else ""}>new</option>
    <option value="contacted" {"selected" if status=="contacted" else ""}>contacted</option>
    <option value="confirmed" {"selected" if status=="confirmed" else ""}>confirmed</option>
    <option value="cancelled" {"selected" if status=="cancelled" else ""}>cancelled</option>
  </select>
</td>
</tr>
"""

    html += f"""
</tbody></table></div>
<p class="sub" style="margin-top:14px;">Tip: bookmark <span class="pill">/admin?token=...</span></p>

<script>
async function updateStatus(id, status){{
  try{{
    await fetch('/admin/status?token={esc(token)}', {{
      method:'POST',
      headers:{{'Content-Type':'application/json'}},
      body: JSON.stringify({{id, status}})
    }});
  }}catch(e){{}}
}}
</script>

</body></html>"""
    return HTMLResponse(content=html)

@app.post("/admin/status")
def admin_status(payload: StatusIn, token: str = ""):
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if payload.status not in ("new","contacted","confirmed","cancelled"):
        raise HTTPException(status_code=400, detail="Invalid status")
    con = db()
    con.execute("UPDATE booking_requests SET status=? WHERE id=?", (payload.status, payload.id))
    con.commit()
    con.close()
    return {"ok": True}

@app.get("/admin/export.csv")
def admin_export_csv(token: str = ""):
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    rows = fetch_bookings(5000)
    # CSV minimal
    def safe(v): 
        v = "" if v is None else str(v)
        v = v.replace('"','""')
        return f'"{v}"'
    header = ["id","created_at","name","phone","is_new","class_type","preferred_day","status"]
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join([
            safe(r.get("id")), safe(r.get("created_at")), safe(r.get("name")),
            safe(r.get("phone")), safe(r.get("is_new")), safe(r.get("class_type")),
            safe(r.get("preferred_day")), safe(r.get("status")),
        ]))
    csv_data = "\n".join(lines)
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=booking_requests.csv"}
    )
