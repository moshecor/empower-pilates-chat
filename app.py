import os, json, re, sqlite3, smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Tuple, Dict, Any, List

import requests
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI

# =========================
# ENV / CONFIG
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=OPENAI_API_KEY)

APP_TOKEN   = os.getenv("APP_TOKEN", "changeme")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "changeme")

DB_PATH      = os.getenv("DB_PATH", "chat.db")

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
    "classes": ["Pilates"],
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
# PARSING / INTENTS
# =========================
def normalize_phone(text: str) -> str:
    digits = re.sub(r"\D", "", text or "")
    return digits[-10:] if len(digits) >= 10 else ""

def extract_name(text: str) -> str:
    m = re.search(r"\b(my name is|i am|this is)\s+([A-Za-z][A-Za-z\s'-]{1,40})\b", text or "", re.IGNORECASE)
    return m.group(2).strip() if m else ""

def looks_like_new(text: str) -> bool:
    return bool(re.search(r"\b(new|first time|first-time|i'?m new|yes i'?m new)\b", text or "", re.IGNORECASE))

def looks_like_returning(text: str) -> bool:
    return bool(re.search(r"\b(returning|been before|existing|not new)\b", text or "", re.IGNORECASE))

def is_booking_intent(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ["book", "booking", "schedule", "appointment", "reserve", "sign up", "class", "trial"])

def is_hours_intent(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ["hours", "open", "close", "opening", "when are you open"])

def is_location_intent(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ["where", "location", "address", "directions"])

def is_free_trial_intent(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ["free trial", "trial", "first class free"])

WEEKDAYS = {
    "monday":"Monday","mon":"Monday",
    "tuesday":"Tuesday","tue":"Tuesday","tues":"Tuesday",
    "wednesday":"Wednesday","wed":"Wednesday",
    "thursday":"Thursday","thu":"Thursday","thur":"Thursday","thurs":"Thursday",
    "friday":"Friday","fri":"Friday",
    "saturday":"Saturday","sat":"Saturday",
    "sunday":"Sunday","sun":"Sunday",
}

def parse_day_time(text: str) -> Tuple[str, str]:
    """Very small heuristic to extract preferred day + time window.
    Returns (day, time_window) where either may be "" if unknown."""
    t = (text or "").lower()
    day = ""
    for k, v in WEEKDAYS.items():
        if re.search(rf"\b{k}\b", t):
            day = v
            break

    # time window
    # recognize "morning/afternoon/evening" or times like 6pm, 18:30, etc.
    time_window = ""
    if re.search(r"\bmorning\b", t):
        time_window = "morning"
    elif re.search(r"\bafternoon\b", t):
        time_window = "afternoon"
    elif re.search(r"\bevening\b|\btonight\b", t):
        time_window = "evening"

    # specific time?
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", t)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2) or "00")
        ap = m.group(3).lower()
        # normalize to e.g. 6:30 PM
        time_window = f"{hh}:{mm:02d} {ap.upper()}"

    m2 = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", t)
    if m2 and not time_window:
        time_window = f"{m2.group(1)}:{m2.group(2)}"

    return day, time_window

def build_suggestions(state: str) -> list[str]:
    # more "WhatsApp-ish" quick replies
    if state == "idle":
        return ["Book a class", "Free trial", "Hours", "Address"]
    if state == "collect_is_new":
        return ["New client", "Returning"]
    if state == "collect_day_time":
        return ["Tuesday evening", "Thursday morning", "Sunday morning"]
    if state == "collect_name_phone":
        return ["My name is Jane Doe, 9085551234"]
    return ["Book a class"]

# =========================
# LLM CORE (human-like, controlled)
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
        # last resort: treat it as text
        return {"reply": text[:900] if text else "Sorry — something went wrong."}

def _human_system_base() -> str:
    return f"""
You are a warm, human-sounding WhatsApp assistant for {BUSINESS["name"]}.
Write in English.

Voice & style (hard rules):
- Keep it short: 1–2 sentences max.
- Natural, friendly, not salesy.
- Mirror the user's message briefly (1 short clause) before guiding.
- Ask exactly ONE question (or none if you are giving info).
- No emojis spam: at most 1 emoji, and only sometimes.
- Never mention you're an AI. Never over-explain.

Safety & accuracy (hard rules):
- Do NOT invent prices, availability, classes, promotions, policies beyond the facts provided.
- If asked for something unknown, offer a call to the studio phone number.
- If user seems upset/confused: apologize briefly and offer human handoff.

Business facts:
- Address: {BUSINESS["address"]}
- Phone: {BUSINESS["phone"]}
- Email: {BUSINESS["email"]}
- Hours: {", ".join(BUSINESS["hours"])}
- Offers: {", ".join(BUSINESS["offers"])}
""".strip()

def _repair_reply_if_needed(text: str) -> str:
    """Enforce: short + max 1 question mark."""
    if not text:
        return "Sorry — can you rephrase that?"
    t = text.strip()

    # hard cap
    if len(t) > 420:
        t = t[:420].rstrip()

    # keep at most 1 question
    if t.count("?") > 1:
        first = t.find("?")
        t = t[: first + 1].strip()

    # remove super-salesy openings
    t = re.sub(r"^(welcome to|thank you for reaching out to|we are delighted to)", "Hi —", t, flags=re.I).strip()

    return t

def humanize_text(session_id: str, raw_assistant_text: str, state: str, must_ask: Optional[str] = None) -> str:
    """
    Takes a deterministic bot line and rewrites it to sound human,
    while preserving the intent and keeping it short.
    must_ask: if provided, force that exact question to be the only question.
    """
    if not OPENAI_API_KEY:
        # fallback: just return raw
        return _repair_reply_if_needed(raw_assistant_text)

    system = _human_system_base() + f"\n\nCurrent flow state: {state}\nReturn ONLY JSON: {{\"reply\":\"...\"}}"
    user = f"""
Rewrite the following draft message to match the voice rules.
Keep the same meaning. Do not add new facts.

DRAFT:
{raw_assistant_text}

{"You MUST end with this exact question (and no other questions): " + must_ask if must_ask else ""}
""".strip()

    data = call_openai_json(system, user)
    reply = (data.get("reply") or "").strip()
    reply = _repair_reply_if_needed(reply)

    # if we must ask something, enforce it hard
    if must_ask:
        # strip trailing questions and append must_ask
        reply = re.sub(r"\s*\?+.*$", "", reply).strip()
        # keep it short
        if len(reply) > 260:
            reply = reply[:260].rstrip()
        reply = (reply + ("\n" if reply and not reply.endswith("\n") else "") + must_ask).strip()

        # ensure only one question mark
        if reply.count("?") > 1:
            # keep everything up to the must_ask only
            reply = reply.split("\n")[-1].strip() if reply.split("\n")[-1].strip().endswith("?") else must_ask

    return reply

def qa_reply(session_id: str, user_msg: str, state: str = "idle") -> str:
    history = fetch_recent_messages(session_id, limit=16)
    transcript = "\n".join(
        [f"{'USER' if m['role']=='user' else 'ASSISTANT'}: {m['content']}" for m in history]
    )

    system = _human_system_base() + f"""

You must be helpful and concise.
If the user is asking to book, move them toward booking in the current state.

Hard rules:
- Ask ONLY one question.
- If you don't know, offer calling {BUSINESS["phone"]}.
Return ONLY JSON: {{"reply":"..."}}
""".strip()

    user = f"""CHAT HISTORY:
{transcript}

USER MESSAGE:
{user_msg}
""".strip()

    data = call_openai_json(system, user)
    return _repair_reply_if_needed((data.get("reply") or "").strip())[:900] or "Sorry — can you rephrase that?"

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
    # OpenAI isn't pinged here; keep it simple
    return JSONResponse({"ok": db_ok, "db_ok": db_ok, "model": OPENAI_MODEL})

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
        s = get_session(session_id)
        return {
            "reply": "One sec 🙂 You’re sending messages too fast. Try again in a moment?",
            "suggestions": build_suggestions(s.get("state","idle"))
        }

    add_message(session_id, "user", msg)
    s = get_session(session_id)
    state = s.get("state","idle")

    # -------------------------
    # Fast intents (no LLM)
    # -------------------------
    if is_hours_intent(msg):
        raw = "Our hours are:\n- " + "\n- ".join(BUSINESS["hours"])
        reply = humanize_text(session_id, raw, state="idle")
        add_message(session_id, "assistant", reply)
        return {"reply": reply, "suggestions": build_suggestions(state)}

    if is_location_intent(msg):
        raw = f"We’re at {BUSINESS['address']}. If you want, you can call {BUSINESS['phone']} for directions."
        reply = humanize_text(session_id, raw, state="idle")
        add_message(session_id, "assistant", reply)
        return {"reply": reply, "suggestions": build_suggestions(state)}

    if is_free_trial_intent(msg):
        raw = "Yes — we do a free trial for newcomers (25-minute intro)."
        # push booking flow
        if state == "idle":
            update_session(session_id, state="collect_is_new", class_type="Pilates")
            state = "collect_is_new"
        reply = humanize_text(session_id, raw, state=state, must_ask="Are you a new client or returning?")
        add_message(session_id, "assistant", reply)
        return {"reply": reply, "suggestions": build_suggestions(state)}

    # -------------------------
    # Start booking flow
    # -------------------------
    if state == "idle" and is_booking_intent(msg):
        update_session(session_id, state="collect_is_new", class_type="Pilates")
        reply = humanize_text(
            session_id,
            "I can help you get scheduled.",
            state="collect_is_new",
            must_ask="Are you a new client or returning?"
        )
        add_message(session_id, "assistant", reply)
        return {"reply": reply, "suggestions": build_suggestions("collect_is_new")}

    # Smart "yes" in idle
    if state == "idle" and re.search(r"\b(yes|yeah|yep|sure)\b", msg.lower()):
        reply = humanize_text(
            session_id,
            "Got it.",
            state="idle",
            must_ask="Are you trying to book a class, or did you need our hours/address?"
        )
        add_message(session_id, "assistant", reply)
        return {"reply": reply, "suggestions": build_suggestions("idle")}

    # -------------------------
    # Booking flow steps
    # -------------------------
    if state == "collect_is_new":
        is_new = ""
        if looks_like_new(msg):
            is_new = "new"
        elif looks_like_returning(msg):
            is_new = "returning"

        if not is_new:
            # keep it human but force the question
            hint = qa_reply(session_id, msg, state="collect_is_new")
            reply = humanize_text(
                session_id,
                hint if hint else "No problem.",
                state="collect_is_new",
                must_ask="Just to confirm — are you a new client or returning?"
            )
            add_message(session_id, "assistant", reply)
            return {"reply": reply, "suggestions": build_suggestions("collect_is_new")}

        update_session(session_id, is_new=is_new, state="collect_day_time")
        reply = humanize_text(
            session_id,
            "Perfect.",
            state="collect_day_time",
            must_ask="What day works best, and what time window do you prefer?"
        )
        add_message(session_id, "assistant", reply)
        return {"reply": reply, "suggestions": build_suggestions("collect_day_time")}

    if state == "collect_day_time":
        day, t_window = parse_day_time(msg)
        # store best effort; keep whatever user typed too, but split if possible
        update_session(
            session_id,
            preferred_day=day or (s.get("preferred_day") or msg),
            preferred_time=t_window or (s.get("preferred_time") or ""),
            state="collect_name_phone"
        )
        reply = humanize_text(
            session_id,
            "Got it.",
            state="collect_name_phone",
            must_ask="What’s your full name and best phone number for confirmation?"
        )
        add_message(session_id, "assistant", reply)
        return {"reply": reply, "suggestions": build_suggestions("collect_name_phone")}

    if state == "collect_name_phone":
        name = extract_name(msg) or (s.get("name","") or "")
        phone = normalize_phone(msg) or (s.get("phone","") or "")
        update_session(session_id, name=name, phone=phone)

        if not name or not phone:
            hint = qa_reply(session_id, msg, state="collect_name_phone")
            reply = humanize_text(
                session_id,
                hint if hint else "Almost there.",
                state="collect_name_phone",
                must_ask="Please share your full name and a 10-digit phone number."
            )
            add_message(session_id, "assistant", reply)
            return {"reply": reply, "suggestions": build_suggestions("collect_name_phone")}

        # finalize booking request
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
            f"Preference day: {s2.get('preferred_day','')}\n"
            f"Preference time: {s2.get('preferred_time','')}\n"
            f"Created: {utc_now()} UTC\n"
        )
        send_email(subject, body)

        raw = (
            "All set — I’ve got your request. ✅ "
            "Our team will confirm the exact time shortly."
        )
        reply = humanize_text(
            session_id,
            raw,
            state="idle",
            must_ask=f"If you’d like, you can also call {BUSINESS['phone']} — would you like the team to text or call you?"
        )
        add_message(session_id, "assistant", reply)
        return {"reply": reply, "suggestions": build_suggestions("idle")}

    # -------------------------
    # Normal Q&A (still controlled)
    # -------------------------
    reply = qa_reply(session_id, msg, state="idle")
    add_message(session_id, "assistant", reply)
    return {"reply": reply, "suggestions": build_suggestions("idle")}

# =========================
# ADMIN (HTML + CSV + STATUS UPDATE)  (unchanged)
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
            pref = (r.get("preferred_day","") or "") + (" " + (r.get("preferred_time","") or "") if r.get("preferred_time") else "")
            html += f"""
<tr>
<td>{rid}</td>
<td class="muted">{esc(r.get("created_at",""))}</td>
<td>{esc(r.get("name",""))}</td>
<td>{esc(r.get("phone",""))}</td>
<td>{esc(r.get("is_new",""))}</td>
<td>{esc(pref.strip())}</td>
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
    def safe(v):
        v = "" if v is None else str(v)
        v = v.replace('"','""')
        return f'"{v}"'
    header = ["id","created_at","name","phone","is_new","class_type","preferred_day","preferred_time","status"]
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join([
            safe(r.get("id")), safe(r.get("created_at")), safe(r.get("name")),
            safe(r.get("phone")), safe(r.get("is_new")), safe(r.get("class_type")),
            safe(r.get("preferred_day")), safe(r.get("preferred_time")), safe(r.get("status")),
        ]))
    csv_data = "\n".join(lines)
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=booking_requests.csv"}
    )
