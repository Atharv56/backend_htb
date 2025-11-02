# app.py
import os
import json
import smtplib
import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Optional, List, Literal, TypedDict, Dict, Any

from flask import Flask, request, jsonify
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# LangGraph / LLM imports (kept from your server)
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode  # kept if you add tools later
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Your function (assumed available on PYTHONPATH)
from queryGraph import build_reasoning_path_for_query

# =============================================================================
# SMTP / INVITE CONFIG (env-first; safe defaults)
# =============================================================================
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))

# Use env vars in prod; fall back to your provided values for convenience
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "hacktheburgh005@gmail.com")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD", "cxtrnaexngslrfbe")  # Gmail App Password, not the account password

DEFAULT_TZ = os.getenv("DEFAULT_TZ", "Europe/London")
CRLF = "\r\n"

# =============================================================================
# LLM Client (unchanged from your server)
# =============================================================================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY", "AIzaSyCRtIOzit-vgIiE-7tvvSQAcCYoVJlO3UE"),
    temperature=0.7,
    max_tokens=1000,
)

# =============================================================================
# EventExtraction graph (your existing code, unchanged)
# =============================================================================

class EventExtractionState(TypedDict):
    query: str
    organizer: Optional[str]
    event_type: Optional[str]
    attendees: Optional[int]
    requirements: List[str]
    constraints: List[str]
    raw_extraction: Optional[str]
    error: Optional[str]
    retry_count: int
    needs_enrichment: bool
    enriched_constraints: List[str]

def extract_intent_node(state: EventExtractionState) -> EventExtractionState:
    query = state["query"]

    system_prompt = """You are an expert event coordinator specializing in matching events to appropriate venues.

    Your task is to analyze an event booking query and extract ONLY information relevant to the **venue side**.

    Return JSON in this exact format:
    {
    "organizer": "string",
    "event_type": "string",
    "attendees": number,
    "requirements": ["..."],
    "constraints": ["..."]
    }

    Clarifications:
    - Focus **only** on what the *venue must provide* or *accommodate* — not what the organizer brings.
    - Exclude non-venue requirements such as snacks, judges, prizes, or staff.
    - Include elements like:
      - space type (stage, hall, classroom, gym, outdoor area)
      - equipment/facilities (lighting, power, Wi-Fi, seating, mats, projectors, sound system)
      - environment (quiet, ventilation, acoustics, weather protection)
      - safety, accessibility, and comfort.
    - ENSURE THAT THE REQUIREMENTS ARE LOWERCASED AND BELONG TO THE SET ('microphones', 'conference table', 'wi-fi', 'tables', 'chairs', 'power outlets', 'catering area', 'conference phone', 'goal posts', 'floodlights', 'goals', 'stage area', 'lighting', 'security', 'rotary evaporators', 'spectrophotometer', 'incubators', 'microscopes', 'autoclave', 'laminar flow hood', 'balances', 'ph meters', 'magnetic stirrers', 'stage', 'sound system', 'catering facilities', 'sprung floor', 'barres', 'easels', 'pottery wheels', 'kiln', 'ventilation system', 'starting blocks', 'lane ropes', 'shallow water', 'warm water', 'av/projector', 'accessibility', 'audio-visual facilities', 'induction loop', 'fume hoods', 'eye wash stations', 'chemical storage', 'gas taps', 'distilled water', 'large fume hood', 'projector', 'whiteboard', 'demonstration bench', 'pa system', 'chemical storage cabinets', 'fume hood', 'balance', 'nitrogen gas', 'spectrometer', 'chromatograph', 'computer workstations', 'badminton nets', 'basketball hoops', 'volleyball nets', 'scoreboard', 'changing rooms', 'mats', 'mirrors', 'training equipment', 'first aid kit', 'showers', 'lockers', 'toilets', 'benches', 'desks', 'podium, 'air conditioning', 'projector screen', 'monitor', 'seating area', 'sinks', 'safety signs', 'ventilation', 'lifeguard chair', 'water filtration system', 'decorative plants').
    - IF THE REQUIREMENTS DO NOT BELONG TO THE ABOVE SET, ADD THE MOST RELEVANT ONES FROM THE SET ELSE 'NOT PRESENT'

    Guidelines by event type:
    - Tech events: power outlets, Wi-Fi, tables, seating, projectors, whiteboards, overnight access, ventilation, accessibility, safety.
    - Drama rehearsals: stage, lighting, prop storage, soundproofing, acoustics, seating, accessibility, safety.
    - Physical activities (e.g., Karate): open space, mats, ventilation, safety gear storage, first aid, water access.
    - Lectures: seating, projector, screen, microphone, lighting, acoustics, accessibility.
    - Exhibitions: display stands, lighting, open area, accessibility, weather protection, security.
    - Food events: kitchen access, ventilation, hygiene, tables, accessibility, safety.
    - Outdoor events: open area, weather protection, restrooms, accessibility, safety.

    IF YOU ARE UNSURE ABOUT ANY FIELD, JUST GENERALIZE IT AND ADD RELEVANT AMINITIES FROM THE ABOVE SET.
    Always include accessibility and safety considerations.
    """

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Query: {query}")
        ])
        raw_output = response.content or ""

        json_str = raw_output
        if "```json" in json_str:
            json_str = json_str.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```", 1)[1].split("```", 1)[0].strip()

        extracted = json.loads(json_str)

        return {
            **state,
            "organizer": extracted.get("organizer"),
            "event_type": extracted.get("event_type"),
            "attendees": extracted.get("attendees"),
            "requirements": extracted.get("requirements", []),
            "constraints": extracted.get("constraints", []),
            "raw_extraction": raw_output,
            "error": None,
            "needs_enrichment": len(extracted.get("constraints", [])) < 2
        }

    except Exception as e:
        return {
            **state,
            "error": f"Extraction failed: {e}",
            "raw_extraction": locals().get("raw_output"),
            "retry_count": state.get("retry_count", 0) + 1
        }

def validate_extraction_node(state: EventExtractionState) -> EventExtractionState:
    errors = []
    if not state.get("organizer"):
        errors.append("Organizer not identified")
    if not state.get("event_type"):
        errors.append("Event type not identified")
    if not state.get("attendees") or (isinstance(state["attendees"], int) and state["attendees"] <= 0):
        errors.append("Invalid or missing attendee count")
    return {**state, "error": "; ".join(errors)} if errors else state

def enrich_constraints_node(state: EventExtractionState) -> EventExtractionState:
    event_type = state.get("event_type", "")
    attendees = state.get("attendees", 0)
    existing = state.get("constraints", [])

    prompt = f"""Given:
- Event Type: {event_type}
- Attendees: {attendees}
- Existing Constraints: {', '.join(existing) or 'None'}

Provide 3–5 additional constraints as a JSON array:
["constraint 1", "constraint 2", "constraint 3"]
"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = (response.content or "").strip()

        if "```json" in content:
            content = content.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in content:
            content = content.split("```", 1)[1].split("```", 1)[0].strip()

        enriched = json.loads(content)
        combined = list({*existing, *enriched})  # de-duplicate

        return {
            **state,
            "enriched_constraints": enriched,
            "constraints": combined,
            "needs_enrichment": False
        }

    except Exception as e:
        return {
            **state,
            "needs_enrichment": False,
            "error": f"{state.get('error', '')} (Enrichment failed: {e})"
        }

def format_output_node(state: EventExtractionState) -> EventExtractionState:
    return state

def should_retry(state: EventExtractionState) -> Literal["extract_intent", "validate"]:
    if state.get("error") and state.get("retry_count", 0) < 2:
        return "extract_intent"
    return "validate"

def should_enrich(state: EventExtractionState) -> Literal["enrich_constraints", "format_output"]:
    if state.get("needs_enrichment", False) and not state.get("error"):
        return "enrich_constraints"
    return "format_output"

def create_advanced_extraction_graph():
    workflow = StateGraph(EventExtractionState)
    workflow.add_node("extract_intent", extract_intent_node)
    workflow.add_node("validate", validate_extraction_node)
    workflow.add_node("enrich_constraints", enrich_constraints_node)
    workflow.add_node("format_output", format_output_node)

    workflow.set_entry_point("extract_intent")

    workflow.add_conditional_edges(
        "extract_intent",
        should_retry,
        {"extract_intent": "extract_intent", "validate": "validate"},
    )
    workflow.add_conditional_edges(
        "validate",
        should_enrich,
        {"enrich_constraints": "enrich_constraints", "format_output": "format_output"},
    )

    workflow.add_edge("enrich_constraints", "format_output")
    workflow.add_edge("format_output", END)
    return workflow.compile()

def extract_event_intent(query: str, use_enrichment: bool = True) -> EventExtractionState:
    graph = create_advanced_extraction_graph()
    initial: EventExtractionState = {
        "query": query,
        "organizer": None,
        "event_type": None,
        "attendees": None,
        "requirements": [],
        "constraints": [],
        "raw_extraction": None,
        "error": None,
        "retry_count": 0,
        "needs_enrichment": use_enrichment,
        "enriched_constraints": [],
    }
    return graph.invoke(initial)

# =============================================================================
# ICS BUILD + EMAIL SENDER
# =============================================================================

def build_ics_request(event_info: Dict[str, Any],
                      organizer_email: str,
                      recipient_email: str,
                      tz_name: str = DEFAULT_TZ) -> str:
    """
    Build an iCalendar invite (METHOD:REQUEST) with proper UTC timestamps.
    event_info requires: name, date (e.g. 'December 15, 2025'), time (e.g. '10:00 AM'),
    location, duration_minutes, description
    """
    start_local = datetime.strptime(
        f"{event_info['date']} {event_info['time']}",
        "%B %d, %Y %I:%M %p"
    ).replace(tzinfo=ZoneInfo(tz_name))

    end_local = start_local + timedelta(minutes=event_info.get("duration_minutes", 60))
    dtstart = start_local.astimezone(ZoneInfo("UTC")).strftime("%Y%m%dT%H%M%SZ")
    dtend = end_local.astimezone(ZoneInfo("UTC")).strftime("%Y%m%dT%H%M%SZ")
    dtstamp = datetime.now(tz=ZoneInfo("UTC")).strftime("%Y%m%dT%H%M%SZ")
    uid = f"{uuid.uuid4()}@{organizer_email.split('@')[-1]}"

    lines = [
        "BEGIN:VCALENDAR",
        "PRODID:-//Python RSVP Invite//EN",
        "VERSION:2.0",
        "CALSCALE:GREGORIAN",
        "METHOD:REQUEST",
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTAMP:{dtstamp}",
        f"DTSTART:{dtstart}",
        f"DTEND:{dtend}",
        f"SUMMARY:{event_info['name']}",
        f"LOCATION:{event_info['location']}",
        f"DESCRIPTION:{event_info.get('description','')}",
        f"ORGANIZER;CN=Event Team:mailto:{organizer_email}",
        f"ATTENDEE;CN={recipient_email};ROLE=REQ-PARTICIPANT;PARTSTAT=NEEDS-ACTION;RSVP=TRUE:mailto:{recipient_email}",
        "SEQUENCE:0",
        "STATUS:CONFIRMED",
        "TRANSP:OPAQUE",
        "END:VEVENT",
        "END:VCALENDAR"
    ]
    return CRLF.join(lines) + CRLF

def send_event_email(recipient_email: str,
                     recipient_name: str,
                     event_info: Dict[str, Any],
                     organizer_email: str,
                     organizer_password: str) -> Dict[str, Any]:
    """
    Sends a multipart email with a text body and a calendar invite attachment.
    Returns a per-recipient result dict.
    """
    msg = MIMEMultipart("mixed")
    msg["From"] = organizer_email
    msg["To"] = recipient_email
    msg["Subject"] = f"Invitation: {event_info['name']}"

    body = f"""
Dear {recipient_name},

You're invited to {event_info['name']}!

Event Details:
- Date: {event_info['date']}
- Time: {event_info['time']}
- Location: {event_info['location']}

Please use the Accept/Decline buttons to RSVP.

We look forward to seeing you there!

Best regards,
Event Team
    """.strip()

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(body, "plain"))
    msg.attach(alt)

    # Calendar invite (text/calendar)
    ics = build_ics_request(event_info, organizer_email, recipient_email)
    cal_part = MIMEText(ics, _subtype="calendar", _charset="utf-8")
    cal_part.replace_header("Content-Type", "text/calendar; method=REQUEST; charset=utf-8; name=invite.ics")
    cal_part.add_header("Content-Class", "urn:content-classes:calendarmessage")
    cal_part.add_header("Content-Disposition", 'attachment; filename="invite.ics"')
    msg.attach(cal_part)

    # Optional .ics attachment for broad client compatibility
    ical_atch = MIMEBase("application", "ics")
    ical_atch.set_payload(ics.encode("utf-8"))
    encoders.encode_base64(ical_atch)
    ical_atch.add_header("Content-Disposition", 'attachment; filename="invite.ics"')
    msg.attach(ical_atch)

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(organizer_email, organizer_password)
            server.send_message(msg)
        return {"email": recipient_email, "status": "sent"}
    except Exception as e:
        return {"email": recipient_email, "status": "failed", "error": str(e)}

# =============================================================================
# Flask app + routes
# =============================================================================

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

# ---------- Your extraction endpoints (unchanged output) ----------

@app.route("/extract", methods=["POST"])
def extract():
    """
    Request JSON:
    {
      "query": "Computer Society hosting hackathon for 200 members",
      "use_enrichment": true
    }
    """
    try:
        payload = request.get_json(force=True) or {}
        query = payload.get("query")
        use_enrichment = bool(payload.get("use_enrichment", True))
        if not query or not isinstance(query, str):
            return jsonify({"success": False, "error": "Invalid or missing 'query'"}), 400

        result = extract_event_intent(query, use_enrichment=use_enrichment)
        requirements = [req.lower() for req in result['requirements']]
        attendees = result['attendees']
        graph_payload = build_reasoning_path_for_query(
            requirements,
            attendees=attendees,
            min_coverage=0.3,
            topk_for_context=5
        )
        return graph_payload, 200

    except Exception as e:
        return jsonify({"success": False, "error": f"Server error: {e}"}), 500

@app.route("/extract/batch", methods=["POST"])
def extract_batch():
    """
    Request JSON:
    {
      "queries": [
        "Computer Society hosting hackathon for 200 members",
        "Karate Society hosting sparring session for 20 members"
      ],
      "use_enrichment": true
    }
    """
    try:
        payload = request.get_json(force=True) or {}
        queries = payload.get("queries", [])
        use_enrichment = bool(payload.get("use_enrichment", True))

        if not isinstance(queries, list) or not all(isinstance(q, str) and q.strip() for q in queries):
            return jsonify({"success": False, "error": "Invalid or missing 'queries' (list of non-empty strings)"}), 400

        # Return the last built reasoning payload (matching your current behavior)
        last_payload = None
        for q in queries:
            r = extract_event_intent(q, use_enrichment=use_enrichment)
            requirements = [req.lower() for req in r['requirements']]
            attendees = r['attendees'] if isinstance(r.get('attendees'), int) else 0
            last_payload = build_reasoning_path_for_query(
                requirements,
                attendees=attendees or 200,
                min_coverage=0.6,
                topk_for_context=5
            )

        return last_payload or jsonify({"success": False, "error": "No payload produced"}), 200

    except Exception as e:
        return jsonify({"success": False, "error": f"Server error: {e}"}), 500

# ---------- NEW: Email invite endpoint ----------

@app.route("/invite", methods=["POST"])
def invite():
    try:
        payload = request.get_json(force=True) or {}
        event_info = payload.get("event_info") or {}
        tz_name = payload.get("timezone") or DEFAULT_TZ
        print(event_info)
        # Basic validation of event_info
        required_fields = ["name", "date", "time", "location"]
        missing = [f for f in required_fields if f not in event_info]
        if missing:
            return jsonify({"success": False, "error": f"Missing fields in event_info: {', '.join(missing)}"}), 400

        if "duration_minutes" not in event_info:
            event_info["duration_minutes"] = 60
        if "description" not in event_info:
            event_info["description"] = ""

        # Recipients
        recipients: List[str] = payload.get("recipients") or []
        mailing_list = payload.get("mailing_list")
        if mailing_list:
            recipients.extend([e.strip() for e in mailing_list.split(",") if e.strip()])

        recipients = sorted(set([r for r in recipients if isinstance(r, str) and "@" in r]))
        if not recipients:
            return jsonify({"success": False, "error": "No valid recipients provided"}), 400

        # Sender override or env/defaults
        organizer_email = payload.get("sender_email") or SENDER_EMAIL
        organizer_password = payload.get("sender_password") or SENDER_PASSWORD

        # Build per-recipient names from emails (simple heuristic)
        participants = [{"email": e, "name": e.split("@")[0].replace(".", " ").title()} for e in recipients]

        # Send
        results = []
        for p in participants:
            # ensure ICS uses the requested timezone
            local_event = dict(event_info)
            # build_ics_request reads DEFAULT_TZ, but we can pass tz explicitly by temporarily overriding DEFAULT_TZ
            # Instead, we pass tz to build_ics_request by wrapping it here:
            ics = build_ics_request(local_event, organizer_email, p["email"], tz_name=tz_name)
            # Reuse the email function but bypass a second ICS generation by temporarily injecting via headers?
            # Simpler: keep send_event_email which regenerates ICS identically (cheap), consistency > micro-opt
            result = send_event_email(
                recipient_email=p["email"],
                recipient_name=p["name"],
                event_info=local_event,
                organizer_email=organizer_email,
                organizer_password=organizer_password
            )
            results.append(result)

        success_count = sum(1 for r in results if r["status"] == "sent")
        return jsonify({
            "success": success_count == len(results),
            "sent": success_count,
            "failed": len(results) - success_count,
            "results": results
        }), 200

    except Exception as e:
        return jsonify({"success": False, "error": f"Server error: {e}"}), 500

# =============================================================================

if __name__ == "__main__":
    # Run: (set env) GOOGLE_API_KEY=... SENDER_EMAIL=... SENDER_PASSWORD=... python app.py
    app.run(host="0.0.0.0", port=8000, debug=True)
