# app.py
import os
import json
from typing import Optional, List, Literal, TypedDict

from flask import Flask, request, jsonify
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode  # kept if you add tools later
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------- Types ----------

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

# ---------- LLM Client ----------

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # set this in your environment
# if not GOOGLE_API_KEY:
#     # Fail fast with a clear message if key is missing
#     raise RuntimeError("Missing GOOGLE_API_KEY environment variable.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key="AIzaSyCRtIOzit-vgIiE-7tvvSQAcCYoVJlO3UE",
    temperature=0.7,
    max_tokens=1000,
)

# ---------- Nodes ----------

def extract_intent_node(state: EventExtractionState) -> EventExtractionState:
    """Extracts structured event info from natural language query."""
    query = state["query"]

    system_prompt = """You are an expert event coordinator specializing in matching events to appropriate venues.

Your task is to analyze an event booking query and extract ONLY information relevant to the venue side.

Return JSON in this exact format:
{
  "organizer": "string",
  "event_type": "string",
  "attendees": number,
  "requirements": ["..."],
  "constraints": ["..."]
}

Clarifications:
- Focus only on what the venue must provide or accommodate — not what the organizer brings.
- Exclude non-venue requirements such as snacks, judges, prizes, or staff.
- Include elements like: space type, equipment/facilities (lighting, power, Wi-Fi, seating, mats, projectors, sound), environment (quiet, ventilation, acoustics), safety, accessibility, comfort.
- Always include accessibility and safety considerations.

Guidelines by event type:
- Tech: power outlets, Wi-Fi, tables, seating, projectors, whiteboards, overnight access, ventilation, accessibility, safety.
- Drama: stage, lighting, prop storage, soundproofing, acoustics, seating, accessibility, safety.
- Physical (Karate): open space, mats, ventilation, safety gear storage, first aid, water access.
- Lectures: seating, projector, screen, microphone, lighting, acoustics, accessibility.
- Exhibitions: display stands, lighting, open area, accessibility, weather protection, security.
- Food: kitchen access, ventilation, hygiene, tables, accessibility, safety.
- Outdoor: open area, weather protection, restrooms, accessibility, safety.
"""

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Query: {query}")
        ])
        raw_output = response.content or ""

        # Extract JSON payload if fenced
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
    """Validates extracted data for completeness and correctness."""
    errors = []
    if not state.get("organizer"):
        errors.append("Organizer not identified")
    if not state.get("event_type"):
        errors.append("Event type not identified")
    if not state.get("attendees") or (isinstance(state["attendees"], int) and state["attendees"] <= 0):
        errors.append("Invalid or missing attendee count")
    return {**state, "error": "; ".join(errors)} if errors else state

def enrich_constraints_node(state: EventExtractionState) -> EventExtractionState:
    """Adds contextual enrichment to constraints."""
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
    """Pass-through; final state is already JSON-serializable."""
    return state

# ---------- Routing Logic ----------

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

# ---------- Flask App ----------

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/extract", methods=["POST"])
def extract():
    """
    Request JSON:
    {
      "query": "Computer Society hosting hackathon for 200 members",
      "use_enrichment": true  // optional (default: true)
    }
    """
    try:
        payload = request.get_json(force=True) or {}
        query = payload.get("query")
        use_enrichment = bool(payload.get("use_enrichment", True))
        print("hello")
        print(query)
        if not query or not isinstance(query, str):
            return jsonify({"success": False, "error": "Invalid or missing 'query'"}), 400

        result = extract_event_intent(query, use_enrichment=use_enrichment)
        print(result)
        # Always return JSON
        return jsonify({
            "success": result.get("error") is None,
            "data": {
                "query": result.get("query"),
                "organizer": result.get("organizer"),
                "event_type": result.get("event_type"),
                "attendees": result.get("attendees"),
                "requirements": result.get("requirements", []),
                "constraints": result.get("constraints", []),
                "enriched_constraints": result.get("enriched_constraints", [])
            },
            "error": result.get("error"),
            # For debugging you can expose raw_extraction, but keep it off by default
            # "raw_extraction": result.get("raw_extraction"),
        }), 200

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
      "use_enrichment": true  // optional
    }
    """
    try:
        payload = request.get_json(force=True) or {}
        queries = payload.get("queries", [])
        use_enrichment = bool(payload.get("use_enrichment", True))

        if not isinstance(queries, list) or not all(isinstance(q, str) and q.strip() for q in queries):
            return jsonify({"success": False, "error": "Invalid or missing 'queries' (list of non-empty strings)"}), 400

        results = []
        for q in queries:
            r = extract_event_intent(q, use_enrichment=use_enrichment)
            results.append({
                "query": r.get("query"),
                "organizer": r.get("organizer"),
                "event_type": r.get("event_type"),
                "attendees": r.get("attendees"),
                "requirements": r.get("requirements", []),
                "constraints": r.get("constraints", []),
                "enriched_constraints": r.get("enriched_constraints", []),
                "error": r.get("error"),
                "success": r.get("error") is None
            })

        return jsonify({"success": True, "results": results}), 200

    except Exception as e:
        return jsonify({"success": False, "error": f"Server error: {e}"}), 500

if __name__ == "__main__":
    # Run: GOOGLE_API_KEY=... python app.py
    app.run(host="0.0.0.0", port=8000, debug=True)
