import json
from neo4j import GraphDatabase

# --- Neo4j connection (use your creds) ---
uri = "neo4j+s://30b84a7a.databases.neo4j.io"
user = "neo4j"
password = "YO9d3i-lA9EIXNwBKamz_d1lRwSsfMbxaugODbpbQ3c"
driver = GraphDatabase.driver(uri, auth=(user, password))

# Count after capacity
CYPHER_COUNT_CAPACITY = """
MATCH (v:Venue)
RETURN count(v) AS total,
       count { MATCH (v:Venue) WHERE v.capacity >= $attendees RETURN v } AS pass_capacity
"""

# Count after coverage (>= min_coverage)
CYPHER_COUNT_COVERAGE = """
MATCH (v:Venue)
WHERE v.capacity >= $attendees
OPTIONAL MATCH (v)-[:HAS_AMENITY]->(a:Amenity)
WITH v, collect(DISTINCT toLower(a.name)) AS ams, $req_names AS req
WITH v,
     CASE WHEN size(req)=0 THEN 1.0
          ELSE toFloat(size([x IN req WHERE x IN ams])) / toFloat(size(req))
     END AS coverage
RETURN count(v) AS pass_coverage
"""

# Scored top-k (we'll take top-1 for the path, optionally list 2–3 for context)
CYPHER_SCORED = """
MATCH (v:Venue)
WHERE v.capacity >= $attendees
OPTIONAL MATCH (v)-[:HAS_AMENITY]->(a:Amenity)
WITH v, collect(DISTINCT toLower(a.name)) AS ams, $req_names AS req
WITH v, ams, req,
     size([x IN req WHERE x IN ams]) AS matched,
     CASE
       WHEN size(req)=0 THEN 1.0
       ELSE toFloat(size([x IN req WHERE x IN ams])) / toFloat(size(req))
     END AS coverage,
     (v.capacity - $attendees) AS slack
WHERE coverage >= $min_coverage
WITH v, ams, req, matched, coverage, slack,
     (1.0/(1.0 + exp(-0.15*slack))) AS slack_penalty
WITH v, ams, req, matched, coverage, slack, slack_penalty,
     (0.65*coverage + 0.15*coverage - 0.20*slack_penalty) AS final_score,
     [x IN req WHERE x IN ams] AS matched_list,
     [x IN req WHERE NOT x IN ams] AS missing_list
RETURN v.name AS venue,
       v.canonical_name AS id,
       v.capacity AS capacity,
       matched,
       round(coverage,3) AS coverage,
       slack,
       round(final_score,3) AS score,
       matched_list,
       missing_list
ORDER BY score DESC, slack ASC, capacity ASC
LIMIT $k
"""

def _add(nodes, links, nid, label, group=None, size=None, meta=None, parent_id=None):
    node = {"id": nid, "label": label}
    if group is not None: node["group"] = group
    if size is not None: node["val"] = size
    if meta is not None: node["meta"] = meta
    nodes.append(node)
    if parent_id:
        links.append({"source": parent_id, "target": nid})
    return node

def build_reasoning_path_for_query(required_amenities, attendees, min_coverage=0.6, topk_for_context=1):
    req = [r.strip().lower() for r in required_amenities]

    with driver.session() as session:
        # step counts
        counts = session.run(CYPHER_COUNT_CAPACITY, attendees=attendees).data()[0]
        total = counts["total"]
        pass_capacity = counts["pass_capacity"]

        pass_coverage = session.run(
            CYPHER_COUNT_COVERAGE,
            attendees=attendees,
            req_names=req
        ).data()[0]["pass_coverage"]

        # top-k scored (we'll show only best in the path, optionally list 1–3)
        rows = session.run(
            CYPHER_SCORED,
            attendees=attendees,
            req_names=req,
            min_coverage=min_coverage,
            k=max(1, min(topk_for_context, 5))
        ).data()

    nodes, links = [], []

    # Root
    root_id = "root"
    _add(nodes, links, root_id, "Query", group="root", size=8,
         meta={"attendees": attendees, "min_coverage": min_coverage, "required": req})

    # Step 1: Capacity filter
    s1 = "step_capacity"
    _add(nodes, links, s1, f"Capacity ≥ {attendees}  (kept {pass_capacity}/{total})",
         group="step", size=6, parent_id=root_id)

    # Step 2: Coverage filter
    s2 = "step_coverage"
    _add(nodes, links, s2, f"Coverage ≥ {min_coverage}  (kept {pass_coverage}/{pass_capacity})",
         group="step", size=6, parent_id=s1)

    # Step 3: Scoring rule summary
    s3 = "step_scoring"
    _add(nodes, links, s3, "Score = 0.65*Coverage + 0.15*Coverage - 0.20*SlackPenalty",
         group="step", size=6, parent_id=s2)

    # Final: best venue + show present amenities for ALL shortlisted venues
    final_id = "final"
    if rows:
        best = rows[0]
        best_vid = f"venue_{best['id']}"

        # List short-listed venues (topk_for_context) under step s3
        for i, r in enumerate(rows):
            vid = f"venue_{r['id']}"
            label = f"{r['venue']} · cap {r['capacity']} · score {r['score']}"
            _add(
                nodes, links, vid, label,
                group=("best" if i == 0 else "shortlist"),
                size=(7 if i == 0 else 4),
                parent_id=s3,
                meta={
                    "coverage": r["coverage"],
                    "slack": r["slack"],
                    "matched": r["matched"],
                    "score": r["score"],
                    "capacity": r["capacity"],
                }
            )

            # Matched (present) amenities for EVERY shortlisted venue
            matched_group = f"{vid}_matched"
            _add(nodes, links, matched_group, "Matched amenities", group="explain", parent_id=vid)
            for a in r["matched_list"]:
                _add(nodes, links, f"{vid}_m_{a}", a, group="amenity", parent_id=matched_group)

            # (Optional) Missing amenities ONLY for the best to keep graph small
            if i == 0:
                missing_group = f"{vid}_missing"
                _add(nodes, links, missing_group, "Missing amenities", group="explain", parent_id=vid)
                for a in r["missing_list"]:
                    _add(nodes, links, f"{vid}_x_{a}", a, group="amenity_missing", parent_id=missing_group)

        # Final selection node attached to the best venue
        _add(
            nodes, links, final_id,
            f"Selected → {best['venue']} · score {best['score']}",
            group="final", size=8, parent_id=best_vid
        )
        # Optional back edge so you can highlight the full reasoning loop
        links.append({"source": final_id, "target": root_id})

    else:
        _add(nodes, links, final_id, "Selected → none (no candidate over threshold)", group="final", size=8, parent_id=s3)

    # Minimal path list you can highlight in UI
    path = [
        {"source": root_id, "target": s1},
        {"source": s1, "target": s2},
        {"source": s2, "target": s3},
    ]
    if rows:
        best = rows[0]
        path.append({"source": s3, "target": f"venue_{best['id']}"})
        path.append({"source": f"venue_{best['id']}", "target": final_id})

    # Text block
    text = (
        f"Reasoning: start → capacity filter (kept {pass_capacity}/{total}) "
        f"→ coverage ≥ {min_coverage} (kept {pass_coverage}/{pass_capacity}) "
        f"→ score (coverage vs slack) → select best.\n"
        f"Req amenities: {', '.join(req) or '(none)'}."
    )

    return {"nodes": nodes, "links": links, "path": path, "textInformation": text}


# # --- Example run (prints the minimal payload) ---
# if __name__ == "__main__":
#     requirements = ["Wi-Fi", "av/Projector", "Accessibility", "Power outlets", "Whiteboard"]
#     requirements = [req.lower() for req in requirements]
#     payload = build_reasoning_path_for_query(requirements, attendees=200, min_coverage=0.6, topk_for_context=3)
#     print(json.dumps(payload, indent=2))

#     with open("example_reasoning_path_output.json", "w", encoding="utf-8") as f:
#         json.dump(payload, f, indent=2)