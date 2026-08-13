"""
FastAPI service for the AI-Powered Trade Route Recommendation System.

Endpoints:
    GET  /health              - service status, graph/model info
    GET  /ports?query=singapore - search ports by name (for picking origin/destination)
    POST /recommend            - get ranked route recommendations

Run from project root with:
    uvicorn src.api.main:app --reload
"""

import pickle
from itertools import islice
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import lightgbm as lgb
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Paths (relative to project root, resolved from this file's location)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]  # project root
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_PATH = PROCESSED_DIR / "models" / "lgbm_ranker.txt"
GRAPH_PATH = PROCESSED_DIR / "maritime_graph_disrupted.gpickle"

FEATURE_COLS = [
    "total_distance_nm", "num_hops", "disruption_exposure_max",
    "disruption_exposure_sum", "bottleneck_depth_m", "avg_harbor_score",
    "num_disrupted_ports",
]
HARBOR_SIZE_SCORE = {"Large": 3, "Medium": 2, "Small": 1, "Very Small": 0, "Unknown": 0}
PENALTY_MULTIPLIER = 30.0  # must match Phase 3 calibration
K_CANDIDATES = 4  # per weight strategy, matches Phase 5 training

# ---------------------------------------------------------------------------
# App + startup state
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI-Powered Trade Route Recommendation System",
    description="Recommends ranked alternative maritime routes under disruption scenarios.",
    version="0.1.0",
)

state = {"graph": None, "model": None, "loaded": False}


@app.on_event("startup")
def load_artifacts():
    """Loads the graph and trained model once at service startup."""
    try:
        with open(GRAPH_PATH, "rb") as f:
            state["graph"] = pickle.load(f)
        state["model"] = lgb.Booster(model_file=str(MODEL_PATH))
        state["loaded"] = True
        print(f"Loaded graph ({state['graph'].number_of_nodes()} nodes, "
              f"{state['graph'].number_of_edges()} edges) and model from {MODEL_PATH}")
    except Exception as e:
        state["loaded"] = False
        print(f"WARNING: failed to load artifacts at startup: {e}")


# ---------------------------------------------------------------------------
# Core routing / feature logic (mirrors notebooks 04 and 05)
# ---------------------------------------------------------------------------
def port_lookup_by_name(G, name_substring: str):
    return [
        {"port_id": node_id, "port_name": attrs["port_name"], "country": attrs["country"]}
        for node_id, attrs in G.nodes(data=True)
        if name_substring.lower() in attrs["port_name"].lower()
    ]


def k_shortest_paths(G, origin_id, dest_id, k=5, weight_key="disrupted_weight"):
    if origin_id not in G or dest_id not in G:
        return []
    if not nx.has_path(G, origin_id, dest_id):
        return []
    try:
        paths_generator = nx.shortest_simple_paths(G, origin_id, dest_id, weight=weight_key)
        return list(islice(paths_generator, k))
    except Exception:
        return []


def compute_route_features(G, path, weight_key_for_severity="disruption_severity_edge"):
    """Computes the same 7 features used in Phase 5 training."""
    total_distance = 0.0
    max_exposure = 0.0
    sum_exposure = 0.0

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge = G[u][v]
        total_distance += edge.get("distance_nm", 0.0)
        sev = edge.get(weight_key_for_severity, 0.0)
        max_exposure = max(max_exposure, sev)
        sum_exposure += sev

    depths = [G.nodes[p].get("cargo_pier_depth_m", np.nan) for p in path]
    depths = [d for d in depths if not (d is None or (isinstance(d, float) and np.isnan(d)))]
    bottleneck_depth = min(depths) if depths else 0.0

    harbor_scores = [HARBOR_SIZE_SCORE.get(G.nodes[p].get("harbor_size", "Unknown"), 0) for p in path]
    avg_harbor_score = float(np.mean(harbor_scores)) if harbor_scores else 0.0

    num_disrupted_ports = sum(
        1 for p in path if G.nodes[p].get("disruption_severity", 0.0) > 0
    )

    return {
        "total_distance_nm": total_distance,
        "num_hops": len(path) - 1,
        "disruption_exposure_max": max_exposure,
        "disruption_exposure_sum": sum_exposure,
        "bottleneck_depth_m": bottleneck_depth,
        "avg_harbor_score": avg_harbor_score,
        "num_disrupted_ports": num_disrupted_ports,
    }


def generate_and_rank_candidates(G, model, origin_id, dest_id, k=K_CANDIDATES, top_n=5):
    """
    Generates candidates from both weight strategies (disruption-averse + pure-distance),
    scores them with the trained LightGBM ranker, and returns the top_n ranked routes.
    """
    paths_safe = k_shortest_paths(G, origin_id, dest_id, k=k, weight_key="disrupted_weight")
    paths_short = k_shortest_paths(G, origin_id, dest_id, k=k, weight_key="base_weight")

    seen = set()
    combined_paths = []
    for path in paths_safe + paths_short:
        key = tuple(path)
        if key not in seen:
            seen.add(key)
            combined_paths.append(path)

    if not combined_paths:
        return []

    rows = []
    for path in combined_paths:
        feats = compute_route_features(G, path)
        feats["path"] = path
        rows.append(feats)

    df = pd.DataFrame(rows)
    scores = model.predict(df[FEATURE_COLS])
    df["score"] = scores
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    results = []
    for _, row in df.head(top_n).iterrows():
        path = row["path"]
        results.append({
            "route": [G.nodes[p]["port_name"] for p in path],
            "port_ids": path,
            "coordinates": [
                {
                    "name": G.nodes[p]["port_name"],
                    "latitude": G.nodes[p]["latitude"],
                    "longitude": G.nodes[p]["longitude"],
                    "disrupted": G.nodes[p].get("disruption_severity", 0.0) > 0,
                }
                for p in path
            ],
            "score": round(float(row["score"]), 4),
            "total_distance_nm": round(float(row["total_distance_nm"]), 1),
            "num_hops": int(row["num_hops"]),
            "disruption_exposure_max": round(float(row["disruption_exposure_max"]), 3),
            "num_disrupted_ports": int(row["num_disrupted_ports"]),
        })
    return results


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class RecommendRequest(BaseModel):
    origin_port_id: Optional[int] = Field(None, description="Port ID for origin (use /ports to look up)")
    destination_port_id: Optional[int] = Field(None, description="Port ID for destination")
    origin_name: Optional[str] = Field(None, description="Alternative: origin port name substring")
    destination_name: Optional[str] = Field(None, description="Alternative: destination port name substring")
    top_n: int = Field(5, ge=1, le=10, description="Number of ranked routes to return")


class RouteResult(BaseModel):
    route: list
    port_ids: list
    coordinates: list
    score: float
    total_distance_nm: float
    num_hops: int
    disruption_exposure_max: float
    num_disrupted_ports: int


class RecommendResponse(BaseModel):
    origin: str
    destination: str
    num_candidates_considered: int
    recommendations: list[RouteResult]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    G = state["graph"]
    return {
        "status": "ok" if state["loaded"] else "degraded",
        "graph_loaded": G is not None,
        "num_nodes": G.number_of_nodes() if G is not None else 0,
        "num_edges": G.number_of_edges() if G is not None else 0,
        "model_loaded": state["model"] is not None,
    }


@app.get("/ports")
def search_ports(query: str = Query(..., min_length=2, description="Port name substring, e.g. 'singapore'")):
    G = state["graph"]
    if G is None:
        raise HTTPException(status_code=503, detail="Graph not loaded")
    matches = port_lookup_by_name(G, query)
    if not matches:
        raise HTTPException(status_code=404, detail=f"No ports found matching '{query}'")
    return {"query": query, "count": len(matches), "results": matches}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    G = state["graph"]
    model = state["model"]
    if G is None or model is None:
        raise HTTPException(status_code=503, detail="Service artifacts not loaded. Check /health.")

    # Resolve origin
    origin_id = req.origin_port_id
    if origin_id is None and req.origin_name:
        matches = port_lookup_by_name(G, req.origin_name)
        if not matches:
            raise HTTPException(status_code=404, detail=f"No port found matching origin '{req.origin_name}'")
        origin_id = matches[0]["port_id"]
    if origin_id is None:
        raise HTTPException(status_code=400, detail="Provide either origin_port_id or origin_name")

    # Resolve destination
    dest_id = req.destination_port_id
    if dest_id is None and req.destination_name:
        matches = port_lookup_by_name(G, req.destination_name)
        if not matches:
            raise HTTPException(status_code=404, detail=f"No port found matching destination '{req.destination_name}'")
        dest_id = matches[0]["port_id"]
    if dest_id is None:
        raise HTTPException(status_code=400, detail="Provide either destination_port_id or destination_name")

    if origin_id not in G:
        raise HTTPException(status_code=404, detail=f"Origin port_id {origin_id} not found in graph")
    if dest_id not in G:
        raise HTTPException(status_code=404, detail=f"Destination port_id {dest_id} not found in graph")
    if not nx.has_path(G, origin_id, dest_id):
        raise HTTPException(status_code=422, detail="No feasible route exists between these ports")

    recommendations = generate_and_rank_candidates(G, model, origin_id, dest_id, top_n=req.top_n)
    if not recommendations:
        raise HTTPException(status_code=422, detail="No candidate routes could be generated")

    return {
        "origin": G.nodes[origin_id]["port_name"],
        "destination": G.nodes[dest_id]["port_name"],
        "num_candidates_considered": len(recommendations),
        "recommendations": recommendations,
    }
