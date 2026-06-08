"""
=============================================================
  DeliveryRL — FastAPI V2
  Modèle : PPO V2 (5 véhicules, contraintes capacité + froid)
  Inférence : pur numpy (sans PyTorch)
  Routes :
    GET  /health          → statut de l'API
    POST /assign          → 1 commande → véhicule + ETA
    POST /assign/batch    → N commandes → véhicules + tournée VRP + ETAs
=============================================================
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import math, os

# ─────────────────────────────────────────────
#  CONFIGURATION FLOTTE V2
# ─────────────────────────────────────────────
FLEET_CONFIG = [
    {"id": "V0", "max_capacity": 15, "is_refrigerated": True,
     "depot_lat": 48.7566, "depot_lon": 2.3522},
    {"id": "V1", "max_capacity": 15, "is_refrigerated": True,
     "depot_lat": 48.7551, "depot_lon": 2.4804},
    {"id": "V2", "max_capacity": 20, "is_refrigerated": False,
     "depot_lat": 48.7773, "depot_lon": 2.4567},
    {"id": "V3", "max_capacity": 20, "is_refrigerated": False,
     "depot_lat": 48.8134, "depot_lon": 2.3845},
    {"id": "V4", "max_capacity": 25, "is_refrigerated": False,
     "depot_lat": 48.7262, "depot_lon": 2.3652},
]
NUM_VEHICLES  = len(FLEET_CONFIG)
VEHICLE_NAMES = [v["id"] for v in FLEET_CONFIG]

# ─────────────────────────────────────────────
#  CHARGEMENT DES POIDS NUMPY
# ─────────────────────────────────────────────
WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "delivery_weights_v2.npz")

print(f"⏳ Chargement des poids : {WEIGHTS_PATH}")
try:
    w        = np.load(WEIGHTS_PATH)
    PI_L0_W  = w["pi_l0_w"];  PI_L0_B  = w["pi_l0_b"]
    PI_L2_W  = w["pi_l2_w"];  PI_L2_B  = w["pi_l2_b"]
    PI_L4_W  = w["pi_l4_w"];  PI_L4_B  = w["pi_l4_b"] 
    ACTION_W = w["action_w"]; ACTION_B = w["action_b"]
    MODEL_LOADED = True
    print(f"✅ Poids V2 chargés — action shape: {ACTION_W.shape}")
except Exception as e:
    MODEL_LOADED = False
    print(f"❌ Erreur chargement poids : {e}")
print(f"PI_L0_W shape: {PI_L0_W.shape}")
print(f"PI_L2_W shape: {PI_L2_W.shape}")
print(f"ACTION_W shape: {ACTION_W.shape}")
# ─────────────────────────────────────────────
#  IMPORT ENVIRONNEMENT V2
# ─────────────────────────────────────────────
from delivery_routing_env_v2 import (
    DeliveryRoutingEnvV2,
    _parse_temp,
    _needs_refrigeration,
    _haversine,
)

app = FastAPI(
    title       = "DeliveryRL API V2",
    description = "Affectation RL + VRP OR-Tools | 5 véhicules | contraintes capacité + froid",
    version     = "2.0.0",
)

# ─────────────────────────────────────────────
#  OR-TOOLS (optionnel)
# ─────────────────────────────────────────────
try:
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp
    ORTOOLS_AVAILABLE = True
    print("✅ OR-Tools disponible")
except ImportError:
    ORTOOLS_AVAILABLE = False
    print("⚠ OR-Tools non disponible — fallback nearest-neighbor")


# ─────────────────────────────────────────────
#  SCHÉMAS PYDANTIC
# ─────────────────────────────────────────────
class OrderIn(BaseModel):
    shippement_number     : str
    pickup_location       : Optional[str]   = ""
    pickup_address        : Optional[str]   = ""
    expected_pickup_time  : Optional[str]   = ""
    temp_control          : Optional[str]   = "Ambient"
    quantity              : Optional[float] = 1.0
    destination_store_name: Optional[str]   = ""
    destination_address   : Optional[str]   = ""
    expected_delivery_time: Optional[str]   = ""
    sender_name           : Optional[str]   = ""
    delivery_long         : Optional[float] = 2.3522
    delivery_lat          : Optional[float] = 48.8566
    pickup_long           : Optional[float] = 2.3522
    pickup_lat            : Optional[float] = 48.8566
    delivery_distance     : Optional[float] = 10.0
    delivery_time         : Optional[float] = 30.0
    visibility            : Optional[float] = 10000.0
    wind_speed            : Optional[float] = 10.0
    main_str              : Optional[str]   = "Clear"
    currentspeed_kmph     : Optional[float] = 50.0
    freeflowspeed         : Optional[float] = 80.0

class BatchIn(BaseModel):
    orders: List[OrderIn]

class AssignmentResult(BaseModel):
    shippement_number  : str
    assigned_vehicle   : str
    vehicle_index      : int
    is_refrigerated    : bool
    vehicle_capacity   : int
    eta_minutes        : float
    temp_control       : str
    delivery_distance  : float
    traffic_status     : str
    constraint_ok      : bool

class RouteStop(BaseModel):
    stop_order        : int
    shippement_number : str
    eta_minutes       : float
    delivery_lat      : float
    delivery_lon      : float

class VehicleRoute(BaseModel):
    vehicle           : str
    vehicle_index     : int
    is_refrigerated   : bool
    total_orders      : int
    total_load        : float
    max_capacity      : int
    total_distance_km : float
    route             : List[RouteStop]

class BatchResult(BaseModel):
    assignments    : List[AssignmentResult]
    routes         : List[VehicleRoute]
    summary        : dict
    vrp_solver     : str


# ─────────────────────────────────────────────
#  UTILITAIRES
# ─────────────────────────────────────────────
def _predict(obs: np.ndarray) -> int:
    x = obs.astype(np.float32).flatten()
    x = np.tanh(PI_L0_W @ x + PI_L0_B)   # 84 → 256
    x = np.tanh(PI_L2_W @ x + PI_L2_B)   # 256 → 256
    x = np.tanh(PI_L4_W @ x + PI_L4_B)   # 256 → 128
    return int(np.argmax(ACTION_W @ x + ACTION_B))  # 128 → 6

def _order_to_dict(o: OrderIn) -> dict:
    return {
        "shippement_number"   : o.shippement_number,
        "delivery_distance"   : o.delivery_distance,
        "delivery_time"       : o.delivery_time,
        "visibility"          : o.visibility,
        "wind speed"          : o.wind_speed,
        "currentspeed (kmph)" : o.currentspeed_kmph,
        "freeflowspeed"       : o.freeflowspeed,
        "quantity"            : o.quantity,
        "temp_control"        : o.temp_control,
        "delivery_lat"        : o.delivery_lat,
        "delivery_long"       : o.delivery_long,
        "pickup_lat"          : o.pickup_lat,
        "pickup_long"         : o.pickup_long,
    }

def _eta(o: OrderIn) -> float:
    t = (o.delivery_distance / max(o.currentspeed_kmph, 5.0)) * 60.0
    if o.main_str in ("Rain", "Snow"):   t *= 1.15
    elif o.main_str in ("Fog", "Mist"):  t *= 1.10
    return round(t, 1)

def _traffic(cur, free) -> str:
    r = cur / max(free, 1.0)
    return "🟢 Fluide" if r >= 0.85 else ("🟡 Modéré" if r >= 0.60 else "🔴 Dense")

def _dist_matrix(orders):
    n   = len(orders) + 1
    mat = [[0] * n for _ in range(n)]
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                oi, oj = orders[i-1], orders[j-1]
                mat[i][j] = int(_haversine(
                    oi["delivery_lat"], oi["delivery_long"],
                    oj["delivery_lat"], oj["delivery_long"]
                ) * 1000)
    return mat

def _vrp_ortools(orders: list) -> list:
    if len(orders) <= 1:
        return orders
    mat     = _dist_matrix(orders)
    n_nodes = len(mat)
    manager = pywrapcp.RoutingIndexManager(n_nodes, 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    cb_idx  = routing.RegisterTransitCallback(
        lambda f, t: mat[manager.IndexToNode(f)][manager.IndexToNode(t)]
    )
    routing.SetArcCostEvaluatorOfAllVehicles(cb_idx)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    params.time_limit.seconds = 5
    solution = routing.SolveWithParameters(params)
    if not solution:
        return orders
    route, idx = [], routing.Start(0)
    while not routing.IsEnd(idx):
        node = manager.IndexToNode(idx)
        if node != 0:
            route.append(orders[node - 1])
        idx = solution.Value(routing.NextVar(idx))
    return route

def _vrp_nearest(orders: list) -> list:
    if len(orders) <= 1:
        return orders
    unvisited = orders.copy()
    route     = [unvisited.pop(0)]
    while unvisited:
        last    = route[-1]
        nearest = min(unvisited, key=lambda o: _haversine(
            last["delivery_lat"], last["delivery_long"],
            o["delivery_lat"],    o["delivery_long"]
        ))
        route.append(nearest)
        unvisited.remove(nearest)
    return route

def _optimize_route(orders):
    if ORTOOLS_AVAILABLE and len(orders) > 1:
        return _vrp_ortools(orders), "OR-Tools"
    return _vrp_nearest(orders), "nearest-neighbor"


# ─────────────────────────────────────────────
#  ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status"        : "ok",
        "model_loaded"  : MODEL_LOADED,
        "model_version" : "V2",
        "ortools"       : ORTOOLS_AVAILABLE,
        "num_vehicles"  : NUM_VEHICLES,
        "fleet"         : [
            {
                "id"            : v["id"],
                "max_capacity"  : v["max_capacity"],
                "is_refrigerated": v["is_refrigerated"],
            }
            for v in FLEET_CONFIG
        ],
    }


@app.post("/assign", response_model=AssignmentResult)
def assign_single(order: OrderIn):
    """
    Affecte UNE commande à un véhicule.
    Respecte contraintes : capacité + chaîne du froid.
    """
    if not MODEL_LOADED:
        raise HTTPException(503, "Poids non chargés.")

    order_dict = _order_to_dict(order)
    env        = DeliveryRoutingEnvV2(
        orders_data=[order_dict],
        render_mode=None,
    )
    obs, _     = env.reset(seed=42)
    action     = _predict(obs)
    v_idx      = action % NUM_VEHICLES
    vehicle    = FLEET_CONFIG[v_idx]

    # Vérifie contraintes
    temp_val       = _parse_temp(order.temp_control)
    needs_refrig   = _needs_refrigeration(temp_val)
    constraint_ok  = (
        (not needs_refrig or vehicle["is_refrigerated"])
        and order.quantity <= vehicle["max_capacity"]
    )

    return AssignmentResult(
        shippement_number  = order.shippement_number,
        assigned_vehicle   = vehicle["id"],
        vehicle_index      = v_idx,
        is_refrigerated    = vehicle["is_refrigerated"],
        vehicle_capacity   = vehicle["max_capacity"],
        eta_minutes        = _eta(order),
        temp_control       = order.temp_control or "Ambient",
        delivery_distance  = order.delivery_distance,
        traffic_status     = _traffic(order.currentspeed_kmph, order.freeflowspeed),
        constraint_ok      = constraint_ok,
    )


@app.post("/assign/batch", response_model=BatchResult)
def assign_batch(batch: BatchIn):
    """
    Affecte N commandes + optimise la tournée par véhicule (VRP).
    Retourne : assignations + routes optimisées + ETAs + respect contraintes.
    """
    if not MODEL_LOADED:
        raise HTTPException(503, "Poids non chargés.")
    if not batch.orders:
        raise HTTPException(400, "Liste de commandes vide.")

    orders_dicts = [_order_to_dict(o) for o in batch.orders]
    env          = DeliveryRoutingEnvV2(
        orders_data=orders_dicts,
        render_mode=None,
    )
    obs, _       = env.reset(seed=42)

    # ── Affectation PPO ──────────────────────────────────────
    assignments_raw = []
    for i, order in enumerate(batch.orders):
        action  = _predict(obs)
        v_idx   = action % NUM_VEHICLES
        obs, _, terminated, truncated, _ = env.step(action)
        assignments_raw.append((order, v_idx))
        if terminated or truncated:
            # Commandes restantes → véhicule le moins chargé
            loads = [0] * NUM_VEHICLES
            for _, vi in assignments_raw:
                loads[vi] += 1
            for j in range(i + 1, len(batch.orders)):
                best = loads.index(min(loads))
                assignments_raw.append((batch.orders[j], best))
                loads[best] += 1
            break

    # ── Résultats d'assignation ───────────────────────────────
    results = []
    for o, v_idx in assignments_raw:
        vehicle       = FLEET_CONFIG[v_idx]
        temp_val      = _parse_temp(o.temp_control)
        needs_refrig  = _needs_refrigeration(temp_val)
        constraint_ok = (
            (not needs_refrig or vehicle["is_refrigerated"])
            and o.quantity <= vehicle["max_capacity"]
        )
        results.append(AssignmentResult(
            shippement_number  = o.shippement_number,
            assigned_vehicle   = vehicle["id"],
            vehicle_index      = v_idx,
            is_refrigerated    = vehicle["is_refrigerated"],
            vehicle_capacity   = vehicle["max_capacity"],
            eta_minutes        = _eta(o),
            temp_control       = o.temp_control or "Ambient",
            delivery_distance  = o.delivery_distance,
            traffic_status     = _traffic(o.currentspeed_kmph, o.freeflowspeed),
            constraint_ok      = constraint_ok,
        ))

    # ── VRP par véhicule ─────────────────────────────────────
    vehicle_orders: dict = {i: [] for i in range(NUM_VEHICLES)}
    vehicle_loads:  dict = {i: 0.0 for i in range(NUM_VEHICLES)}
    for order, v_idx in assignments_raw:
        vehicle_orders[v_idx].append(_order_to_dict(order))
        vehicle_loads[v_idx] += order.quantity

    routes     = []
    solver_used = "none"
    for v_idx in range(NUM_VEHICLES):
        v_orders = vehicle_orders[v_idx]
        if not v_orders:
            continue
        optimized, solver_used = _optimize_route(v_orders)
        cumeta = 0.0
        stops  = []
        for i, o in enumerate(optimized):
            leg    = (o["delivery_distance"] / max(o["currentspeed (kmph)"], 5.0)) * 60.0
            cumeta += leg
            stops.append(RouteStop(
                stop_order        = i + 1,
                shippement_number = o["shippement_number"],
                eta_minutes       = round(cumeta, 1),
                delivery_lat      = o["delivery_lat"],
                delivery_lon      = o["delivery_long"],
            ))
        total_dist = sum(
            _haversine(
                optimized[i]["delivery_lat"],  optimized[i]["delivery_long"],
                optimized[i+1]["delivery_lat"],optimized[i+1]["delivery_long"]
            ) for i in range(len(optimized)-1)
        ) if len(optimized) > 1 else optimized[0]["delivery_distance"]

        cfg = FLEET_CONFIG[v_idx]
        routes.append(VehicleRoute(
            vehicle           = cfg["id"],
            vehicle_index     = v_idx,
            is_refrigerated   = cfg["is_refrigerated"],
            total_orders      = len(v_orders),
            total_load        = vehicle_loads[v_idx],
            max_capacity      = cfg["max_capacity"],
            total_distance_km = round(total_dist, 2),
            route             = stops,
        ))

    # Taux de respect des contraintes
    ok_count = sum(1 for r in results if r.constraint_ok)

    return BatchResult(
        assignments = results,
        routes      = routes,
        summary     = {
            "total_orders"       : len(batch.orders),
            "assigned"           : len(results),
            "vehicles_used"      : len(routes),
            "constraints_ok_rate": f"{ok_count}/{len(results)}",
            "avg_eta_minutes"    : round(
                sum(r.eta_minutes for r in results) / len(results), 1
            ),
        },
        vrp_solver = solver_used,
    )
