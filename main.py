"""
=============================================================
  DeliveryRL — FastAPI v2 (inférence numpy, sans PyTorch)
  Routes :
    GET  /health          → statut de l'API
    POST /assign          → 1 commande  → véhicule + ETA
    POST /assign/batch    → N commandes → véhicules + tournée VRP + ETAs
=============================================================
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import math
import os

# ─────────────────────────────────────────────
#  CHARGEMENT DES POIDS (pur numpy, sans torch)
# ─────────────────────────────────────────────
WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "delivery_weights.npz")

print(f"⏳ Chargement des poids : {WEIGHTS_PATH}")
try:
    w = np.load(WEIGHTS_PATH)
    PI_L0_W  = w["pi_l0_w"]
    PI_L0_B  = w["pi_l0_b"]
    PI_L2_W  = w["pi_l2_w"]
    PI_L2_B  = w["pi_l2_b"]
    ACTION_W = w["action_w"]
    ACTION_B = w["action_b"]
    print(f"✅ Poids chargés — action_net shape: {ACTION_W.shape}")
    MODEL_LOADED = True
except Exception as e:
    print(f"❌ Erreur chargement poids : {e}")
    MODEL_LOADED = False


def predict_action(obs: np.ndarray) -> int:
    """
    Inférence manuelle du réseau MLP PPO en pur numpy.
    Reproduit : obs → MLP → logits → argmax → action
    """
    x = obs.astype(np.float32).flatten()

    # Couche 1
    x = np.tanh(PI_L0_W @ x + PI_L0_B)
    # Couche 2
    x = np.tanh(PI_L2_W @ x + PI_L2_B)
    # Logits actions
    logits = ACTION_W @ x + ACTION_B
    # Action = argmax (mode déterministe)
    return int(np.argmax(logits))


# ─────────────────────────────────────────────
#  IMPORT ENV (pour construire l'observation)
# ─────────────────────────────────────────────
from delivery_routing_env import DeliveryRoutingEnv

NUM_VEHICLES  = 3
VEHICLE_NAMES = ["Véhicule 0", "Véhicule 1", "Véhicule 2"]

app = FastAPI(
    title       = "DeliveryRL API",
    description = "Affectation RL + optimisation de tournée VRP",
    version     = "2.0.0",
)


# ─────────────────────────────────────────────
#  SCHÉMAS PYDANTIC
# ─────────────────────────────────────────────
class OrderIn(BaseModel):
    shippement_number     : str
    pickup_location       : Optional[str]   = ""
    pickup_address        : Optional[str]   = ""
    expected_pickup_time  : Optional[str]   = ""
    temp_control          : Optional[str]   = "Ambient"
    quantity              : Optional[str]   = "1 pallets"
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
    shippement_number : str
    assigned_vehicle  : str
    vehicle_index     : int
    eta_minutes       : float
    temp_control      : str
    delivery_distance : float
    traffic_status    : str

class RouteStop(BaseModel):
    stop_order        : int
    shippement_number : str
    destination       : str
    eta_minutes       : float
    delivery_lat      : float
    delivery_lon      : float

class VehicleRoute(BaseModel):
    vehicle           : str
    vehicle_index     : int
    total_orders      : int
    total_distance_km : float
    route             : List[RouteStop]

class BatchResult(BaseModel):
    assignments : List[AssignmentResult]
    routes      : List[VehicleRoute]
    summary     : dict


# ─────────────────────────────────────────────
#  UTILITAIRES
# ─────────────────────────────────────────────
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
    }

def _compute_eta(order: OrderIn) -> float:
    speed = max(order.currentspeed_kmph, 5.0)
    eta   = (order.delivery_distance / speed) * 60.0
    if order.main_str in ("Rain", "Snow"):   eta *= 1.15
    elif order.main_str in ("Fog", "Mist"):  eta *= 1.10
    return round(eta, 1)

def _traffic_label(current: float, freeflow: float) -> str:
    ratio = current / max(freeflow, 1.0)
    if ratio >= 0.85:   return "🟢 Fluide"
    elif ratio >= 0.60: return "🟡 Modéré"
    else:               return "🔴 Dense"

def _haversine(lat1, lon1, lat2, lon2) -> float:
    R    = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a    = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * \
           math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def _vrp_nearest_neighbor(orders: list) -> list:
    if len(orders) <= 1:
        return orders
    unvisited = orders.copy()
    route     = [unvisited.pop(0)]
    while unvisited:
        last    = route[-1]
        nearest = min(
            unvisited,
            key=lambda o: _haversine(
                last["delivery_lat"],  last["delivery_long"],
                o["delivery_lat"],     o["delivery_long"]
            )
        )
        route.append(nearest)
        unvisited.remove(nearest)
    return route


# ─────────────────────────────────────────────
#  ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status"      : "ok",
        "model_loaded": MODEL_LOADED,
        "weights_path": WEIGHTS_PATH,
        "num_vehicles": NUM_VEHICLES,
    }


@app.post("/assign", response_model=AssignmentResult)
def assign_single(order: OrderIn):
    """Affecte UNE commande à un véhicule via le réseau PPO."""
    if not MODEL_LOADED:
        raise HTTPException(503, "Poids non chargés. Vérifiez delivery_weights.npz")

    order_dict  = _order_to_dict(order)
    env         = DeliveryRoutingEnv(orders_data=[order_dict], num_vehicles=NUM_VEHICLES)
    obs, _      = env.reset(seed=42)

    action      = predict_action(obs)
    vehicle_idx = action % NUM_VEHICLES

    return AssignmentResult(
        shippement_number = order.shippement_number,
        assigned_vehicle  = VEHICLE_NAMES[vehicle_idx],
        vehicle_index     = vehicle_idx,
        eta_minutes       = _compute_eta(order),
        temp_control      = order.temp_control or "Ambient",
        delivery_distance = order.delivery_distance,
        traffic_status    = _traffic_label(order.currentspeed_kmph, order.freeflowspeed),
    )


@app.post("/assign/batch", response_model=BatchResult)
def assign_batch(batch: BatchIn):
    """Affecte un BATCH de commandes + optimise la tournée par véhicule (VRP)."""
    if not MODEL_LOADED:
        raise HTTPException(503, "Poids non chargés. Vérifiez delivery_weights.npz")
    if not batch.orders:
        raise HTTPException(400, "Liste de commandes vide.")

    orders_dicts = [_order_to_dict(o) for o in batch.orders]
    env          = DeliveryRoutingEnv(orders_data=orders_dicts, num_vehicles=NUM_VEHICLES)
    obs, _       = env.reset(seed=42)

    # ── Affectation PPO ──
    assignments_raw = []
    for i, order in enumerate(batch.orders):
        action      = predict_action(obs)
        v_idx       = action % NUM_VEHICLES
        obs, _, terminated, truncated, _ = env.step(action)
        assignments_raw.append((order, v_idx))
        if terminated or truncated:
            for j in range(i + 1, len(batch.orders)):
                loads = [0] * NUM_VEHICLES
                for _, vi in assignments_raw:
                    loads[vi] += 1
                best = loads.index(min(loads))
                assignments_raw.append((batch.orders[j], best))
            break

    # ── Résultats d'assignation ──
    assignment_results = [
        AssignmentResult(
            shippement_number = o.shippement_number,
            assigned_vehicle  = VEHICLE_NAMES[v],
            vehicle_index     = v,
            eta_minutes       = _compute_eta(o),
            temp_control      = o.temp_control or "Ambient",
            delivery_distance = o.delivery_distance,
            traffic_status    = _traffic_label(o.currentspeed_kmph, o.freeflowspeed),
        )
        for o, v in assignments_raw
    ]

    # ── VRP par véhicule ──
    vehicle_orders: dict = {i: [] for i in range(NUM_VEHICLES)}
    for order, v_idx in assignments_raw:
        vehicle_orders[v_idx].append(_order_to_dict(order))

    vehicle_routes = []
    for v_idx in range(NUM_VEHICLES):
        v_orders = vehicle_orders[v_idx]
        if not v_orders:
            continue
        optimized      = _vrp_nearest_neighbor(v_orders)
        cumulative_eta = 0.0
        stops          = []
        for i, o in enumerate(optimized):
            leg = (o["delivery_distance"] / max(o["currentspeed (kmph)"], 5.0)) * 60.0
            cumulative_eta += leg
            stops.append(RouteStop(
                stop_order        = i + 1,
                shippement_number = o["shippement_number"],
                destination       = "",
                eta_minutes       = round(cumulative_eta, 1),
                delivery_lat      = o["delivery_lat"],
                delivery_lon      = o["delivery_long"],
            ))
        total_dist = sum(
            _haversine(
                optimized[i]["delivery_lat"],  optimized[i]["delivery_long"],
                optimized[i+1]["delivery_lat"],optimized[i+1]["delivery_long"]
            ) for i in range(len(optimized) - 1)
        ) if len(optimized) > 1 else optimized[0]["delivery_distance"]

        vehicle_routes.append(VehicleRoute(
            vehicle           = VEHICLE_NAMES[v_idx],
            vehicle_index     = v_idx,
            total_orders      = len(v_orders),
            total_distance_km = round(total_dist, 2),
            route             = stops,
        ))

    return BatchResult(
        assignments = assignment_results,
        routes      = vehicle_routes,
        summary     = {
            "total_orders"   : len(batch.orders),
            "vehicles_used"  : len(vehicle_routes),
            "avg_eta_minutes": round(
                sum(a.eta_minutes for a in assignment_results) / len(assignment_results), 1
            ),
        },
    )
