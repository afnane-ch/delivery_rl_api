"""
=============================================================
  DeliveryRoutingEnv V2 — Gymnasium Custom Environment
  
  Nouveautés vs V1 :
    - Flotte de 5 véhicules (2 réfrigérés + 3 standard)
    - État des véhicules dans l'observation
    - Contrainte capacité palettes
    - Contrainte chaîne du froid (Chilled/Frozen)
    - Distance véhicule → pickup dans le reward
    - Compatible Stable Baselines3 + dataset CSV réel
=============================================================
"""

import math
import numpy as np
import pandas as pd
from typing import Optional, List, Dict
import gymnasium as gym
from gymnasium import spaces

# ─────────────────────────────────────────────
#  CONFIGURATION FLOTTE
# ─────────────────────────────────────────────
FLEET_CONFIG = [
    {"id": "V0", "max_capacity": 15, "is_refrigerated": True,
     "depot_lat": 48.7566, "depot_lon": 2.3522},  # Rungis
    {"id": "V1", "max_capacity": 15, "is_refrigerated": True,
     "depot_lat": 48.7551, "depot_lon": 2.4804},  # Limeil-Brévannes
    {"id": "V2", "max_capacity": 20, "is_refrigerated": False,
     "depot_lat": 48.7773, "depot_lon": 2.4567},  # Créteil
    {"id": "V3", "max_capacity": 20, "is_refrigerated": False,
     "depot_lat": 48.8134, "depot_lon": 2.3845},  # Ivry-sur-Seine
    {"id": "V4", "max_capacity": 25, "is_refrigerated": False,
     "depot_lat": 48.7262, "depot_lon": 2.3652},  # Orly
]

NUM_VEHICLES = len(FLEET_CONFIG)  # 5
MAX_ORDERS   = 10                 # commandes max par épisode

# ─────────────────────────────────────────────
#  CONSTANTES DE NORMALISATION
# ─────────────────────────────────────────────
MAX_DISTANCE_KM  = 80.0
MAX_DELIVERY_MIN = 180.0
MAX_SPEED_KMH    = 90.0
MAX_VISIBILITY   = 10000.0
MAX_WIND_KMH     = 80.0
MAX_QUANTITY     = 20.0
MAX_DIST_VEHICLE = 50.0   # distance max véhicule → pickup (km)

# ─────────────────────────────────────────────
#  CONSTANTES DE RÉCOMPENSE
# ─────────────────────────────────────────────
REWARD_ON_TIME          = +100.0
REWARD_LATE             = -20.0
REWARD_ASSIGN_BONUS     = +30.0
REWARD_IDLE             = -25.0
REWARD_CAPACITY_VIOL    = -80.0   # dépassement capacité → très pénalisé
REWARD_TEMP_VIOL        = -60.0   # Chilled/Frozen sur véhicule non réfrigéré
REWARD_PER_KM           = -0.05   # coût distance
REWARD_VEHICLE_CLOSE    = +15.0   # bonus si véhicule proche du pickup
REWARD_COMPLETION       = +200.0  # bonus fin épisode × taux complétion


# ─────────────────────────────────────────────
#  UTILITAIRES
# ─────────────────────────────────────────────
def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance à vol d'oiseau entre deux points GPS (km)."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(max(0, a)))


def _parse_float(val, default=0.0) -> float:
    try:
        return float(str(val).replace(",", ".").strip())
    except (ValueError, AttributeError):
        return default


def _parse_quantity(val) -> float:
    try:
        return float(str(val).split()[0].replace(",", "."))
    except (ValueError, IndexError):
        return 1.0


def _parse_temp(val) -> float:
    """Ambient=0.0, Chilled=0.5, Frozen=1.0"""
    v = str(val).strip().lower()
    if v in ("frozen", "surgelé", "congele"):
        return 1.0
    elif v in ("chilled", "froid", "refrigerated", "réfrigéré"):
        return 0.5
    return 0.0


def _needs_refrigeration(temp_val: float) -> bool:
    return temp_val > 0.0   # Chilled ou Frozen


# ─────────────────────────────────────────────
#  ENVIRONNEMENT PRINCIPAL
# ─────────────────────────────────────────────
class DeliveryRoutingEnvV2(gym.Env):
    """
    Environnement RL V2 pour le routage de livraisons.

    OBSERVATION (Box float32, shape=120) :
      - 10 commandes × 9 features  = 90 dims
      - 5  véhicules × 6 features  = 30 dims

    ACTION (Discrete 6) :
      0..4 → affecter au véhicule V0..V4
      5    → reporter la commande

    REWARD :
      Ponctualité + respect chaîne du froid + capacité + distance
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        orders_data: Optional[List[Dict]] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.render_mode  = render_mode
        self._raw_orders  = orders_data or []

        # ── Spaces ──────────────────────────────────────────────
        # 10 commandes × 9 + 5 véhicules × 6 = 120
        obs_dim = MAX_ORDERS * 9 + NUM_VEHICLES * 6
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        # V0, V1, V2, V3, V4 ou reporter (5)
        self.action_space = spaces.Discrete(NUM_VEHICLES + 1)

        # État interne
        self._orders       : List[Dict] = []
        self._vehicles     : List[Dict] = []
        self._current_idx  : int        = 0
        self._total_reward : float      = 0.0
        self._step_count   : int        = 0

    # ── RESET ────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Charger les commandes
        if self._raw_orders:
            raw = self._raw_orders.copy()
            self.np_random.shuffle(raw)
            batch = raw[:MAX_ORDERS]
            self._orders = self._parse_orders(batch)
        else:
            n = int(self.np_random.integers(3, MAX_ORDERS + 1))
            self._orders = self._generate_synthetic(n)

        # Padding jusqu'à MAX_ORDERS
        while len(self._orders) < MAX_ORDERS:
            self._orders.append(self._empty_order())

        # Initialiser la flotte depuis FLEET_CONFIG
        self._vehicles = [
            {
                "id"            : cfg["id"],
                "max_capacity"  : cfg["max_capacity"],
                "is_refrigerated": float(cfg["is_refrigerated"]),
                "current_load"  : 0.0,
                "current_lat"   : cfg["depot_lat"],
                "current_lon"   : cfg["depot_lon"],
                "is_available"  : 1.0,
            }
            for cfg in FLEET_CONFIG
        ]

        self._current_idx  = 0
        self._total_reward = 0.0
        self._step_count   = 0

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), self._get_info()

    # ── STEP ─────────────────────────────────────────────────────
    def step(self, action: int):
        self._step_count += 1
        reward     = 0.0
        terminated = False
        truncated  = False

        order = self._orders[self._current_idx]

        if action < NUM_VEHICLES and not order["is_empty"]:
            vehicle = self._vehicles[action]

            # ── Vérification contraintes ──────────────────────────
            capacity_ok = (
                vehicle["current_load"] + order["quantity"]
                <= vehicle["max_capacity"]
            )
            temp_ok = (
                not _needs_refrigeration(order["temp_control"])
                or vehicle["is_refrigerated"]
            )

            if capacity_ok and temp_ok and vehicle["is_available"]:
                reward += REWARD_ASSIGN_BONUS
                reward += self._compute_reward(order, vehicle)

                # Mise à jour état véhicule
                vehicle["current_load"] += order["quantity"]
                vehicle["current_lat"]   = order["delivery_lat"]
                vehicle["current_lon"]   = order["delivery_lon"]
                order["assigned"]        = True
                order["vehicle_id"]      = vehicle["id"]

            elif not capacity_ok:
                # Dépassement capacité — grosse pénalité
                reward += REWARD_CAPACITY_VIOL

            elif not temp_ok:
                # Violation chaîne du froid — grosse pénalité
                reward += REWARD_TEMP_VIOL

            else:
                reward += REWARD_IDLE

            self._current_idx += 1

        elif action == NUM_VEHICLES or order["is_empty"]:
            # Reporter ou commande vide
            reward += REWARD_IDLE
            self._current_idx += 1

        # ── Fin d'épisode ────────────────────────────────────────
        real_orders = [o for o in self._orders if not o["is_empty"]]
        assigned    = [o for o in real_orders if o.get("assigned")]

        if self._current_idx >= MAX_ORDERS:
            terminated = True
            rate = len(assigned) / max(len(real_orders), 1)
            reward += REWARD_COMPLETION * rate

        if self._step_count >= MAX_ORDERS * 3:
            truncated = True

        self._total_reward += reward

        info = self._get_info()
        info["assigned_count"] = len(assigned)
        info["total_orders"]   = len(real_orders)

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, info

    # ── RENDER ───────────────────────────────────────────────────
    def render(self):
        if self.render_mode != "human":
            return

        real = [o for o in self._orders if not o["is_empty"]]
        done = [o for o in real if o.get("assigned")]
        print(f"\n{'═'*60}")
        print(f"  Step {self._step_count:>2} | "
              f"{len(done)}/{len(real)} assignées | "
              f"Reward: {self._total_reward:+.1f}")

        # Commande actuelle
        if self._current_idx < len(self._orders):
            o = self._orders[self._current_idx]
            if not o["is_empty"]:
                tc_map = {0.0: "📦 Ambient", 0.5: "🌡 Chilled", 1.0: "❄ Frozen"}
                tc = tc_map.get(o["temp_control"], "?")
                print(f"  Commande: {o['shipment_number']} | "
                      f"{o['delivery_distance']:.1f}km | "
                      f"{tc} | {o['quantity']:.0f} palettes")

        # État flotte
        print("  ── Flotte ──")
        for v in self._vehicles:
            refr = "🌡" if v["is_refrigerated"] else "📦"
            load_pct = v["current_load"] / v["max_capacity"] * 100
            bar = "█" * int(load_pct // 10) + "░" * (10 - int(load_pct // 10))
            print(f"  {v['id']} {refr} [{bar}] "
                  f"{v['current_load']:.0f}/{v['max_capacity']} palettes")
        print("═" * 60)

    def close(self):
        pass

    # ─────────────────────────────────────────────────────────────
    #  INTERNES
    # ─────────────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        """
        Vecteur d'observation normalisé [0, 1].
        
        Structure :
          [commandes: MAX_ORDERS × 9]
          [véhicules: NUM_VEHICLES × 6]
        """
        order_feats = []
        for o in self._orders[:MAX_ORDERS]:
            order_feats.extend([
                min(o["delivery_distance"] / MAX_DISTANCE_KM, 1.0),
                min(o["delivery_time"]     / MAX_DELIVERY_MIN, 1.0),
                min(o["currentspeed"]      / MAX_SPEED_KMH, 1.0),
                min(o["freeflowspeed"]     / MAX_SPEED_KMH, 1.0),
                min(o["visibility"]        / MAX_VISIBILITY, 1.0),
                min(o["wind_speed"]        / MAX_WIND_KMH, 1.0),
                min(o["quantity"]          / MAX_QUANTITY, 1.0),
                o["temp_control"],
                float(o.get("assigned", False)),
            ])

        vehicle_feats = []
        for v in self._vehicles:
            # Distance du véhicule au pickup de la commande actuelle
            if self._current_idx < len(self._orders):
                curr_order = self._orders[self._current_idx]
                dist_to_pickup = _haversine(
                    v["current_lat"], v["current_lon"],
                    curr_order["pickup_lat"], curr_order["pickup_lon"]
                ) if not curr_order["is_empty"] else 0.0
            else:
                dist_to_pickup = 0.0

            vehicle_feats.extend([
                v["current_load"] / v["max_capacity"],   # charge normalisée
                v["is_refrigerated"],                     # 0 ou 1
                v["is_available"],                        # 0 ou 1
                (v["current_lat"] - 48.6) / 0.6,        # lat normalisée région Paris
                (v["current_lon"] - 2.0)  / 0.8,        # lon normalisée région Paris
                min(dist_to_pickup / MAX_DIST_VEHICLE, 1.0),  # distance au pickup
            ])

        obs = np.array(order_feats + vehicle_feats, dtype=np.float32)
        return np.clip(obs, 0.0, 1.0)

    def _get_info(self) -> dict:
        return {
            "step"         : self._step_count,
            "current_order": self._current_idx,
            "total_reward" : self._total_reward,
            "vehicle_loads": {
                v["id"]: f"{v['current_load']:.0f}/{v['max_capacity']}"
                for v in self._vehicles
            },
        }

    def _compute_reward(self, order: dict, vehicle: dict) -> float:
        """Reward basé sur ponctualité, distance et proximité véhicule."""
        reward = 0.0

        # Ponctualité : rapport vitesse actuelle / freeflow
        traffic_ratio  = order["currentspeed"] / max(order["freeflowspeed"], 1.0)
        estimated_time = order["delivery_time"] / max(traffic_ratio, 0.1)

        if estimated_time <= order["delivery_time"]:
            reward += REWARD_ON_TIME
        else:
            late_factor = (estimated_time - order["delivery_time"]) / max(order["delivery_time"], 1.0)
            reward += REWARD_LATE * min(late_factor, 3.0)

        # Coût distance
        reward += REWARD_PER_KM * order["delivery_distance"]

        # Bonus si le véhicule est proche du pickup (< 5 km)
        dist_to_pickup = _haversine(
            vehicle["current_lat"], vehicle["current_lon"],
            order["pickup_lat"],    order["pickup_lon"]
        )
        if dist_to_pickup < 5.0:
            reward += REWARD_VEHICLE_CLOSE

        # Pénalité météo défavorable
        if order["visibility"] < 1000:
            reward -= 10.0

        return reward

    def _parse_orders(self, raw: list) -> list:
        """Convertit les lignes CSV/dict en format interne."""
        orders = []
        for r in raw:
            sn = str(r.get("shippement_number", "")).strip()
            if sn in ("", "nan", "NaN"):
                continue
            orders.append({
                "shipment_number"  : sn,
                "delivery_distance": _parse_float(r.get("delivery_distance", 0)),
                "delivery_time"    : _parse_float(r.get("delivery_time", 60)),
                "visibility"       : _parse_float(r.get("visibility", 10000)),
                "wind_speed"       : _parse_float(r.get("wind speed", r.get("wind_speed", 0))),
                "currentspeed"     : _parse_float(r.get("currentspeed (kmph)", r.get("currentspeed", 50))),
                "freeflowspeed"    : _parse_float(r.get("freeflowspeed", 80)),
                "quantity"         : _parse_quantity(r.get("quantity", 1)),
                "temp_control"     : _parse_temp(r.get("temp_control", "Ambient")),
                "delivery_lat"     : _parse_float(r.get("delivery_lat", 48.8566)),
                "delivery_lon"     : _parse_float(r.get("delivery_long", r.get("delivery_lon", 2.3522))),
                "pickup_lat"       : _parse_float(r.get("pickup_lat", 48.8566)),
                "pickup_lon"       : _parse_float(r.get("pickup_long", r.get("pickup_lon", 2.3522))),
                "is_empty"         : False,
                "assigned"         : False,
                "vehicle_id"       : None,
            })
        return orders

    def _generate_synthetic(self, n: int) -> list:
        """Commandes synthétiques pour test rapide sans CSV."""
        orders = []
        for _ in range(n):
            orders.append({
                "shipment_number"  : f"SH-{self.np_random.integers(1000, 9999)}",
                "delivery_distance": float(self.np_random.uniform(1, 60)),
                "delivery_time"    : float(self.np_random.uniform(10, 90)),
                "visibility"       : float(self.np_random.uniform(500, 10000)),
                "wind_speed"       : float(self.np_random.uniform(0, 50)),
                "currentspeed"     : float(self.np_random.uniform(10, 85)),
                "freeflowspeed"    : float(self.np_random.uniform(50, 90)),
                "quantity"         : float(self.np_random.integers(2, 21)),
                "temp_control"     : float(self.np_random.choice([0.0, 0.5, 1.0])),
                "delivery_lat"     : float(self.np_random.uniform(48.70, 48.90)),
                "delivery_lon"     : float(self.np_random.uniform(2.20, 2.60)),
                "pickup_lat"       : float(self.np_random.uniform(48.70, 48.90)),
                "pickup_lon"       : float(self.np_random.uniform(2.20, 2.60)),
                "is_empty"         : False,
                "assigned"         : False,
                "vehicle_id"       : None,
            })
        return orders

    def _empty_order(self) -> dict:
        return {
            "shipment_number": "", "delivery_distance": 0.0,
            "delivery_time": 0.0, "visibility": MAX_VISIBILITY,
            "wind_speed": 0.0, "currentspeed": 0.0, "freeflowspeed": 0.0,
            "quantity": 0.0, "temp_control": 0.0,
            "delivery_lat": 0.0, "delivery_lon": 0.0,
            "pickup_lat": 0.0, "pickup_lon": 0.0,
            "is_empty": True, "assigned": False, "vehicle_id": None,
        }
