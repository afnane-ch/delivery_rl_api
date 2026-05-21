"""
=============================================================
  DeliveryRoutingEnv — Gymnasium Custom Environment v2
  Adapté aux données réelles Google Sheets "email content"
  Compatible Stable Baselines3
=============================================================
"""

from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ─────────────────────────────────────────────
#  CONSTANTES
# ─────────────────────────────────────────────
MAX_ORDERS       = 10
NUM_VEHICLES     = 3
MAX_DISTANCE_KM  = 100.0
MAX_DELIVERY_MIN = 120.0
MAX_SPEED_KMH    = 130.0
MAX_VISIBILITY_M = 10000.0
MAX_WIND_KMH     = 80.0
MAX_QUANTITY     = 50.0
MAX_VEHICLE_LOAD = 50.0

REWARD_ON_TIME       = +100.0
REWARD_ASSIGN_BONUS  = +30.0
REWARD_COMPLETION    = +200.0
REWARD_LATE          = -20.0
REWARD_TEMP_VIOLATED = -15.0
REWARD_PER_KM        = -0.05
REWARD_IDLE          = -25.0


def _parse_float(val) -> float:
    """Gère les décimales françaises (virgule) et les valeurs manquantes."""
    try:
        return float(str(val).replace(",", ".").strip())
    except (ValueError, AttributeError):
        return 0.0


def _parse_quantity(val) -> float:
    """Ex: '12 pallets' → 12.0"""
    try:
        return float(str(val).split()[0].replace(",", "."))
    except (ValueError, IndexError):
        return 1.0


def _parse_temp_control(val) -> bool:
    """'Chilled' / 'Frozen' → True  |  'Ambient' / '' → False"""
    return str(val).strip().lower() in ("chilled", "frozen", "refrigerated", "froid")


# ─────────────────────────────────────────────
#  ENVIRONNEMENT
# ─────────────────────────────────────────────
class DeliveryRoutingEnv(gym.Env):
    """
    Environnement RL pour le routage de livraisons.
    Observation : vecteur float32 normalisé [0,1]
    Action      : Discrete → affecter commande au véhicule X ou reporter
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(self, orders_data=None, num_vehicles=NUM_VEHICLES, render_mode=None):
        super().__init__()
        self.num_vehicles = num_vehicles
        self.render_mode  = render_mode
        self._raw_orders  = orders_data or []

        obs_size = MAX_ORDERS * 9 + self.num_vehicles * 4
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        # 0..N-1 = affecter au véhicule X  |  N = reporter
        self.action_space = spaces.Discrete(self.num_vehicles + 1)

        self._orders       = []
        self._vehicles     = []
        self._current_idx  = 0
        self._total_reward = 0.0
        self._step_count   = 0

    # ── RESET ────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self._raw_orders:
            self._orders = self._parse_orders(self._raw_orders)
        else:
            self._orders = self._generate_synthetic_orders(
                n=int(self.np_random.integers(3, MAX_ORDERS + 1))
            )

        while len(self._orders) < MAX_ORDERS:
            self._orders.append(self._empty_order())

        self._vehicles = [
            {
                "lat": float(self.np_random.uniform(48.7, 49.0)),
                "lon": float(self.np_random.uniform(2.1, 2.6)),
                "current_load": 0.0,
                "is_available": 1.0,
            }
            for _ in range(self.num_vehicles)
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

        if action < self.num_vehicles:
            vehicle = self._vehicles[action]
            if vehicle["is_available"] and not order["is_empty"]:
                reward += REWARD_ASSIGN_BONUS          # bonus fixe pour avoir assigné
                reward += self._compute_reward(order, vehicle)
                vehicle["current_load"] += order["quantity"] / MAX_VEHICLE_LOAD
                vehicle["current_load"]  = min(vehicle["current_load"], 1.0)
                vehicle["lat"]           = order["delivery_lat"]
                vehicle["lon"]           = order["delivery_lon"]
                order["assigned"]        = True
            else:
                reward += REWARD_IDLE
            self._current_idx += 1
        else:
            reward += REWARD_IDLE                      # très coûteux de reporter
            self._current_idx += 1

        real_orders = [o for o in self._orders if not o["is_empty"]]
        assigned    = [o for o in real_orders if o.get("assigned")]

        if self._current_idx >= MAX_ORDERS:
            terminated = True
            # Bonus de fin d'épisode : récompense proportionnelle aux commandes assignées
            if len(real_orders) > 0:
                completion_rate = len(assigned) / len(real_orders)
                reward += REWARD_COMPLETION * completion_rate
        if self._step_count >= MAX_ORDERS * 2:
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
        if self.render_mode not in ("human", "ansi"):
            return
        real = [o for o in self._orders if not o["is_empty"]]
        done = [o for o in real if o.get("assigned")]
        print(f"\n{'═'*50}")
        print(f"  Step {self._step_count} | {len(done)}/{len(real)} assignées | "
              f"Reward: {self._total_reward:+.1f}")
        idx   = min(self._current_idx, len(self._orders) - 1)
        order = self._orders[idx]
        if not order["is_empty"]:
            tc = "🌡 Chilled" if order["temp_control"] else "📦 Ambient"
            print(f"  Commande: {order['shipment_number']} | "
                  f"{order['delivery_distance']:.1f}km | {tc} | "
                  f"qté={order['quantity']:.0f}")
        for i, v in enumerate(self._vehicles):
            print(f"  Véhicule {i}: charge={v['current_load']*100:.0f}% | "
                  f"dispo={'✓' if v['is_available'] else '✗'}")
        print("═"*50)

    def close(self):
        pass

    # ── INTERNES ─────────────────────────────────────────────────
    def _get_obs(self):
        order_feats = []
        for o in self._orders[:MAX_ORDERS]:
            order_feats.extend([
                min(o["delivery_distance"] / MAX_DISTANCE_KM, 1.0),
                min(o["delivery_time"]     / MAX_DELIVERY_MIN, 1.0),
                min(o["currentspeed"]      / MAX_SPEED_KMH, 1.0),
                min(o["freeflowspeed"]     / MAX_SPEED_KMH, 1.0),
                min(o["visibility"]        / MAX_VISIBILITY_M, 1.0),
                min(o["wind_speed"]        / MAX_WIND_KMH, 1.0),
                min(o["quantity"]          / MAX_QUANTITY, 1.0),
                float(o["temp_control"]),
                float(o.get("assigned", False)),
            ])
        vehicle_feats = []
        for v in self._vehicles:
            vehicle_feats.extend([
                (v["lat"] + 90)  / 180.0,
                (v["lon"] + 180) / 360.0,
                v["current_load"],
                v["is_available"],
            ])
        return np.clip(
            np.array(order_feats + vehicle_feats, dtype=np.float32), 0.0, 1.0
        )

    def _get_info(self):
        return {
            "step": self._step_count,
            "current_order": self._current_idx,
            "total_reward": self._total_reward,
        }

    def _compute_reward(self, order, vehicle):
        reward = 0.0
        traffic_ratio  = order["currentspeed"] / max(order["freeflowspeed"], 1.0)
        estimated_time = order["delivery_time"] / max(traffic_ratio, 0.1)
        if estimated_time <= order["delivery_time"]:
            reward += REWARD_ON_TIME
        else:
            late = (estimated_time - order["delivery_time"]) / max(order["delivery_time"], 1.0)
            reward += REWARD_LATE * late
        if order["temp_control"] and vehicle["current_load"] > 0.8:
            reward += REWARD_TEMP_VIOLATED
        reward += REWARD_PER_KM * order["delivery_distance"]
        if order["visibility"] < 1000:
            reward -= 10.0
        return reward

    def _parse_orders(self, raw: list) -> list:
        """
        Convertit les lignes Google Sheets en format interne.
        Gère : virgules décimales, 'Chilled', '12 pallets'.
        """
        orders = []
        # Filtre les lignes vides/NaN (Google Sheets envoie souvent des lignes vides)
        raw_clean = [
            r for r in raw
            if str(r.get("shippement_number", "")).strip() not in ("", "nan", "NaN", "None")
        ]
        for r in raw_clean[:MAX_ORDERS]:
            orders.append({
                "shipment_number"  : str(r.get("shippement_number", "")),
                "delivery_distance": _parse_float(r.get("delivery_distance ", r.get("delivery_distance", 0))),
                "delivery_time"    : _parse_float(r.get("delivery_time", 60)),
                "visibility"       : _parse_float(r.get("visibility", 10000)),
                "wind_speed"       : _parse_float(r.get("wind speed ", r.get("wind speed", 0))),
                "currentspeed"     : _parse_float(r.get("currentspeed (kmph)", 50)),
                "freeflowspeed"    : _parse_float(r.get("freeflowspeed ", r.get("freeflowspeed", 80))),
                "quantity"         : _parse_quantity(r.get("quantity", 1)),
                "temp_control"     : _parse_temp_control(r.get("temp_control", "")),
                "delivery_lat"     : _parse_float(r.get("delivery_lat", 48.8566)),
                "delivery_lon"     : _parse_float(r.get("delivery_long", 2.3522)),
                "is_empty"         : False,
                "assigned"         : False,
            })
        return orders

    def _generate_synthetic_orders(self, n: int) -> list:
        orders = []
        weather = ["Clear", "Clouds", "Rain"]
        for _ in range(n):
            orders.append({
                "shipment_number"  : f"SH-{self.np_random.integers(1000,9999)}",
                "delivery_distance": float(self.np_random.uniform(1, 80)),
                "delivery_time"    : float(self.np_random.uniform(10, 90)),
                "visibility"       : float(self.np_random.uniform(500, 10000)),
                "wind_speed"       : float(self.np_random.uniform(0, 50)),
                "currentspeed"     : float(self.np_random.uniform(10, 120)),
                "freeflowspeed"    : float(self.np_random.uniform(50, 130)),
                "quantity"         : float(self.np_random.uniform(1, 30)),
                "temp_control"     : bool(self.np_random.integers(0, 2)),
                "delivery_lat"     : float(self.np_random.uniform(48.7, 49.0)),
                "delivery_lon"     : float(self.np_random.uniform(2.1, 2.6)),
                "is_empty"         : False,
                "assigned"         : False,
            })
        return orders

    def _empty_order(self):
        return {
            "shipment_number": "", "delivery_distance": 0.0,
            "delivery_time": 0.0, "visibility": MAX_VISIBILITY_M,
            "wind_speed": 0.0, "currentspeed": 0.0, "freeflowspeed": 0.0,
            "quantity": 0.0, "temp_control": False,
            "delivery_lat": 0.0, "delivery_lon": 0.0,
            "is_empty": True, "assigned": False,
        }
