"""Declared synthetic physics and simulated spraying. None is field calibration."""
import math
from itertools import product
import numpy as np
import torch
from torch import nn

torch.set_num_threads(2)


class PlumePINN(nn.Module):
    """Unit-pulse, free-space, 2-D advection/diffusion/reaction surrogate.

    Inputs x,y,t,east_velocity,north_velocity,diffusivity,removal are normalized
    dimensionless simulation coordinates. Never interpret them as a site map.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor([1, 1, 1, .5, .5, .08, 1.]))
        self.network = nn.Sequential(nn.Linear(7, 48), nn.Tanh(), nn.Linear(48, 48),
                                     nn.Tanh(), nn.Linear(48, 48), nn.Tanh(), nn.Linear(48, 1))

    def forward(self, x):
        return torch.sigmoid(self.network(x / self.scale))


def plume_truth(x):
    xx, yy, t, u, v, diffusion, removal = x.T
    variance = .2 ** 2 + 2 * diffusion * t
    return (.2**2 / variance * np.exp(-removal*t)
            * np.exp(-((xx-u*t)**2+(yy-v*t)**2)/(2*variance)))[:, None]


def plume_points(rng, count):
    return rng.uniform([-1, -1, 0, -.5, -.5, .005, 0], [1, 1, 1, .5, .5, .08, 1],
                       size=(count, 7)).astype(np.float32)


def efficiency_points(rng, count):
    # RH %, temperature C, particle diameter um, droplet diameter um, wind m/s.
    return rng.uniform([20, 5, .2, 20, .5], [95, 45, 15, 200, 8], size=(count, 5))


def efficiency_truth(x, stress_factor=1.):
    humidity, temperature, particle, droplet, wind = x.T
    # A bounded TOY response, not a measured or validated scavenging equation.
    size_capture = 1 - np.exp(-particle / (droplet / 60 + 1))
    diffusion_proxy = .06 / np.sqrt(particle)
    moisture = .75 + .25 * humidity / 100
    evaporation = np.exp(-np.maximum(temperature - 20, 0) / 100)
    residence = np.exp(-wind / 25)
    return np.clip(stress_factor * (size_capture + diffusion_proxy) * moisture * evaporation * residence, 0, 1)


# A finite action space contains OFF plus each binary zone mask at 50% or 100%.
ACTIONS = np.array([[0., 0., 0., 0.]] +
                   [[int(mask & (1 << j)) > 0 for j in range(4)] for mask in range(1, 16)] +
                   [[.5 * (int(mask & (1 << j)) > 0) for j in range(4)] for mask in range(1, 16)], dtype=np.float32)
ZONES = ["A_north", "B_east", "C_south", "D_west"]
FEATURES = ["source_pm10/1500", "observed_source_change/500", "A_pm10/150", "B_pm10/150",
            "C_pm10/150", "D_pm10/150", "wind_to_east_unit", "wind_to_north_unit",
            "wind_speed/8", "RH/100", "temperature/50", "last_A_duty", "last_B_duty", "last_C_duty", "last_D_duty"]


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(len(FEATURES), 64), nn.ReLU(),
                                     nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, len(ACTIONS)))

    def forward(self, x):
        return self.network(x)


class SprayEnvironment:
    """5-second toy transport/response steps. No physical pump is connected."""
    def __init__(self, seed, efficiency_model, stress_factor=1.):
        self.rng = np.random.default_rng(seed)
        self.efficiency_model = efficiency_model
        self.stress_factor = stress_factor
        self.reset()

    def reset(self):
        self.step_number = 0
        self.source = float(self.rng.uniform(100, 500))
        self.previous_source = self.source
        self.theta = float(self.rng.uniform(0, 2*np.pi))  # wind-TO bearing clockwise from north
        self.speed = float(self.rng.uniform(.5, 8))
        self.humidity = float(self.rng.uniform(20, 95))
        self.temperature = float(self.rng.uniform(5, 45))
        self.concentration = self.rng.uniform(0, 40, 4)
        self.last_duty = np.zeros(4)
        self.pulse = 0.
        # Train and deploy the same learned synthetic efficiency model.
        e = [[self.humidity, self.temperature, 6., 80., self.speed]]
        self.efficiency = float(np.clip(self.efficiency_model.predict(e)[0], 0, 1)) * self.stress_factor
        return self.observe()

    def observe(self):
        return np.array([self.source/1500, (self.source-self.previous_source)/500,
                         *self.concentration/150, np.sin(self.theta), np.cos(self.theta),
                         self.speed/8, self.humidity/100, self.temperature/50, *self.last_duty], dtype=np.float32)

    def step(self, action):
        duty = ACTIONS[action]
        alignment = np.maximum(np.cos(self.theta - np.arange(4)*np.pi/2), 0)**2
        gain = .06 + .55 * alignment
        target = self.source * gain
        self.concentration = (.75*self.concentration + .25*target) * np.exp(-.6*duty*self.efficiency)
        risk = float(np.maximum(self.concentration/150 - 1, 0).mean())
        # Both terms are declared design choices, not site costs or legal limits.
        reward = -(10*risk + .35*float(duty.mean()))
        water = float(duty.sum()) * .5 / 60 * 5  # assumed .5 L/min per full-duty zone
        self.last_duty = duty.copy()
        self.previous_source = self.source
        if self.rng.random() < .055:
            self.pulse = float(self.rng.uniform(400, 1500))
        self.pulse *= .94
        self.source = float(np.clip(.95*self.source + .05*(80 + self.pulse) + self.rng.normal(0, 35), 0, 2000))
        self.theta += float(self.rng.normal(0, .035))
        self.step_number += 1
        return self.observe(), reward, self.step_number >= 120, {
            "water_litres": water, "zone_seconds_above_150": int((self.concentration>150).sum())*5,
            "mean_excess_ug_m3": float(np.maximum(self.concentration - 150, 0).mean())}


def reactive_action(environment):
    mask = sum((1 << j) for j, value in enumerate(environment.concentration) if value > 120)
    return mask


def evaluate_policy(network, efficiency, seed, episodes=40, stress_factor=1.):
    rows = []
    for episode in range(episodes):
        for name in ["no_control", "continuous", "reactive", "dqn"]:
            env = SprayEnvironment(seed+episode, efficiency, stress_factor)
            obs = env.observe()
            reward_sum, water, seconds, excess = 0., 0., 0, 0.
            actions = []
            for _ in range(120):
                if name == "no_control":
                    action = 0
                elif name == "continuous":
                    action = 15
                elif name == "reactive":
                    action = reactive_action(env)
                else:
                    with torch.no_grad():
                        action = int(network(torch.from_numpy(obs[None])).argmax(1).item())
                obs, reward, done, info = env.step(action)
                actions.append(action)
                reward_sum += reward
                water += info["water_litres"]
                seconds += info["zone_seconds_above_150"]
                excess += info["mean_excess_ug_m3"]
            rows.append({"episode_seed": seed+episode, "policy": name, "reward": reward_sum,
                         "water_litres": water, "zone_seconds_above_150": seconds,
                         "mean_excess_ug_m3": excess/120, "actions": actions})
    summary = {}
    for name in ["no_control", "continuous", "reactive", "dqn"]:
        matching = [r for r in rows if r["policy"] == name]
        summary[name] = {key: float(np.mean([r[key] for r in matching])) for key in
                         ["reward", "water_litres", "zone_seconds_above_150", "mean_excess_ug_m3"]}
    return summary, rows
