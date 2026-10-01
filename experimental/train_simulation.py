"""Build seeded synthetic datasets; train a PINN, efficiency model and DQN."""
import copy
import gzip
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import torch
from torch import nn
from sklearn.ensemble import HistGradientBoostingRegressor

from common import ARTIFACTS, CONFIG, REPO, REPORTS, digest, error_metrics, write_json
from simulation_models import (ACTIONS, FEATURES, PlumePINN, QNetwork, SprayEnvironment,
    efficiency_points, efficiency_truth, evaluate_policy, plume_points, plume_truth)


def train_pinn():
    sc = CONFIG["simulation"]
    torch.manual_seed(sc["train_seed"])
    rng = np.random.default_rng(sc["train_seed"])
    data = plume_points(rng, 30000)
    label = plume_truth(data).astype(np.float32)
    val_x = plume_points(np.random.default_rng(sc["validation_seed"]), 5000)
    val_y = plume_truth(val_x)
    model = PlumePINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    best, chosen_step, saved = np.inf, 0, None
    for step in range(sc["pinn_steps"]):
        indices = rng.integers(len(data), size=256)
        x = torch.from_numpy(data[indices]).requires_grad_(True)
        y = torch.from_numpy(label[indices])
        prediction = model(x)
        gradient = torch.autograd.grad(prediction.sum(), x, create_graph=True)[0]
        xx = torch.autograd.grad(gradient[:, 0].sum(), x, create_graph=True)[0][:, 0]
        yy = torch.autograd.grad(gradient[:, 1].sum(), x, create_graph=True)[0][:, 1]
        residual = gradient[:, 2] + x[:, 3]*gradient[:, 0] + x[:, 4]*gradient[:, 1] - x[:, 5]*(xx+yy) + x[:, 6]*prediction[:, 0]
        supervised = ((prediction-y)**2 * (1 + 10*y)).mean()
        loss = supervised + .02*(residual**2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step+1) % 200 == 0:
            with torch.no_grad():
                error = float(((model(torch.from_numpy(val_x)).numpy()-val_y)**2).mean())
            if error < best:
                best, chosen_step, saved = error, step+1, copy.deepcopy(model.state_dict())
            print("PINN step", step+1, "validation RMSE", round(np.sqrt(error), 5), flush=True)
    model.load_state_dict(saved)
    torch.save(model.state_dict(), ARTIFACTS / "plume-pinn.pt")
    frozen_hash = digest(ARTIFACTS / "plume-pinn.pt")
    # Independent seeds test only this idealized equation, not a physical site.
    test_x = plume_points(np.random.default_rng(sc["test_seed"]), 5000)
    test_y = plume_truth(test_x)
    x = torch.from_numpy(test_x).requires_grad_(True)
    pred = model(x)
    g = torch.autograd.grad(pred.sum(), x, create_graph=True)[0]
    xx = torch.autograd.grad(g[:, 0].sum(), x, create_graph=True)[0][:, 0]
    yy = torch.autograd.grad(g[:, 1].sum(), x, create_graph=True)[0][:, 1]
    residual = g[:, 2] + x[:, 3]*g[:, 0] + x[:, 4]*g[:, 1] - x[:, 5]*(xx+yy) + x[:, 6]*pred[:, 0]
    report = {"dataset_kind": "synthetic_known_equation", "training_rows": len(data), "validation_rows": len(val_x),
        "test_rows": len(test_x), "checkpoint_selected_by_validation_step": chosen_step,
        "test": error_metrics(test_y, pred.detach().numpy()),
        "test_pde_residual_rmse": float(torch.sqrt((residual**2).mean()).detach()),
        "artifact_sha256_frozen_before_test": frozen_hash,
        "equation": "dC/dt + u*dC/dx + v*dC/dy - D*(d2C/dx2+d2C/dy2) + k*C = 0",
        "initial_condition": "Gaussian pulse sigma=0.2 and peak=1",
        "domain": {"x_y": [-1, 1], "t": [0, 1], "u_v": [-.5, .5], "D": [.005, .08], "k": [0, 1]},
        "physics_loss_weight": .02, "limits": ["Dimensionless free-space pulse only", "Uniform diffusivity/removal",
             "No site geometry, turbulence, obstacles, calibrated emissions or absolute concentration", "Approximate footprint, not exact"]}
    write_json(REPORTS / "plume-evaluation.json", report)
    return model, {"plume_training_sha256": __import__('hashlib').sha256(data.tobytes()+label.tobytes()).hexdigest()}


def train_efficiency():
    sc = CONFIG["simulation"]
    x = efficiency_points(np.random.default_rng(sc["train_seed"]), 30000)
    y = efficiency_truth(x)
    model = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15, early_stopping=False, random_state=CONFIG["seed"])
    model.fit(x, y)
    joblib.dump(model, ARTIFACTS / "spray-efficiency.joblib", compress=3)
    frozen_hash = digest(ARTIFACTS / "spray-efficiency.joblib")
    test_x = efficiency_points(np.random.default_rng(sc["test_seed"]), 5000)
    pred = np.clip(model.predict(test_x), 0, 1)
    report = {"dataset_kind": "synthetic_assumed_scavenging_response", "train_rows": len(x), "test_rows": len(test_x),
        "test": error_metrics(efficiency_truth(test_x), pred),
        "changed_efficiency_coefficient_stress_test": error_metrics(efficiency_truth(test_x, .7), pred),
        "artifact_sha256_frozen_before_test": frozen_hash, "features": ["RH_percent", "temperature_C", "particle_um", "droplet_um", "wind_m_s"],
        "limits": ["Toy capture/evaporation/residence function", "No measured droplet capture or field spraying treatments",
                   "Particle/droplet diameters are supplied simulator assumptions, not inferred from PM10 alone"]}
    write_json(REPORTS / "efficiency-evaluation.json", report)
    return model, {"efficiency_training_sha256": __import__('hashlib').sha256(x.tobytes()+y.tobytes()).hexdigest()}


def train_dqn(efficiency):
    sc = CONFIG["simulation"]
    rng = np.random.default_rng(sc["train_seed"])
    torch.manual_seed(sc["train_seed"])
    network = QNetwork()
    target = copy.deepcopy(network)
    optimizer = torch.optim.Adam(network.parameters(), lr=.001)
    capacity = sc["dqn_steps"]
    state = np.empty((capacity, len(FEATURES)), np.float32)
    next_state = np.empty_like(state)
    action_buffer = np.empty(capacity, np.int64)
    reward_buffer = np.empty(capacity, np.float32)
    done_buffer = np.empty(capacity, np.float32)
    episode = 0
    env = SprayEnvironment(sc["train_seed"], efficiency)
    observation = env.observe()
    best, best_step, saved, history = -np.inf, 0, None, []
    for step in range(capacity):
        epsilon = max(.05, 1-step/(capacity*.8))
        if rng.random() < epsilon:
            action = int(rng.integers(len(ACTIONS)))
        else:
            with torch.no_grad():
                action = int(network(torch.from_numpy(observation[None])).argmax(1).item())
        new_observation, reward, done, info = env.step(action)
        state[step], next_state[step] = observation, new_observation
        action_buffer[step], reward_buffer[step], done_buffer[step] = action, reward, float(done)
        observation = new_observation
        if done:
            episode += 1
            env = SprayEnvironment(sc["train_seed"]+episode, efficiency)
            observation = env.observe()
        if step >= 1000 and step % 4 == 0:
            indices = rng.integers(step+1, size=128)
            s, ns = torch.from_numpy(state[indices]), torch.from_numpy(next_state[indices])
            actions = torch.from_numpy(action_buffer[indices])
            rewards, dones = torch.from_numpy(reward_buffer[indices]), torch.from_numpy(done_buffer[indices])
            q = network(s).gather(1, actions[:, None])[:, 0]
            with torch.no_grad():
                # Double-DQN action selection; a separate target network evaluates it.
                selected = network(ns).argmax(1)
                value = target(ns).gather(1, selected[:, None])[:, 0]
                expected = rewards + .98*(1-dones)*value
            loss = nn.functional.smooth_l1_loss(q, expected)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 5)
            optimizer.step()
        if (step+1) % 500 == 0:
            target.load_state_dict(network.state_dict())
        if (step+1) % sc["dqn_validation_every"] == 0:
            summary, _ = evaluate_policy(network, efficiency, sc["validation_seed"]+10000, episodes=12)
            value = summary["dqn"]["reward"]
            history.append({"training_step": step+1, "validation": summary})
            if value > best:
                best, best_step, saved = value, step+1, copy.deepcopy(network.state_dict())
            print("DQN step", step+1, "validation reward", round(value, 2), flush=True)
    network.load_state_dict(saved)
    torch.save(network.state_dict(), ARTIFACTS / "spraying-dqn.pt")
    frozen_hash = digest(ARTIFACTS / "spraying-dqn.pt")
    summary, rows = evaluate_policy(network, efficiency, sc["test_seed"]+20000, sc["test_episodes"])
    stress, _ = evaluate_policy(network, efficiency, sc["test_seed"]+30000, sc["test_episodes"], .7)
    write_json(REPORTS / "control-evaluation.json", {"dataset_kind": "synthetic_transport_and_learned_toy_efficiency",
        "training_transitions": capacity, "checkpoint_selected_by_validation_step": best_step,
        "artifact_sha256_frozen_before_test": frozen_hash, "validation": history, "test_episodes": sc["test_episodes"],
        "paired_test": summary, "30pct_lower_capture_stress": stress,
        "features": FEATURES, "action_duties": ACTIONS.tolist(), "dt_seconds": 5, "episode_seconds": 600,
        "reward": "-(10*mean(max(PM10/150-1,0)) + 0.35*mean(duty))",
        "full_zone_flow_L_min": .5, "zone_convention": "A north, B east, C south, D west; meteorological wind FROM",
        "limits": ["One training seed family and one selected checkpoint; not robust RL benchmarking",
                   "Threshold 150 is illustrative simulation setting", "All flow/suppression/transport constants assumed",
                   "Water and concentration outcomes are simulation only", "No commands sent to physical hardware"]})
    with gzip.open(REPORTS / "control-test-episodes.json.gz", "wt") as stream:
        json.dump(rows, stream, allow_nan=False)
    # Rebuildable small datasets remain ignored; tracked reports identify their bytes.
    generated = REPO / "data/processed/experimental"
    generated.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(generated / "control-training.npz", state=state, next_state=next_state,
                        action=action_buffer, reward=reward_buffer, done=done_buffer)
    return network, {"control_training_sha256": digest(generated / "control-training.npz")}


def export_and_verify(plume, policy):
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic
    report = {}
    rng = np.random.default_rng(8181)
    for name, model, inputs in [
        ("plume", plume, plume_points(rng, 1000)),
        ("spraying", policy, rng.uniform(0, 1, (1000, len(FEATURES))).astype(np.float32))]:
        path = ARTIFACTS / (name + ".onnx")
        model.eval()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            torch.onnx.export(model, torch.from_numpy(inputs[:1]), str(path), input_names=["features"],
                              output_names=["prediction"], dynamic_axes={"features": {0: "batch"}, "prediction": {0: "batch"}},
                              opset_version=17, dynamo=False)
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            native = model(torch.from_numpy(inputs)).numpy()
        exported = session.run(None, {"features": inputs})[0]
        parity = float(np.max(np.abs(native-exported)))
        if parity > 1e-4:
            raise ValueError("ONNX export parity failed")
        report[name] = {"onnx_bytes": path.stat().st_size, "float32_max_absolute_difference": parity,
                        "verified_runtime": "macOS arm64 CPUExecutionProvider; not a microcontroller"}
        if name == "spraying":
            quantized = ARTIFACTS / "spraying-int8.onnx"
            quantize_dynamic(str(path), str(quantized), weight_type=QuantType.QInt8)
            q_session = ort.InferenceSession(str(quantized), providers=["CPUExecutionProvider"])
            q_value = q_session.run(None, {"features": inputs})[0]
            report[name]["int8_bytes"] = quantized.stat().st_size
            report[name]["int8_action_agreement_on_random_states"] = float((q_value.argmax(1)==native.argmax(1)).mean())
            report[name]["int8_max_q_value_difference"] = float(np.abs(q_value-native).max())
    write_json(REPORTS / "onnx-verification.json", report)


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    if (ARTIFACTS / "spraying-dqn.pt").exists() or (ARTIFACTS / "plume-pinn.pt").exists():
        raise SystemExit("Fitted models exist; do not tune against a revealed synthetic test. Start a newly declared experiment.")
    start = time.monotonic()
    efficiency, e_meta = train_efficiency()
    plume, p_meta = train_pinn()
    policy, c_meta = train_dqn(efficiency)
    export_and_verify(plume, policy)
    write_json(REPORTS / "synthetic-dataset.json", {"kind": "generated_simulation_only", "config": CONFIG["simulation"],
        **e_meta, **p_meta, **c_meta, "training_seconds": time.monotonic()-start,
        "generator": "simulation_models.py", "generator_sha256": digest(Path(__file__).with_name("simulation_models.py")),
        "reuse": "Generated data derived solely from the project's declared toy equations; no measured treatment labels",
        "validation_test_note": "Separate seeds within an assumed distribution plus changed-efficiency stress test; no field validity"})
    print("Simulation model training complete", flush=True)


if __name__ == "__main__":
    main()
