"""Optional experimental service. Frozen /v1 runtime is not modified."""
from contextlib import asynccontextmanager
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from common import ROOT
from runtime import Prototype


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Pollution(Strict):
    pm25: float = Field(ge=0, le=10000)
    pm10: float = Field(ge=0, le=10000)
    temperature: float = Field(ge=-40, le=80)
    humidity: float = Field(ge=0, le=100)


class ShortSnapshot(Pollution):
    second: int = Field(ge=0, strict=True)


class ShortRequest(Strict):
    history: list[ShortSnapshot] = Field(min_length=121, max_length=121)
    history_kind: Literal["measured_validation_recording", "caller_supplied_observations"] = "caller_supplied_observations"

    @model_validator(mode="after")
    def causal_clock(self):
        times = [h.second for h in self.history]
        if any(b-a != 1 for a, b in zip(times, times[1:])):
            raise ValueError("Exactly 121 ascending one-second snapshots required; no resampling in the service")
        return self


class HourlySnapshot(Pollution):
    timestamp: datetime
    wind_east: float = Field(ge=-40, le=40)
    wind_north: float = Field(ge=-40, le=40)
    pressure: float = Field(ge=800, le=1100)
    rain: float = Field(ge=0, le=500)


class HourlyRequest(Strict):
    station: str
    history: list[HourlySnapshot] = Field(min_length=25, max_length=25)
    history_kind: Literal["measured_validation_recording_with_derived_RH", "caller_supplied_observations"] = "caller_supplied_observations"

    @model_validator(mode="after")
    def hourly_clock(self):
        times = [h.timestamp for h in self.history]
        if any(t.tzinfo is not None for t in times):
            raise ValueError("Use original station local calendar times without timezone conversion")
        if any((b-a).total_seconds() != 3600 for a,b in zip(times,times[1:])):
            raise ValueError("25 consecutive hourly observations required")
        return self


class PlumePoint(Strict):
    x: float = Field(ge=-1, le=1)
    y: float = Field(ge=-1, le=1)
    t: float = Field(ge=0, le=1)
    east_velocity: float = Field(ge=-.5, le=.5)
    north_velocity: float = Field(ge=-.5, le=.5)
    diffusivity: float = Field(ge=.005, le=.08)
    removal: float = Field(ge=0, le=1)


class PlumeRequest(Strict):
    simulation_only: Literal[True]
    points: list[PlumePoint] = Field(min_length=1, max_length=1600)


class EfficiencyRequest(Strict):
    simulation_only: Literal[True]
    humidity: float = Field(ge=20, le=95)
    temperature: float = Field(ge=5, le=45)
    particle_um: float = Field(ge=.2, le=15)
    droplet_um: float = Field(ge=20, le=200)
    wind_speed_m_s: float = Field(ge=.5, le=8)


class ControlRequest(Strict):
    simulation_only: Literal[True]
    policy: Literal["reactive", "dqn"] = "reactive"
    source_pm10: float = Field(ge=0, le=2000)
    previous_source_pm10: float = Field(ge=0, le=2000)
    zone_pm10: list[Annotated[float, Field(ge=0, le=2000)]] = Field(min_length=4, max_length=4)
    wind_from_degrees: float = Field(ge=0, lt=360)
    wind_speed_m_s: float = Field(ge=.5, le=8)
    humidity: float = Field(ge=20, le=95)
    temperature: float = Field(ge=5, le=45)
    last_duties: list[Annotated[float, Field(ge=0, le=1)]] = Field(min_length=4, max_length=4)
    rollouts: int = Field(default=64, ge=8, le=128, strict=True)


def create_app(root=ROOT):
    @asynccontextmanager
    async def lifespan(app):
        app.state.prototype = Prototype(root)
        yield
    app = FastAPI(title="DustTwin experimental AI", version="0.1.0", lifespan=lifespan)
    origins = [o.strip() for o in os.environ.get("DUSTTWIN_PROTOTYPE_ORIGINS", "").split(",") if o.strip()]
    if "*" in origins:
        raise ValueError("Explicit prototype frontend origins required")
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    @app.exception_handler(RequestValidationError)
    async def invalid(request, error):
        return JSONResponse(status_code=422, content={"error": "invalid_input", "fields": [
            {"field": ".".join(map(str,e["loc"])), "message": e["msg"]} for e in error.errors()]})

    def invoke(method, *args):
        try:
            return getattr(app.state.prototype, method)(*args)
        except RuntimeError as error:
            raise HTTPException(503, str(error)) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.get("/health")
    def health():
        p = app.state.prototype
        return {"experiment": "environment_and_simulated_control_v1", "ready": not p.errors,
            "components": {name: name in p.models for name in ["outdoor", "hourly", "plume", "efficiency", "control"]},
            "errors": p.errors, "field_validated": False, "physical_actuation": False}

    @app.post("/prototype/forecast/30s")
    def short(request: ShortRequest):
        return invoke("short", [h.model_dump() for h in request.history])

    @app.post("/prototype/forecast/hourly")
    def hourly(request: HourlyRequest):
        return invoke("hourly", request.station, [h.model_dump() for h in request.history])

    @app.post("/prototype/plume")
    def plume(request: PlumeRequest):
        return invoke("plume", [[p.x,p.y,p.t,p.east_velocity,p.north_velocity,p.diffusivity,p.removal] for p in request.points])

    @app.post("/prototype/efficiency")
    def efficiency(request: EfficiencyRequest):
        return invoke("efficiency", [request.humidity,request.temperature,request.particle_um,request.droplet_um,request.wind_speed_m_s])

    @app.post("/prototype/control")
    def control(request: ControlRequest):
        return invoke("control", request.model_dump())

    @app.get("/prototype/evidence")
    def evidence():
        path = Path(root)/"reports/capabilities.json"
        if not path.exists():
            raise HTTPException(503, "Verified evidence unavailable")
        return json.loads(path.read_text())
    return app
