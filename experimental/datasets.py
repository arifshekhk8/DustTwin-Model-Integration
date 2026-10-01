"""Acquire attributed source bytes; construct causal features without interpolation."""
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json
import subprocess

import numpy as np
import pandas as pd

from common import ARTIFACTS, CONFIG, RAW, REPO, REPORTS, digest, write_json

UCI_URL = "https://archive.ics.uci.edu/static/public/501/beijing%2Bmulti%2Bsite%2Bair%2Bquality%2Bdata.zip"
MENDELEY_URL = "https://data.mendeley.com/public-files/datasets/7f22n9v7hp/files/07e65063-2634-47ee-92aa-c73cf71d5168/file_downloaded"
MENDELEY_SHA = "7aa66322a8440e1e353440616685ece43d9abb2af9d6a63971f224896816eadb"
COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
STATIONS = ["Aotizhongxin", "Changping", "Dingling", "Dongsi", "Guanyuan",
            "Gucheng", "Huairou", "Nongzhanguan", "Shunyi", "Tiantan", "Wanliu", "Wanshouxigong"]
CHANNELS = ["pm25", "pm10", "temperature", "humidity", "wind_east", "wind_north", "pressure", "rain"]
SHORT_CHANNELS = ["pm25", "pm10", "temperature", "humidity"]


def acquire(name, url, expected=None):
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / name
    if not path.exists():
        print("Downloading", name, flush=True)
        temporary = path.with_suffix(".pending")
        subprocess.run(["curl","--silent","--show-error","--fail","--location","--retry","2",
            "--connect-timeout","20","--max-time","300","--proto","=https","--proto-redir","=https",
            "--user-agent","Mozilla/5.0","--header","Accept: application/vnd.mendeley-public-dataset.1+json",
            "--output",str(temporary),url],check=True)
        if expected and digest(temporary) != expected:
            raise ValueError("Downloaded source checksum mismatch: " + name)
        temporary.replace(path)
    if expected and digest(path) != expected:
        raise ValueError("Source checksum mismatch: " + name)
    return path


def humidity_from_dewpoint(temp, dewpoint):
    # Magnus approximation over water. This is DERIVED RH, not a measured channel.
    return np.clip(100 * np.exp(17.625 * dewpoint / (243.04 + dewpoint)
                                - 17.625 * temp / (243.04 + temp)), 0, 100)


def uci_frames():
    manifest_path = REPORTS / "datasets.json"
    expected = None
    if manifest_path.exists():
        expected = json.loads(manifest_path.read_text()).get("uci", {}).get("sha256")
    path = acquire("uci-beijing-501.zip", UCI_URL, expected)
    with ZipFile(path) as outer:
        inner = ZipFile(BytesIO(outer.read("PRSA2017_Data_20130301-20170228.zip")))
        frames = []
        for name in sorted(inner.namelist()):
            if not name.endswith(".csv"):
                continue
            raw = pd.read_csv(BytesIO(inner.read(name)), na_values="NA")
            ts = pd.to_datetime(raw[["year", "month", "day", "hour"]])
            if ts.duplicated().any() or not (ts.diff().dropna() == pd.Timedelta(hours=1)).all():
                raise ValueError("Non-hourly or duplicate source clock")
            angle = raw.wd.map({value: i * np.pi / 8 for i, value in enumerate(COMPASS)})
            f = pd.DataFrame({"pm25": raw["PM2.5"], "pm10": raw.PM10,
                "temperature": raw.TEMP, "humidity": humidity_from_dewpoint(raw.TEMP, raw.DEWP),
                "wind_east": -raw.WSPM * np.sin(angle), "wind_north": -raw.WSPM * np.cos(angle),
                "pressure": raw.PRES, "rain": raw.RAIN})
            f.index = pd.DatetimeIndex(ts)
            f.attrs["station"] = raw.station.iloc[0]
            if f.attrs["station"] not in STATIONS:
                raise ValueError("Unknown station")
            frames.append(f)
    return frames, {"title": "Beijing Multi-Site Air Quality", "doi": "10.24432/C5RK5G",
        "source_url": "https://archive.ics.uci.edu/dataset/501/beijingmultisiteairqualitydata",
        "download_url": UCI_URL, "creator": "Song Chen (2017)", "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/", "sha256": digest(path),
        "bytes": path.stat().st_size, "rows": sum(map(len, frames)), "stations": STATIONS,
        "native_cadence_seconds": 3600, "humidity": "derived with Magnus approximation from measured temperature and dewpoint",
        "wind": "16 compass sectors converted to meteorological wind-to east/north velocity components",
        "time": "Original station local calendar; no fabricated UTC timestamps",
        "alterations": "Causal lag/rolling features; no backfill or interpolation; incomplete windows/targets dropped"}


def hourly_features(frame, station, stations=None):
    values = {}
    for channel in CHANNELS:
        for lag in CONFIG["forecast"]["lags_hours"]:
            values[f"{channel}_lag{lag}"] = frame[channel].shift(lag)
    for channel in ["pm25", "pm10"]:
        for window in [6, 24]:
            r = frame[channel].rolling(window, min_periods=window)
            values[f"{channel}_mean{window}"] = r.mean()
            values[f"{channel}_std{window}"] = r.std(ddof=0)
    for known in (STATIONS if stations is None else stations):
        values[f"station_{known}"] = float(known == station)
    values["hour_sin"] = np.sin(frame.index.hour * 2 * np.pi / 24)
    values["hour_cos"] = np.cos(frame.index.hour * 2 * np.pi / 24)
    values["year_sin"] = np.sin(frame.index.dayofyear * 2 * np.pi / 365.25)
    values["year_cos"] = np.cos(frame.index.dayofyear * 2 * np.pi / 365.25)
    return pd.DataFrame(values, index=frame.index)


def short_frames():
    # Reading an existing source archive is fine; never copy installed runtimes or bulk data into Git.
    source_cache = REPO.parent / "DustTwin-AI/data/raw/mendeley-7f22n9v7hp-v1"
    existing = list(source_cache.glob("*.zip")) if source_cache.exists() else []
    path = existing[0] if existing else acquire("mendeley-2024.zip", MENDELEY_URL, MENDELEY_SHA)
    if digest(path) != MENDELEY_SHA:
        raise ValueError("Mendeley source checksum mismatch")
    with ZipFile(path) as outer:
        inner = ZipFile(BytesIO(outer.read(outer.namelist()[0])))
        frames, files = [], []
        for day in [1, 2]:
            name = next(n for n in inner.namelist() if f"Outdoor experiment data/Day {day}/Raw/" in n
                        and "opc" in n.lower() and n.endswith(".csv"))
            contents = inner.read(name)
            lines = contents.decode("utf-8-sig").splitlines()
            header = next(i for i, line in enumerate(lines) if line.startswith("OADateTime,"))
            raw = pd.read_csv(BytesIO(contents), skiprows=header)
            clock = raw.OADateTime.astype(str)
            if clock.str.contains(":").all():
                time = pd.to_timedelta(clock)
                resolution = "one-second time-of-day labels, repeated native timestamps"
            else:
                time = pd.to_timedelta(pd.to_numeric(clock), unit="D")
                resolution = "OLE fractional-day numeric timestamps"
            seconds = (time - time.iloc[0]).dt.total_seconds().to_numpy()
            if np.any(np.diff(seconds) < 0):
                raise ValueError("Backwards source clock")
            grid = np.arange(0, int(np.floor(seconds[-1])) + 1, dtype=np.int64)
            indices = np.searchsorted(seconds, grid, side="right") - 1
            ages = grid - seconds[indices]
            native = raw[["PM2.5(ug/m3)", "PM10(ug/m3)", "Temperature(C)", "RelativeHumidity(%)"]].to_numpy(float)
            f = pd.DataFrame(native[indices], columns=SHORT_CHANNELS, index=grid)
            f.loc[ages > CONFIG["short_forecast"]["maximum_observation_age_seconds"], :] = np.nan
            frames.append(f)
            files.append({"path": name, "sha256": hashlib.sha256(contents).hexdigest(),
                "native_rows": len(raw), "grid_rows": len(grid), "duration_seconds": int(grid[-1]),
                "clock_resolution": resolution, "maximum_observation_age_seconds": 1.5})
    return frames, {"title": "Data on different particulate matter profiles produced in laboratory from construction activity and outdoor monitoring",
        "authors": ["Komiljon Askarov", "Jae-ho Choi"], "year": 2024, "doi": "10.17632/7f22n9v7hp.1",
        "source_url": "https://data.mendeley.com/datasets/7f22n9v7hp/1", "download_url": MENDELEY_URL,
        "license": "CC BY 4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "sha256": digest(path), "recordings": files, "recording_order": "Day labels; Day 1 lacks a calendar date",
        "alterations": "Outdoor OPC-N3 only; causal latest-past one-second grid; no lab recordings, wind joining, interpolation or smoothing columns"}


def short_features(frame):
    values = {}
    for channel in SHORT_CHANNELS:
        for lag in [0, 1, 5, 10, 30, 60, 120]:
            values[f"{channel}_lag{lag}"] = frame[channel].shift(lag)
        for window in [31, 61, 121]:
            r = frame[channel].rolling(window, min_periods=window)
            values[f"{channel}_mean{window}"] = r.mean()
            values[f"{channel}_std{window}"] = r.std(ddof=0)
            values[f"{channel}_change{window}"] = (frame[channel] - frame[channel].shift(window - 1)) / (window - 1)
    return pd.DataFrame(values, index=frame.index)
