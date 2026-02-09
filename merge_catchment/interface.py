import json
import sqlite3
from pathlib import Path

def calculate_upstream_area(divides, headwaters):
    for id in headwaters:
        toid = divides[id]["toid"]
        area = divides[id]["area"]

        while toid in divides.keys():
            divides[toid]["upstream_area"] += area
            toid = divides[toid]["toid"]

    return divides


def recalculate_upstream_area(divides, start_point: int):
    removed_area = divides[start_point]["upstream_area"]
    next_id = start_point
    while next_id in divides.keys():
        divides[next_id]["upstream_area"] = max(0, divides[next_id]["upstream_area"] - removed_area)
        next_id = divides[next_id]["toid"]

    return divides


def group_catchments(hf_path: Path, target_area: float = 330.0):
    with sqlite3.connect(hf_path) as conn:
        result = conn.execute("SELECT divide_id, toid, areasqkm FROM divides").fetchall()
    divides = {}
    headwaters = set()
    for divide_id, nex_id, area in result:
        id = int(divide_id.split("-")[-1])
        toid = int(nex_id.split("-")[-1])
        area = float(area)
        headwaters.add(id)
        divide = {"toid": toid, "area": area, "upstream_area": area}
        divides[id] = divide

    for id in divides.keys():
        toid = divides[id]["toid"]
        if toid in headwaters:
            headwaters.remove(toid)
        try:
            upstreams = divides[toid].get("upstreams", [])
            upstreams.append(id)
            divides[toid]["upstreams"] = upstreams
        except KeyError:
            print(f"no downstream found for {toid}")

    divides = calculate_upstream_area(divides, set(divides.keys()))
    merges = []
    while max(divide["upstream_area"] for divide in divides.values()) > target_area:
        # find the upstream area closes to target_area
        closest_id = None
        closest_area = float("inf")
        for id, divide in divides.items():
            area_diff = abs(divide["upstream_area"] - target_area)
            if area_diff < closest_area:
                closest_id = id
                closest_area = area_diff

        to_check = divides[closest_id]["upstreams"]
        upstreams = []
        while len(to_check) > 0:
            next_check = []
            for id in to_check:
                if id in divides.keys():
                    next_check.extend(divides[id].get("upstreams", []))
                    upstreams.append(id)
            to_check = next_check
        merges.append(upstreams)
        divides = recalculate_upstream_area(divides, closest_id)

        for id in upstreams:
            divides.pop(id)

        # recursively get every id upstream of that id
        # remove all those ids from the divides dict
        # recalculate upstream area t rerun
    # one final merge to pick up stragglers
    merges.append(list(divides.keys()))
    print(f"{len(merges)} cats remaining")
    return merges


def backup(original: Path):
    backup_path = original.with_suffix(".bak")
    if original.exists() and not backup_path.exists():
        original.rename(f"{backup_path.expanduser().resolve().absolute()}")


def restore(original: Path):
    backup_path = original.with_suffix(".bak")
    if original.exists():
        original.unlink()
    if backup_path.exists():
        backup_path.rename(f"{original.expanduser().resolve().absolute()}")


def get_dates(realization: Path):
    with open(realization, "r") as f:
        realization = json.load(f)
        return (
            realization["time"]["start_time"].split(" ")[0],
            realization["time"]["end_time"].split(" ")[0],
        )

