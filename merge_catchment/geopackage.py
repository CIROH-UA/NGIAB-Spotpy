import sqlite3
from enum import Enum
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from .weighted_functions import weighted_circ_mean, weighted_mode


class Layers(Enum):
    DIVIDES = "divides"
    SUBDIVIDES = "subdivides"
    DIVIDE_ATTRIBUTES = "divide-attributes"
    FLOWPATH_ATTRIBUTES = "flowpath-attributes"
    FLOWPATHS = "flowpaths"
    NEXUS = "nexus"
    NETWORK = "network"


class GeoPackage:
    gpkg_path: Path
    divides: gpd.GeoDataFrame
    subdivides: gpd.GeoDataFrame | None
    divide_attributes: pd.DataFrame
    flowpath_attributes: pd.DataFrame
    flowpaths: gpd.GeoDataFrame
    nexus: gpd.GeoDataFrame
    network: pd.DataFrame | None

    def __init__(self, gpkg_path: Path):
        self.gpkg_path = gpkg_path
        self.divides = gpd.read_file(gpkg_path, layer=Layers.DIVIDES.value)
        self.divide_attributes = gpd.read_file(
            gpkg_path, layer=Layers.DIVIDE_ATTRIBUTES.value
        ).set_index("divide_id")
        self.flowpath_attributes = (
            gpd.read_file(gpkg_path, layer=Layers.FLOWPATH_ATTRIBUTES.value)
            .set_index("id")
            .drop(columns=["link", "to", "toid", "gage_nex_id", "vpuid"], errors="ignore")
        )

        self.flowpaths = gpd.read_file(gpkg_path, layer=Layers.FLOWPATHS.value)
        self.nexus = gpd.read_file(gpkg_path, layer=Layers.NEXUS.value)
        self.network = pd.DataFrame(columns=["new_id", "id"])

    @property
    def layers(self) -> dict[str, gpd.GeoDataFrame]:
        layers = {
            Layers.DIVIDES.value: self.divides,
            Layers.DIVIDE_ATTRIBUTES.value: self.divide_attributes,
            Layers.FLOWPATHS.value: self.flowpaths,
            Layers.FLOWPATH_ATTRIBUTES.value: self.flowpath_attributes,
            Layers.NEXUS.value: self.nexus,
            Layers.NETWORK.value: self.network,
        }
        return layers

    def save(self, save_as: Path = Path("/dev/null")):
        if save_as == Path("/dev/null"):
            save_as = Path(f"{self.gpkg_path.stem}_modified.gpkg")

        for name, gdf in self.layers.items():
            if name not in ["divides", "nexus", "divide-attributes"]:
                continue
            columns_to_drop = gdf.columns.tolist()
            if name == "divides":
                columns_to_drop.remove("divide_id")
                columns_to_drop.remove("toid")
            elif name == "divide-attributes":
                columns_to_drop = []
            else:
                columns_to_drop.remove("id")
                columns_to_drop.remove("toid")

            if isinstance(gdf, gpd.GeoDataFrame):
                gdf.to_file(save_as, layer=name, driver="GPKG")
            elif isinstance(gdf, pd.DataFrame):
                with sqlite3.connect(save_as) as conn:
                    gdf.to_sql(name, conn, if_exists="replace", index=True)

    def execute_sql(self, sql: str) -> list:
        with sqlite3.connect(self.gpkg_path) as conn:
            return conn.execute(sql).fetchall()

    def _get_areasqkm_dict(self):
        sql = "SELECT divide_id, areasqkm FROM 'divides'"
        results = self.execute_sql(sql)
        return dict(results)

    def _get_areasqkm_xarray(self):
        weights_dict = self._get_areasqkm_dict()
        da = xr.DataArray.from_dict(
            {
                "dims": ["divide_id"],
                "coords": {"divide_id": {"dims": ["divide_id"], "data": list(weights_dict.keys())}},
                "data": list(weights_dict.values()),
            }
        )
        return da

    def _merge_divide_attributes(self, ids: list[str], new_id: str):
        if not new_id:
            raise ValueError("new_id must be provided")
        df = self.divide_attributes.loc[ids]
        xr_ds = xr.Dataset().from_dataframe(df)
        # make the coordinate be divide_id
        all_weights = self._get_areasqkm_dict()
        for id in list(all_weights.keys()):
            if id not in ids:
                all_weights.pop(id)
        weights = [all_weights[id] for id in ids]
        # all_weights = self._get_areasqkm_xarray()
        # xr_weights = all_weights.sel

        # variables are named mode.var mean.var circ_mean.var geom_mean.var etc depending on how they should be aggregated
        # new_entry = {}
        largest_cat = max(ids, key=lambda x: all_weights[x])
        for name in xr_ds.data_vars.keys():
            if name.startswith("mean."):
                self.divide_attributes.loc[new_id, name] = (
                    # xr_ds[name].weighted(xr_weights).mean().values
                    np.average(xr_ds[name].values, weights=weights)
                )
            elif name.startswith("mode."):
                self.divide_attributes.loc[new_id, name] = weighted_mode(df[name], all_weights)
            elif name.startswith("geom_mean."):
                self.divide_attributes.loc[new_id, name] = (
                    # xr_ds[name].weighted(xr_weights).mean().values
                    np.average(xr_ds[name].values, weights=weights)
                )
            elif name.startswith("circ_mean."):
                self.divide_attributes.loc[new_id, name] = weighted_circ_mean(df[name], all_weights)
            elif name.startswith("dist_4.twi"):
                self.divide_attributes.loc[new_id, name] = df.loc[largest_cat][name]
            elif name.startswith("vpuid"):
                self.divide_attributes.loc[new_id, name] = weighted_mode(df[name], all_weights)
            elif name.startswith("centroid"):
                self.divide_attributes.loc[new_id, name] = (
                    # xr_ds[name].weighted(xr_weights).mean().values
                    np.average(xr_ds[name].values, weights=weights)
                )
            else:
                print(f"Unknown variable type: {name}")

        # remove the old divides
        if new_id in ids:
            ids.remove(new_id)
        self.divide_attributes.drop(ids, inplace=True)

    def _merge_divides(self, ids: list[str], new_id: str):
        if not new_id:
            raise ValueError("new_id must be provided")

        aggregation = {
            # "divide_id": "last",
            "toid": "first",
            "type": "last",
            "ds_id": "last",
            "areasqkm": "sum",
            "id": "last",
            "lengthkm": "sum",
            "tot_drainage_areasqkm": "sum",
            "has_flowline": "max",
            "vpuid": "last",
        }

        geom_to_merge = self.divides[self.divides["divide_id"].isin(ids)]

        # merge the attributes first
        self._merge_divide_attributes(ids, new_id)

        if geom_to_merge.empty:
            print(f"No divides found matching IDs: {ids}")
            return

        temp_divides = self.divides.copy()

        temp_divides.loc[temp_divides["divide_id"].isin(ids), "divide_id"] = new_id
        result = temp_divides.dissolve(by="divide_id", aggfunc=aggregation, as_index=False)
        self.divides = result
        for id in ids:
            self.network.loc[-1] = [new_id, id]

    def _rename_divide(self, old_id: str, new_id: str):
        if not new_id:
            raise ValueError("new_id must be provided")
        if old_id == new_id:
            return

        self.divides.loc[self.divides["divide_id"] == old_id, "divide_id"] = new_id
        self.divide_attributes.rename(index={old_id: new_id}, inplace=True)
        self.network.loc[-1] = [new_id, old_id]

    def merge(self, ids: list[str | list[str]]) -> None:
        # convert wb ids to cat ids

        for i, id_sublist in enumerate(ids):
            sub = ["cat-" + str(id).split("-")[0] for id in id_sublist]

            cat_ids = list(dict.fromkeys(sub))

            if len(cat_ids) > 1:
                self._merge_divides(cat_ids, f"cat-{i}")
            elif len(cat_ids) == 1:
                self._rename_divide(cat_ids[0], f"cat-{i}")

        self.divides["toid"] = "nex-1"

        self.nexus = self.nexus.drop(self.nexus.index.to_list()[1:], axis=0)
        self.nexus["id"] = "nex-1"
