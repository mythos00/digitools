"""OSM PBF faylından çay, kanal, göl, su anbarı və bulaqları çıxarır.

İşə salma:
    python prepare_water.py azerbaijan-261028.osm.pbf
Nəticə: data/water.parquet
"""
import sys
from pathlib import Path
import osmium
import geopandas as gpd
from shapely.geometry import LineString, Polygon, Point

LINE_WATERWAYS = {"river", "stream", "canal", "ditch", "drain", "brook", "tidal_channel"}
AREA_WATERS = {"lake", "reservoir", "pond", "basin", "lagoon", "oxbow", "water"}

class WaterHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.rows = []

    def way(self, way):
        tags = way.tags
        waterway = tags.get("waterway")
        natural = tags.get("natural")
        water = tags.get("water")
        landuse = tags.get("landuse")
        # Saltwater/coastal features are intentionally excluded where explicitly tagged.
        salt = " ".join(str(tags.get(k, "")) for k in ("salt", "salinity", "water", "natural")).lower()
        if "salt" in salt or natural == "coastline" or waterway == "seaway":
            return
        include_line = waterway in LINE_WATERWAYS
        include_area = (natural == "water" or water in AREA_WATERS or landuse == "reservoir")
        coords = []
        for node in way.nodes:
            try:
                if node.location.valid():
                    coords.append((node.location.lon, node.location.lat))
            except Exception:
                pass
        if len(coords) < 2:
            return
        geom = None
        kind = waterway or water or natural or landuse or "water"
        try:
            if include_area and len(coords) >= 4 and coords[0] == coords[-1]:
                poly = Polygon(coords)
                if poly.is_valid and not poly.is_empty and poly.area > 0:
                    geom = poly
            if geom is None and include_line:
                geom = LineString(coords)
        except Exception:
            geom = None
        if geom is None or geom.is_empty:
            return
        self.rows.append({
            "osm_id": int(way.id), "name": tags.get("name"), "waterway": waterway,
            "natural": natural, "water": water, "landuse": landuse,
            "water_kind": kind, "geometry": geom,
        })

    def node(self, node):
        tags = node.tags
        if tags.get("natural") != "spring":
            return
        try:
            if not node.location.valid():
                return
            point = Point(node.location.lon, node.location.lat)
        except Exception:
            return
        self.rows.append({
            "osm_id": int(node.id), "name": tags.get("name"), "waterway": None,
            "natural": "spring", "water": None, "landuse": None,
            "water_kind": "spring", "geometry": point,
        })

def main():
    if len(sys.argv) < 2:
        print("İstifadə: python prepare_water.py azerbaijan-261028.osm.pbf")
        raise SystemExit(2)
    pbf = Path(sys.argv[1]).expanduser().resolve()
    if not pbf.exists():
        print(f"PBF faylı tapılmadı: {pbf}")
        raise SystemExit(1)
    handler = WaterHandler()
    print(f"OSM PBF oxunur: {pbf}")
    handler.apply_file(str(pbf), locations=True, idx="flex_mem")
    if not handler.rows:
        print("Su obyekti tapılmadı.")
        raise SystemExit(1)
    gdf = gpd.GeoDataFrame(handler.rows, geometry="geometry", crs="EPSG:4326")
    out = Path("data") / "water.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(out, index=False)
    print(f"Yaradıldı: {out.resolve()}")
    print(f"Obyekt sayı: {len(gdf):,}")
    print("Qeyd: OSM-də olmayan və ya yalnız relation kimi xəritələnmiş bəzi su obyektləri buraxıla bilər.")

if __name__ == "__main__":
    main()
