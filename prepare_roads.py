"""OSM PBF faylından yol xətlərini çıxarıb data/roads.parquet yaradır.

İşə salma nümunəsi:
    python prepare_roads.py azerbaijan-261028.osm.pbf
"""
import sys
from pathlib import Path

import geopandas as gpd
import osmium
import pandas as pd
from shapely.geometry import LineString

# Highway etiketi olan bütün OSM yolları saxlanılır; ilkin mərhələdə filtr yoxdur.


class RoadHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.rows = []

    def way(self, way):
        highway = way.tags.get("highway")
        if not highway:
            return
        coords = []
        for node in way.nodes:
            try:
                if node.location.valid():
                    coords.append((node.location.lon, node.location.lat))
            except Exception:
                continue
        if len(coords) < 2:
            return
        try:
            geom = LineString(coords)
        except Exception:
            return
        if geom.is_empty or not geom.is_valid:
            return
        self.rows.append({
            "osm_id": int(way.id),
            "highway": highway,
            "name": way.tags.get("name"),
            "ref": way.tags.get("ref"),
            "surface": way.tags.get("surface"),
            "oneway": way.tags.get("oneway"),
            "geometry": geom,
        })

def main():
    if len(sys.argv) < 2:
        print("İstifadə: python prepare_roads.py azerbaijan-261028.osm.pbf")
        raise SystemExit(2)
    pbf_path = Path(sys.argv[1]).expanduser().resolve()
    if not pbf_path.exists():
        print(f"Fayl tapılmadı: {pbf_path}")
        raise SystemExit(1)

    print(f"PBF oxunur: {pbf_path}")
    print("Bu proses kompüterdən asılı olaraq bir neçə dəqiqə çəkə bilər...")
    handler = RoadHandler()
    handler.apply_file(str(pbf_path), locations=True, idx="flex_mem")
    if not handler.rows:
        print("Heç bir yol xətti çıxarılmadı.")
        raise SystemExit(1)

    gdf = gpd.GeoDataFrame(handler.rows, geometry="geometry", crs="EPSG:4326")
    out_path = Path("data") / "roads.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(out_path, index=False)
    print(f"Hazırdır: {out_path.resolve()}")
    print(f"Yol xəttlərinin sayı: {len(gdf):,}")
    print("Highway etiketi olan bütün OSM yol növləri filtrsiz saxlanılıb.")
    print(f"Fayl ölçüsü: {out_path.stat().st_size / (1024 * 1024):.1f} MB")
    print("İndi data/roads.parquet faylını Streamlit layihənə əlavə et.")

if __name__ == "__main__":
    main()
