import math
import heapq
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
import requests
import streamlit as st
from geopy.distance import geodesic
from pyproj import Transformer
from shapely.geometry import Point, box
from shapely.ops import linemerge, nearest_points, transform as shapely_transform, unary_union
from streamlit_folium import st_folium

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Torpaq Sahəsi Analizi", page_icon="🌍", layout="wide")
st.title("🌍 Torpaq Sahəsinin Ətraf Analizi")
st.caption("Azərbaycan üzrə torpaq sahəsinin ətrafındakı obyektləri, yaxın küçələri və əsas yola bağlı tam yol xətlərini araşdırın.")
st.info(
    "Məlumatlar Geoapify və OpenStreetMap məlumatlarına əsaslanır. "
    "Nəticələr rəsmi kadastr və ya mühəndislik ölçməsi deyil."
)

# ---------------- SETTINGS ----------------
API_URL = "https://api.geoapify.com/v2/places"
GEOCODE_URL = "https://api.geoapify.com/v1/geocode/search"
ROADS_FILE = Path(__file__).parent / "data" / "roads.parquet"
ROAD_CATEGORY = "Yol xətləri"
WATER_CATEGORY = "Su obyektləri"
WATER_FILE = Path(__file__).parent / "data" / "water.parquet"

# Streamlit Community Cloud -> App settings -> Secrets bölməsinə əlavə et:
# GEOAPIFY_API_KEY = "sənin_yeni_api_açarın"
try:
    API_KEY = "b8f6c459e6434e7582d251e5cf2de93e"
except Exception:
    API_KEY = ""

CATEGORY_MAP = {
    WATER_CATEGORY: [],
    "Kəndlər və şəhərlər": ["populated_place"],
    "Məktəblər": ["education.school"],
    "Xəstəxana və klinikalar": ["healthcare"],
    "Mağazalar və marketlər": ["commercial"],
    "Yanacaqdoldurma məntəqələri": ["service.vehicle.fuel"],
    "Avtobus dayanacaqları": ["public_transport.bus"],
    "Hava limanları": ["airport"],
    "Yanğın və polis xidmətləri": ["service.fire_station", "service.police"],
}
ALL_CATEGORIES = [ROAD_CATEGORY] + list(CATEGORY_MAP.keys())
UNSUPPORTED_CATEGORIES = ["Elektrik infrastrukturu", "Dəmir yolu"]

COLORS = {
    ROAD_CATEGORY: "blue",
    WATER_CATEGORY: "cadetblue",
    "Kəndlər və şəhərlər": "green",
    "Məktəblər": "purple",
    "Xəstəxana və klinikalar": "red",
    "Mağazalar və marketlər": "darkgreen",
    "Yanacaqdoldurma məntəqələri": "darkred",
    "Avtobus dayanacaqları": "lightblue",
    "Hava limanları": "pink",
    "Yanğın və polis xidmətləri": "gray",
}

ROAD_TYPE_NAMES = {
    "motorway": "Magistral yol",
    "motorway_link": "Magistral yola giriş/çıxış",
    "trunk": "Əsas magistral yol",
    "trunk_link": "Əsas yola giriş/çıxış",
    "primary": "Əsas yol",
    "primary_link": "Əsas yola giriş/çıxış",
    "secondary": "İkinci dərəcəli yol",
    "secondary_link": "İkinci dərəcəli yola giriş/çıxış",
    "tertiary": "Yerli əsas yol",
    "tertiary_link": "Yerli yola giriş/çıxış",
    "residential": "Yaşayış küçəsi",
    "living_street": "Yaşayış küçəsi",
    "unclassified": "Digər yol",
    "service": "Xidmət yolu",
    "track": "Torpaq/kənd yolu",
    "path": "Cığır",
    "footway": "Piyada yolu",
    "pedestrian": "Piyada küçəsi",
    "cycleway": "Velosiped yolu",
    "steps": "Pilləkənli piyada yolu",
}

RESULT_COLUMNS = [
    "Kateqoriya", "Obyekt", "Yol növü", "Məsafə (km)", "Enlik", "Uzunluq",
    "Xəritə", "_lat", "_lon", "_id", "_line_coords", "_geojson"
]

# ---------------- GENERAL HELPERS ----------------
def distance_text(km):
    return f"{km * 1000:.0f} m" if km < 1 else f"{km:.2f} km"


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

# ---------------- GEOCODING ----------------
@st.cache_data(ttl=3600, show_spinner=False)
def geocode_location(query, api_key):
    response = requests.get(
        GEOCODE_URL,
        params={
            "text": query,
            "filter": "countrycode:az",
            "format": "json",
            "limit": 1,
            "apiKey": api_key,
        },
        timeout=(5, 15),
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    if not results:
        return None
    item = results[0]
    return {
        "name": item.get("formatted", query),
        "lat": float(item["lat"]),
        "lon": float(item["lon"]),
    }

# ---------------- GEOAPIFY PLACES API ----------------
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_places(lat, lon, radius_m, categories, api_key):
    api_categories = []
    for category in categories:
        # Yol xətləri OSM faylından gəlir, Geoapify Places API-yə göndərilmir.
        api_categories.extend(CATEGORY_MAP.get(category, []))
    api_categories = list(dict.fromkeys(api_categories))
    if not api_categories:
        return []

    response = requests.get(
        API_URL,
        params={
            "categories": ",".join(api_categories),
            "filter": f"circle:{lon},{lat},{radius_m}",
            "bias": f"proximity:{lon},{lat}",
            "limit": 500,
            "lang": "az",
            "apiKey": api_key,
        },
        timeout=(5, 25),
    )
    response.raise_for_status()
    return response.json().get("features", [])

# ---------------- GEOAPIFY RESULT PROCESSING ----------------
def analyze_features(features, origin_lat, origin_lon, selected_categories):
    rows, seen = [], set()
    for feature in features:
        props = feature.get("properties", {}) or {}
        geometry = feature.get("geometry", {}) or {}
        coords = geometry.get("coordinates")
        if coords and len(coords) >= 2:
            obj_lon, obj_lat = safe_float(coords[0]), safe_float(coords[1])
        else:
            obj_lat, obj_lon = safe_float(props.get("lat")), safe_float(props.get("lon"))
        if obj_lat is None or obj_lon is None:
            continue

        place_id = props.get("place_id") or f"{obj_lat:.6f},{obj_lon:.6f},{props.get('name', '')}"
        if place_id in seen:
            continue
        seen.add(place_id)

        api_cats = props.get("categories", [])
        if isinstance(api_cats, str):
            api_cats = [api_cats]
        matched_category = None
        for category in selected_categories:
            wanted = CATEGORY_MAP.get(category, [])
            if any(
                api_cat == wanted_cat or api_cat.startswith(wanted_cat + ".")
                for api_cat in api_cats if isinstance(api_cat, str)
                for wanted_cat in wanted
            ):
                matched_category = category
                break
        if matched_category is None:
            continue

        km = geodesic((origin_lat, origin_lon), (obj_lat, obj_lon)).km
        name = props.get("name") or props.get("address_line1") or props.get("formatted") or matched_category
        rows.append({
            "Kateqoriya": matched_category,
            "Obyekt": name,
            "Yol növü": "",
            "Məsafə (km)": round(km, 3),
            "Enlik": round(obj_lat, 6),
            "Uzunluq": round(obj_lon, 6),
            "Xəritə": f"https://www.google.com/maps?q={obj_lat},{obj_lon}",
            "_lat": obj_lat,
            "_lon": obj_lon,
            "_id": str(place_id),
            "_line_coords": None,
            "_geojson": None,
        })
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)

# ---------------- LOCAL ROAD LINES ----------------
@st.cache_resource(show_spinner="Yol xətləri yüklənir...")
def load_road_lines():
    """Hazır GeoParquet faylını bir dəfə yaddaşa yükləyir."""
    if not ROADS_FILE.exists():
        return None
    roads = gpd.read_parquet(ROADS_FILE)
    if roads.empty:
        return roads
    if roads.crs is None:
        roads = roads.set_crs("EPSG:4326")
    elif roads.crs.to_string() != "EPSG:4326":
        roads = roads.to_crs("EPSG:4326")
    required = {"highway", "geometry"}
    if not required.issubset(roads.columns):
        raise ValueError("roads.parquet faylında 'highway' və 'geometry' sütunları olmalıdır.")
    roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty].copy()
    # Hazırda heç bir bağlantı/yol-növü filtri tətbiq edilmir: OSM-də olan
    # bütün highway xətləri xəritədə göstərilə bilər.
    return roads.reset_index(drop=True)


def utm_epsg_for_lon(lon):
    zone = int((lon + 180) // 6) + 1
    return 32600 + zone


def find_nearest_road_lines(origin_lat, origin_lon, radius_km, roads):
    """Radius daxilində əsas yol şəbəkəsinə bağlı bütün tam OSM yol xətlərini göstərir.

    Hər xətt üçün torpaq sahəsinə ən yaxın nöqtə ayrıca hesablanır. Əvvəlki
    kimi hər highway növündən yalnız bir xətt seçilmir; buna görə ara küçələr
    və xidmət yolları da xəritədə qalır.
    """
    if roads is None or roads.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    radius_m = radius_km * 1000
    lat_delta = radius_m / 110_800
    cos_lat = max(0.15, math.cos(math.radians(origin_lat)))
    lon_delta = radius_m / (111_000 * cos_lat)
    search_box = box(
        origin_lon - lon_delta, origin_lat - lat_delta,
        origin_lon + lon_delta, origin_lat + lon_delta,
    )
    try:
        positions = roads.sindex.query(search_box, predicate="intersects")
        candidates = roads.iloc[list(positions)].copy()
    except Exception:
        bounds = roads.bounds
        mask = (
            (bounds.minx <= origin_lon + lon_delta) & (bounds.maxx >= origin_lon - lon_delta) &
            (bounds.miny <= origin_lat + lat_delta) & (bounds.maxy >= origin_lat - lat_delta)
        )
        candidates = roads.loc[mask].copy()

    if candidates.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    # Əvvəlcə magistral/əsas yolları və torpaq sahəsinə ən yaxın yoldan
    # həmin yollara gedən ən qısa əlaqəli yol zəncirini seç.
    major_types = {
        "motorway", "motorway_link", "trunk", "trunk_link",
        "primary", "primary_link",
    }
    candidates = candidates.copy()
    candidates["_road_key"] = candidates.get("osm_id", pd.Series(candidates.index, index=candidates.index)).astype(str)
    candidates["_is_major"] = candidates["highway"].astype(str).isin(major_types)

    # OSM way xətlərinin hər koordinatını qovşaq sayaraq qraf qurulur.
    # Eyni koordinata gələn yollar bir-birinə bağlanır.
    graph = {}
    node_way_ids = {}
    line_nodes = {}
    def node_key(coord):
        return (round(float(coord[0]), 6), round(float(coord[1]), 6))
    def edge_length(a, b):
        mean_lat = math.radians((a[1] + b[1]) / 2)
        dx = (b[0] - a[0]) * 111320 * math.cos(mean_lat)
        dy = (b[1] - a[1]) * 110540
        return math.hypot(dx, dy)

    for idx, road in candidates.iterrows():
        geom = road.geometry
        parts = [geom] if geom.geom_type == "LineString" else list(geom.geoms) if geom.geom_type == "MultiLineString" else []
        way_key = str(road.get("osm_id", idx))
        way_nodes = []
        for part in parts:
            coords = list(part.coords)
            keys = [node_key(c) for c in coords]
            way_nodes.extend(keys)
            for a, b in zip(keys, keys[1:]):
                if a == b:
                    continue
                length = edge_length(a, b)
                graph.setdefault(a, []).append((b, length, way_key))
                graph.setdefault(b, []).append((a, length, way_key))
                node_way_ids.setdefault(a, set()).add(way_key)
                node_way_ids.setdefault(b, set()).add(way_key)
        line_nodes[way_key] = way_nodes

    major_way_keys = set(candidates.loc[candidates["_is_major"], "_road_key"].astype(str))
    selected_way_keys = set(major_way_keys)
    if graph and major_way_keys:
        # Ən yaxın yol xəttini və onun üzərində torpaq sahəsinə ən yaxın qovşağı tap.
        nearest_way = None
        nearest_vertex = None
        nearest_vertex_dist = float("inf")
        for idx, road in candidates.iterrows():
            way_key = str(road.get("osm_id", idx))
            geom = road.geometry
            if geom is None or geom.is_empty:
                continue
            try:
                projected = shapely_transform(
                    Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg_for_lon(origin_lon)}", always_xy=True).transform,
                    geom,
                )
                origin_projected = shapely_transform(
                    Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg_for_lon(origin_lon)}", always_xy=True).transform,
                    Point(origin_lon, origin_lat),
                )
                if origin_projected.distance(projected) > radius_m:
                    continue
                for node in line_nodes.get(way_key, []):
                    dlon = (node[0] - origin_lon) * 111320 * math.cos(math.radians(origin_lat))
                    dlat = (node[1] - origin_lat) * 110540
                    d = math.hypot(dlon, dlat)
                    if d < nearest_vertex_dist:
                        nearest_vertex_dist = d
                        nearest_way = way_key
                        nearest_vertex = node
            except Exception:
                continue

        # Dijkstra: ən yaxın yol qovşağından istənilən magistral/əsas yola qədər.
        if nearest_vertex is not None:
            distances = {nearest_vertex: 0.0}
            previous = {}
            queue = [(0.0, nearest_vertex)]
            target = None
            major_nodes = {
                node for way_key in major_way_keys
                for node in line_nodes.get(way_key, [])
            }
            while queue:
                dist, node = heapq.heappop(queue)
                if dist != distances.get(node):
                    continue
                if node in major_nodes:
                    target = node
                    break
                for nxt, weight, way_key in graph.get(node, []):
                    new_dist = dist + weight
                    if new_dist < distances.get(nxt, float("inf")):
                        distances[nxt] = new_dist
                        previous[nxt] = (node, way_key)
                        heapq.heappush(queue, (new_dist, nxt))
            if target is not None:
                node = target
                while node != nearest_vertex and node in previous:
                    prev_node, way_key = previous[node]
                    selected_way_keys.add(way_key)
                    node = prev_node
                if nearest_way is not None:
                    selected_way_keys.add(nearest_way)

    # Əsas yolların hamısı saxlanır; digər yol növlərindən isə yalnız
    # ən yaxın yoldan əsas yola gedən seçilmiş bağlantı qalır.
    candidates = candidates[
        candidates["_road_key"].isin(selected_way_keys) | candidates["_is_major"]
    ].copy()
    if candidates.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    # Eyni adlı yolun hissələrini ref fərqli olsa belə birləşdiririk.
    # OSM-də eyni yolun bəzi hissələrində ref etiketi fərqli/boş ola bilər;
    # ref-ə görə ayrıca qruplaşdırmaq xəritədə təkrar xətlər yarada bilər.
    # Həndəsi union üst-üstə düşən hissələri təmizləyir, linemerge isə birləşdirir.
    named_groups = []
    for _, row in candidates.iterrows():
        name = row.get("name")
        if pd.isna(name) or not str(name).strip():
            continue
        named_groups.append(str(name).strip())
    named_groups = list(dict.fromkeys(named_groups))

    merged_rows = []
    roads_source = roads.copy()
    for name in named_groups:
        mask = roads_source["name"].astype("string").str.strip().eq(name).fillna(False)
        same_name = roads_source.loc[mask].copy()
        if same_name.empty:
            continue
        try:
            merged_geom = linemerge(unary_union(list(same_name.geometry)))
        except Exception:
            merged_geom = unary_union(list(same_name.geometry))
        if merged_geom is None or merged_geom.is_empty:
            continue
        first = same_name.iloc[0].copy()
        first["geometry"] = merged_geom
        first["osm_id"] = "joined-" + "-".join(same_name["osm_id"].astype(str).head(20).tolist())
        merged_rows.append(first)

    # Adı olmayan seçilmiş bağlantılar da itirilməsin.
    unnamed = candidates[candidates["name"].isna() | candidates["name"].astype(str).str.strip().eq("")].copy()
    if merged_rows:
        named_gdf = gpd.GeoDataFrame(merged_rows, geometry="geometry", crs=roads.crs)
        candidates = pd.concat([named_gdf, unnamed], ignore_index=True)
    else:
        candidates = unnamed
    if candidates.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    epsg = utm_epsg_for_lon(origin_lon)
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    inverse = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    origin_point = shapely_transform(transformer.transform, Point(origin_lon, origin_lat))
    rows = []

    for idx, road in candidates.iterrows():
        geom = road.geometry
        highway = str(road.get("highway", "unknown"))
        if geom is None or geom.is_empty:
            continue
        try:
            projected_line = shapely_transform(transformer.transform, geom)
            if projected_line.is_empty:
                continue
            distance_m = float(origin_point.distance(projected_line))
            if distance_m > radius_m:
                continue
            _, nearest_point = nearest_points(origin_point, projected_line)
            nearest_lon, nearest_lat = inverse.transform(nearest_point.x, nearest_point.y)
            name = road.get("name")
            ref = road.get("ref")
            display_name = str(name).strip() if pd.notna(name) and str(name).strip() else (
                str(ref).strip() if pd.notna(ref) and str(ref).strip() else ROAD_TYPE_NAMES.get(highway, highway)
            )
            # Hər yolun bütün həndəsi hissələrini saxla; MultiLineString olduqda
            # artıq yalnız ən uzun hissə deyil, bütün hissələr xəritədə çəkilir.
            if geom.geom_type == "LineString":
                line_coords = [[(float(y), float(x)) for x, y in geom.coords]]
            elif geom.geom_type == "MultiLineString":
                line_coords = [
                    [(float(y), float(x)) for x, y in part.coords]
                    for part in geom.geoms if len(part.coords) >= 2
                ]
                if not line_coords:
                    continue
            else:
                continue
            osm_id = road.get("osm_id", idx)
            rows.append({
                "Kateqoriya": ROAD_CATEGORY,
                "Obyekt": display_name,
                "Yol növü": ROAD_TYPE_NAMES.get(highway, highway),
                "Məsafə (km)": round(distance_m / 1000, 3),
                "Enlik": round(float(nearest_lat), 6),
                "Uzunluq": round(float(nearest_lon), 6),
                "Xəritə": f"https://www.google.com/maps?q={nearest_lat},{nearest_lon}",
                "_lat": float(nearest_lat),
                "_lon": float(nearest_lon),
                "_id": f"osm-road-{osm_id}",
                "_line_coords": line_coords,
            })
        except Exception:
            continue

    return pd.DataFrame(rows, columns=RESULT_COLUMNS).sort_values("Məsafə (km)").reset_index(drop=True) if rows else pd.DataFrame(columns=RESULT_COLUMNS)

# ---------------- LOCAL FRESHWATER FEATURES ----------------
@st.cache_resource(show_spinner="Şirin su obyektləri yüklənir...")
def load_water_features():
    """data/water.parquet faylından OSM su obyektlərini yükləyir."""
    if not WATER_FILE.exists():
        return None
    water = gpd.read_parquet(WATER_FILE)
    if water.empty:
        return water
    if water.crs is None:
        water = water.set_crs("EPSG:4326")
    elif water.crs.to_string() != "EPSG:4326":
        water = water.to_crs("EPSG:4326")
    water = water[water.geometry.notna() & ~water.geometry.is_empty].copy()
    # Köhnə water.parquet faylında adsız obyektlər qalıbsa belə göstərmə.
    if "name" in water.columns:
        named_mask = water["name"].notna() & water["name"].astype(str).str.strip().ne("")
        water = water.loc[named_mask].copy()
    return water.reset_index(drop=True)


def find_water_features(origin_lat, origin_lon, radius_km, water):
    """Yaxın OSM şirin su namizədlərini xətt və sahə həndəsələri ilə qaytarır."""
    if water is None or water.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    radius_m = radius_km * 1000
    lat_delta = radius_m / 110800
    lon_delta = radius_m / (111000 * max(0.15, math.cos(math.radians(origin_lat))))
    search_box = box(origin_lon-lon_delta, origin_lat-lat_delta, origin_lon+lon_delta, origin_lat+lat_delta)
    try:
        candidates = water.iloc[list(water.sindex.query(search_box, predicate="intersects"))].copy()
    except Exception:
        b = water.bounds
        mask = (b.minx <= origin_lon+lon_delta) & (b.maxx >= origin_lon-lon_delta) & (b.miny <= origin_lat+lat_delta) & (b.maxy >= origin_lat-lat_delta)
        candidates = water.loc[mask].copy()
    if candidates.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    epsg = utm_epsg_for_lon(origin_lon)
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    inverse = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    origin = shapely_transform(transformer.transform, Point(origin_lon, origin_lat))
    rows = []
    for idx, row in candidates.iterrows():
        geom = row.geometry
        try:
            projected = shapely_transform(transformer.transform, geom)
            dist = float(origin.distance(projected))
            if dist > radius_m:
                continue
            nearest = nearest_points(origin, projected)[1]
            lon, lat = inverse.transform(nearest.x, nearest.y)
            name = row.get("name")
            kind = row.get("water_kind") or row.get("waterway") or row.get("water") or row.get("natural") or "Su obyekti"
            if pd.isna(name) or not str(name).strip():
                continue
            label = str(name).strip()
            gj = geom.__geo_interface__
            rows.append({
                "Kateqoriya": WATER_CATEGORY, "Obyekt": label, "Yol növü": str(kind),
                "Məsafə (km)": round(dist/1000, 3), "Enlik": round(float(lat), 6), "Uzunluq": round(float(lon), 6),
                "Xəritə": f"https://www.google.com/maps?q={lat},{lon}", "_lat": float(lat), "_lon": float(lon),
                "_id": f"osm-water-{row.get('osm_id', idx)}", "_line_coords": None, "_geojson": gj,
            })
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    return pd.DataFrame(rows, columns=RESULT_COLUMNS).sort_values("Məsafə (km)").reset_index(drop=True)


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Axtarış parametrləri")
    location_query = st.text_input(
        "Torpaq sahəsinin lokasiyası",
        placeholder="Qəbələ, Azərbaycan və ya 40.4093, 49.8671",
        key="land_location_query",
    )
    radius_km = st.select_slider(
        "Axtarış radiusu", options=[1, 2, 3, 5, 7, 10], value=2,
        format_func=lambda x: f"{x} km", key="land_radius_select",
    )
    selected_categories = st.multiselect(
        "Obyekt kateqoriyaları",
        options=ALL_CATEGORIES,
        default=[ROAD_CATEGORY, "Məktəblər", "Su obyektləri", "Kəndlər və şəhərlər"],
        key="land_categories_select",
    )
    search_button = st.button("🔎 Analiz et", type="primary", use_container_width=True, key="land_search_button")
    st.caption("Ətraf obyektlər 30 dəqiqə, yol faylı isə tətbiq işlədiyi müddətdə yaddaşda saxlanılır.")

# ---------------- SEARCH ----------------
if search_button:
    if not location_query.strip():
        st.error("Lokasiya daxil edin.")
    elif not selected_categories:
        st.error("Ən azı bir kateqoriya seçin.")
    elif any(c not in {ROAD_CATEGORY, WATER_CATEGORY} for c in selected_categories) and not API_KEY:
        st.error("Geoapify API açarı tapılmadı. Streamlit Cloud Secrets bölməsində GEOAPIFY_API_KEY əlavə edin.")
    else:
        try:
            query = location_query.strip()
            location = None
            # Koordinat daxil edilibsə, əvvəl API çağırmadan yoxla.
            parts = [p.strip() for p in query.split(",")]
            if len(parts) == 2:
                lat, lon = safe_float(parts[0]), safe_float(parts[1])
                if lat is not None and lon is not None and 38 <= lat <= 42.5 and 44.5 <= lon <= 51:
                    location = {"name": f"Koordinatlar: {lat}, {lon}", "lat": lat, "lon": lon}
            if location is None:
                if not API_KEY:
                    st.error("Lokasiya adını koordinata çevirmək üçün Geoapify açarı lazımdır. Koordinatları 'enlik, uzunluq' formatında da daxil edə bilərsiniz.")
                    st.stop()
                with st.spinner("Lokasiya koordinatları tapılır..."):
                    location = geocode_location(query, API_KEY)
            if location is None:
                st.error("Lokasiya tapılmadı. Şəhər/kənd adı və ya 40.4093, 49.8671 formatında koordinat daxil edin.")
            else:
                road_results = pd.DataFrame(columns=RESULT_COLUMNS)
                if ROAD_CATEGORY in selected_categories:
                    roads = load_road_lines()
                    if roads is None:
                        st.session_state["land_search_error"] = (
                            "Yol xətti faylı tapılmadı: data/roads.parquet. Əvvəl azerbaijan-261028.osm.pbf "
                            "faylından roads.parquet hazırlayan prepare_roads.py skriptini işə salın."
                        )
                        st.rerun()
                    with st.spinner("Ən yaxın yol xətləri hesablanır..."):
                        road_results = find_nearest_road_lines(location["lat"], location["lon"], radius_km, roads)
                water_results = pd.DataFrame(columns=RESULT_COLUMNS)
                if WATER_CATEGORY in selected_categories:
                    water_data = load_water_features()
                    if water_data is None:
                        st.session_state["land_search_error"] = (
                            "Şirin su məlumatı tapılmadı: data/water.parquet. "
                            "Əvvəl prepare_water.py skriptini PBF faylı ilə işə salın."
                        )
                        st.rerun()
                    with st.spinner("Yaxın şirin su obyektləri axtarılır..."):
                        water_results = find_water_features(location["lat"], location["lon"], radius_km, water_data)
                place_results = pd.DataFrame(columns=RESULT_COLUMNS)
                non_local_categories = [c for c in selected_categories if c not in {ROAD_CATEGORY, WATER_CATEGORY}]
                if non_local_categories:
                    with st.spinner("Ətraf obyektlər axtarılır..."):
                        features = fetch_places(location["lat"], location["lon"], radius_km * 1000, non_local_categories, API_KEY)
                    place_results = analyze_features(features, location["lat"], location["lon"], non_local_categories)
                results = pd.concat([road_results, water_results, place_results], ignore_index=True)
                if not results.empty:
                    results = results.sort_values("Məsafə (km)").reset_index(drop=True)
                st.session_state["land_location"] = location
                st.session_state["land_radius"] = radius_km
                st.session_state["land_results"] = results
                st.session_state["land_unsupported"] = [c for c in UNSUPPORTED_CATEGORIES if c in selected_categories]
                st.session_state["land_search_error"] = None
        except requests.HTTPError as exc:
            details = exc.response.text[:400] if exc.response is not None else ""
            st.session_state["land_search_error"] = f"Geoapify HTTP xətası: {exc}. {details}"
        except requests.RequestException as exc:
            st.session_state["land_search_error"] = f"Geoapify serverinə qoşulmaq mümkün olmadı: {exc}"
        except Exception as exc:
            st.session_state["land_search_error"] = f"Xəta: {exc}"

# ---------------- RESULTS ----------------
if st.session_state.get("land_search_error"):
    st.error(st.session_state["land_search_error"])

if "land_location" in st.session_state and "land_results" in st.session_state:
    location = st.session_state["land_location"]
    results = st.session_state["land_results"]
    radius_used = st.session_state["land_radius"]
    st.subheader("📍 Analiz edilən lokasiya")
    st.write(location["name"])
    st.caption(f"Koordinatlar: {location['lat']:.6f}, {location['lon']:.6f} | Radius: {radius_used} km")

    if results.empty:
        st.warning("Bu radiusda seçilmiş kateqoriyalar üzrə nəticə tapılmadı. Radiusu artırın və ya kateqoriyaları yoxlayın.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Tapılan nəticələr", len(results))
        c2.metric("Kateqoriyalar", results["Kateqoriya"].nunique())
        c3.metric("Ən yaxın nəticə", distance_text(float(results.iloc[0]["Məsafə (km)"])))

        if ROAD_CATEGORY in results["Kateqoriya"].values:
            st.subheader("🛣️ OSM-dəki yol xətləri")
            st.caption("Bu mərhələdə yol növü və əsas yola bağlantı filtri yoxdur. Radius daxilində tapılan bütün OSM yol xətləri göstərilir.")
            nearest_roads = results[results["Kateqoriya"] == ROAD_CATEGORY].sort_values("Məsafə (km)")
            if nearest_roads.empty:
                st.warning("Seçilmiş radiusda yol xətti tapılmadı. Radiusu artırın.")
            else:
                road_display = nearest_roads[["Yol növü", "Obyekt", "Məsafə (km)", "Enlik", "Uzunluq", "Xəritə"]].copy()
                road_display["Məsafə"] = road_display["Məsafə (km)"].apply(distance_text)
                st.dataframe(
                    road_display[["Yol növü", "Obyekt", "Məsafə", "Enlik", "Uzunluq", "Xəritə"]],
                    use_container_width=True, hide_index=True,
                    column_config={"Xəritə": st.column_config.LinkColumn("Xəritədə aç", display_text="Google Maps")},
                )

        st.subheader("⭐ Kateqoriyalar üzrə ən yaxın obyekt")
        nearest = results.sort_values("Məsafə (km)").groupby("Kateqoriya", as_index=False).first()
        nearest_display = nearest[["Kateqoriya", "Obyekt", "Məsafə (km)", "Xəritə"]].copy()
        nearest_display["Məsafə"] = nearest_display["Məsafə (km)"].apply(distance_text)
        st.dataframe(
            nearest_display[["Kateqoriya", "Obyekt", "Məsafə", "Xəritə"]],
            use_container_width=True, hide_index=True,
            column_config={"Xəritə": st.column_config.LinkColumn("Xəritədə aç", display_text="Google Maps")},
        )

    # ---------------- MAP ----------------
    st.subheader("🗺️ Xəritə")
    map_object = folium.Map(location=[location["lat"], location["lon"]], zoom_start=13, control_scale=True)
    folium.Marker(
        [location["lat"], location["lon"]], tooltip="Torpaq sahəsinin lokasiyası", popup=location["name"],
        icon=folium.Icon(color="red", icon="home"),
    ).add_to(map_object)
    folium.Circle(
        location=[location["lat"], location["lon"]], radius=radius_used * 1000,
        color="blue", fill=True, fill_opacity=0.06,
    ).add_to(map_object)

    if not results.empty:
        for _, row in results.head(1500).iterrows():
            if row["Kateqoriya"] == WATER_CATEGORY and isinstance(row.get("_geojson"), dict):
                style = {"color": "#1677c8", "weight": 3, "opacity": 0.9, "fillColor": "#38a9e8", "fillOpacity": 0.35}
                folium.GeoJson(
                    row["_geojson"], name=row["Obyekt"], style_function=lambda _feature, stl=style: stl,
                    tooltip=f"{row['Obyekt']} — {row['Yol növü']} — {distance_text(float(row['Məsafə (km)']))}",
                    popup=folium.Popup(f"<b>{row['Obyekt']}</b><br>Növ: {row['Yol növü']}<br>Məsafə: {distance_text(float(row['Məsafə (km)']))}", max_width=320),
                ).add_to(map_object)
                continue
            if row["Kateqoriya"] == ROAD_CATEGORY and isinstance(row.get("_line_coords"), list) and row["_line_coords"]:
                for line_segment in row["_line_coords"]:
                    if isinstance(line_segment, list) and len(line_segment) >= 2:
                        folium.PolyLine(
                            locations=line_segment, color="blue", weight=3, opacity=0.85,
                            tooltip=f"{row['Yol növü']} — ən yaxın məsafə: {distance_text(float(row['Məsafə (km)']))}",
                        ).add_to(map_object)
                # Yol xətlərinin hər biri üçün ayrıca CircleMarker çəkmirik:
                # dairəvi qovşaqlarda eyni magistral üzərində 3-4 nöqtə görünməsin.
                # Yolun özü tam mavi xətt kimi qalır.
                continue
            popup_html = (
                f"<b>{row['Obyekt']}</b><br>Kateqoriya: {row['Kateqoriya']}<br>"
                f"Məsafə: {distance_text(float(row['Məsafə (km)']))}<br>"
                f"<a href='{row['Xəritə']}' target='_blank'>Xəritədə aç</a>"
            )
            if row["Kateqoriya"] == ROAD_CATEGORY:
                popup_html += f"<br>Yol növü: {row['Yol növü']}<br>Ən yaxın nöqtə: {row['Enlik']}, {row['Uzunluq']}"
            folium.CircleMarker(
                location=[row["_lat"], row["_lon"]], radius=6 if row["Kateqoriya"] == ROAD_CATEGORY else 5,
                color=COLORS.get(row["Kateqoriya"], "blue"), fill=True, fill_opacity=0.9,
                tooltip=f"{row['Obyekt']} — {distance_text(float(row['Məsafə (km)']))}",
                popup=folium.Popup(popup_html, max_width=320),
            ).add_to(map_object)
    st_folium(map_object, use_container_width=True, height=560, returned_objects=[], key="land_analysis_map")

    if not results.empty:
        st.subheader("📋 Bütün nəticələr")
        category_filter = st.selectbox("Kateqoriya üzrə filtr", ["Hamısı"] + sorted(results["Kateqoriya"].unique()), key="land_results_category_filter")
        filtered = results if category_filter == "Hamısı" else results[results["Kateqoriya"] == category_filter]
        table_columns = ["Kateqoriya", "Obyekt", "Yol növü", "Məsafə (km)", "Enlik", "Uzunluq", "Xəritə"]
        table = filtered[table_columns].copy()
        table["Məsafə"] = table["Məsafə (km)"].apply(distance_text)
        display_columns = ["Kateqoriya", "Obyekt"]
        if table["Yol növü"].astype(str).str.len().gt(0).any():
            display_columns.append("Yol növü")
        display_columns += ["Məsafə", "Enlik", "Uzunluq", "Xəritə"]
        st.dataframe(
            table[display_columns], use_container_width=True, hide_index=True,
            column_config={"Xəritə": st.column_config.LinkColumn("Xəritə linki", display_text="Aç")},
        )
        csv_data = filtered.drop(columns=["_lat", "_lon", "_id", "_line_coords", "_geojson"], errors="ignore").to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ Nəticələri CSV kimi yüklə", data=csv_data, file_name="torpaq_etraf_analizi.csv", mime="text/csv", key="land_download_csv")

    st.caption("Seçilmiş, eyni adlı yol hissələri birləşdirilərək bütöv xətt kimi göstərilir. Yol məsafəsi avtomobil marşrutu deyil. © OpenStreetMap contributors.")
else:
    st.markdown("### Necə istifadə etməli?")
    st.markdown(
        "1. Lokasiya və ya koordinat daxil edin.\n"
        "2. Axtarış radiusunu seçin.\n"
        "3. Yol xətləri və ya digər obyekt kateqoriyalarını seçin.\n"
        "4. **Analiz et** düyməsinə basın.\n"
        "5. Yol xətləri seçilibsə, proqram hər yol tipi üçün xəttə ən yaxın nöqtəni tapacaq."
    )
st.caption("Məkan məlumatları: Geoapify və © OpenStreetMap contributors. 
su obyektləri yerli OSM PBF faylından götürülür; tamlıq OSM xəritələnməsindən asılıdır.")
