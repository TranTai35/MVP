import os
import time
import json
import requests
import streamlit as st
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from streamlit_folium import st_folium
import folium
import io
import csv

# Optional SerpApi client
try:
    from serpapi import GoogleSearch
    _HAS_SERPAPI_CLIENT = True
except Exception:
    GoogleSearch = None
    _HAS_SERPAPI_CLIENT = False

# =======================
# Cấu hình chung
# =======================
st.set_page_config(page_title="Smart Tourism System ", layout="wide")

DEFAULT_OLLAMA_BASE = os.environ.get("OLLAMA_API_BASE", "https://tjadc-34-11-34-242.a.free.pinggy.link")
NOMINATIM = "https://nominatim.openstreetmap.org"

# =======================
# Cấu hình OSRM
# =======================

OSRM = "https://router.project-osrm.org"
# OSRM = "https://localhost:5000"

# SerpAPI cho Google Maps search
SERPAPI_ENDPOINT = "https://serpapi.com/search.json"
DEFAULT_SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "fdcea49178237153de98821d877265b20649dadd015fddac5a28c2482873a7d3")

UA = {
    "User-Agent": "SmartTourism/1.0",
    "Accept": "application/json",
}

# =======================
# OSM / OSRM UTILITIES
# =======================
def geocode(q: str) -> Tuple[float, float, str]:
    time.sleep(1.0)
    r = requests.get(
        f"{NOMINATIM}/search",
        params={"q": q, "format": "jsonv2", "limit": 1},
        headers=UA,
        timeout=60,
    )
    r.raise_for_status()
    j = r.json()
    if not j:
        raise ValueError("Không tìm thấy vị trí.")
    return float(j[0]["lat"]), float(j[0]["lon"]), j[0].get("display_name", q)


def reverse_geocode(lat: float, lon: float) -> str:
    time.sleep(1.0)
    r = requests.get(
        f"{NOMINATIM}/reverse",
        params={"lat": lat, "lon": lon, "format": "jsonv2"},
        headers=UA,
        timeout=60,
    )
    r.raise_for_status()
    j = r.json()
    return j.get("display_name", f"{lat:.5f},{lon:.5f}")


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    from math import radians, sin, cos, asin, sqrt

    R = 6371.0088
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return R * c


def osrm_geom(lon1, lat1, lon2, lat2):
    url = f"{OSRM}/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"
    params = {"overview": "full", "geometries": "geojson"}

    # Try OSRM first; on failure, fall back to straight-line geometry so UI vẫn hiển thị được
    last_err = None
    for _ in range(2):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=60)
            r.raise_for_status()
            data = r.json()
            route = data["routes"][0]
            return route["geometry"], route["distance"] / 1000.0, route["duration"] / 3600.0
        except Exception as e:
            last_err = e
            time.sleep(1.0)

    # Fallback: straight line using haversine distance
    km = haversine_km(lat1, lon1, lat2, lon2)
    hrs = km / 40.0  # giả định tốc độ 40 km/h nếu không tính được route
    geom = {"type": "LineString", "coordinates": [[lon1, lat1], [lon2, lat2]]}
    return geom, km, hrs

# =======================
# SerpAPI Google Maps Search
# =======================
def build_serpapi_query(main_cat: str, cuisine: str) -> str:
    cuisine = (cuisine or "").strip()
    if main_cat == "Ăn uống":
        base = "nhà hàng"
    else:
        base = "địa điểm vui chơi"

    if cuisine:
        return f"{base} {cuisine}"
    return base


def search_places_serpapi(
    lat: float,
    lon: float,
    radius_km: float,
    main_cat: str,
    detail_filters: Dict,
    api_key: str,
    min_rating: float = 3.5,
    min_reviews: int = 10,
    top_n: int = 10,
    price_range: Tuple[int, int] = (0, 10**9),
    fetch_price_details: bool = False,
) -> List[Dict]:
    if not api_key:
        raise ValueError("Chưa có SERPAPI_KEY.")

    query = build_serpapi_query(main_cat, detail_filters.get("cuisine", ""))
    # Try using the official SerpApi Python client if available (better error handling)
    if _HAS_SERPAPI_CLIENT:
        params = {
            "engine": "google_maps",
            "type": "search",
            "q": query,
            "ll": f"@{lat},{lon},14z",
            "hl": "vi",
            "api_key": api_key,
        }
        try:
            client = GoogleSearch(params)
            data = client.get_dict()
        except Exception as e:
            # Fallback to direct HTTP if client fails
            params["api_key"] = api_key
            r = requests.get(SERPAPI_ENDPOINT, params=params, headers=UA, timeout=60)
            r.raise_for_status()
            data = r.json()
    else:
        params = {
            "engine": "google_maps",
            "type": "search",
            "q": query,
            "ll": f"@{lat},{lon},14z",
            "hl": "vi",
            "api_key": api_key,
        }
        r = requests.get(SERPAPI_ENDPOINT, params=params, headers=UA, timeout=60)
        r.raise_for_status()
        data = r.json()
    results = data.get("local_results", [])

    items = []
    for res in results:
        title = res.get("title", "(Không tên)")
        rating = res.get("rating")
        reviews = res.get("reviews")
        address = res.get("address", "")
        coords = res.get("gps_coordinates") or {}
        lat2 = coords.get("latitude")
        lon2 = coords.get("longitude")

        if lat2 is None or lon2 is None:
            continue

        if rating is not None and rating < min_rating:
            continue
        if reviews is not None and reviews < min_reviews:
            continue

        # Price filtering: attempt to infer a numeric price from result fields.
        # If price cannot be determined, optionally try a secondary SerpApi query
        # to find menu/price info (this consumes extra SerpApi quota).
        try:
            import re
        except Exception:
            re = None

        pmin, pmax = price_range
        price_val = None
        # Common fields that may contain price info
        price_field = res.get("price") or res.get("price_range") or res.get("price_level") or res.get("price_description")
        if price_field is not None:
            # numeric
            if isinstance(price_field, (int, float)):
                price_val = float(price_field)
            elif isinstance(price_field, str):
                # try to extract digits like '₫120.000' or '120,000 VND'
                if re:
                    m = re.search(r"(\d[\d,\. ]+)", price_field)
                    if m:
                        s = m.group(1)
                        s = s.replace(',', '').replace('.', '').replace(' ', '')
                        try:
                            price_val = float(s)
                        except Exception:
                            price_val = None
                    else:
                        # fallback: detect price-level symbols like '$', '$$'
                        dollar_count = price_field.count('$')
                        if dollar_count:
                            mapping = {1: 50000, 2: 150000, 3: 400000, 4: 1000000}
                            price_val = mapping.get(min(dollar_count, 4))
            # Secondary lookup: if still unknown and allowed, try querying SerpApi
            if price_val is None and fetch_price_details:
                # small delay to avoid hitting rate limits
                time.sleep(0.5)
                try:
                    q = f"{title} menu giá {address}"
                    params2 = {
                        "engine": "google_maps",
                        "type": "search",
                        "q": q,
                        "ll": f"@{lat},{lon},14z",
                        "hl": "vi",
                        "api_key": api_key,
                    }
                    if _HAS_SERPAPI_CLIENT:
                        try:
                            client2 = GoogleSearch(params2)
                            data2 = client2.get_dict()
                        except Exception:
                            r2 = requests.get(SERPAPI_ENDPOINT, params=params2, headers=UA, timeout=30)
                            r2.raise_for_status()
                            data2 = r2.json()
                    else:
                        r2 = requests.get(SERPAPI_ENDPOINT, params=params2, headers=UA, timeout=30)
                        r2.raise_for_status()
                        data2 = r2.json()

                    # try to extract numeric price from returned text fields
                    text_blob = json.dumps(data2)
                    if re:
                        m = re.search(r"(\d[\d,\. ]{2,})", text_blob)
                        if m:
                            s = m.group(1)
                            s = s.replace(',', '').replace('.', '').replace(' ', '')
                            try:
                                price_val = float(s)
                            except Exception:
                                price_val = None
                except Exception:
                    # ignore errors from secondary lookup; keep price_val as None
                    price_val = None

        # If we could determine a price and it's outside range, skip
        if price_val is not None and not (pmin <= price_val <= pmax):
            continue

        dist = haversine_km(lat, lon, lat2, lon2)

        items.append(
            {
                "name": title,
                "address": address,
                "rating": rating,
                "reviews": reviews,
                "lat": lat2,
                "lon": lon2,
                "distance_km": dist,
                "price": price_val,
            }
        )

    items.sort(key=lambda x: (-(x["rating"] or 0), -(x["reviews"] or 0), x["distance_km"]))
    return items[:top_n]

# =======================
# Build Route A → B → C
# =======================
def build_route_segments(origin: Dict, schedule: List[Dict]) -> List[Dict]:
    waypoints = []
    if origin:
        waypoints.append(origin)
    for blk in schedule:
        if blk.get("place"):
            waypoints.append(blk["place"])

    segments = []
    for i in range(len(waypoints) - 1):
        a = waypoints[i]
        b = waypoints[i+1]
        geom, km, hrs = osrm_geom(a["lon"], a["lat"], b["lon"], b["lat"])
        segments.append({
            "from": a["name"],
            "to": b["name"],
            "geom": geom,
            "km": km,
            "hrs": hrs,
        })
    return segments


# =======================
# Chatbot Ollama
# =======================
def ollama_chat(messages: List[Dict], base_url: str, model: str = "llama3.2:1b"):
    try:
        url = f"{base_url.rstrip('/')}/api/chat"
        payload = {"model": model, "messages": messages, "stream": False}
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        if "message" in data:
            return data["message"]["content"]
        return ""
    except Exception as e:
        return f"(Chatbot offline) {e}"


# Serialize itinerary
def serialize_itinerary(name, origin, schedule):
    return {
        "name": name,
        "origin": origin,
        "schedule": schedule,
    }


# Sanitizer for assistant message headers coming from some models
def sanitize_assistant_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    # Replace the special header token with a friendly Vietnamese label
    return text.replace("<|start_header_id|>assistant<|end_header_id|>", "Chatbot trả lời")


def auto_generate_schedule_variants(
    start_time_str: str,
    end_time_str: str,
    preferences: List[str],
    origin: Dict,
    center: Dict,
    radius_km: float,
    serpapi_key: str,
    ollama_base: str,
    model: str,
) -> Tuple[List[List[Dict]], str]:
    """Auto-generate đến 3 lịch trình theo sở thích.

    - Mỗi sở thích = 1 khung giờ.
    - SerpAPI: lấy ứng viên cho từng sở thích; Ollama chọn điểm phù hợp nhất (tránh trùng lặp trong lịch).
    - Sinh 3 biến thể: mixed, ưu tiên rating, ưu tiên quãng đường.
    - Các lịch sau chỉ trùng tối đa 2 địa điểm với lịch trước.
    - Các mục ăn/uống cách nhau tối thiểu 4 tiếng.
    """
    t0 = datetime.strptime(start_time_str, "%H:%M")
    t1 = datetime.strptime(end_time_str, "%H:%M")
    if t1 <= t0:
        t1 = t1.replace(day=t1.day + 1)

    total_minutes = int((t1 - t0).total_seconds() / 60)
    if total_minutes < 30:
        raise ValueError("Khoảng thời gian quá ngắn để tự động tạo lịch.")

    pref_list = preferences or ["Khám phá"]
    block_count = max(1, len(pref_list))
    block_minutes = max(30, total_minutes // block_count)

    food_keywords = ["ăn", "cơm", "phở", "bún", "bánh", "cafe", "cà phê", "ăn sáng", "ăn trưa", "ăn tối", "nhậu", "lẩu", "buffet", "pizza", "sushi", "bánh mì", "cơm tấm"]

    def is_food_pref(pref: str) -> bool:
        pref_l = (pref or "").lower()
        return any(k in pref_l for k in food_keywords)

    def matches_pref(place: Dict, pref: str) -> bool:
        if not pref:
            return True
        text = f"{place.get('name','')} {place.get('address','')}".lower()
        pref_l = pref.lower().strip()
        if pref_l and pref_l in text:
            return True
        tokens = [t for t in pref_l.replace(",", " ").split() if len(t) >= 3]
        return any(t in text for t in tokens)

    def fetch_pref_candidates(pref: str) -> List[Dict]:
        try:
            main_cat = "Ăn uống" if is_food_pref(pref) else "Vui chơi"
            detail = {"cuisine": pref}
            res = search_places_serpapi(
                center.get("lat", origin.get("lat")),
                center.get("lon", origin.get("lon")),
                radius_km,
                main_cat,
                detail,
                serpapi_key,
                min_rating=0.0,
                min_reviews=0,
                top_n=12,
            )
        except Exception:
            res = []
        filtered = [r for r in res if matches_pref(r, pref)]
        return filtered if filtered else res

    def choose_with_ollama(pref: str, cands: List[Dict], used_names: set) -> Optional[Dict]:
        short = cands[:5]
        if not short:
            return None
        short = [c for c in short if c.get("name") not in used_names] or short

        options_text = "\n".join(
            [
                f"{i+1}. {c.get('name','(không tên)')} — {c.get('address','')} — rating {c.get('rating','?')}"
                for i, c in enumerate(short)
            ]
        )
        prompt = (
            "Bạn là trợ lý du lịch. Chọn duy nhất 1 địa điểm phù hợp nhất với sở thích sau, ưu tiên đúng sở thích, rating cao, gần trung tâm.\n"
            f"Sở thích: {pref}\n"
            "Danh sách ứng viên:\n"
            f"{options_text}\n"
            "Chỉ trả về số thứ tự (ví dụ: 2)."
        )
        try:
            resp = ollama_chat([
                {"role": "system", "content": "Bạn là trợ lý du lịch, trả lời cực ngắn."},
                {"role": "user", "content": prompt},
            ], ollama_base, model)
            import re
            m = re.search(r"(\d+)", resp)
            if m:
                idx = int(m.group(1)) - 1
                if 0 <= idx < len(short):
                    return short[idx]
        except Exception:
            pass
        for c in short:
            if c.get("name") not in used_names:
                return c
        return short[0]

    def build_schedule(strategy_config) -> List[Dict]:
        """
        strategy_config: str hoặc list
        - str "rating": toàn bộ ưu tiên đánh giá
        - str "distance": toàn bộ ưu tiên quãng đường
        - list (e.g. ["rating", "distance"]): áp dụng rating cho nửa đầu, distance cho nửa sau
        """
        schedule: List[Dict] = []
        current_start = t0
        last_eat_time = None
        used_names = set()

        for idx, pref_goal in enumerate(pref_list):
            cands = fetch_pref_candidates(pref_goal)
            if not cands:
                raise ValueError(f"Không tìm thấy địa điểm cho sở thích: {pref_goal}")

            if isinstance(strategy_config, list):
                mid = len(pref_list) // 2
                strategy = strategy_config[0] if idx < mid else strategy_config[1]
            else:
                strategy = strategy_config

            if strategy == "rating":
                cands = sorted(cands, key=lambda x: (-(x.get("rating") or 0), -(x.get("reviews") or 0)))
            elif strategy == "distance":
                cands = sorted(cands, key=lambda x: x.get("distance_km", 9999))

            place = choose_with_ollama(pref_goal, cands, used_names)
            if not place:
                raise ValueError(f"Không chọn được địa điểm cho: {pref_goal}")

            is_eat = is_food_pref(pref_goal)
            if is_eat and last_eat_time is not None:
                gap = (current_start - last_eat_time).total_seconds() / 3600.0
                if gap < 4:
                    delta = timedelta(hours=4 - gap)
                    current_start = current_start + delta

            end_time = current_start + timedelta(minutes=block_minutes)
            if end_time > t1:
                raise ValueError("Không đủ thời gian để xếp các sở thích (yêu cầu giãn cách 4h cho ăn uống).")

            schedule.append({
                "start": current_start.strftime("%H:%M"),
                "end": end_time.strftime("%H:%M"),
                "goal": pref_goal,
                "place": place,
            })

            used_names.add(place.get("name"))
            if is_eat:
                last_eat_time = current_start
            current_start = end_time

        return schedule

    schedules: List[List[Dict]] = []

    for strategy in [["rating", "distance"], "rating", "distance"]:
        try:
            sched = build_schedule(strategy)
            schedules.append(sched)
        except Exception:
            continue

    if not schedules:
        raise ValueError("Không tạo được lịch trình nào. Hãy thử nới thời gian hoặc giảm số sở thích.")

    try:
        places_text = "; ".join([f"{b['place'].get('name','')}" for b in schedules[0]])
        system = {"role": "system", "content": "Bạn là trợ lý lập kế hoạch du lịch ngắn gọn bằng tiếng Việt."}
        user_msg = {
            "role": "user",
            "content": f"Tóm tắt ngắn gọn (2-3 câu) về lịch trình này: {places_text}",
        }
        note = ollama_chat([system, user_msg], ollama_base, model)
    except Exception:
        note = ""

    return schedules, note

# =======================
# UI CHÍNH
# ======================="
st.title("🗺️ Smart Tourism System")


# -------- SIDEBAR --------
with st.sidebar:
    st.header("⚙️ Cấu hình")
    ollama_base = st.text_input("OLLAMA_API_BASE", value=DEFAULT_OLLAMA_BASE)
    model = st.text_input("Model", value="llama3.2:1b")

    serpapi_key = st.text_input(
        "SERPAPI_KEY",
        value=DEFAULT_SERPAPI_KEY,
        type="password",
    )

    st.divider()
    st.subheader("Vị trí gốc")
    locate_method = st.radio("Chọn cách nhập", ["Nhập địa chỉ", "Chọn trên bản đồ"], horizontal=True)
    #radius_km = st.slider("Bán kính tìm kiếm (km)", 1, 20, 10)

    default_center = [21.0278, 105.8342]  # Hà Nội

    # ----- Nhập địa chỉ -----
    if locate_method == "Nhập địa chỉ":
        addr = st.text_input("Địa chỉ", value="Hà Nội")
        if st.button("📍 Lấy địa chỉ"):
            try:
                lat, lon, disp = geocode(addr)
                st.session_state["origin"] = {
                    "lat": lat, "lon": lon,
                    "name": disp,
                    "latlon" : [lat, lon],
                }
                
                origin = {
                    "lat": lat,
                    "lon": lon,
                    "name": disp,
                    "latlon": [lat, lon],
                }
                
                st.session_state["origin"] = origin
                # 🔁 trung tâm tìm kiếm ban đầu = origin
                st.session_state["search_center"] = {
                    "lat": lat,
                    "lon": lon,
                    "name": disp,
                }
                st.success(f"Đã xác định: {disp}")
            except Exception as e:
                st.error(str(e))

    # ----- Chọn vị trí trên bản đồ -----
    else:
        st.caption("Nhấp chuột vào vị trí cần chọn rồi nhấn nút Lấy vị trí.")
        m = folium.Map(
            location=st.session_state.get("origin", {}).get("latlon", default_center)
            if isinstance(st.session_state.get("origin"), dict)
            else default_center,
            zoom_start=13,
        )

        orig = st.session_state.get("origin")
        if isinstance(orig, dict) and "lat" in orig and "lon" in orig:
            loc = orig.get("latlon") or [orig["lat"], orig["lon"]]
            folium.Marker(
                loc,
                popup="Vị trí hiện tại",
            ).add_to(m)

        map_state = st_folium(m, height=300, returned_objects=["last_clicked", "center"])

        if st.button("📍 Lấy địa chỉ"):
            if map_state:
                click = map_state.get("last_clicked")
                center = map_state.get("center")
                if click:
                    lat, lon = click["lat"], click["lng"]
                elif center:
                    lat, lon = center["lat"], center["lng"]
                else:
                    lat, lon = default_center

                try:
                    disp = reverse_geocode(lat, lon)
                except:
                    disp = f"{lat:.5f},{lon:.5f}"

                origin = {
                    "lat": lat,
                    "lon": lon,
                    "name": disp,
                    "latlon": [lat, lon],
                }
                st.session_state["origin"] = origin
                st.session_state["search_center"] = {
                    "lat": lat,
                    "lon": lon,
                    "name": disp,
                }
                st.success(f"Đã chọn: {disp}")

    radius_km = st.slider("Bán kính tìm kiếm (km)", 1, 20, 10)


    # Initialize session state for schedule mode
    if "schedule_mode" not in st.session_state:
        st.session_state["schedule_mode"] = "Nhập từng cái"

    # Manual mode filters will be shown conditionally in the schedule section below
    min_rating = 0.0
    price_range = (0, 500000)
    detail_filters = {"cuisine": ""}
    fetch_price_details = False


# =======================
# TẠO LỊCH TRÌNH
# =======================
st.subheader("🗂️ Lịch trình")

if "itin_name" not in st.session_state:
    st.session_state["itin_name"] = "Đi chơi sáng"
if "schedule" not in st.session_state:
    st.session_state["schedule"] = []
if "schedule_mode" not in st.session_state:
    st.session_state["schedule_mode"] = "Nhập từng cái"

itin_name = st.text_input("Tên lịch trình", value=st.session_state["itin_name"])
st.session_state["itin_name"] = itin_name

# Mode selector
mode = st.radio("Chế độ tạo lịch", ["Nhập từng cái", "Tự động tạo lịch"], horizontal=True)

# Reset schedule when switching modes so timeline starts empty
if mode == "Tự động tạo lịch" and st.session_state.get("schedule_mode") != "Tự động tạo lịch":
    st.session_state["schedule"] = []
elif mode == "Nhập từng cái" and st.session_state.get("schedule_mode") != "Nhập từng cái":
    st.session_state["schedule"] = []

st.session_state["schedule_mode"] = mode

if mode == "Nhập từng cái":
    # Show manual filters in sidebar for manual mode
    with st.sidebar:
        st.divider()
        st.subheader("Nhập mong muốn của bạn!")
        cuisine = st.text_input("(Ăn gì/Đi chơi ở đâu/Làm gì)", value="", key="sidebar_cuisine")
        detail_filters = {"cuisine": cuisine.strip()}

        min_rating = st.slider("Rating tối thiểu", 0.0, 5.0, 0.0, 0.1, key="sidebar_rating")
        min_reviews = 0

        price_range = st.slider(
            "Khoảng giá (₫)",
            0,
            2000000,
            (0, 500000),
            step=10000,
            key="sidebar_price"
        )
        fetch_price_details = False
    
    # Manual mode UI
    colA, colB = st.columns([2, 1])
    
    with colA:
        start_time = st.time_input("Bắt đầu", datetime.strptime("6:00", "%H:%M").time(), key="manual_start_main")
        end_time = st.time_input("Kết thúc", datetime.strptime("7:00", "%H:%M").time(), key="manual_end_main")
        goal = st.text_input("Mục tiêu", value="Ăn sáng", key="manual_goal_main")

        def add_block(start, end, goal):
            if st.session_state["schedule"]:
                last = st.session_state["schedule"][-1]["end"]
                if start <= last:
                    st.warning("Khung giờ mới phải sau khung giờ cuối.")
                    return
            st.session_state["schedule"].append(
                {"start": start, "end": end, "goal": goal, "place": None}
            )

    with colB:
        if st.button("➕ Thêm khung giờ", key="btn_add_manual"):
            add_block(
                start_time.strftime("%H:%M"),
                end_time.strftime("%H:%M"),
                goal,
            )

else:
    # Auto mode: time range + preferences
    st.markdown("Tự động tạo lịch: SerpAPI tìm địa điểm, Ollama chọn phù hợp nhất cho từng sở thích")
    auto_start = st.time_input("Bắt đầu", datetime.strptime("9:00", "%H:%M").time(), key="auto_start_main")
    auto_end = st.time_input("Kết thúc", datetime.strptime("12:00", "%H:%M").time(), key="auto_end_main")
    
    # Dynamic preferences list
    if "auto_prefs" not in st.session_state:
        st.session_state["auto_prefs"] = [""]
    
    st.markdown("**Sở thích của bạn** (ví dụ: 'ăn sáng', 'cà phê', 'tham quan')")
    pref_inputs = []
    for i in range(len(st.session_state["auto_prefs"])):
        col1, col2 = st.columns([4, 1])
        with col1:
            val = st.text_input(f"Sở thích #{i+1}", value=st.session_state["auto_prefs"][i], key=f"auto_pref_{i}")
        pref_inputs.append(val)
        with col2:
            if st.button("✕", key=f"del_pref_{i}"):
                st.session_state["auto_prefs"].pop(i)
                st.rerun()
    
    if st.button("➕ Thêm sở thích", key="btn_add_pref"):
        st.session_state["auto_prefs"].append("")
        st.rerun()
    
    # Generate button
    if st.button("🚀 Tạo lịch tự động", key="btn_generate"):
        prefs = [v.strip() for v in pref_inputs if v and v.strip()]
        if not prefs:
            st.warning("Vui lòng nhập ít nhất một sở thích.")
        else:
            with st.spinner("Đang tạo lịch tự động..."):
                try:
                    origin = st.session_state.get("origin", {})
                    center = st.session_state.get("search_center", origin)
                    schedules, note = auto_generate_schedule_variants(
                        auto_start.strftime("%H:%M"),
                        auto_end.strftime("%H:%M"),
                        prefs,
                        origin,
                        center,
                        radius_km,
                        serpapi_key,
                        ollama_base,
                        model,
                    )
                    st.session_state["_auto_schedules"] = schedules
                    st.session_state["_auto_note"] = note
                except Exception as e:
                    st.error(str(e))
    
    # Display generated schedule(s)
    if st.session_state.get("_auto_schedules"):
        st.divider()
        st.markdown("### Lịch trình được tạo")
        
        schedules = st.session_state["_auto_schedules"]
        
        for idx, sched in enumerate(schedules, 1):
            st.markdown(f"**Lịch trình #{idx}**")
            
            # Show blocks and places
            for blk in sched:
                place = blk.get("place") or {}
                st.write(f"**{blk['start']}–{blk['end']}**: {blk['goal']} — *{place.get('name', '(Không tên)')}*")
                if place.get("address"):
                    st.caption(f"📍 {place['address']}")
            
            # Map for this schedule
            if sched:
                with st.expander(f"Xem bản đồ lịch trình #{idx}", expanded=False):
                    origin = st.session_state.get("origin", {})
                    m_auto = folium.Map(
                        location=[origin.get("lat", 21.0278), origin.get("lon", 105.8342)],
                        zoom_start=13
                    )
                    
                    # Origin marker (numbered 1)
                    if origin:
                        origin_icon = folium.DivIcon(
                            html=f"""
                            <div style="display:flex;align-items:center;justify-content:center;width:28px;height:28px;border-radius:50%;background:#2ecc71;color:white;font-weight:bold;">
                                1
                            </div>
                            """
                        )
                        folium.Marker(
                            [origin["lat"], origin["lon"]],
                            popup="Vị trí gốc",
                            icon=origin_icon,
                        ).add_to(m_auto)
                    
                    # Place markers and draw routes
                    waypoints = []
                    if origin:
                        waypoints.append(origin)
                    
                    for blk in sched:
                        if blk.get("place"):
                            waypoints.append(blk["place"])
                    
                    # Draw routes between waypoints
                    for i in range(len(waypoints) - 1):
                        try:
                            a = waypoints[i]
                            b = waypoints[i + 1]
                            geom, km, hrs = osrm_geom(a["lon"], a["lat"], b["lon"], b["lat"])
                            coords = [(lat, lon) for lon, lat in geom["coordinates"]]
                            folium.PolyLine(coords, weight=5, color="blue").add_to(m_auto)
                        except Exception:
                            pass
                    
                    # Place markers with numbers
                    for i, blk in enumerate(sched, 1):
                        place = blk.get("place", {})
                        if place.get("lat") and place.get("lon"):
                            stop_icon = folium.DivIcon(
                                html=f"""
                                <div style="display:flex;align-items:center;justify-content:center;width:28px;height:28px;border-radius:50%;background:#e74c3c;color:white;font-weight:bold;">
                                    {i + 1}
                                </div>
                                """
                            )
                            folium.Marker(
                                [place["lat"], place["lon"]],
                                popup=f"{i + 1}. {place.get('name', '')}",
                                icon=stop_icon,
                            ).add_to(m_auto)
                    
                    st_folium(m_auto, height=400, key=f"map_auto_{idx}")
            
            # Select button
            if st.button(f"✅ Chọn lịch trình #{idx}", key=f"select_auto_{idx}"):
                st.session_state["schedule"] = sched
                st.session_state["_auto_schedules"] = None
                st.success(f"Đã chọn lịch trình #{idx}. Bạn có thể lưu hoặc chỉnh sửa nó.")
                st.rerun()
            
            st.divider()


# =======================
# TÌM ĐỊA ĐIỂM (SERPAPI) - MANUAL MODE ONLY
# =======================
if st.session_state.get("schedule_mode") == "Nhập từng cái":
    st.subheader("🔎 Tìm địa điểm (SerpAPI)")
    origin = st.session_state.get("origin")

    if not origin:
        st.info("Hãy chọn vị trí gốc ở sidebar.")
        st.stop()
    else:
        # 🔁 Trung tâm tìm kiếm hiện tại: ưu tiên địa điểm vừa được gán vào khung giờ (nếu có)
        # Nếu không có, dùng `search_center` (do người dùng đặt) hoặc fallback về `origin`.
        schedule = st.session_state.get("schedule", [])
        last_assigned_place = None
        if schedule:
            for blk in reversed(schedule):
                if blk.get("place") and blk["place"].get("lat") and blk["place"].get("lon"):
                    last_assigned_place = blk["place"]
                    break

        if last_assigned_place:
            # Khi người dùng vừa gán quán vào khung giờ, dùng quán đó làm tâm tìm kiếm
            st.session_state["search_center"] = {
                "lat": last_assigned_place["lat"],
                "lon": last_assigned_place["lon"],
                "name": last_assigned_place.get("name", "Địa điểm đã chọn"),
            }

        center = st.session_state.get("search_center", origin)
        st.write(f"**Trung tâm tìm kiếm hiện tại**: {center['name']}")
        st.write(f"**Bán kính**: {radius_km} km")

        if st.button("Tìm địa điểm"):
            if not serpapi_key:
                st.error("Chưa nhập SERPAPI_KEY.")
            else:
                with st.spinner("Đang tìm trên Google Maps..."):
                    try:
                        results = search_places_serpapi(
                            center["lat"], center["lon"], radius_km,
                            "Ăn uống", detail_filters, serpapi_key,
                            min_rating=min_rating, min_reviews=min_reviews, top_n=10,
                            price_range=price_range,
                            fetch_price_details=fetch_price_details,
                        )
                        st.session_state["results"] = results
                        st.success(f"Tìm thấy {len(results)} địa điểm.")
                    except Exception as e:
                        st.error(str(e))


        # ----- MAP -----
        m = folium.Map(location=[center["lat"], center["lon"]], zoom_start=13)

        # origin A (màu xanh)
        if origin:
            folium.Marker(
                [origin["lat"], origin["lon"]],
                popup="Vị trí gốc",
                icon=folium.Icon(color="green")
        ).add_to(m)


        # trung tâm tìm kiếm hiện tại (có thể là B, C,...)
        folium.Circle(
        location=[center["lat"], center["lon"]],
        radius=radius_km * 1000,
        fill=True,
        color="#3186cc",
        fill_opacity=0.1,
    ).add_to(m)
    folium.Marker(
        [center["lat"], center["lon"]],
        popup="Trung tâm tìm kiếm",
        icon=folium.Icon(color="orange"),
    ).add_to(m)


    results = st.session_state.get("results", [])
    for i, r in enumerate(results):
        price_val = r.get('price')
        if price_val is None:
            price_str = "Không rõ"
        else:
            try:
                price_str = f"₫{int(price_val):,}"
            except Exception:
                price_str = str(price_val)

        popup = f"""
        <b>{i+1}. {r['name']}</b><br>
        ⭐ {r.get('rating', '?')} ({r.get('reviews', '?')} review)<br>
        Giá trung bình: {price_str}<br>
        {r['address']}<br>
        {r['distance_km']:.1f} km
        """
        folium.Marker([r["lat"], r["lon"]], popup=popup).add_to(m)

    st_folium(m, height=400)

    # ----- Danh sách gợi ý -----
    if results:
        st.markdown("### 📋 Danh sách gợi ý (Top theo rating)")

        table = []
        for i, r in enumerate(results, 1):
            table.append({
                "STT": i,
                "Tên": r["name"],
                "Rating": r["rating"],
                "Reviews": r["reviews"],
                "Giá trung bình (₫)": (f"{int(r['price']):,}" if r.get("price") is not None else "Không rõ"),
                "Khoảng cách": round(r["distance_km"], 1),
                "Địa chỉ": r["address"],
            })

        st.dataframe(table, hide_index=True)

        choice = st.selectbox(
            "Chọn địa điểm để gán vào khung giờ cuối",
            ["(Không)"] + [f"{i+1}. {r['name']}" for i, r in enumerate(results)]
        )

        if st.button("📌 Gán địa điểm"):
            if choice != "(Không)" and st.session_state["schedule"]:
                idx = int(choice.split(".")[0]) - 1
                chosen_place = results[idx]
                st.session_state["schedule"][-1]["place"] = chosen_place

                # 🔁 Từ giờ trở đi, tâm tìm kiếm = địa điểm vừa chọn
                st.session_state["search_center"] = {
                    "lat": chosen_place["lat"],
                    "lon": chosen_place["lon"],
                    "name": chosen_place["name"],
                }

                st.success(f"Đã gán vào khung giờ cuối và đặt '{chosen_place['name']}' làm trung tâm tìm kiếm tiếp theo.")



# =======================
# HIỂN THỊ TIMELINE
# =======================
st.subheader("🕒 Timeline")

if not st.session_state["schedule"]:
    st.info("Chưa có khung giờ.")
else:
    for blk in st.session_state["schedule"]:
        place = blk.get("place")
        c1, c2, c3 = st.columns([1, 3, 4])

        with c1:
            st.write(f"**{blk['start']}–{blk['end']}**")

        with c2:
            st.write(f"**{blk['goal']}**")

        with c3:
            if place:
                st.write(f"**{place['name']}** — ⭐ {place.get('rating','?')}")
                st.caption(f"{place.get('address','')} — {place.get('distance_km',0):.1f} km")
            else:
                st.caption("_Chưa chọn địa điểm_")


# =======================
# LƯU LỊCH TRÌNH
# =======================
if "saved_itineraries" not in st.session_state:
    st.session_state["saved_itineraries"] = []

st.subheader("💾 Lưu Lịch Trình")

if st.button("Lưu lịch trình"):
    origin = st.session_state.get("origin")
    if origin and st.session_state["schedule"]:
        st.session_state["saved_itineraries"].append(
            {
                "name": st.session_state["itin_name"],
                "origin": origin,
                "schedule": st.session_state["schedule"].copy(),
            }
        )
        st.success("Đã lưu.")
    else:
        st.warning("Thiếu vị trí gốc hoặc khung giờ.")


# =======================
# DANH SÁCH LỊCH TRÌNH ĐÃ LƯU
# =======================
st.subheader("📚 Lịch trình đã lưu")

for i, it in enumerate(st.session_state["saved_itineraries"], 1):
    with st.expander(f"{i}. {it['name']}"):
        st.write(f"**Vị trí gốc:** {it['origin']['name']}")

        rows = []
        for blk in it["schedule"]:
            p = blk.get("place") or {}
            rows.append({
                "Bắt đầu": blk["start"],
                "Kết thúc": blk["end"],
                "Mục tiêu": blk["goal"],
                "Địa điểm": p.get("name", ""),
                "Địa chỉ": p.get("address", ""),
            })
        st.table(rows)

        if st.button(f"📥 Tải lịch trình này", key=f"load_{i}"):
            st.session_state["origin"] = it["origin"]
            st.session_state["schedule"] = it["schedule"].copy()
            st.session_state["itin_name"] = it["name"]
            st.success("Đã tải lịch trình.")


# =======================
# XUẤT FILE
# =======================
st.subheader("⬇️ Xuất lịch trình")

origin = st.session_state.get("origin")
if origin and st.session_state["schedule"]:
    export_type = st.selectbox("Định dạng", ["JSON", "CSV", "TXT"])
    data = serialize_itinerary(st.session_state["itin_name"], origin, st.session_state["schedule"])

    if export_type == "JSON":
        st.download_button(
            "📥 Tải JSON",
            json.dumps(data, ensure_ascii=False, indent=2),
            file_name="itinerary.json",
        )

    elif export_type == "CSV":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["start","end","goal","place","address"])
        for blk in st.session_state["schedule"]:
            p = blk.get("place") or {}
            w.writerow([blk["start"],blk["end"],blk["goal"],p.get("name",""),p.get("address","")])
        st.download_button("📥 Tải CSV", buf.getvalue(), file_name="itinerary.csv")

    else:
        lines = [f"# {st.session_state['itin_name']}"]
        for blk in st.session_state["schedule"]:
            p = blk.get("place") or {}
            place_name = p.get('name','?')
            place_addr = p.get('address','')
            if place_addr:
                lines.append(f"- {blk['start']}–{blk['end']}: {blk['goal']} tại {place_name} — {place_addr}")
            else:
                lines.append(f"- {blk['start']}–{blk['end']}: {blk['goal']} tại {place_name}")
        st.download_button("📥 Tải TXT", "\n".join(lines), file_name="itinerary.txt")


# =======================
# VẼ LỘ TRÌNH A→B→C
# =======================
st.subheader("🧭 Tuyến đường")

places = [b["place"] for b in st.session_state["schedule"] if b.get("place")]
if origin and places:
    try:
        segs = build_route_segments(origin, st.session_state["schedule"])
        m2 = folium.Map(location=[origin["lat"], origin["lon"]], zoom_start=12)

        # origin marker with index 1
        origin_icon = folium.DivIcon(
            html=f"""
            <div style="display:flex;align-items:center;justify-content:center;width:28px;height:28px;border-radius:50%;background:#2ecc71;color:white;font-weight:bold;">
                1
            </div>
            """
        )
        folium.Marker([origin["lat"], origin["lon"]], popup=f"1. {origin.get('name','Start')}", icon=origin_icon).add_to(m2)

        total_km = 0
        for idx, s in enumerate(segs, start=1):
            total_km += s["km"]
            coords = [(lat, lon) for lon, lat in s["geom"]["coordinates"]]
            folium.PolyLine(coords, weight=5, color="blue").add_to(m2)

            end_lat, end_lon = coords[-1]
            num = idx + 1  # numbering: origin=1, first segment end=2, ...
            stop_icon = folium.DivIcon(
                html=f"""
                <div style="display:flex;align-items:center;justify-content:center;width:28px;height:28px;border-radius:50%;background:#e74c3c;color:white;font-weight:bold;">
                    {num}
                </div>
                """
            )
            folium.Marker([end_lat, end_lon], popup=f"{num}. {s['to']}", icon=stop_icon).add_to(m2)

        st_folium(m2, height=400)
        st.success(f"Tổng quãng đường: {total_km:.1f} km")
    except Exception as e:
        st.error(str(e))


# =======================
# Chatbot Ollama
# =======================
st.subheader("💬 Chatbot gợi ý")

if "chat" not in st.session_state:
    st.session_state["chat"] = [
        {"role": "system", "content": "You are a helpful travel assistant."}
    ]

# Hiển thị lịch sử
for msg in st.session_state["chat"]:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            content = msg.get("content")
            if msg.get("role") == "assistant":
                content = sanitize_assistant_text(content)
            st.write(content)

# Input người dùng
txt = st.chat_input("Nhập câu hỏi về du lịch...")
if txt:
    st.session_state["chat"].append({"role": "user", "content": txt})
    with st.chat_message("user"):
        st.write(txt)

        # Build a temporary context message that provides location info
    # Prefer the place assigned to the last schedule block; otherwise use origin.
    loc_name = None
    loc_lat = None
    loc_lon = None
    origin = st.session_state.get("origin")
    schedule = st.session_state.get("schedule", [])
    if schedule:
        last_place = schedule[-1].get("place")
        if last_place and last_place.get("lat") and last_place.get("lon"):
            loc_name = last_place.get("name")
            loc_lat = last_place.get("lat")
            loc_lon = last_place.get("lon")
    if loc_name is None and origin:
        loc_name = origin.get("name")
        loc_lat = origin.get("lat")
        loc_lon = origin.get("lon")

    if loc_name is None:
        # No location available — inform the user and proceed without context
        with st.chat_message("assistant"):
            msg = "(Gợi ý) Tôi chưa có vị trí gốc hoặc địa điểm nào trong lịch trình. Vui lòng chọn vị trí để có gợi ý chính xác.\n" + "Đang gửi câu hỏi cho chatbot không có bối cảnh vị trí..."
            st.write(msg)
        # call without added context
        messages_for_call = list(st.session_state["chat"])
    else:
        # create a system message with explicit location search instructions (not persisted)
        loc_text = (
            f"Bạn là trợ lý du lịch chuyên gợi ý địa điểm gần vị trí cho trước. "
            f"Hãy dùng tọa độ sau để tìm các quán/cửa hàng quanh đó và trả lời ngắn gọn bằng tiếng Việt:\n"
            f"- Vị trí tham chiếu: {loc_name}\n"
            f"- Tọa độ: lat={loc_lat}, lon={loc_lon}\n"
            "Yêu cầu khi trả lời:\n"
            "1) Không suy diễn tên tỉnh/thành hay giới thiệu hành chính (ví dụ 'Long Xuyên thuộc tỉnh...') — chỉ dùng tọa độ để tìm địa điểm gần đó.\n"
            "2) Trả về danh sách tối đa 5 quán phù hợp (tên, địa chỉ ngắn, khoảng cách ước tính, rating nếu có).\n"
            "3) Nếu cần hỏi thêm (ví dụ muốn loại hình, khoảng giá), đặt một câu hỏi ngắn gọn để xác nhận.\n"
            "4) Trả lời ngắn gọn, rõ ràng, không thêm phần tử thừa."
        )
        messages_for_call = list(st.session_state["chat"]) + [{"role": "system", "content": loc_text}]

    # Call Ollama with the temporary messages (location context included)
    with st.chat_message("assistant"):
        reply = ollama_chat(messages_for_call, ollama_base, model)
        reply_s = sanitize_assistant_text(reply)
        st.session_state["chat"].append({"role": "assistant", "content": reply_s})
        st.write(reply_s)

st.caption("⚡ Tìm kiếm: SerpAPI — Bản đồ: OSM — Route: OSRM — Chat: Ollama")


