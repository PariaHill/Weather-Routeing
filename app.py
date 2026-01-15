import streamlit as st
import gpxpy
import requests
from datetime import datetime, timedelta, timezone
import math
import pandas as pd
from typing import List, Tuple, Dict, Optional
import json
import folium
from streamlit_folium import st_folium
from folium.plugins import AntPath
import tempfile
import os

# GRIB2 처리용 (NOAA GFS)
try:
    import xarray as xr
    import cfgrib
    GRIB_AVAILABLE = True
except ImportError:
    GRIB_AVAILABLE = False

# Page config - must be first Streamlit command
st.set_page_config(page_title="Weather Routing Calculator", layout="wide")

# Initialize localStorage (after set_page_config)
try:
    from streamlit_local_storage import LocalStorage
    local_storage = LocalStorage()
    LOCAL_STORAGE_AVAILABLE = True
except:
    LOCAL_STORAGE_AVAILABLE = False

def load_from_storage(key: str, default):
    """localStorage에서 값 로드"""
    if not LOCAL_STORAGE_AVAILABLE:
        return default
    try:
        value = local_storage.getItem(key)
        if value is not None:
            return value
    except:
        pass
    return default

def save_to_storage(key: str, value):
    """localStorage에 값 저장"""
    if not LOCAL_STORAGE_AVAILABLE:
        return
    try:
        local_storage.setItem(key, value, key=f"save_{key}")
    except:
        pass

# 선종별 경험적 파라미터
VESSEL_TYPE_PARAMS = {
    'Bulk Carrier': {
        'windage_front_ratio': 0.08,   # 정면 풍압면적 / (LOA * Depth)
        'windage_side_ratio': 0.35,    # 측면 풍압면적 / (LOA * Depth)
        'typical_cb': 0.85,            # 일반적인 방형비척계수
        'wave_resistance_factor': 1.0  # 파랑저항 계수
    },
    'Tanker': {
        'windage_front_ratio': 0.06,
        'windage_side_ratio': 0.30,
        'typical_cb': 0.82,
        'wave_resistance_factor': 0.95
    },
    'Container': {
        'windage_front_ratio': 0.12,
        'windage_side_ratio': 0.50,
        'typical_cb': 0.65,
        'wave_resistance_factor': 1.1
    },
    'Special Purpose': {
        'windage_front_ratio': 0.15,   # Cable layer, Survey vessel 등
        'windage_side_ratio': 0.55,
        'typical_cb': 0.70,
        'wave_resistance_factor': 1.15
    },
    'RoRo': {
        'windage_front_ratio': 0.18,
        'windage_side_ratio': 0.60,
        'typical_cb': 0.60,
        'wave_resistance_factor': 1.2
    },
    'General Cargo': {
        'windage_front_ratio': 0.10,
        'windage_side_ratio': 0.40,
        'typical_cb': 0.75,
        'wave_resistance_factor': 1.05
    }
}

class VesselData:
    """선박 제원 데이터"""
    def __init__(self, vessel_type, displacement, windage_area_side, 
                 loa, breadth, draft, speed_knots):
        self.vessel_type = vessel_type
        self.displacement = displacement  # 톤
        self.loa = loa  # m
        self.breadth = breadth  # m
        self.draft = draft  # m
        self.speed_knots = speed_knots  # 노트
        
        # 선종별 파라미터 가져오기
        params = VESSEL_TYPE_PARAMS.get(vessel_type, VESSEL_TYPE_PARAMS['General Cargo'])
        self.wave_resistance_factor = params['wave_resistance_factor']
        
        # Depth 추정 (Draft의 약 1.5~2배)
        estimated_depth = draft * 1.8
        
        # 풍압면적 계산 (선종별 경험적 비율 사용)
        self.windage_area_front = params['windage_front_ratio'] * loa * estimated_depth
        self.windage_area_side = windage_area_side  # 측면은 사용자 입력 유지
        
        # 방형비척계수(Cb) 계산: Cb = Displacement / (L × B × d × ρ)
        # ρ = 1.025 (표준해수 비중)
        seawater_density = 1.025  # ton/m³
        underwater_volume = loa * breadth * draft  # m³ (이론적 최대 부피)
        calculated_cb = displacement / (underwater_volume * seawater_density)
        
        # Cb가 현실적인 범위(0.4~0.95)인지 확인
        if 0.4 <= calculated_cb <= 0.95:
            self.cb = calculated_cb
        else:
            # 비현실적인 값이면 선종별 기본값 사용
            self.cb = params['typical_cb']

class WeatherPoint:
    """기상 데이터 포인트"""
    def __init__(self, time, lat, lon, pressure=None, wind_dir=None, wind_speed=None,
                 wind_gust=None, wave_dir=None, wave_height=None, swell_dir=None, swell_height=None):
        self.time = time
        self.lat = lat
        self.lon = lon
        self.pressure = pressure
        self.wind_dir = wind_dir  # degrees, coming from
        self.wind_speed = wind_speed  # m/s
        self.wind_gust = wind_gust  # m/s
        self.wave_dir = wave_dir  # degrees, coming from
        self.wave_height = wave_height  # m
        self.swell_dir = swell_dir  # degrees, coming from
        self.swell_height = swell_height  # m

def parse_gpx(gpx_file) -> List[Tuple[float, float]]:
    """GPX 파일에서 포인트 추출 (트랙, 루트, 웨이포인트 모두 지원)"""
    try:
        # 파일 내용 읽기
        if hasattr(gpx_file, 'read'):
            content = gpx_file.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            gpx_file.seek(0)  # 파일 포인터 리셋
        
        gpx = gpxpy.parse(gpx_file)
    except Exception as e:
        st.error(f"GPX parsing error: {str(e)}")
        return []
    
    points = []
    
    # 1. 트랙 포인트 (tracks > segments > points)
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append((point.latitude, point.longitude))
    
    # 2. 트랙이 없으면 루트 포인트 시도 (routes > points)
    if not points:
        for route in gpx.routes:
            for point in route.points:
                points.append((point.latitude, point.longitude))
    
    # 3. 루트도 없으면 웨이포인트 시도
    if not points:
        for waypoint in gpx.waypoints:
            points.append((waypoint.latitude, waypoint.longitude))
    
    return points

def calculate_distance(lat1, lon1, lat2, lon2) -> float:
    """두 지점 간 거리 계산 (해리)"""
    R = 3440.065  # 지구 반경 (해리)
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    distance = R * c
    return distance

def calculate_bearing(lat1, lon1, lat2, lon2) -> float:
    """두 지점 간 방위각 계산 (진방위, 0-360도)"""
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lon = math.radians(lon2 - lon1)
    
    y = math.sin(delta_lon) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360) % 360

def rhumb_line_destination(lat, lon, bearing, distance) -> Tuple[float, float]:
    """Rhumb line으로 목적지 계산"""
    R = 3440.065  # 해리
    lat_rad = math.radians(lat)
    bearing_rad = math.radians(bearing)
    
    delta_lat = distance * math.cos(bearing_rad)
    lat2_rad = lat_rad + delta_lat / R
    
    delta_psi = math.log(math.tan(lat2_rad/2 + math.pi/4) / math.tan(lat_rad/2 + math.pi/4))
    q = delta_lat / delta_psi if abs(delta_psi) > 1e-12 else math.cos(lat_rad)
    
    delta_lon = distance * math.sin(bearing_rad) / q
    lon2_rad = math.radians(lon) + delta_lon / R
    
    lat2 = math.degrees(lat2_rad)
    lon2 = math.degrees(lon2_rad)
    lon2 = ((lon2 + 540) % 360) - 180  # Normalize to -180 to 180
    
    return lat2, lon2

class TrackLine:
    """트랙 라인을 따라 위치를 계산하는 헬퍼 클래스"""
    
    def __init__(self, track_points: List[Tuple[float, float]]):
        self.track_points = track_points
        self.segment_distances = []
        self.cumulative_distances = [0]
        
        # 각 세그먼트 거리와 누적 거리 계산
        for i in range(len(track_points) - 1):
            dist = calculate_distance(track_points[i][0], track_points[i][1],
                                     track_points[i+1][0], track_points[i+1][1])
            self.segment_distances.append(dist)
            self.cumulative_distances.append(self.cumulative_distances[-1] + dist)
        
        self.total_distance = self.cumulative_distances[-1]
    
    def get_position_at_distance(self, distance: float) -> Tuple[float, float, float]:
        """
        트랙 상의 주어진 거리에서의 위치와 heading 반환 (선형 보간 사용)
        Returns: (lat, lon, heading)
        """
        # 출발점
        if distance <= 0:
            heading = calculate_bearing(self.track_points[0][0], self.track_points[0][1],
                                       self.track_points[1][0], self.track_points[1][1])
            return self.track_points[0][0], self.track_points[0][1], heading
        
        # 도착점
        if distance >= self.total_distance:
            heading = calculate_bearing(self.track_points[-2][0], self.track_points[-2][1],
                                       self.track_points[-1][0], self.track_points[-1][1])
            return self.track_points[-1][0], self.track_points[-1][1], heading
        
        # 해당 거리가 속한 세그먼트 찾기
        for i in range(len(self.cumulative_distances) - 1):
            seg_start_dist = self.cumulative_distances[i]
            seg_end_dist = self.cumulative_distances[i + 1]
            
            if distance <= seg_end_dist:
                # 이 세그먼트 안에 위치
                segment_length = self.segment_distances[i]
                distance_in_segment = distance - seg_start_dist
                
                # 선형 보간 비율 (0.0 ~ 1.0)
                if segment_length > 0:
                    ratio = distance_in_segment / segment_length
                else:
                    ratio = 0
                
                # 시작점과 끝점
                start_lat, start_lon = self.track_points[i]
                end_lat, end_lon = self.track_points[i + 1]
                
                # 선형 보간으로 위치 계산 (정확히 트랙 위)
                lat = start_lat + ratio * (end_lat - start_lat)
                lon = start_lon + ratio * (end_lon - start_lon)
                
                # Heading
                heading = calculate_bearing(start_lat, start_lon, end_lat, end_lon)
                
                return lat, lon, heading
        
        # fallback (도착점)
        heading = calculate_bearing(self.track_points[-2][0], self.track_points[-2][1],
                                   self.track_points[-1][0], self.track_points[-1][1])
        return self.track_points[-1][0], self.track_points[-1][1], heading

def calculate_dr_on_track(track: TrackLine, start_time: datetime, 
                          speed_knots: float, interval_hours: int) -> List[Dict]:
    """
    Step 1 & 2: 정해진 속도로 트랙을 따라 DR 위치 계산
    """
    dr_positions = []
    current_time = start_time
    distance_sailed = 0
    
    # 출발점
    lat, lon, heading = track.get_position_at_distance(0)
    dr_positions.append({
        'time': current_time,
        'lat': lat,
        'lon': lon,
        'distance_sailed': 0,
        'distance_remaining': track.total_distance,
        'heading': heading
    })
    
    # interval_hours 간격으로 위치 계산
    while distance_sailed < track.total_distance:
        current_time += timedelta(hours=interval_hours)
        distance_sailed += speed_knots * interval_hours
        
        if distance_sailed >= track.total_distance:
            distance_sailed = track.total_distance
        
        lat, lon, heading = track.get_position_at_distance(distance_sailed)
        
        dr_positions.append({
            'time': current_time,
            'lat': lat,
            'lon': lon,
            'distance_sailed': distance_sailed,
            'distance_remaining': track.total_distance - distance_sailed,
            'heading': heading
        })
        
        if distance_sailed >= track.total_distance:
            break
    
    return dr_positions

def fetch_weather_for_positions(dr_positions: List[Dict], api_key: str, 
                                 start_time: datetime) -> List[Dict]:
    """
    Step 3 & 5: DR 위치들의 기상 데이터 조회 (NOAA GFS 사용)
    NOAA GFS는 최대 384시간(16일) 예보 제공
    api_key 파라미터는 호환성을 위해 유지 (NOAA는 키 불필요)
    """
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # NOAA GFS 예보 한계 (약 16일 = 384시간)
    FORECAST_LIMIT_HOURS = 384
    
    # GFS 사이클 캐시 (한 번만 찾기)
    gfs_cache = {}
    
    # GRIB 라이브러리 가용성 체크
    if not GRIB_AVAILABLE:
        st.warning("⚠️ GRIB libraries (xarray, cfgrib) not available. Install with: pip install xarray cfgrib eccodes")
    
    for i, point in enumerate(dr_positions):
        status_text.text(f"Fetching NOAA GFS data: {i+1}/{len(dr_positions)}")
        progress_bar.progress((i + 1) / len(dr_positions))
        
        # 출발 시간 대비 경과 시간 계산
        hours_from_start = (point['time'] - start_time).total_seconds() / 3600
        
        if hours_from_start <= FORECAST_LIMIT_HOURS and GRIB_AVAILABLE:
            # 예보 범위 내: NOAA GFS 조회
            weather_data = get_noaa_weather(point['lat'], point['lon'], 
                                           point['time'], gfs_cache)
            weather = parse_noaa_data(weather_data, point['time'])
            point['weather'] = weather
            point['weather_available'] = True
            point['gfs_cycle'] = weather_data.get('cycle', 'N/A')
            point['gfs_fhour'] = weather_data.get('fhour', 0)
            point['raw_weather_data'] = weather_data  # 디버그용
        else:
            # 예보 범위 초과 또는 GRIB 불가: NIL 처리
            point['weather'] = WeatherPoint(point['time'], point['lat'], point['lon'])
            point['weather_available'] = False
    
    progress_bar.empty()
    status_text.empty()
    
    return dr_positions

def recalculate_dr_with_weather(dr_positions: List[Dict], track: TrackLine,
                                vessel: VesselData, start_time: datetime,
                                interval_hours: int) -> List[Dict]:
    """
    Step 4: 기상 및 해류 영향을 반영하여 DR 재계산 (트랙 라인 위에서만)
    
    계산 흐름:
    1. 바람/파도 저항 → 실효 대수속력(STW) 계산
    2. RTOFS 해류 데이터 조회 → 대지속력(SOG) 계산
    3. SOG 기반으로 실제 이동 거리 및 ETA 계산
    
    기상 데이터가 없는 구간(weather_available=False)은 대수속력으로 계산
    """
    new_dr = []
    current_time = start_time
    distance_sailed = 0
    
    # 출발점 (기상 데이터 복사)
    lat, lon, heading = track.get_position_at_distance(0)
    new_dr.append({
        'time': current_time,
        'lat': lat,
        'lon': lon,
        'distance_sailed': 0,
        'distance_remaining': track.total_distance,
        'heading': heading,
        'weather': dr_positions[0].get('weather'),
        'weather_available': dr_positions[0].get('weather_available', True),
        'stw': vessel.speed_knots,  # Speed Through Water
        'sog': vessel.speed_knots,  # Speed Over Ground
        'speed_loss': 0,
        'current_effect': 0
    })
    
    # 각 구간별로 속도 계산하여 위치 재계산
    for i in range(1, len(dr_positions)):
        prev_point = new_dr[-1]
        orig_point = dr_positions[i]
        
        # 기상 데이터 가용 여부 확인
        weather_available = prev_point.get('weather_available', True)
        weather = prev_point.get('weather')
        
        # 해류 데이터 조회 (RTOFS)
        current_data = None
        current_effect = 0.0
        try:
            current_data = get_rtofs_current(prev_point['lat'], prev_point['lon'], 
                                            prev_point['time'])
        except:
            pass
        
        if weather_available and weather:
            # 기상 데이터 있음: 속도 손실 및 해류 영향 계산
            speed_loss, current_effect = calculate_speed_loss(vessel, weather, 
                                                              prev_point['heading'],
                                                              current_data)
        else:
            # 기상 데이터 없음
            speed_loss = 0
            # 해류만 있으면 해류 영향 계산
            if current_data and 'u_current' in current_data:
                u = current_data.get('u_current', 0)
                v = current_data.get('v_current', 0)
                heading_rad = math.radians(prev_point['heading'])
                current_along = u * math.sin(heading_rad) + v * math.cos(heading_rad)
                current_effect = current_along * 1.94384
        
        # 실효 대수속력 (STW) - 바람/파도 영향
        # speed_loss가 음수면 추진력으로 속력 증가
        stw = vessel.speed_knots - speed_loss
        stw = max(stw, 3.0)  # 최소 3노트 (조종 가능 속력)
        stw = min(stw, vessel.speed_knots * 1.05)  # 최대 5% 증가 제한
        
        # 대지속력 (SOG) - 해류 영향 추가
        sog = stw + current_effect
        sog = max(sog, 1.0)  # 최소 1노트 (극단적 역조에서도 전진)
        
        # 이 구간 동안 항해한 거리 (대지속력 기준)
        distance_this_interval = sog * interval_hours
        prev_distance = distance_sailed
        distance_sailed += distance_this_interval
        
        # 트랙 끝에 도달했는지 확인
        if distance_sailed >= track.total_distance:
            # 정확한 도착 시간 계산
            remaining_distance = track.total_distance - prev_distance
            time_to_arrival = remaining_distance / sog  # 시간 (hours)
            arrival_time = prev_point['time'] + timedelta(hours=time_to_arrival)
            
            # 도착점 추가
            lat, lon, heading = track.get_position_at_distance(track.total_distance)
            new_dr.append({
                'time': arrival_time,
                'lat': lat,
                'lon': lon,
                'distance_sailed': track.total_distance,
                'distance_remaining': 0,
                'heading': heading,
                'weather': orig_point.get('weather'),
                'weather_available': orig_point.get('weather_available', True),
                'stw': stw,
                'sog': sog,
                'actual_speed': sog,  # 호환성 유지
                'speed_loss': speed_loss,
                'current_effect': current_effect,
                'current_data': current_data
            })
            break
        
        # 트랙 상의 새 위치
        lat, lon, heading = track.get_position_at_distance(distance_sailed)
        
        # 시간도 재계산 (실제 속도 기반)
        current_time += timedelta(hours=interval_hours)
        
        new_dr.append({
            'time': current_time,
            'lat': lat,
            'lon': lon,
            'distance_sailed': distance_sailed,
            'distance_remaining': track.total_distance - distance_sailed,
            'heading': heading,
            'weather': orig_point.get('weather'),
            'weather_available': orig_point.get('weather_available', True),
            'stw': stw,
            'sog': sog,
            'actual_speed': sog,  # 호환성 유지
            'speed_loss': speed_loss,
            'current_effect': current_effect,
            'current_data': current_data
        })
    
    # 마지막 포인트가 도착점이 아니면 도착점 추가
    last_point = new_dr[-1]
    if last_point['distance_remaining'] > 0.1:  # 0.1nm 이상 남았으면
        # 마지막 구간의 속도로 도착 시간 계산
        weather_available = last_point.get('weather_available', True)
        weather = last_point.get('weather')
        
        # 해류 데이터
        current_data = None
        current_effect = 0.0
        try:
            current_data = get_rtofs_current(last_point['lat'], last_point['lon'],
                                            last_point['time'])
        except:
            pass
        
        if weather_available and weather:
            speed_loss, current_effect = calculate_speed_loss(vessel, weather, 
                                                              last_point['heading'],
                                                              current_data)
        else:
            speed_loss = 0
            if current_data and 'u_current' in current_data:
                u = current_data.get('u_current', 0)
                v = current_data.get('v_current', 0)
                heading_rad = math.radians(last_point['heading'])
                current_along = u * math.sin(heading_rad) + v * math.cos(heading_rad)
                current_effect = current_along * 1.94384
        
        stw = max(vessel.speed_knots - speed_loss, 3.0)
        stw = min(stw, vessel.speed_knots * 1.05)
        sog = max(stw + current_effect, 1.0)
        
        time_to_arrival = last_point['distance_remaining'] / sog
        arrival_time = last_point['time'] + timedelta(hours=time_to_arrival)
        
        lat, lon, heading = track.get_position_at_distance(track.total_distance)
        new_dr.append({
            'time': arrival_time,
            'lat': lat,
            'lon': lon,
            'distance_sailed': track.total_distance,
            'distance_remaining': 0,
            'heading': heading,
            'weather': last_point.get('weather'),
            'weather_available': last_point.get('weather_available', True),
            'stw': stw,
            'sog': sog,
            'actual_speed': sog,
            'speed_loss': speed_loss,
            'current_effect': current_effect,
            'current_data': current_data
        })
    
    return new_dr

########################################
# NOAA GFS 데이터 관련 함수들
########################################

def find_latest_gfs_cycle() -> Tuple[Optional[str], Optional[int], Optional[datetime]]:
    """
    최신 GFS 사이클 찾기 (00Z, 06Z, 12Z, 18Z)
    NOAA 서버에서 사용 가능한 최신 사이클 확인
    """
    now = datetime.now(timezone.utc)
    
    # 최근 24시간 내 사이클 확인 (최신 순)
    for hours_ago in range(0, 25, 6):
        check_time = now - timedelta(hours=hours_ago)
        date_str = check_time.strftime('%Y%m%d')
        
        # 해당 날짜의 사이클 확인 (18, 12, 06, 00)
        for cycle in [18, 12, 6, 0]:
            cycle_time = check_time.replace(hour=cycle, minute=0, second=0, microsecond=0)
            if cycle_time > now:
                continue
            
            # 데이터 가용 여부 확인 (HEAD 요청으로 빠르게)
            url = (f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?"
                   f"dir=%2Fgfs.{date_str}%2F{cycle:02d}%2Fatmos&"
                   f"file=gfs.t{cycle:02d}z.pgrb2.0p25.f000&"
                   f"var_PRMSL=on&lev_mean_sea_level=on&"
                   f"subregion=&toplat=32&leftlon=126&rightlon=127&bottomlat=31")
            
            try:
                resp = requests.head(url, timeout=10)
                if resp.status_code == 200:
                    return date_str, cycle, cycle_time
            except:
                continue
    
    return None, None, None

def build_subregion_params(lat: float, lon: float, margin: float = 0.5) -> str:
    """입력 좌표 기준 서브리전 파라미터 생성 (0.25도 그리드에 맞춤)"""
    lat_min = math.floor((lat - margin) * 4) / 4
    lat_max = math.ceil((lat + margin) * 4) / 4
    lon_min = math.floor((lon - margin) * 4) / 4
    lon_max = math.ceil((lon + margin) * 4) / 4
    
    return f"subregion=&toplat={lat_max}&leftlon={lon_min}&rightlon={lon_max}&bottomlat={lat_min}"

def fetch_gfs_atmosphere(date_str: str, cycle: int, fhour: int, lat: float, lon: float) -> Optional[bytes]:
    """GFS Atmosphere 모델에서 PRMSL(기압), GUST(돌풍) 가져오기"""
    subregion = build_subregion_params(lat, lon)
    url = (f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?"
           f"dir=%2Fgfs.{date_str}%2F{cycle:02d}%2Fatmos&"
           f"file=gfs.t{cycle:02d}z.pgrb2.0p25.f{fhour:03d}&"
           f"var_PRMSL=on&var_GUST=on&"
           f"lev_mean_sea_level=on&lev_surface=on&"
           f"{subregion}")
    
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200 and len(resp.content) > 100:
            return resp.content
    except:
        pass
    return None

def fetch_gfswave(date_str: str, cycle: int, fhour: int, lat: float, lon: float) -> Optional[bytes]:
    """GFS Wave 모델에서 바람 및 파도 데이터 가져오기"""
    subregion = build_subregion_params(lat, lon)
    
    url = (f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfswave.pl?"
           f"dir=%2Fgfs.{date_str}%2F{cycle:02d}%2Fwave%2Fgridded&"
           f"file=gfswave.t{cycle:02d}z.global.0p25.f{fhour:03d}.grib2&"
           f"var_WIND=on&var_WDIR=on&var_UGRD=on&var_VGRD=on&"
           f"var_HTSGW=on&var_DIRPW=on&var_PERPW=on&"
           f"var_SWELL=on&var_SWDIR=on&var_SWPER=on&"
           f"lev_surface=on&lev_1_in_sequence=on&"
           f"{subregion}")
    
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200 and len(resp.content) > 100:
            return resp.content
    except:
        pass
    return None

def parse_grib_data(grib_bytes: Optional[bytes], lat: float, lon: float) -> Dict:
    """GRIB2 데이터 파싱하여 딕셔너리로 반환"""
    if grib_bytes is None or len(grib_bytes) < 100:
        return {'error': f'No data or too small ({len(grib_bytes) if grib_bytes else 0} bytes)'}
    
    if not GRIB_AVAILABLE:
        return {'error': 'GRIB libraries not available'}
    
    result = {}
    result['_raw_size'] = len(grib_bytes)
    temp_path = None
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.grib2', delete=False) as f:
            f.write(grib_bytes)
            temp_path = f.name
        
        # 여러 필터 설정으로 시도
        filter_configs = [
            {'typeOfLevel': 'surface'},
            {'typeOfLevel': 'meanSea'},
            {'typeOfLevel': 'orderedSequence'},
            {},
        ]
        
        datasets_tried = 0
        vars_found = []
        
        for filter_keys in filter_configs:
            try:
                if filter_keys:
                    ds = xr.open_dataset(temp_path, engine='cfgrib',
                                       backend_kwargs={'filter_by_keys': filter_keys, 'errors': 'ignore'})
                else:
                    ds = xr.open_dataset(temp_path, engine='cfgrib',
                                       backend_kwargs={'errors': 'ignore'})
                datasets_tried += 1
            except Exception as e:
                continue
            
            if ds is None:
                continue
            
            # 좌표 이름 확인
            lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
            lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'
            
            # 가장 가까운 좌표로 데이터 추출
            try:
                ds_point = ds.sel({lat_name: lat, lon_name: lon}, method='nearest')
            except Exception as e:
                ds.close()
                continue
            
            # 변수들 추출
            var_mapping = {
                # GFS Atmosphere
                'prmsl': 'pressure',
                'gust': 'gust',
                # GFS Wave - 다양한 변수명 처리
                'wind': 'wind_speed',
                'wdir': 'wind_dir',
                'u10': 'wind_u',
                'v10': 'wind_v',
                'u': 'wind_u',
                'v': 'wind_v',
                'ugrd': 'wind_u',
                'vgrd': 'wind_v',
                'htsgw': 'wave_height',
                'swh': 'wave_height',
                'dirpw': 'wave_dir',
                'mwd': 'wave_dir',
                'perpw': 'wave_period',
                'mwp': 'wave_period',
                'swell': 'swell_height',
                'shts': 'swell_height',
                'swdir': 'swell_dir',
                'mdts': 'swell_dir',
                'swper': 'swell_period',
                'mpts': 'swell_period',
            }
            
            for var_name in ds_point.data_vars:
                var_lower = var_name.lower()
                vars_found.append(var_name)
                for grib_var, result_key in var_mapping.items():
                    if grib_var in var_lower:
                        try:
                            val = float(ds_point[var_name].values)
                            if not math.isnan(val):
                                result[result_key] = val
                        except:
                            pass
                        break
            
            ds.close()
        
        result['_datasets_tried'] = datasets_tried
        result['_vars_found'] = vars_found
        
    except Exception as e:
        result['parse_error'] = str(e)
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
    
    return result

def get_noaa_weather(lat: float, lon: float, target_time: datetime, 
                     gfs_cache: Dict) -> Dict:
    """
    NOAA GFS에서 특정 위치/시간의 기상 데이터 조회
    gfs_cache: {'date_str': str, 'cycle': int, 'cycle_time': datetime} 캐시
    """
    weather_data = {}
    
    # 캐시된 사이클 정보 사용 또는 새로 찾기
    if not gfs_cache.get('date_str'):
        date_str, cycle, cycle_time = find_latest_gfs_cycle()
        if date_str:
            gfs_cache['date_str'] = date_str
            gfs_cache['cycle'] = cycle
            gfs_cache['cycle_time'] = cycle_time
    
    date_str = gfs_cache.get('date_str')
    cycle = gfs_cache.get('cycle')
    cycle_time = gfs_cache.get('cycle_time')
    
    if not date_str or cycle is None:
        weather_data['error'] = 'No GFS cycle available'
        return weather_data
    
    # target_time을 UTC aware로 변환 (naive datetime인 경우)
    if target_time.tzinfo is None:
        target_time_utc = target_time.replace(tzinfo=timezone.utc)
    else:
        target_time_utc = target_time
    
    # 예보 시간 계산 (cycle_time 기준 몇 시간 후인지)
    hours_from_cycle = (target_time_utc - cycle_time).total_seconds() / 3600
    
    # 3시간 간격으로 반올림
    fhour = round(hours_from_cycle / 3) * 3
    fhour = max(0, min(fhour, 384))  # 0~384시간 범위
    
    weather_data['fhour'] = fhour
    weather_data['cycle'] = f"{date_str} {cycle:02d}Z"
    
    # GFS Atmosphere (기압, 돌풍)
    atm_data = fetch_gfs_atmosphere(date_str, cycle, fhour, lat, lon)
    if atm_data:
        parsed_atm = parse_grib_data(atm_data, lat, lon)
        weather_data.update(parsed_atm)
    
    # GFS Wave (바람, 파도, 너울)
    wave_data = fetch_gfswave(date_str, cycle, fhour, lat, lon)
    if wave_data:
        parsed_wave = parse_grib_data(wave_data, lat, lon)
        weather_data.update(parsed_wave)
    
    # wind_u, wind_v가 있으면 풍속/풍향 계산
    if 'wind_u' in weather_data and 'wind_v' in weather_data:
        u = weather_data['wind_u']
        v = weather_data['wind_v']
        weather_data['wind_speed'] = math.sqrt(u**2 + v**2)
        weather_data['wind_dir'] = (math.degrees(math.atan2(u, v)) + 180) % 360
    
    return weather_data

def parse_noaa_data(weather_data: Dict, target_time: datetime) -> WeatherPoint:
    """NOAA GFS 데이터를 WeatherPoint 객체로 변환"""
    result = WeatherPoint(target_time, 0, 0)
    
    # 기압 (Pa -> hPa 변환)
    if 'pressure' in weather_data:
        pressure = weather_data['pressure']
        if pressure > 10000:  # Pa 단위면 hPa로 변환
            pressure = pressure / 100
        result.pressure = pressure * 100  # 다시 Pa로 (기존 로직과 호환)
    
    # 바람
    if 'wind_speed' in weather_data:
        result.wind_speed = weather_data['wind_speed']
    if 'wind_dir' in weather_data:
        result.wind_dir = weather_data['wind_dir']
    
    # 돌풍
    if 'gust' in weather_data:
        result.wind_gust = weather_data['gust']
    
    # 파도
    if 'wave_height' in weather_data:
        result.wave_height = weather_data['wave_height']
    if 'wave_dir' in weather_data:
        result.wave_dir = weather_data['wave_dir']
    
    # 너울
    if 'swell_height' in weather_data:
        result.swell_height = weather_data['swell_height']
    if 'swell_dir' in weather_data:
        result.swell_dir = weather_data['swell_dir']
    
    return result

# 기존 Windy 함수들은 NOAA로 대체됨 (호환성을 위해 래퍼 함수 제공)
def get_windy_weather(lat: float, lon: float, api_key: str) -> Dict:
    """[DEPRECATED] Windy API 대신 NOAA GFS 사용 - 호환성 래퍼"""
    # 이 함수는 더 이상 사용되지 않음
    # 새 코드는 get_noaa_weather() 사용
    return {}

def parse_windy_data(weather_data: Dict, target_time: datetime) -> WeatherPoint:
    """[DEPRECATED] Windy 파싱 대신 NOAA 파싱 사용 - 호환성 래퍼"""
    return parse_noaa_data(weather_data, target_time)

########################################
# NOAA RTOFS 해류 데이터 (OpenDAP)
########################################

def get_rtofs_current(lat: float, lon: float, target_time: datetime) -> Dict:
    """
    NOAA RTOFS에서 해류 데이터 가져오기 (OpenDAP)
    Returns: {'u_current': m/s, 'v_current': m/s} 또는 빈 딕셔너리
    """
    result = {}
    
    try:
        # RTOFS는 경도를 0~360 범위로 사용
        lon_360 = lon if lon >= 0 else lon + 360
        
        # 현재 시간 기준 가장 가까운 RTOFS 예보 시간 찾기
        # RTOFS는 하루 2회 (00Z, 12Z) 업데이트
        now_utc = datetime.now(timezone.utc)
        
        # 최근 RTOFS 날짜 (1-2일 전 데이터가 안정적)
        rtofs_date = (now_utc - timedelta(days=1)).strftime('%Y%m%d')
        
        # target_time과 RTOFS 기준시간의 차이로 forecast hour 계산
        # 간단히 24시간 예보 사용 (f024)
        forecast_hour = 24
        
        # OpenDAP URL (2D 표층 해류)
        # RTOFS 2D diagnostic fields
        base_url = f"https://nomads.ncep.noaa.gov/dods/rtofs/rtofs_global{rtofs_date}/rtofs_glo_2ds_f{forecast_hour:03d}_daily_diag"
        
        # 위경도 인덱스 계산 (RTOFS 해상도: 1/12도 ≈ 0.083도)
        # 위도: -80 ~ 90, 경도: 0 ~ 360
        lat_idx = int((lat + 80) / 0.083)
        lon_idx = int(lon_360 / 0.083)
        
        # 범위 제한
        lat_idx = max(0, min(lat_idx, 2040))
        lon_idx = max(0, min(lon_idx, 4319))
        
        # U 성분 (동서 방향 해류)
        u_url = f"{base_url}.ascii?u_velocity[0][{lat_idx}][{lon_idx}]"
        resp_u = requests.get(u_url, timeout=15)
        
        if resp_u.status_code == 200:
            # OpenDAP ASCII 응답 파싱
            lines = resp_u.text.strip().split('\n')
            for line in lines:
                if line.startswith('u_velocity'):
                    continue
                if ',' in line or line.replace('.','').replace('-','').replace('e','').replace('+','').isdigit():
                    try:
                        # 마지막 숫자값 추출
                        val = float(line.split(',')[-1].strip() if ',' in line else line.strip())
                        if abs(val) < 10:  # 현실적인 범위 (10 m/s 미만)
                            result['u_current'] = val
                    except:
                        pass
                    break
        
        # V 성분 (남북 방향 해류)
        v_url = f"{base_url}.ascii?v_velocity[0][{lat_idx}][{lon_idx}]"
        resp_v = requests.get(v_url, timeout=15)
        
        if resp_v.status_code == 200:
            lines = resp_v.text.strip().split('\n')
            for line in lines:
                if line.startswith('v_velocity'):
                    continue
                if ',' in line or line.replace('.','').replace('-','').replace('e','').replace('+','').isdigit():
                    try:
                        val = float(line.split(',')[-1].strip() if ',' in line else line.strip())
                        if abs(val) < 10:
                            result['v_current'] = val
                    except:
                        pass
                    break
        
        # 해류 속력 및 방향 계산
        if 'u_current' in result and 'v_current' in result:
            u = result['u_current']
            v = result['v_current']
            result['current_speed'] = math.sqrt(u**2 + v**2)
            result['current_dir'] = (math.degrees(math.atan2(u, v)) + 360) % 360  # 흐르는 방향
            
    except Exception as e:
        result['rtofs_error'] = str(e)
    
    return result

########################################
# 개선된 저항 및 속력 계산 (물리 법칙 기반)
########################################

def calculate_wind_resistance(vessel: VesselData, wind_speed_ms: float, 
                              wind_dir: float, vessel_heading: float) -> float:
    """
    풍압저항 계산 (kN) - 물리 법칙 기반
    
    음수 반환 가능 (선미풍 = 추진력)
    cos(θ) 기반 연속 방향 계수 사용
    """
    if wind_speed_ms is None or wind_speed_ms < 0.1:
        return 0.0
    
    # 상대풍향 계산 (0° = 정선수풍, 180° = 정선미풍)
    relative_angle = (wind_dir - vessel_heading + 360) % 360
    if relative_angle > 180:
        relative_angle = 360 - relative_angle
    
    relative_angle_rad = math.radians(relative_angle)
    
    # cos 기반 방향 계수: 정선수(0°) = +1, 횡풍(90°) = 0, 정선미(180°) = -1
    # 추진력 반영을 위해 cos 함수 직접 사용
    direction_factor = math.cos(relative_angle_rad)
    
    # 항력계수 (상대풍향에 따라 연속적으로 변화)
    # 정면: 0.9, 측면: 0.6, 후면: 0.4
    Cd = 0.9 - 0.5 * (1 - abs(direction_factor))
    
    # 풍압면적: 상대풍향에 따른 가중 평균
    # 정면풍: front area, 횡풍: side area, 선미풍: front area
    area_weight = abs(direction_factor)
    area = (vessel.windage_area_front * area_weight + 
            vessel.windage_area_side * (1 - area_weight))
    
    rho_air = 1.225  # kg/m³
    
    # 풍압저항 (N)
    # 양수 = 저항, 음수 = 추진력
    R_wind = 0.5 * rho_air * Cd * area * (wind_speed_ms ** 2) * direction_factor
    
    return R_wind / 1000  # kN (음수 허용)


def calculate_wave_resistance(vessel: VesselData, wave_height: float, 
                              wave_dir: float, vessel_heading: float) -> float:
    """
    파랑저항 계산 (kN) - 파랑 에너지 밀도 이론 기반
    
    공식: R_wave = C × ρ × g × H² × (B/L) × direction_factor × type_factor
    
    - H²: 파랑 에너지는 파고의 제곱에 비례
    - B/L: 선폭/전장 비율 (넓고 짧은 선박일수록 저항 증가)
    - cos 기반 방향 계수로 선미파 추진 효과 반영
    
    음수 반환 가능 (선미파 = 추진력)
    """
    if wave_height is None or wave_height < 0.3:
        return 0.0
    
    # 상대파향 계산
    relative_angle = (wave_dir - vessel_heading + 360) % 360
    if relative_angle > 180:
        relative_angle = 360 - relative_angle
    
    relative_angle_rad = math.radians(relative_angle)
    
    # cos 기반 방향 계수
    # 정선수파(0°) = +1.0 (최대 저항)
    # 횡파(90°) = 0 (저항 없음, 실제로는 롤링만)
    # 정선미파(180°) = -0.3 (추진력, 하지만 서핑 효과는 제한적)
    if relative_angle <= 90:
        direction_factor = math.cos(relative_angle_rad)
    else:
        # 선미파: 추진 효과 있으나 제한적 (서핑 효과)
        direction_factor = -0.3 * math.cos(math.pi - relative_angle_rad)
    
    # 물리 상수
    rho_water = 1025  # kg/m³
    g = 9.81  # m/s²
    
    # 선형 계수
    B = vessel.breadth
    L = vessel.loa
    BL_ratio = B / L  # 일반적으로 0.1 ~ 0.2
    
    # 방형비척계수 보정 (비대선일수록 저항 증가)
    cb_factor = 0.8 + (vessel.cb * 0.4)  # Cb 0.5 → 1.0, Cb 0.85 → 1.14
    
    # 선종별 계수
    type_factor = getattr(vessel, 'wave_resistance_factor', 1.0)
    
    # 경험 계수 (튜닝 파라미터)
    C = 0.5
    
    # 파랑저항 공식: R = C × ρ × g × H² × B × (B/L) × factors
    # 단위: kg/m³ × m/s² × m² × m × 무차원 = N
    R_wave = C * rho_water * g * (wave_height ** 2) * B * BL_ratio * direction_factor * cb_factor * type_factor
    
    return R_wave / 1000  # kN (음수 허용)


def calculate_swell_resistance(vessel: VesselData, swell_height: float,
                               swell_dir: float, vessel_heading: float) -> float:
    """
    너울 저항 계산 (kN)
    
    너울은 파장이 길어 wind wave보다 영향이 적음
    파랑저항의 40% 수준으로 계산
    """
    if swell_height is None or swell_height < 0.5:
        return 0.0
    
    # 기본 파랑저항 계산
    R_base = calculate_wave_resistance(vessel, swell_height, swell_dir, vessel_heading)
    
    # 너울 감쇠 계수 (파장이 길어 저항 감소)
    swell_factor = 0.4
    
    return R_base * swell_factor


def calculate_base_resistance(vessel: VesselData) -> float:
    """
    동적 기저 저항 계산 (kN)
    
    Froude의 저항 이론 기반:
    R_base ∝ Displacement^(2/3) × V²
    
    단위 변환 및 경험 계수 포함
    """
    # 배수량 (톤 → kg)
    disp_kg = vessel.displacement * 1000
    
    # 속력 (knots → m/s)
    V_ms = vessel.speed_knots * 0.5144
    
    # Froude 기반 저항 추정
    # R = k × Δ^(2/3) × V²
    # k는 선형에 따른 경험 계수
    k = 0.0012  # 튜닝 파라미터
    
    # 방형비척계수 보정 (비대선 = 저항 증가)
    cb_factor = 0.85 + (vessel.cb * 0.3)
    
    R_base = k * (disp_kg ** (2/3)) * (V_ms ** 2) * cb_factor
    
    return R_base / 1000  # kN


def calculate_speed_loss(vessel: VesselData, weather: WeatherPoint, 
                        vessel_heading: float, current_data: Dict = None) -> Tuple[float, float]:
    """
    속력 손실 및 해류 영향 계산
    
    Returns: (speed_loss_knots, current_effect_knots)
    
    - speed_loss: 바람/파도에 의한 대수속력 손실 (음수 = 속력 증가)
    - current_effect: 해류에 의한 대지속력 변화 (음수 = 역조)
    
    물리 법칙:
    1. 추가 저항 계산 (음수 = 추진력)
    2. ΔV/V = (1/3) × (ΔR/R_base)
    3. 최대 손실 18% 제한 (상업 항로 기준)
    """
    total_added_resistance = 0.0
    
    # 풍압저항 (음수 = 추진력)
    if weather.wind_speed is not None and weather.wind_speed > 0.1:
        R_wind = calculate_wind_resistance(vessel, weather.wind_speed, 
                                          weather.wind_dir or 0, vessel_heading)
        total_added_resistance += R_wind
    
    # 파랑저항 (음수 = 추진력)
    if weather.wave_height is not None and weather.wave_height > 0.3:
        R_wave = calculate_wave_resistance(vessel, weather.wave_height,
                                          weather.wave_dir or 0, vessel_heading)
        total_added_resistance += R_wave
    
    # 너울저항 (음수 = 추진력)
    if weather.swell_height is not None and weather.swell_height > 0.5:
        R_swell = calculate_swell_resistance(vessel, weather.swell_height,
                                            weather.swell_dir or weather.wave_dir or 0,
                                            vessel_heading)
        total_added_resistance += R_swell
    
    # 동적 기저 저항
    base_resistance = calculate_base_resistance(vessel)
    base_resistance = max(base_resistance, 10.0)  # 최소값 보장
    
    # 저항 비율
    resistance_ratio = total_added_resistance / base_resistance
    
    # 속력 변화: ΔV/V ≈ (1/3) × (ΔR/R)
    # 양수 = 감속, 음수 = 가속
    speed_change_ratio = resistance_ratio / 3.0
    speed_loss = vessel.speed_knots * speed_change_ratio
    
    # 상한/하한 제한 (상업 항로 기준)
    max_loss = vessel.speed_knots * 0.18  # 최대 18% 감속
    max_gain = vessel.speed_knots * 0.05  # 최대 5% 가속 (추진력)
    
    speed_loss = max(-max_gain, min(speed_loss, max_loss))
    
    # 해류 영향 계산
    current_effect = 0.0
    if current_data:
        u_curr = current_data.get('u_current', 0)
        v_curr = current_data.get('v_current', 0)
        
        if u_curr != 0 or v_curr != 0:
            # 선박 진행 방향 단위 벡터
            heading_rad = math.radians(vessel_heading)
            ship_u = math.sin(heading_rad)  # 동쪽 성분
            ship_v = math.cos(heading_rad)  # 북쪽 성분
            
            # 해류의 선박 진행 방향 성분 (내적)
            # 양수 = 순조 (속력 증가), 음수 = 역조 (속력 감소)
            current_along_track = u_curr * ship_u + v_curr * ship_v
            
            # m/s → knots 변환
            current_effect = current_along_track * 1.94384
    
    return speed_loss, current_effect


def ms_to_knots(ms: float) -> float:
    """m/s를 노트로 변환"""
    return ms * 1.94384

def decimal_to_dms(decimal_deg: float, is_lat: bool) -> str:
    """십진수 좌표를 ddd mm.mm N/S/E/W 형식으로 변환"""
    if is_lat:
        direction = 'N' if decimal_deg >= 0 else 'S'
    else:
        direction = 'E' if decimal_deg >= 0 else 'W'
    
    decimal_deg = abs(decimal_deg)
    degrees = int(decimal_deg)
    minutes = (decimal_deg - degrees) * 60
    
    if is_lat:
        return f"{degrees:02d} {minutes:05.2f} {direction}"
    else:
        return f"{degrees:03d} {minutes:05.2f} {direction}"

def create_arrow_svg(degrees: float, size: int = 16) -> str:
    """방향(degrees)에 해당하는 회전된 SVG 화살표 생성 (바람/파도가 오는 방향)"""
    if degrees is None:
        return ""
    
    # SVG 화살표 - 아래를 가리키는 화살표 (0° = 북에서 오는 바람)
    # degrees 만큼 회전
    svg = f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" style="vertical-align: middle; transform: rotate({degrees}deg);">
        <path d="M12 2 L12 22 M12 22 L6 16 M12 22 L18 16" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>'''
    return svg

def create_results_table_html(dr_positions: List[Dict], speed_knots: float = None) -> str:
    """결과 테이블을 HTML로 생성 (SVG 화살표 포함)
    기상 데이터가 없는 구간(weather_available=False)은 NIL로 표시
    STW (Speed Through Water), SOG (Speed Over Ground), Current 표시
    """
    
    html = '''
    <style>
        .weather-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        .weather-table th {
            background-color: #f0f2f6;
            padding: 6px 8px;
            text-align: left;
            border-bottom: 2px solid #ddd;
            white-space: nowrap;
        }
        .weather-table td {
            padding: 6px 8px;
            border-bottom: 1px solid #eee;
            white-space: nowrap;
        }
        .weather-table tr:hover {
            background-color: #f8f9fa;
        }
        .weather-table tr.no-weather {
            background-color: #fff8e6;
        }
        .weather-table tr.no-weather:hover {
            background-color: #fff3cd;
        }
        .nil-cell {
            color: #999;
            font-style: italic;
        }
        .positive-current {
            color: #28a745;
        }
        .negative-current {
            color: #dc3545;
        }
        .arrow-cell {
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }
        .arrow-svg {
            display: inline-block;
            vertical-align: middle;
        }
    </style>
    <table class="weather-table">
        <thead>
            <tr>
                <th>ETA (UTC)</th>
                <th>Latitude</th>
                <th>Longitude</th>
                <th>Course</th>
                <th>Pressure</th>
                <th>Wind</th>
                <th>Wave</th>
                <th>Max Wave</th>
                <th>Sailed</th>
                <th>Remain</th>
                <th>STW</th>
                <th>Current</th>
                <th>SOG</th>
            </tr>
        </thead>
        <tbody>
    '''
    
    for i, point in enumerate(dr_positions):
        weather = point.get('weather')
        weather_available = point.get('weather_available', True)
        utc_time = point['time'].strftime('%Y-%m-%d %H:%M')
        lat_str = decimal_to_dms(point['lat'], is_lat=True)
        lon_str = decimal_to_dms(point['lon'], is_lat=False)
        
        # 기상 데이터 없는 행 스타일
        row_class = '' if weather_available else 'no-weather'
        nil_class = '' if weather_available else 'nil-cell'
        
        # Course (heading) - 화살표 없이 숫자만
        heading = point.get('heading')
        if heading is not None:
            course_str = f"{heading:.0f}°"
        else:
            course_str = "N/A"
        
        if not weather_available:
            # 기상 데이터 없음: NIL 표시
            pressure = f'<span class="{nil_class}">NIL</span>'
            wind_str = f'<span class="{nil_class}">NIL</span>'
            wave_str = f'<span class="{nil_class}">NIL</span>'
            max_wave_str = f'<span class="{nil_class}">NIL</span>'
            # STW는 대수속력 사용
            stw_str = f"{speed_knots:.1f}" if speed_knots else "N/A"
            current_str = f'<span class="{nil_class}">NIL</span>'
            sog_str = f"{speed_knots:.1f}" if speed_knots else "N/A"
        else:
            # Pressure (Pa -> hPa 변환, 소수점 없이)
            if weather and weather.pressure:
                pressure_val = weather.pressure
                if pressure_val > 10000:
                    pressure_val = pressure_val / 100
                pressure = f"{pressure_val:.0f}"
            else:
                pressure = "N/A"
            
            # Wind with arrow (오는 방향 그대로 표시)
            if weather and weather.wind_dir is not None and weather.wind_speed is not None:
                wind_arrow = f'<span class="arrow-svg" style="display:inline-block; transform:rotate({weather.wind_dir}deg);">↓</span>'
                wind_str = f'{wind_arrow} {weather.wind_dir:.0f}° / {ms_to_knots(weather.wind_speed):.1f}kt'
            else:
                wind_str = "N/A"
            
            # Wave with arrow (오는 방향 그대로 표시)
            if weather and weather.wave_dir is not None and weather.wave_height is not None:
                wave_arrow = f'<span class="arrow-svg" style="display:inline-block; transform:rotate({weather.wave_dir}deg);">↓</span>'
                wave_str = f'{wave_arrow} {weather.wave_dir:.0f}° / {weather.wave_height:.1f}m'
                # Max Wave (레일리 분포 x1.6)
                max_wave = weather.wave_height * 1.6
                max_wave_str = f"{max_wave:.1f}m"
            else:
                wave_str = "N/A"
                max_wave_str = "N/A"
            
            # STW (Speed Through Water)
            stw = point.get('stw', point.get('actual_speed', 0))
            stw_str = f"{stw:.1f}"
            
            # Current effect
            current_effect = point.get('current_effect', 0)
            if current_effect > 0.1:
                current_str = f'<span class="positive-current">+{current_effect:.1f}</span>'
            elif current_effect < -0.1:
                current_str = f'<span class="negative-current">{current_effect:.1f}</span>'
            else:
                current_str = "0.0"
            
            # SOG (Speed Over Ground)
            sog = point.get('sog', point.get('actual_speed', 0))
            sog_str = f"{sog:.1f}"
        
        sailed = f"{point['distance_sailed']:.1f}"
        remaining = f"{point['distance_remaining']:.1f}"
        
        html += f'''
            <tr class="{row_class}">
                <td>{utc_time}</td>
                <td>{lat_str}</td>
                <td>{lon_str}</td>
                <td>{course_str}</td>
                <td>{pressure}</td>
                <td>{wind_str}</td>
                <td>{wave_str}</td>
                <td>{max_wave_str}</td>
                <td>{sailed}</td>
                <td>{remaining}</td>
                <td>{stw_str}</td>
                <td>{current_str}</td>
                <td>{sog_str}</td>
            </tr>
        '''
    
    html += '''
        </tbody>
    </table>
    '''
    
    return html

def create_results_table(dr_positions: List[Dict]) -> pd.DataFrame:
    """결과 테이블 생성 (DataFrame 버전 - fallback용)"""
    rows = []
    
    for i, point in enumerate(dr_positions):
        weather = point.get('weather')
        utc_time = point['time']
        lat_str = decimal_to_dms(point['lat'], is_lat=True)
        lon_str = decimal_to_dms(point['lon'], is_lat=False)
        
        row = {
            'ETA (UTC)': utc_time.strftime('%Y-%m-%d %H:%M'),
            'Latitude': lat_str,
            'Longitude': lon_str,
            'Pressure (hPa)': f"{weather.pressure:.1f}" if weather and weather.pressure else "N/A",
            'Wind': f"{weather.wind_dir:.0f}° / {ms_to_knots(weather.wind_speed):.1f}kt" if weather and weather.wind_dir and weather.wind_speed else "N/A",
            'Wave': f"{weather.wave_dir:.0f}° / {weather.wave_height:.1f}m" if weather and weather.wave_dir and weather.wave_height else "N/A",
            'Sailed (nm)': f"{point['distance_sailed']:.1f}",
            'Remaining (nm)': f"{point['distance_remaining']:.1f}",
            'Est. Speed (kt)': f"{point.get('actual_speed', 0):.1f}" if 'actual_speed' in point else "N/A"
        }
        
        rows.append(row)
    
    return pd.DataFrame(rows)

def create_route_map(track_points: List[Tuple[float, float]], dr_positions: List[Dict]) -> folium.Map:
    """GPX 트랙과 DR 위치를 표시하는 지도 생성"""
    
    # 지도 중심점 계산
    all_lats = [p[0] for p in track_points] + [p['lat'] for p in dr_positions]
    all_lons = [p[1] for p in track_points] + [p['lon'] for p in dr_positions]
    center_lat = sum(all_lats) / len(all_lats)
    center_lon = sum(all_lons) / len(all_lons)
    
    # 지도 생성
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles='CartoDB positron'
    )
    
    # 경계 맞추기
    sw = [min(all_lats), min(all_lons)]
    ne = [max(all_lats), max(all_lons)]
    m.fit_bounds([sw, ne], padding=[20, 20])
    
    # GPX 트랙 라인 (계획 항로) - 회색 점선
    track_coords = [[p[0], p[1]] for p in track_points]
    folium.PolyLine(
        track_coords,
        weight=3,
        color='gray',
        dash_array='10',
        opacity=0.7,
        tooltip='Planned Route'
    ).add_to(m)
    
    # DR 항로 라인 - 트랙을 따라가도록 구성
    # DR 위치들 사이의 트랙 구간을 포함하여 라인 생성
    track_line = TrackLine(track_points)
    dr_route_coords = []
    
    for i, dr_point in enumerate(dr_positions):
        dr_distance = dr_point['distance_sailed']
        
        if i == 0:
            # 첫 DR 위치
            dr_route_coords.append([dr_point['lat'], dr_point['lon']])
        else:
            prev_distance = dr_positions[i-1]['distance_sailed']
            
            # 이전 DR과 현재 DR 사이의 트랙 경유점들 추가
            for j, cum_dist in enumerate(track_line.cumulative_distances):
                if prev_distance < cum_dist < dr_distance:
                    # 이 경유점은 두 DR 사이에 있음
                    dr_route_coords.append([track_points[j][0], track_points[j][1]])
            
            # 현재 DR 위치 추가
            dr_route_coords.append([dr_point['lat'], dr_point['lon']])
    
    # DR 항로 애니메이션 라인
    AntPath(
        dr_route_coords,
        weight=4,
        color='#2E86AB',
        pulse_color='#A5D8FF',
        delay=1000,
        opacity=0.8
    ).add_to(m)
    
    # DR 위치 마커
    for i, point in enumerate(dr_positions):
        weather = point.get('weather')
        
        # 좌표 포맷
        lat_str = decimal_to_dms(point['lat'], is_lat=True)
        lon_str = decimal_to_dms(point['lon'], is_lat=False)
        
        # 팝업 내용 생성
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; font-size: 12px; min-width: 200px;">
            <b style="font-size: 14px;">DR Position #{i}</b><br>
            <hr style="margin: 5px 0;">
            <b>ETA (UTC):</b> {point['time'].strftime('%Y-%m-%d %H:%M')}<br>
            <b>Position:</b> {lat_str}, {lon_str}<br>
            <b>Course:</b> {point.get('heading', 0):.0f}°<br>
            <b>Distance Sailed:</b> {point['distance_sailed']:.1f} nm<br>
            <b>Remaining:</b> {point['distance_remaining']:.1f} nm<br>
        """
        
        if weather:
            # Pressure 변환
            pressure_val = weather.pressure if weather.pressure else 0
            if pressure_val > 10000:
                pressure_val = pressure_val / 100
            
            popup_html += f"""
            <hr style="margin: 5px 0;">
            <b style="color: #2E86AB;">Weather Forecast</b><br>
            <b>Pressure:</b> {pressure_val:.0f} hPa<br>
            """
            
            if weather.wind_dir is not None and weather.wind_speed is not None:
                popup_html += f"<b>Wind:</b> {weather.wind_dir:.0f}° / {ms_to_knots(weather.wind_speed):.1f} kt<br>"
            
            if weather.wave_dir is not None and weather.wave_height is not None:
                max_wave = weather.wave_height * 1.6  # 레일리 분포
                popup_html += f"<b>Wave:</b> {weather.wave_dir:.0f}° / {weather.wave_height:.1f} m<br>"
                popup_html += f"<b>Max Wave:</b> {max_wave:.1f} m<br>"
            
            if point.get('actual_speed'):
                popup_html += f"<b>Est. Speed:</b> {point['actual_speed']:.1f} kt<br>"
        
        popup_html += "</div>"
        
        # Tooltip 내용 생성 (마우스 오버시 표시)
        tooltip_lines = [f"DR #{i}: {point['time'].strftime('%Y-%m-%d %H:%M')} UTC"]
        tooltip_lines.append(f"Position: {lat_str}, {lon_str}")
        
        if weather:
            if weather.wind_dir is not None and weather.wind_speed is not None:
                tooltip_lines.append(f"Wind: {weather.wind_dir:.0f}° / {ms_to_knots(weather.wind_speed):.1f} kt")
            if weather.wave_dir is not None and weather.wave_height is not None:
                max_wave = weather.wave_height * 1.6
                tooltip_lines.append(f"Wave: {weather.wave_height:.1f} m (Max: {max_wave:.1f} m)")
        
        tooltip_text = "<br>".join(tooltip_lines)
        
        # 마커 색상: 출발(녹색), 도착(빨강), 중간(파랑)
        if i == 0:
            icon_color = 'green'
            icon = 'play'
        elif i == len(dr_positions) - 1:
            icon_color = 'red'
            icon = 'flag'
        else:
            icon_color = 'blue'
            icon = 'info-sign'
        
        folium.Marker(
            location=[point['lat'], point['lon']],
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=folium.Tooltip(tooltip_text),
            icon=folium.Icon(color=icon_color, icon=icon)
        ).add_to(m)
    
    # GPX 경유점 마커 (작은 원)
    for i, point in enumerate(track_points):
        folium.CircleMarker(
            location=[point[0], point[1]],
            radius=5,
            color='gray',
            fill=True,
            fill_color='white',
            fill_opacity=0.8,
            tooltip=f"Waypoint #{i+1}"
        ).add_to(m)
    
    return m

# Initialize session state with localStorage values
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.vessel_type_idx = load_from_storage('vessel_type_idx', 3)  # Special Purpose (기본값)
    st.session_state.displacement = load_from_storage('displacement', 5000.0)
    st.session_state.windage_side = load_from_storage('windage_side', 800.0)
    st.session_state.loa = load_from_storage('loa', 115.0)
    st.session_state.breadth = load_from_storage('breadth', 20.0)
    st.session_state.draft = load_from_storage('draft', 5.5)
    st.session_state.speed_knots = load_from_storage('speed_knots', 11.0)
    st.session_state.interval_idx = load_from_storage('interval_idx', 1)  # 6시간 (기본값)
    st.session_state.dep_tz_idx = load_from_storage('dep_tz_idx', 12)  # UTC+0
    st.session_state.arr_tz_idx = load_from_storage('arr_tz_idx', 21)  # UTC+9
    st.session_state.calculation_done = False
    # 출발 날짜/시간 초기화 (현재 시간)
    st.session_state.departure_date = datetime.now().date()
    st.session_state.departure_time = datetime.now().replace(second=0, microsecond=0).time()

# Streamlit UI
st.title("⛵ Weather Routing Calculator")
st.markdown("---")

# 선종 옵션 리스트
VESSEL_TYPES = list(VESSEL_TYPE_PARAMS.keys())

# Interval 옵션
INTERVAL_OPTIONS = [3, 6, 12, 24]

# Sidebar - 선박 데이터 입력
with st.sidebar:
    st.header("Vessel Data")
    
    # 선종 선택 (맨 위에 추가)
    vessel_type_idx = st.selectbox("Vessel Type", options=range(len(VESSEL_TYPES)),
                                    format_func=lambda x: VESSEL_TYPES[x],
                                    index=int(st.session_state.vessel_type_idx),
                                    key="input_vessel_type")
    if vessel_type_idx != st.session_state.vessel_type_idx:
        st.session_state.vessel_type_idx = vessel_type_idx
        save_to_storage('vessel_type_idx', vessel_type_idx)
    vessel_type = VESSEL_TYPES[vessel_type_idx]
    
    displacement = st.number_input("Displacement (ton)", min_value=100.0, 
                                   value=float(st.session_state.displacement), step=100.0,
                                   key="input_displacement")
    if displacement != st.session_state.displacement:
        st.session_state.displacement = displacement
        save_to_storage('displacement', displacement)
    
    windage_side = st.number_input("Windage Area Side (m²)", min_value=10.0, 
                                   value=float(st.session_state.windage_side), step=10.0,
                                   key="input_windage_side")
    if windage_side != st.session_state.windage_side:
        st.session_state.windage_side = windage_side
        save_to_storage('windage_side', windage_side)
    
    loa = st.number_input("LOA (m)", min_value=10.0, 
                          value=float(st.session_state.loa), step=1.0,
                          key="input_loa")
    if loa != st.session_state.loa:
        st.session_state.loa = loa
        save_to_storage('loa', loa)
    
    breadth = st.number_input("Breadth (m)", min_value=5.0, 
                              value=float(st.session_state.breadth), step=0.5,
                              key="input_breadth")
    if breadth != st.session_state.breadth:
        st.session_state.breadth = breadth
        save_to_storage('breadth', breadth)
    
    draft = st.number_input("Draft (m)", min_value=1.0, 
                            value=float(st.session_state.draft), step=0.1,
                            key="input_draft")
    if draft != st.session_state.draft:
        st.session_state.draft = draft
        save_to_storage('draft', draft)
    
    st.markdown("---")
    st.header("Voyage Data")
    
    speed_knots = st.number_input("Speed through water (knots)", min_value=1.0, 
                                  value=float(st.session_state.speed_knots), step=0.5,
                                  key="input_speed")
    if speed_knots != st.session_state.speed_knots:
        st.session_state.speed_knots = speed_knots
        save_to_storage('speed_knots', speed_knots)
    
    # DR Interval 선택
    interval_idx = st.selectbox("DR Interval (hours)", options=range(len(INTERVAL_OPTIONS)),
                                format_func=lambda x: f"{INTERVAL_OPTIONS[x]}h",
                                index=int(st.session_state.interval_idx),
                                key="input_interval")
    if interval_idx != st.session_state.interval_idx:
        st.session_state.interval_idx = interval_idx
        save_to_storage('interval_idx', interval_idx)
    interval_hours = INTERVAL_OPTIONS[interval_idx]
    
    # Time Zone 옵션 생성 (-12 ~ +13)
    tz_options = [f"UTC{'+' if i >= 0 else ''}{i}" for i in range(-12, 14)]
    tz_values = list(range(-12, 14))
    
    col_dep, col_arr = st.columns(2)
    with col_dep:
        dep_tz_idx = st.selectbox("Departure Zone", options=range(len(tz_options)), 
                                   format_func=lambda x: tz_options[x], 
                                   index=int(st.session_state.dep_tz_idx),
                                   key="input_dep_tz")
        if dep_tz_idx != st.session_state.dep_tz_idx:
            st.session_state.dep_tz_idx = dep_tz_idx
            save_to_storage('dep_tz_idx', dep_tz_idx)
        departure_tz = tz_values[dep_tz_idx]
    with col_arr:
        arr_tz_idx = st.selectbox("Arrival Zone", options=range(len(tz_options)), 
                                   format_func=lambda x: tz_options[x], 
                                   index=int(st.session_state.arr_tz_idx),
                                   key="input_arr_tz")
        if arr_tz_idx != st.session_state.arr_tz_idx:
            st.session_state.arr_tz_idx = arr_tz_idx
            save_to_storage('arr_tz_idx', arr_tz_idx)
        arrival_tz = tz_values[arr_tz_idx]
    
    departure_date = st.date_input("Departure Date (LT)", 
                                    value=st.session_state.departure_date,
                                    key="input_departure_date")
    st.session_state.departure_date = departure_date
    
    departure_time = st.time_input("Departure Time (LT)", 
                                    value=st.session_state.departure_time,
                                    key="input_departure_time")
    st.session_state.departure_time = departure_time
    
    # 로컬 시간을 UTC로 변환
    departure_local = datetime.combine(departure_date, departure_time)
    departure_datetime = departure_local - timedelta(hours=departure_tz)
    
    st.markdown("---")
    # NOAA GFS는 API 키 불필요
    st.info("🌐 Data Source: NOAA GFS (No API key required)")
    if GRIB_AVAILABLE:
        st.success("✅ GRIB libraries available")
    else:
        st.warning("⚠️ GRIB libraries not installed. Run: pip install xarray cfgrib eccodes")
    
    # 호환성을 위해 api_key 변수 유지 (NOAA는 사용 안함)
    api_key = "NOAA_GFS"
    
    st.markdown("---")
    st.header("Debug Options")
    show_debug = st.checkbox("Show API response details", value=False)

# Main area - 계산 완료 후에는 접힌 상태로
upload_expanded = not st.session_state.calculation_done
with st.expander("📁 Upload GPX Track & Actions", expanded=upload_expanded):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        gpx_file = st.file_uploader("Choose a GPX file", type=['gpx'])
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # 간격 조정
        calculate_button = st.button("🧭 Calculate Route", type="primary", use_container_width=True)

if calculate_button and gpx_file:
    try:
        # Vessel data 생성 (선종 기반)
        vessel = VesselData(
            vessel_type=vessel_type,
            displacement=displacement,
            windage_area_side=windage_side,
            loa=loa,
            breadth=breadth,
            draft=draft,
            speed_knots=speed_knots
        )
        
        # 계산 과정을 expander 안에 표시
        progress_expander = st.expander("⚙️ Calculation Progress", expanded=True)
        
        with progress_expander:
            # 선박 정보 표시
            st.info(f"🚢 Vessel: {vessel_type} | Cb: {vessel.cb:.3f} | Wave Factor: {vessel.wave_resistance_factor}")
            
            st.info("📍 Parsing GPX track...")
            track_points = parse_gpx(gpx_file)
            
            if len(track_points) == 0:
                st.error("❌ No track points found in GPX file. Please check the file contains tracks, routes, or waypoints.")
                st.stop()
            
            if len(track_points) < 2:
                st.error("❌ At least 2 points are required for routing.")
                st.stop()
            
            st.success(f"✅ Loaded {len(track_points)} track points")
            
            # TrackLine 객체 생성
            track = TrackLine(track_points)
            st.info(f"📏 Total track distance: {track.total_distance:.1f} nm")
            
            # 예상 항해 시간 계산 (대략적)
            estimated_hours = track.total_distance / speed_knots
            if estimated_hours > 240:
                st.warning(f"⚠️ Estimated voyage: {estimated_hours:.0f} hours ({estimated_hours/24:.1f} days). Weather forecast is limited to ~10 days. Data beyond forecast range will be shown as NIL.")
            
            # Step 1 & 2: 초기 DR 위치 계산 (정속 기준, interval 적용)
            st.info(f"🧮 Calculating initial DR positions (interval: {interval_hours}h)...")
            initial_dr = calculate_dr_on_track(track, departure_datetime, speed_knots, interval_hours)
            st.success(f"✅ Generated {len(initial_dr)} DR positions")
            
            # Step 3: 초기 DR 위치들의 기상 데이터 조회 (예보 범위 체크)
            st.info("🌤️ Fetching weather data for initial positions...")
            initial_dr = fetch_weather_for_positions(initial_dr, api_key, departure_datetime)
            
            # 예보 범위 초과 포인트 수 체크
            no_weather_count = sum(1 for p in initial_dr if not p.get('weather_available', True))
            if no_weather_count > 0:
                st.warning(f"⚠️ {no_weather_count} positions are beyond weather forecast range (shown as NIL)")
            
            # 디버그: NOAA GFS 응답 확인
            if show_debug and initial_dr and len(initial_dr) > 1:
                with st.expander("🔍 Debug: NOAA GFS Response", expanded=True):
                    sample_point = initial_dr[1]
                    st.write("**GFS Cycle:**", sample_point.get('gfs_cycle', 'N/A'))
                    st.write("**Forecast Hour:**", sample_point.get('gfs_fhour', 'N/A'))
                    st.write("**Raw Weather Data:**", sample_point.get('raw_weather_data', 'N/A'))
                    
                    weather = sample_point.get('weather')
                    if weather:
                        st.write("**Parsed Weather:**")
                        st.write(f"  - Pressure: {weather.pressure}")
                        st.write(f"  - Wind: {weather.wind_dir}° / {weather.wind_speed} m/s")
                        st.write(f"  - Gust: {weather.wind_gust} m/s")
                        st.write(f"  - Wave: {weather.wave_dir}° / {weather.wave_height} m")
                        st.write(f"  - Swell: {weather.swell_dir}° / {weather.swell_height} m")
                    else:
                        st.write("**Weather object is None**")
                        st.write("**Weather Data:**")
                        st.write(f"  - Pressure: {weather.pressure}")
                        st.write(f"  - Wind: {weather.wind_dir}° / {weather.wind_speed} m/s")
                        st.write(f"  - Wave: {weather.wave_dir}° / {weather.wave_height} m")
                        st.write(f"  - Swell: {weather.swell_dir}° / {weather.swell_height} m")
            
            # Step 4: 기상 영향 반영하여 DR 재계산
            st.info("🔄 Recalculating DR with weather effects...")
            updated_dr = recalculate_dr_with_weather(initial_dr, track, vessel, departure_datetime, interval_hours)
            
            # Step 5: 재계산된 위치의 기상 데이터 다시 조회
            st.info("🌤️ Fetching weather data for updated positions...")
            final_dr = fetch_weather_for_positions(updated_dr, api_key, departure_datetime)
            
            # 결과 표시
            st.success("✅ Weather routing calculation completed!")
        
        # 계산 완료 플래그 설정
        st.session_state.calculation_done = True
        st.session_state.final_dr = final_dr
        st.session_state.track_points = track_points
        st.session_state.departure_datetime = departure_datetime
        st.session_state.arrival_tz = arrival_tz
        st.session_state.speed_knots_saved = speed_knots  # 테이블용
        
        st.markdown("---")
        
        st.header("📊 Routing Results")
        
        # 요약 정보
        eta_utc = final_dr[-1]['time']
        eta_arr_local = eta_utc + timedelta(hours=arrival_tz)
        voyage_time = (eta_utc - departure_datetime).total_seconds() / 3600
        avg_speed = final_dr[-1]['distance_sailed'] / voyage_time if voyage_time > 0 else 0
        
        tz_label = f"UTC{'+' if arrival_tz >= 0 else ''}{arrival_tz}"
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Distance", f"{final_dr[-1]['distance_sailed']:.1f} nm")
        with col2:
            st.metric(f"ETA ({tz_label})", eta_arr_local.strftime('%m/%d %H:%M'))
        with col3:
            st.metric("Voyage Time", f"{voyage_time:.1f} hrs")
        with col4:
            st.metric("Avg Speed", f"{avg_speed:.1f} kt")
        
        # 지도 표시
        st.subheader("🗺️ Route Map")
        route_map = create_route_map(track_points, final_dr)
        st_folium(route_map, width=None, height=500, use_container_width=True)
        
        # 테이블 표시 (HTML with rotated arrows)
        st.subheader("📋 Detailed Forecast")
        table_html = create_results_table_html(final_dr, speed_knots)
        
        # st.components.v1.html 사용하여 HTML 렌더링
        import streamlit.components.v1 as components
        
        # 테이블 행 수에 따라 높이 동적 계산
        table_height = min(600, 50 + len(final_dr) * 40)
        components.html(table_html, height=table_height, scrolling=True)
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

elif calculate_button:
    if not gpx_file:
        st.warning("⚠️ Please upload a GPX file")

# 이전 계산 결과가 있으면 표시 (새로 계산하지 않은 경우)
elif st.session_state.calculation_done and 'final_dr' in st.session_state and not calculate_button:
    final_dr = st.session_state.final_dr
    track_points = st.session_state.get('track_points', [])
    departure_datetime = st.session_state.departure_datetime
    arrival_tz = st.session_state.arrival_tz
    saved_speed = st.session_state.get('speed_knots_saved', speed_knots)
    
    st.markdown("---")
    st.header("📊 Routing Results")
    
    # 요약 정보
    eta_utc = final_dr[-1]['time']
    eta_arr_local = eta_utc + timedelta(hours=arrival_tz)
    voyage_time = (eta_utc - departure_datetime).total_seconds() / 3600
    avg_speed = final_dr[-1]['distance_sailed'] / voyage_time if voyage_time > 0 else 0
    
    tz_label = f"UTC{'+' if arrival_tz >= 0 else ''}{arrival_tz}"
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Distance", f"{final_dr[-1]['distance_sailed']:.1f} nm")
    with col2:
        st.metric(f"ETA ({tz_label})", eta_arr_local.strftime('%m/%d %H:%M'))
    with col3:
        st.metric("Voyage Time", f"{voyage_time:.1f} hrs")
    with col4:
        st.metric("Avg Speed", f"{avg_speed:.1f} kt")
    
    # 지도 표시 (track_points가 있을 때만)
    if track_points:
        st.subheader("🗺️ Route Map")
        route_map = create_route_map(track_points, final_dr)
        st_folium(route_map, width=None, height=500, use_container_width=True)
    
    # 테이블 표시 (HTML with rotated arrows)
    st.subheader("📋 Detailed Forecast")
    table_html = create_results_table_html(final_dr, saved_speed)
    
    import streamlit.components.v1 as components
    table_height = min(600, 50 + len(final_dr) * 40)
    components.html(table_html, height=table_height, scrolling=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
Weather Routing Calculator | Data Source: NOAA GFS & GFS-Wave (0.25° Resolution)
</div>
""", unsafe_allow_html=True)
