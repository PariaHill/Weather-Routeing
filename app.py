import streamlit as st
import gpxpy
import requests
from datetime import datetime, timedelta
import math
import pandas as pd
from typing import List, Tuple, Dict
import json

# Page config
st.set_page_config(page_title="Weather Routing Calculator", layout="wide")

# Initialize session state
if 'vessel_data' not in st.session_state:
    st.session_state.vessel_data = {}

class VesselData:
    """선박 제원 데이터"""
    def __init__(self, displacement, windage_area_front, windage_area_side, 
                 loa, breadth, draft, speed_knots):
        self.displacement = displacement  # 톤
        self.windage_area_front = windage_area_front  # m²
        self.windage_area_side = windage_area_side  # m²
        self.loa = loa  # m
        self.breadth = breadth  # m
        self.draft = draft  # m
        self.speed_knots = speed_knots  # 노트

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
    gpx = gpxpy.parse(gpx_file)
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

def calculate_initial_dr_positions(track_points: List[Tuple[float, float]], 
                                   start_time: datetime, 
                                   speed_knots: float,
                                   interval_hours: int = 6) -> List[Dict]:
    """초기 DR 위치 계산 (6시간 간격)"""
    dr_positions = []
    
    # 전체 항로의 총 거리와 bearing 계산
    total_distance = 0
    for i in range(len(track_points) - 1):
        dist = calculate_distance(track_points[i][0], track_points[i][1],
                                 track_points[i+1][0], track_points[i+1][1])
        total_distance += dist
    
    # 시작점
    current_time = start_time
    current_lat, current_lon = track_points[0]
    dr_positions.append({
        'time': current_time,
        'lat': current_lat,
        'lon': current_lon,
        'distance_sailed': 0,
        'distance_remaining': total_distance
    })
    
    # 6시간 간격으로 DR 계산
    distance_sailed = 0
    track_idx = 0
    
    while distance_sailed < total_distance:
        current_time += timedelta(hours=interval_hours)
        distance_to_sail = speed_knots * interval_hours
        distance_sailed += distance_to_sail
        
        if distance_sailed >= total_distance:
            # 목적지 도달
            current_lat, current_lon = track_points[-1]
            distance_remaining = 0
        else:
            # 현재 구간에서 위치 찾기
            accumulated_dist = 0
            for i in range(track_idx, len(track_points) - 1):
                seg_dist = calculate_distance(track_points[i][0], track_points[i][1],
                                             track_points[i+1][0], track_points[i+1][1])
                
                if accumulated_dist + seg_dist >= distance_to_sail:
                    # 이 구간에 위치
                    remaining_in_seg = distance_to_sail - accumulated_dist
                    bearing = calculate_bearing(track_points[i][0], track_points[i][1],
                                              track_points[i+1][0], track_points[i+1][1])
                    current_lat, current_lon = rhumb_line_destination(
                        track_points[i][0], track_points[i][1], bearing, remaining_in_seg
                    )
                    track_idx = i
                    break
                
                accumulated_dist += seg_dist
            
            distance_remaining = total_distance - distance_sailed
        
        dr_positions.append({
            'time': current_time,
            'lat': current_lat,
            'lon': current_lon,
            'distance_sailed': distance_sailed,
            'distance_remaining': distance_remaining
        })
        
        if distance_sailed >= total_distance:
            break
    
    return dr_positions

def get_windy_weather(lat: float, lon: float, api_key: str) -> Dict:
    """Windy API로 기상 데이터 조회"""
    weather_data = {}
    
    # GFS 모델 (wind, pressure)
    try:
        gfs_payload = {
            "lat": lat,
            "lon": lon,
            "model": "gfs",
            "parameters": ["wind", "windGust", "pressure"],
            "levels": ["surface"],
            "key": api_key
        }
        
        gfs_response = requests.post(
            "https://api.windy.com/api/point-forecast/v2",
            json=gfs_payload,
            timeout=10
        )
        
        if gfs_response.status_code == 200:
            gfs_data = gfs_response.json()
            weather_data['gfs'] = gfs_data
    except Exception as e:
        st.warning(f"GFS data fetch failed: {e}")
    
    # GFS Wave 모델
    try:
        wave_payload = {
            "lat": lat,
            "lon": lon,
            "model": "gfsWave",
            "parameters": ["waves", "swell1", "swell2"],
            "levels": ["surface"],
            "key": api_key
        }
        
        wave_response = requests.post(
            "https://api.windy.com/api/point-forecast/v2",
            json=wave_payload,
            timeout=10
        )
        
        if wave_response.status_code == 200:
            wave_data = wave_response.json()
            weather_data['wave'] = wave_data
        else:
            # 디버그: 응답 상태 확인
            weather_data['wave_error'] = f"Status: {wave_response.status_code}"
    except Exception as e:
        st.warning(f"Wave data fetch failed: {e}")
    
    return weather_data

def parse_windy_data(weather_data: Dict, target_time: datetime) -> WeatherPoint:
    """Windy API 응답에서 가장 가까운 시간의 데이터 추출"""
    result = WeatherPoint(target_time, 0, 0)
    
    if 'gfs' in weather_data:
        gfs = weather_data['gfs']
        timestamps = gfs.get('ts', [])
        
        # 가장 가까운 시간 찾기
        target_ts = int(target_time.timestamp() * 1000)
        closest_idx = 0
        min_diff = abs(timestamps[0] - target_ts)
        
        for i, ts in enumerate(timestamps):
            diff = abs(ts - target_ts)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i
        
        # Wind 데이터
        if 'wind_u-surface' in gfs and 'wind_v-surface' in gfs:
            u = gfs['wind_u-surface'][closest_idx]
            v = gfs['wind_v-surface'][closest_idx]
            wind_speed = math.sqrt(u**2 + v**2)
            wind_dir = (math.degrees(math.atan2(u, v)) + 180) % 360  # Coming from
            result.wind_speed = wind_speed
            result.wind_dir = wind_dir
        
        # Wind gust
        if 'gust-surface' in gfs:
            result.wind_gust = gfs['gust-surface'][closest_idx]
        
        # Pressure
        if 'pressure-surface' in gfs:
            result.pressure = gfs['pressure-surface'][closest_idx]
    
    if 'wave' in weather_data:
        wave = weather_data['wave']
        timestamps = wave.get('ts', [])
        
        if timestamps:
            target_ts = int(target_time.timestamp() * 1000)
            closest_idx = 0
            min_diff = abs(timestamps[0] - target_ts)
            
            for i, ts in enumerate(timestamps):
                diff = abs(ts - target_ts)
                if diff < min_diff:
                    min_diff = diff
                    closest_idx = i
            
            # Wave 높이 - 여러 가능한 키 시도
            wave_height_keys = ['waves_height-surface', 'waves-surface', 'wavesHeight-surface']
            for key in wave_height_keys:
                if key in wave:
                    result.wave_height = wave[key][closest_idx]
                    break
            
            # Wave 방향
            wave_dir_keys = ['waves_direction-surface', 'wavesDirection-surface', 'waves_dir-surface']
            for key in wave_dir_keys:
                if key in wave:
                    result.wave_dir = wave[key][closest_idx]
                    break
            
            # Swell 높이
            swell_height_keys = ['swell1_height-surface', 'swell1-surface', 'swellHeight-surface']
            for key in swell_height_keys:
                if key in wave:
                    result.swell_height = wave[key][closest_idx]
                    break
            
            # Swell 방향
            swell_dir_keys = ['swell1_direction-surface', 'swell1Direction-surface', 'swell1_dir-surface']
            for key in swell_dir_keys:
                if key in wave:
                    result.swell_dir = wave[key][closest_idx]
                    break
    
    return result

def calculate_wind_resistance(vessel: VesselData, wind_speed_ms: float, 
                              wind_dir: float, vessel_heading: float) -> float:
    """풍압저항 계산 (kN)"""
    # Relative wind angle
    relative_angle = (wind_dir - vessel_heading + 360) % 360
    if relative_angle > 180:
        relative_angle = 360 - relative_angle
    
    relative_angle_rad = math.radians(relative_angle)
    
    # 항력계수 (각도에 따라 변화)
    if relative_angle < 45:
        Cd = 0.75
        area = vessel.windage_area_front
    elif relative_angle < 90:
        Cd = 0.6
        area = (vessel.windage_area_front + vessel.windage_area_side) / 2
    else:
        Cd = 0.5
        area = vessel.windage_area_side
    
    # 상대풍속 (선박 속력도 고려해야 하지만 간단히 처리)
    rho_air = 1.225  # kg/m³
    
    # 풍압저항 (N)
    R_wind = 0.5 * rho_air * Cd * area * (wind_speed_ms ** 2) * abs(math.cos(relative_angle_rad))
    
    return R_wind / 1000  # kN

def calculate_wave_resistance(vessel: VesselData, wave_height: float, 
                              wave_dir: float, vessel_heading: float) -> float:
    """파랑저항 계산 (kN) - Kwon 간략식 사용"""
    if wave_height < 0.5:
        return 0
    
    # Relative wave angle
    relative_angle = (wave_dir - vessel_heading + 360) % 360
    if relative_angle > 180:
        relative_angle = 360 - relative_angle
    
    relative_angle_rad = math.radians(relative_angle)
    
    # 간략화된 파랑저항 공식
    # R_wave = C * B * d * H^2 * cos(μ)
    # C는 경험계수 (약 20-30)
    C = 25
    B = vessel.breadth
    d = vessel.draft
    H = wave_height
    
    R_wave = C * B * d * (H ** 2) * abs(math.cos(relative_angle_rad))
    
    return R_wave  # kN

def calculate_speed_loss(vessel: VesselData, weather: WeatherPoint, 
                        vessel_heading: float) -> float:
    """속력 손실 계산 (노트)"""
    total_resistance = 0
    
    # 바람에 의한 저항
    if weather.wind_speed:
        R_wind = calculate_wind_resistance(vessel, weather.wind_speed, 
                                          weather.wind_dir, vessel_heading)
        total_resistance += R_wind
    
    # 파도에 의한 저항
    if weather.wave_height:
        R_wave = calculate_wave_resistance(vessel, weather.wave_height,
                                          weather.wave_dir, vessel_heading)
        total_resistance += R_wave
    
    # Swell도 고려
    if weather.swell_height:
        R_swell = calculate_wave_resistance(vessel, weather.swell_height,
                                           weather.swell_dir or weather.wave_dir or 0,
                                           vessel_heading)
        total_resistance += R_swell * 0.5  # Swell은 wave보다 영향 적음
    
    # 저항을 속력 손실로 변환 (경험식)
    # 간단한 근사: 저항이 두배가 되면 속력이 약 15% 감소
    # ΔV = k * (R_added / R_calm)^0.5 * V_calm
    
    # 평수중 저항 추정 (단순화)
    R_calm = vessel.displacement * 0.01  # 매우 간략한 근사
    
    speed_loss_factor = math.sqrt(total_resistance / max(R_calm, 1))
    speed_loss = speed_loss_factor * vessel.speed_knots * 0.15  # 최대 15% 감소
    
    # 속력 손실 제한 (0 ~ 4 노트)
    speed_loss = max(0, min(speed_loss, 4))
    
    return speed_loss

def recalculate_dr_with_weather(initial_dr: List[Dict], vessel: VesselData,
                                track_points: List[Tuple[float, float]],
                                api_key: str) -> List[Dict]:
    """기상 데이터를 반영하여 DR 재계산"""
    updated_dr = []
    
    # 첫 포인트는 그대로
    updated_dr.append(initial_dr[0].copy())
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(1, len(initial_dr)):
        status_text.text(f"Fetching weather data: {i}/{len(initial_dr)-1}")
        progress_bar.progress(i / (len(initial_dr) - 1))
        
        prev_point = updated_dr[-1]
        current_point = initial_dr[i]
        
        # 기상 데이터 조회
        weather_data = get_windy_weather(prev_point['lat'], prev_point['lon'], api_key)
        weather = parse_windy_data(weather_data, prev_point['time'])
        
        # 현재 DR 위치에서 목적지 방향으로 heading 계산
        # track_points에서 현재 위치보다 앞에 있는 가장 가까운 경유점 찾기
        current_lat, current_lon = prev_point['lat'], prev_point['lon']
        
        # 가장 가까운 다음 경유점 찾기
        target_idx = len(track_points) - 1  # 기본값: 최종 목적지
        for idx in range(len(track_points) - 1):
            # 현재 위치에서 각 경유점까지 거리 확인
            dist_to_waypoint = calculate_distance(current_lat, current_lon,
                                                  track_points[idx + 1][0], track_points[idx + 1][1])
            if dist_to_waypoint > 1:  # 1해리 이상 떨어져 있으면 이 경유점을 목표로
                target_idx = idx + 1
                break
        
        vessel_heading = calculate_bearing(current_lat, current_lon,
                                          track_points[target_idx][0], track_points[target_idx][1])
        
        # 속력 손실 계산
        speed_loss = calculate_speed_loss(vessel, weather, vessel_heading)
        actual_speed = max(vessel.speed_knots - speed_loss, 3)  # 최소 3노트
        
        # 실제 항해 거리
        time_interval = (current_point['time'] - prev_point['time']).total_seconds() / 3600
        distance = actual_speed * time_interval
        
        # 새 위치 계산
        new_lat, new_lon = rhumb_line_destination(prev_point['lat'], prev_point['lon'],
                                                   vessel_heading, distance)
        
        # 누적 거리 계산
        distance_sailed = prev_point['distance_sailed'] + distance
        
        # 남은 거리는 목적지까지 직선거리로 재계산
        distance_remaining = calculate_distance(new_lat, new_lon,
                                               track_points[-1][0], track_points[-1][1])
        
        updated_dr.append({
            'time': current_point['time'],
            'lat': new_lat,
            'lon': new_lon,
            'distance_sailed': distance_sailed,
            'distance_remaining': distance_remaining,
            'weather': weather,
            'heading': vessel_heading,
            'actual_speed': actual_speed,
            'speed_loss': speed_loss
        })
    
    progress_bar.empty()
    status_text.empty()
    
    return updated_dr

def refine_dr_with_updated_positions(dr_positions: List[Dict], vessel: VesselData,
                                     api_key: str) -> List[Dict]:
    """업데이트된 DR 위치로 기상 재조회"""
    refined_dr = []
    refined_dr.append(dr_positions[0].copy())
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(1, len(dr_positions)):
        status_text.text(f"Refining weather data: {i}/{len(dr_positions)-1}")
        progress_bar.progress(i / (len(dr_positions) - 1))
        
        point = dr_positions[i]
        
        # 새 위치에서 기상 재조회
        weather_data = get_windy_weather(point['lat'], point['lon'], api_key)
        weather = parse_windy_data(weather_data, point['time'])
        
        refined_point = point.copy()
        refined_point['weather'] = weather
        refined_dr.append(refined_point)
    
    progress_bar.empty()
    status_text.empty()
    
    return refined_dr

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

def create_results_table_html(dr_positions: List[Dict]) -> str:
    """결과 테이블을 HTML로 생성 (SVG 화살표 포함)"""
    
    html = '''
    <style>
        .weather-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }
        .weather-table th {
            background-color: #f0f2f6;
            padding: 8px 12px;
            text-align: left;
            border-bottom: 2px solid #ddd;
            white-space: nowrap;
        }
        .weather-table td {
            padding: 8px 12px;
            border-bottom: 1px solid #eee;
            white-space: nowrap;
        }
        .weather-table tr:hover {
            background-color: #f8f9fa;
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
                <th>Pressure</th>
                <th>Wind</th>
                <th>Wave</th>
                <th>Sailed</th>
                <th>Remaining</th>
                <th>Est. Speed</th>
            </tr>
        </thead>
        <tbody>
    '''
    
    for point in dr_positions:
        weather = point.get('weather')
        utc_time = point['time'].strftime('%Y-%m-%d %H:%M')
        lat_str = decimal_to_dms(point['lat'], is_lat=True)
        lon_str = decimal_to_dms(point['lon'], is_lat=False)
        
        # Pressure
        pressure = f"{weather.pressure:.1f}" if weather and weather.pressure else "N/A"
        
        # Wind with arrow
        if weather and weather.wind_dir is not None and weather.wind_speed is not None:
            wind_arrow = f'<span class="arrow-svg" style="display:inline-block; transform:rotate({weather.wind_dir}deg);">↓</span>'
            wind_str = f'{wind_arrow} {weather.wind_dir:.0f}° / {ms_to_knots(weather.wind_speed):.1f}kt'
        else:
            wind_str = "N/A"
        
        # Wave with arrow
        if weather and weather.wave_dir is not None and weather.wave_height is not None:
            wave_arrow = f'<span class="arrow-svg" style="display:inline-block; transform:rotate({weather.wave_dir}deg);">↓</span>'
            wave_str = f'{wave_arrow} {weather.wave_dir:.0f}° / {weather.wave_height:.1f}m'
        else:
            wave_str = "N/A"
        
        sailed = f"{point['distance_sailed']:.1f}"
        remaining = f"{point['distance_remaining']:.1f}"
        est_speed = f"{point.get('actual_speed', 0):.1f}" if 'actual_speed' in point else "N/A"
        
        html += f'''
            <tr>
                <td>{utc_time}</td>
                <td>{lat_str}</td>
                <td>{lon_str}</td>
                <td>{pressure}</td>
                <td>{wind_str}</td>
                <td>{wave_str}</td>
                <td>{sailed}</td>
                <td>{remaining}</td>
                <td>{est_speed}</td>
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

# Streamlit UI
st.title("⛵ Weather Routing Calculator")
st.markdown("---")

# Sidebar - 선박 데이터 입력
with st.sidebar:
    st.header("Vessel Data")
    
    displacement = st.number_input("Displacement (ton)", min_value=100.0, value=5000.0, step=100.0)
    windage_front = st.number_input("Windage Area Front (m²)", min_value=10.0, value=500.0, step=10.0)
    windage_side = st.number_input("Windage Area Side (m²)", min_value=10.0, value=800.0, step=10.0)
    loa = st.number_input("LOA (m)", min_value=10.0, value=115.0, step=1.0)
    breadth = st.number_input("Breadth (m)", min_value=5.0, value=20.0, step=0.5)
    draft = st.number_input("Draft (m)", min_value=1.0, value=5.5, step=0.1)
    
    st.markdown("---")
    st.header("Voyage Data")
    
    speed_knots = st.number_input("Speed through water (knots)", min_value=1.0, value=11.0, step=0.5)
    
    # Time Zone 옵션 생성 (-12 ~ +13)
    tz_options = [f"UTC{'+' if i >= 0 else ''}{i}" for i in range(-12, 14)]
    tz_values = list(range(-12, 14))
    
    col_dep, col_arr = st.columns(2)
    with col_dep:
        dep_tz_idx = st.selectbox("Departure Zone", options=range(len(tz_options)), 
                                   format_func=lambda x: tz_options[x], index=12)  # UTC+0 기본값
        departure_tz = tz_values[dep_tz_idx]
    with col_arr:
        arr_tz_idx = st.selectbox("Arrival Zone", options=range(len(tz_options)), 
                                   format_func=lambda x: tz_options[x], index=21)  # UTC+9 기본값 (한국)
        arrival_tz = tz_values[arr_tz_idx]
    
    departure_date = st.date_input("Departure Date (LT)", datetime.now().date())
    departure_time = st.time_input("Departure Time (LT)", datetime.now().time())
    
    # 로컬 시간을 UTC로 변환
    departure_local = datetime.combine(departure_date, departure_time)
    departure_datetime = departure_local - timedelta(hours=departure_tz)
    
    st.markdown("---")
    # Windy API 키는 Streamlit secrets에서만 읽음
    try:
        api_key = st.secrets["WINDY_API_KEY"]
        st.success("✅ API Key loaded")
    except:
        api_key = ""
        st.error("❌ WINDY_API_KEY not found in secrets")
    
    st.markdown("---")
    st.header("Debug Options")
    show_debug = st.checkbox("Show API response keys", value=False)

# Main area - Expander로 접을 수 있게
with st.expander("📁 Upload GPX Track & Actions", expanded=True):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        gpx_file = st.file_uploader("Choose a GPX file", type=['gpx'])
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # 간격 조정
        calculate_button = st.button("🧭 Calculate Route", type="primary", use_container_width=True)

if calculate_button and gpx_file and api_key:
    try:
        # Vessel data 생성
        vessel = VesselData(
            displacement=displacement,
            windage_area_front=windage_front,
            windage_area_side=windage_side,
            loa=loa,
            breadth=breadth,
            draft=draft,
            speed_knots=speed_knots
        )
        
        # 계산 과정을 expander 안에 표시
        progress_expander = st.expander("⚙️ Calculation Progress", expanded=True)
        
        with progress_expander:
            st.info("📍 Parsing GPX track...")
            track_points = parse_gpx(gpx_file)
            
            if len(track_points) == 0:
                st.error("❌ No track points found in GPX file. Please check the file contains tracks, routes, or waypoints.")
                st.stop()
            
            if len(track_points) < 2:
                st.error("❌ At least 2 points are required for routing.")
                st.stop()
            
            st.success(f"✅ Loaded {len(track_points)} track points")
            
            # 초기 DR 계산
            st.info("🧮 Calculating initial DR positions...")
            initial_dr = calculate_initial_dr_positions(track_points, departure_datetime, speed_knots)
            st.success(f"✅ Generated {len(initial_dr)} DR positions")
            
            # 첫번째 반복: 기상 데이터로 DR 재계산
            st.info("🌤️ Fetching weather data and recalculating DR...")
            updated_dr = recalculate_dr_with_weather(initial_dr, vessel, track_points, api_key)
            
            # 디버그: API 응답 키 확인
            if show_debug and updated_dr and len(updated_dr) > 1 and 'weather' in updated_dr[1]:
                # 첫 번째 기상 데이터 포인트에서 원본 데이터 확인을 위해 다시 조회
                test_weather = get_windy_weather(updated_dr[1]['lat'], updated_dr[1]['lon'], api_key)
                with st.expander("🔍 Debug: API Response Keys", expanded=False):
                    if 'gfs' in test_weather:
                        st.write("**GFS Keys:**", list(test_weather['gfs'].keys()))
                    if 'wave' in test_weather:
                        st.write("**Wave Keys:**", list(test_weather['wave'].keys()))
                    if 'wave_error' in test_weather:
                        st.write("**Wave Error:**", test_weather['wave_error'])
            
            # 두번째 반복: 업데이트된 위치에서 기상 재조회
            st.info("🔄 Refining with updated positions...")
            final_dr = refine_dr_with_updated_positions(updated_dr, vessel, api_key)
            
            # 결과 표시
            st.success("✅ Weather routing calculation completed!")
        
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
        
        # 테이블 표시 (HTML with rotated arrows)
        st.subheader("Detailed Forecast")
        table_html = create_results_table_html(final_dr)
        st.markdown(table_html, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

elif calculate_button:
    if not gpx_file:
        st.warning("⚠️ Please upload a GPX file")
    if not api_key:
        st.warning("⚠️ Please provide Windy API key")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
Weather Routing Calculator | Wind/Wave data from Windy.com
</div>
""", unsafe_allow_html=True)
