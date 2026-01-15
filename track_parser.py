"""
Track/Route File Parser Module for Weather Routing Calculator

지원 형식:
- GPX (GPS Exchange Format) - 트랙, 루트, 웨이포인트
- RTZ (IEC 61174 Route Exchange Format) - ECDIS 표준
- Furuno ROU - Furuno ECDIS 전용 형식
- CSV/TXT - 간단한 위도/경도 목록
- JSON - GeoJSON LineString

향후 추가 예정:
- JRC ECDIS 형식
- Transas ECDIS 형식
- KML/KMZ (Google Earth)
"""

import xml.etree.ElementTree as ET
import json
import re
from typing import List, Tuple, Optional, Dict, Any
from io import StringIO, BytesIO


class RouteParseResult:
    """파싱 결과를 담는 클래스"""
    def __init__(self, points: List[Tuple[float, float]], 
                 format_name: str,
                 route_name: Optional[str] = None,
                 waypoint_names: Optional[List[str]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.points = points
        self.format_name = format_name
        self.route_name = route_name
        self.waypoint_names = waypoint_names or []
        self.metadata = metadata or {}
    
    def __len__(self):
        return len(self.points)
    
    @property
    def is_valid(self):
        return len(self.points) >= 2


def detect_format(file_content: bytes, filename: str) -> str:
    """파일 내용과 확장자로 형식 감지"""
    ext = filename.lower().split('.')[-1] if '.' in filename else ''
    
    # 확장자 기반 1차 판단
    if ext == 'gpx':
        return 'gpx'
    elif ext == 'rtz':
        return 'rtz'
    elif ext == 'rou':
        return 'furuno'
    elif ext in ('csv', 'txt'):
        return 'csv'
    elif ext == 'json' or ext == 'geojson':
        return 'geojson'
    
    # 내용 기반 2차 판단
    try:
        content_str = file_content.decode('utf-8', errors='ignore')[:2000]
    except:
        content_str = ''
    
    if '<gpx' in content_str.lower():
        return 'gpx'
    elif '<route' in content_str.lower() and 'xmlns' in content_str:
        return 'rtz'
    elif content_str.startswith('{') or content_str.startswith('['):
        return 'geojson'
    elif re.search(r'-?\d+\.?\d*[,\s]+-?\d+\.?\d*', content_str):
        return 'csv'
    
    return 'unknown'


def parse_route_file(file_obj, filename: str = "route") -> RouteParseResult:
    """
    범용 라우트 파일 파서
    
    Args:
        file_obj: 파일 객체 (Streamlit UploadedFile 또는 일반 파일)
        filename: 파일명 (확장자 감지용)
    
    Returns:
        RouteParseResult: 파싱 결과
    """
    # 파일 내용 읽기
    if hasattr(file_obj, 'read'):
        content = file_obj.read()
        if isinstance(content, str):
            content = content.encode('utf-8')
        # 파일 포인터 리셋
        if hasattr(file_obj, 'seek'):
            file_obj.seek(0)
    else:
        content = file_obj
    
    # 형식 감지
    format_type = detect_format(content, filename)
    
    # 형식별 파서 호출
    parsers = {
        'gpx': parse_gpx,
        'rtz': parse_rtz,
        'furuno': parse_furuno_rou,
        'csv': parse_csv,
        'geojson': parse_geojson,
    }
    
    parser = parsers.get(format_type)
    if parser:
        try:
            return parser(content)
        except Exception as e:
            return RouteParseResult(
                points=[],
                format_name=format_type,
                metadata={'error': str(e)}
            )
    
    return RouteParseResult(
        points=[],
        format_name='unknown',
        metadata={'error': 'Unsupported file format'}
    )


def parse_gpx(content: bytes) -> RouteParseResult:
    """
    GPX (GPS Exchange Format) 파서
    
    지원:
    - <trk>/<trkseg>/<trkpt> - 트랙 포인트
    - <rte>/<rtept> - 루트 포인트
    - <wpt> - 웨이포인트
    """
    try:
        import gpxpy
        
        gpx = gpxpy.parse(content.decode('utf-8'))
        points = []
        waypoint_names = []
        route_name = None
        
        # 1. 트랙 포인트
        for track in gpx.tracks:
            if not route_name and track.name:
                route_name = track.name
            for segment in track.segments:
                for point in segment.points:
                    points.append((point.latitude, point.longitude))
                    waypoint_names.append(point.name or f"TRK{len(points)}")
        
        # 2. 루트 포인트 (트랙이 없을 경우)
        if not points:
            for route in gpx.routes:
                if not route_name and route.name:
                    route_name = route.name
                for point in route.points:
                    points.append((point.latitude, point.longitude))
                    waypoint_names.append(point.name or f"WPT{len(points)}")
        
        # 3. 웨이포인트 (트랙, 루트가 없을 경우)
        if not points:
            for waypoint in gpx.waypoints:
                points.append((waypoint.latitude, waypoint.longitude))
                waypoint_names.append(waypoint.name or f"WPT{len(points)}")
        
        return RouteParseResult(
            points=points,
            format_name='GPX',
            route_name=route_name,
            waypoint_names=waypoint_names,
            metadata={
                'creator': gpx.creator,
                'track_count': len(gpx.tracks),
                'route_count': len(gpx.routes),
                'waypoint_count': len(gpx.waypoints)
            }
        )
    
    except ImportError:
        # gpxpy가 없으면 기본 XML 파싱
        return _parse_gpx_basic(content)


def _parse_gpx_basic(content: bytes) -> RouteParseResult:
    """gpxpy 없이 기본 XML로 GPX 파싱"""
    try:
        root = ET.fromstring(content)
        ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}
        
        points = []
        waypoint_names = []
        
        # 네임스페이스 자동 감지
        if root.tag.startswith('{'):
            ns_uri = root.tag[1:root.tag.index('}')]
            ns = {'gpx': ns_uri}
        
        # trkpt 찾기
        for trkpt in root.findall('.//gpx:trkpt', ns) or root.findall('.//trkpt'):
            lat = float(trkpt.get('lat'))
            lon = float(trkpt.get('lon'))
            points.append((lat, lon))
            name_elem = trkpt.find('gpx:name', ns) or trkpt.find('name')
            waypoint_names.append(name_elem.text if name_elem is not None else f"TRK{len(points)}")
        
        # rtept 찾기
        if not points:
            for rtept in root.findall('.//gpx:rtept', ns) or root.findall('.//rtept'):
                lat = float(rtept.get('lat'))
                lon = float(rtept.get('lon'))
                points.append((lat, lon))
                name_elem = rtept.find('gpx:name', ns) or rtept.find('name')
                waypoint_names.append(name_elem.text if name_elem is not None else f"WPT{len(points)}")
        
        # wpt 찾기
        if not points:
            for wpt in root.findall('.//gpx:wpt', ns) or root.findall('.//wpt'):
                lat = float(wpt.get('lat'))
                lon = float(wpt.get('lon'))
                points.append((lat, lon))
                name_elem = wpt.find('gpx:name', ns) or wpt.find('name')
                waypoint_names.append(name_elem.text if name_elem is not None else f"WPT{len(points)}")
        
        return RouteParseResult(
            points=points,
            format_name='GPX',
            waypoint_names=waypoint_names
        )
    
    except Exception as e:
        return RouteParseResult(
            points=[],
            format_name='GPX',
            metadata={'error': str(e)}
        )


def parse_rtz(content: bytes) -> RouteParseResult:
    """
    RTZ (IEC 61174 Route Exchange Format) 파서
    ECDIS 표준 라우트 교환 형식
    
    구조:
    <route>
      <routeInfo routeName="..." />
      <waypoints>
        <waypoint id="1" name="WPT1">
          <position lat="35.123" lon="129.456" />
        </waypoint>
        ...
      </waypoints>
    </route>
    """
    try:
        root = ET.fromstring(content)
        
        points = []
        waypoint_names = []
        route_name = None
        
        # 네임스페이스 추출
        ns_uri = None
        if root.tag.startswith('{'):
            ns_uri = root.tag[1:root.tag.index('}')]
        
        def find_element(parent, tag):
            """네임스페이스 유무에 관계없이 요소 찾기"""
            if ns_uri:
                elem = parent.find(f'{{{ns_uri}}}{tag}')
                if elem is not None:
                    return elem
            return parent.find(tag) or parent.find(f'.//{tag}')
        
        def find_all_elements(parent, tag):
            """네임스페이스 유무에 관계없이 모든 요소 찾기"""
            if ns_uri:
                elems = parent.findall(f'.//{{{ns_uri}}}{tag}')
                if elems:
                    return elems
            return parent.findall(f'.//{tag}') or []
        
        # routeInfo에서 이름 추출
        route_info = find_element(root, 'routeInfo')
        if route_info is not None:
            route_name = route_info.get('routeName')
        
        # waypoints 추출
        waypoints = find_all_elements(root, 'waypoint')
        
        for wp in waypoints:
            # position 요소 찾기
            pos = find_element(wp, 'position')
            
            if pos is not None:
                lat = pos.get('lat')
                lon = pos.get('lon')
                
                if lat and lon:
                    points.append((float(lat), float(lon)))
                    wp_name = wp.get('name') or wp.get('id') or f"WPT{len(points)}"
                    waypoint_names.append(wp_name)
        
        return RouteParseResult(
            points=points,
            format_name='RTZ (IEC 61174)',
            route_name=route_name,
            waypoint_names=waypoint_names,
            metadata={
                'waypoint_count': len(points)
            }
        )
    
    except Exception as e:
        return RouteParseResult(
            points=[],
            format_name='RTZ',
            metadata={'error': str(e)}
        )


def parse_furuno_rou(content: bytes) -> RouteParseResult:
    """
    Furuno ECDIS ROU 파일 파서
    
    형식 (일반적인 구조):
    Route Name: TEST_ROUTE
    WPT,001,35.12345,129.56789,WPT001
    WPT,002,35.23456,129.67890,WPT002
    ...
    
    또는 XML 기반 형식
    """
    try:
        content_str = content.decode('utf-8', errors='ignore')
        
        points = []
        waypoint_names = []
        route_name = None
        
        # XML 형식 체크
        if '<' in content_str and '>' in content_str:
            return _parse_furuno_xml(content)
        
        # 텍스트 형식 파싱
        lines = content_str.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            # 라우트 이름
            if line.lower().startswith('route name'):
                route_name = line.split(':', 1)[1].strip() if ':' in line else None
                continue
            
            # WPT 라인 파싱
            if line.upper().startswith('WPT'):
                parts = line.split(',')
                if len(parts) >= 4:
                    try:
                        lat = float(parts[2])
                        lon = float(parts[3])
                        points.append((lat, lon))
                        wp_name = parts[4].strip() if len(parts) > 4 else f"WPT{parts[1]}"
                        waypoint_names.append(wp_name)
                    except ValueError:
                        continue
        
        return RouteParseResult(
            points=points,
            format_name='Furuno ROU',
            route_name=route_name,
            waypoint_names=waypoint_names
        )
    
    except Exception as e:
        return RouteParseResult(
            points=[],
            format_name='Furuno ROU',
            metadata={'error': str(e)}
        )


def _parse_furuno_xml(content: bytes) -> RouteParseResult:
    """Furuno XML 형식 ROU 파일 파서"""
    try:
        root = ET.fromstring(content)
        
        points = []
        waypoint_names = []
        route_name = None
        
        # 라우트 이름 찾기
        name_elem = root.find('.//RouteName') or root.find('.//routeName') or root.find('.//name')
        if name_elem is not None:
            route_name = name_elem.text
        
        # 웨이포인트 찾기 (다양한 태그명 시도)
        for tag in ['WayPoint', 'Waypoint', 'waypoint', 'WPT', 'wpt']:
            waypoints = root.findall(f'.//{tag}')
            if waypoints:
                break
        
        for wp in waypoints:
            lat = None
            lon = None
            name = None
            
            # 위도/경도 추출 (다양한 속성/요소명)
            for lat_tag in ['Lat', 'lat', 'latitude', 'Latitude']:
                lat_elem = wp.find(lat_tag) or wp.get(lat_tag)
                if lat_elem is not None:
                    lat = float(lat_elem if isinstance(lat_elem, str) else lat_elem.text)
                    break
            
            for lon_tag in ['Lon', 'lon', 'longitude', 'Longitude', 'Long']:
                lon_elem = wp.find(lon_tag) or wp.get(lon_tag)
                if lon_elem is not None:
                    lon = float(lon_elem if isinstance(lon_elem, str) else lon_elem.text)
                    break
            
            # 이름 추출
            for name_tag in ['Name', 'name', 'WPTName', 'ID', 'id']:
                name_elem = wp.find(name_tag) or wp.get(name_tag)
                if name_elem is not None:
                    name = name_elem if isinstance(name_elem, str) else name_elem.text
                    break
            
            if lat is not None and lon is not None:
                points.append((lat, lon))
                waypoint_names.append(name or f"WPT{len(points)}")
        
        return RouteParseResult(
            points=points,
            format_name='Furuno ROU (XML)',
            route_name=route_name,
            waypoint_names=waypoint_names
        )
    
    except Exception as e:
        return RouteParseResult(
            points=[],
            format_name='Furuno ROU',
            metadata={'error': str(e)}
        )


def parse_csv(content: bytes) -> RouteParseResult:
    """
    CSV/TXT 파일 파서
    
    지원 형식:
    - lat,lon
    - lat,lon,name
    - name,lat,lon
    - 공백/탭 구분
    
    첫 줄이 헤더인 경우 자동 감지
    """
    try:
        content_str = content.decode('utf-8', errors='ignore')
        lines = content_str.strip().split('\n')
        
        if not lines:
            return RouteParseResult(points=[], format_name='CSV')
        
        points = []
        waypoint_names = []
        
        # 구분자 감지
        first_data_line = lines[0]
        if '\t' in first_data_line:
            delimiter = '\t'
        elif ';' in first_data_line:
            delimiter = ';'
        else:
            delimiter = ','
        
        # 헤더 감지
        start_idx = 0
        first_parts = lines[0].lower().split(delimiter)
        if any(h in first_parts for h in ['lat', 'latitude', 'lon', 'longitude', 'name', 'waypoint']):
            start_idx = 1
            # 컬럼 인덱스 파악
            lat_idx = next((i for i, h in enumerate(first_parts) if 'lat' in h), 0)
            lon_idx = next((i for i, h in enumerate(first_parts) if 'lon' in h), 1)
            name_idx = next((i for i, h in enumerate(first_parts) if 'name' in h or 'waypoint' in h), -1)
        else:
            lat_idx, lon_idx, name_idx = 0, 1, 2
        
        for i, line in enumerate(lines[start_idx:], 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = [p.strip() for p in line.split(delimiter)]
            
            if len(parts) >= 2:
                try:
                    # 위도/경도 파싱 시도
                    lat_str = parts[lat_idx] if lat_idx < len(parts) else parts[0]
                    lon_str = parts[lon_idx] if lon_idx < len(parts) else parts[1]
                    
                    lat = _parse_coordinate(lat_str)
                    lon = _parse_coordinate(lon_str)
                    
                    if lat is not None and lon is not None:
                        if -90 <= lat <= 90 and -180 <= lon <= 180:
                            points.append((lat, lon))
                            
                            if name_idx >= 0 and name_idx < len(parts):
                                waypoint_names.append(parts[name_idx])
                            else:
                                waypoint_names.append(f"WPT{len(points)}")
                
                except (ValueError, IndexError):
                    continue
        
        return RouteParseResult(
            points=points,
            format_name='CSV/TXT',
            waypoint_names=waypoint_names
        )
    
    except Exception as e:
        return RouteParseResult(
            points=[],
            format_name='CSV',
            metadata={'error': str(e)}
        )


def _parse_coordinate(coord_str: str) -> Optional[float]:
    """
    다양한 좌표 형식 파싱
    
    지원:
    - 십진수: 35.12345
    - 도분: 35 07.407N
    - 도분초: 35°07'24.4"N
    """
    coord_str = coord_str.strip()
    
    # 십진수
    try:
        return float(coord_str)
    except ValueError:
        pass
    
    # 방향 지시자 처리
    direction = 1
    if coord_str[-1].upper() in 'SW':
        direction = -1
    coord_str = re.sub(r'[NSEW]', '', coord_str, flags=re.IGNORECASE).strip()
    
    # 도분초 형식: 35°07'24.4" 또는 35 07 24.4
    dms_match = re.match(r"(\d+)[°\s]+(\d+)['\s]+(\d+\.?\d*)", coord_str)
    if dms_match:
        d, m, s = map(float, dms_match.groups())
        return direction * (d + m/60 + s/3600)
    
    # 도분 형식: 35 07.407 또는 35°07.407'
    dm_match = re.match(r"(\d+)[°\s]+(\d+\.?\d*)", coord_str)
    if dm_match:
        d, m = map(float, dm_match.groups())
        return direction * (d + m/60)
    
    return None


def parse_geojson(content: bytes) -> RouteParseResult:
    """
    GeoJSON 파일 파서
    
    지원:
    - LineString
    - MultiLineString
    - FeatureCollection with LineString features
    """
    try:
        data = json.loads(content.decode('utf-8'))
        
        points = []
        route_name = None
        
        def extract_linestring(coords):
            """LineString 좌표 추출 (lon, lat 순서 → lat, lon으로 변환)"""
            return [(lat, lon) for lon, lat in coords]
        
        # 직접 geometry인 경우
        if data.get('type') == 'LineString':
            points = extract_linestring(data['coordinates'])
        
        elif data.get('type') == 'MultiLineString':
            for line in data['coordinates']:
                points.extend(extract_linestring(line))
        
        # Feature인 경우
        elif data.get('type') == 'Feature':
            geom = data.get('geometry', {})
            if geom.get('type') == 'LineString':
                points = extract_linestring(geom['coordinates'])
            route_name = data.get('properties', {}).get('name')
        
        # FeatureCollection인 경우
        elif data.get('type') == 'FeatureCollection':
            for feature in data.get('features', []):
                geom = feature.get('geometry', {})
                if geom.get('type') in ('LineString', 'MultiLineString'):
                    if geom['type'] == 'LineString':
                        points.extend(extract_linestring(geom['coordinates']))
                    else:
                        for line in geom['coordinates']:
                            points.extend(extract_linestring(line))
                    if not route_name:
                        route_name = feature.get('properties', {}).get('name')
        
        waypoint_names = [f"WPT{i+1}" for i in range(len(points))]
        
        return RouteParseResult(
            points=points,
            format_name='GeoJSON',
            route_name=route_name,
            waypoint_names=waypoint_names
        )
    
    except Exception as e:
        return RouteParseResult(
            points=[],
            format_name='GeoJSON',
            metadata={'error': str(e)}
        )


def get_supported_formats() -> List[str]:
    """지원되는 파일 확장자 목록 반환"""
    return ['gpx', 'rtz', 'rou', 'csv', 'txt', 'json', 'geojson']


def get_format_descriptions() -> Dict[str, str]:
    """형식별 설명 반환"""
    return {
        'gpx': 'GPS Exchange Format (GPX)',
        'rtz': 'IEC 61174 Route Exchange (ECDIS Standard)',
        'rou': 'Furuno ECDIS Route',
        'csv': 'Comma/Tab Separated Values',
        'txt': 'Plain Text (lat/lon list)',
        'json': 'GeoJSON LineString',
        'geojson': 'GeoJSON LineString'
    }


# 테스트용
if __name__ == "__main__":
    # 간단한 테스트
    test_csv = b"35.1234,129.5678,WPT1\n35.2345,129.6789,WPT2"
    result = parse_csv(test_csv)
    print(f"CSV Test: {len(result)} points")
    
    test_gpx = b'''<?xml version="1.0"?>
    <gpx version="1.1">
        <trk><trkseg>
            <trkpt lat="35.1234" lon="129.5678"/>
            <trkpt lat="35.2345" lon="129.6789"/>
        </trkseg></trk>
    </gpx>'''
    result = parse_gpx(test_gpx)
    print(f"GPX Test: {len(result)} points")
