import requests
import pandas as pd
import numpy as np
import polyline
def get_route_polyline(start_coords, end_coords):
    """Gets the route shape from OSRM"""
    lat1, lon1 = start_coords
    lat2, lon2 = end_coords
    url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=full"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    return data['routes'][0]['geometry']
def get_elevations(points, batch_size=100):
    """Gets elevations for a list of points using the Open-Elevation API"""
    all_elevations = []
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        locations = "|".join([f"{lat},{lon}" for lat, lon in batch])
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={locations}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        this_elevations = [item['elevation'] for item in data['results']]
        all_elevations.extend(this_elevations)
    return all_elevations
def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculates the bearing between two GPS points"""
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dLon = lon2_rad - lon1_rad
    y = np.sin(dLon) * np.cos(lat2_rad)
    x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dLon)
    bearing_rad = np.arctan2(y, x)
    return (np.degrees(bearing_rad)+360) % 360
if __name__ == "__main__":
    CHENNAI_COORDS = (13.0843, 80.2705)
    BANGALORE_COORDS = (12.9629, 77.5775)
    print("Step 1: Fetching route polyline from OSRM:")
    route_poly = get_route_polyline(CHENNAI_COORDS, BANGALORE_COORDS)
    points = polyline.decode(route_poly)
    print(f"Found {len(points)} points in the route.")
    print("Step 2: Fetching elevations from Open-Elevation:")
    elevations = get_elevations(points)
    df = pd.DataFrame(points, columns=['latitude', 'longitude'])
    df['altitude'] = elevations
    bearings = [np.nan]
    for i in range(1, len(df)):
        bearing = calculate_bearing(
            df.iloc[i-1]['latitude'], df.iloc[i-1]['longitude'],
            df.iloc[i]['latitude'], df.iloc[i]['longitude']
        )
        bearings.append(bearing)
    df['bearing'] = bearings
    out = "chennai_to_bangalore_route.csv"
    df.to_csv(out, index=False)
    print(f"Route data saved to '{out}'")
    print(df.head())