import pandas as pd
import pydeck as pdk
import re
import numpy as np
import rasterio
from rasterio.warp import transform
from rasterio.transform import rowcol
import os
from pathlib import Path
try:
    from pyproj import Transformer
except ImportError:
    print("Warning: pyproj not installed. Install with: pip install pyproj")
    Transformer = None

# === DEM Files Configuration ===
DEM_DIR = "/home/iq-sim1/Deekshitha/GPWS_SETUP/data"

def get_dem_filename(lat, lon):
    """
    Get the DEM filename based on latitude and longitude.
    Format: 10N75E.tif covers 10°N to 11°N and 75°E to 76°E
    """
    # Round down to get the base coordinates
    lat_base = int(np.floor(lat))
    lon_base = int(np.floor(lon))
    
    # Format: 10N75E.tif
    if lat_base >= 0:
        lat_str = f"{lat_base}N"
    else:
        lat_str = f"{abs(lat_base)}S"
    
    if lon_base >= 0:
        lon_str = f"{lon_base}E"
    else:
        lon_str = f"{abs(lon_base)}W"
    
    filename = f"{lat_str}{lon_str}.tif"
    return filename

def get_terrain_altitude(lat, lon, dem_cache={}, dem_data_cache={}):
    """
    Get terrain altitude from DEM file for given lat/lon.
    Returns terrain altitude in meters, or 0 if DEM file not found.
    Uses caching to avoid reopening files and re-reading data.
    """
    dem_filename = get_dem_filename(lat, lon)
    dem_path = os.path.join(DEM_DIR, dem_filename)
    
    # Check if file exists
    if not os.path.exists(dem_path):
        # Only print warning once per file
        if dem_filename not in dem_cache:
            print(f"Warning: DEM file not found: {dem_filename}")
            dem_cache[dem_filename] = None
        return 0.0
    
    # Use cache to avoid reopening files and re-reading data
    if dem_filename not in dem_data_cache:
        try:
            with rasterio.open(dem_path) as dem:
                # Read entire DEM data once and cache it
                dem_data_cache[dem_filename] = {
                    'data': dem.read(1),
                    'transform': dem.transform,
                    'crs': dem.crs,
                    'nodata': dem.nodata
                }
        except Exception as e:
            print(f"Error opening DEM file {dem_filename}: {e}")
            dem_data_cache[dem_filename] = None
            return 0.0
    
    if dem_data_cache[dem_filename] is None:
        return 0.0
    
    dem_info = dem_data_cache[dem_filename]
    
    # Transform lat/lon to the DEM's coordinate system
    try:
        # Get transform from cached data
        transform = dem_info['transform']
        crs = dem_info['crs']
        data = dem_info['data']
        nodata = dem_info['nodata']
        
        # Use rasterio's index method if available, otherwise use transform
        if Transformer is not None:
            # Convert lat/lon to the DEM's CRS coordinates
            transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            x, y = transformer.transform(lon, lat)
            # Get row/col indices
            row, col = rowcol(transform, x, y)
        else:
            # Fallback: open file to use index method
            with rasterio.open(dem_path) as dem:
                row, col = dem.index(lon, lat)
        
        # Check bounds
        if row < 0 or row >= data.shape[0] or col < 0 or col >= data.shape[1]:
            return 0.0
        
        value = data[row, col]
        
        if not np.isnan(value) and (nodata is None or value != nodata):
            return float(value)
        else:
            return 0.0
    except Exception as e:
        # Fallback to simpler method
        try:
            with rasterio.open(dem_path) as dem:
                row, col = dem.index(lon, lat)
                if 0 <= row < dem.height and 0 <= col < dem.width:
                    value = dem.read(1)[row, col]
                    if not np.isnan(value) and value != dem.nodata:
                        return float(value)
        except:
            pass
        return 0.0

def meters_to_latlon(ref_lat, ref_lon, Pn, Pe):
    """
    Convert meters (Pn, Pe) to latitude/longitude offset from reference point.
    Pn: North-South displacement in meters (positive = north)
    Pe: East-West displacement in meters (positive = east)
    Returns: (delta_lat, delta_lon) in degrees
    """
    # Approximate conversion factors
    # 1 degree latitude ≈ 111,320 meters (constant)
    # 1 degree longitude ≈ 111,320 * cos(latitude) meters (varies with latitude)
    
    meters_per_degree_lat = 111320.0
    meters_per_degree_lon = 111320.0 * np.cos(np.radians(ref_lat))
    
    delta_lat = Pn / meters_per_degree_lat
    delta_lon = Pe / meters_per_degree_lon
    
    return delta_lat, delta_lon

# === Configuration ===
# CSV file path (will be replaced by generate script)
location_csv_path = "/home/iq-sim1/Desktop/Deekshitha/android_logs/main_log/19_Aug_2025_logs/19_Aug_1/Location.csv"
# StateDataOut.txt file path (will be replaced by generate script)
statedata_path = "/home/iq-sim1/Desktop/Deekshitha/android_logs/main_log/19_Aug_2025_logs/19_Aug_1/DataOut/StateDataOut.txt"
# Output HTML filename (will be replaced by generate script)
output_html = "statedata_19_Aug_1.html"
# View mode: "points" or "direction" (will be replaced by generate script)
view_mode = "points"

# === Load reference point from Location.csv ===
print(f"Loading reference point from: {location_csv_path}")
df_ref = pd.read_csv(location_csv_path)
ref_lat = df_ref.iloc[0]["latitude"]
ref_lon = df_ref.iloc[0]["longitude"]
ref_alt_amsl = df_ref.iloc[0]["altitudeAboveMeanSeaLevel"]

print(f"Reference point: lat={ref_lat:.6f}, lon={ref_lon:.6f}, alt_amsl={ref_alt_amsl:.2f} m")

# === Load StateDataOut.txt ===
print(f"\nLoading StateDataOut from: {statedata_path}")
# Read the data, skipping first and last lines
data = np.genfromtxt(statedata_path, delimiter=' ', skip_header=1, skip_footer=1,
    names=['time', 'q1', 'q2', 'q3', 'q4', 'Vn', 'Ve', 'Vd', 'Pn', 'Pe', 'Pd',
           'Bx', 'By', 'Bz', 'Wn', 'We', 'Mn', 'Me', 'Md', 'Mbn', 'Mbe', 'Mbd'])

print(f"Loaded {len(data)} data points from StateDataOut.txt")

# === Convert Pn, Pe, Pd to lat, lon, alt ===
# Pn, Pe, Pd are in meters relative to reference point
# Pd is down (negative altitude), so alt = ref_alt - Pd
lats = []
lons = []
alts_amsl = []

for i in range(len(data)):
    Pn = data['Pn'][i]
    Pe = data['Pe'][i]
    Pd = data['Pd'][i]
    
    # Convert meters to lat/lon
    delta_lat, delta_lon = meters_to_latlon(ref_lat, ref_lon, Pn, Pe)
    
    # Calculate absolute lat/lon
    lat = ref_lat + delta_lat
    lon = ref_lon + delta_lon
    
    # Pd is down (negative), so altitude = reference altitude - Pd
    alt_amsl = ref_alt_amsl - Pd
    
    lats.append(lat)
    lons.append(lon)
    alts_amsl.append(alt_amsl)

# Create DataFrame
df = pd.DataFrame({
    'lat': lats,
    'lon': lons,
    'alt_amsl': alts_amsl,
    'time': data['time'],
    'Vn': data['Vn'],
    'Ve': data['Ve'],
    'Vd': data['Vd']
})

print(f"\nConverted positions:")
print(f"Original StateData points: {len(df)}")

# Downsample to every 100th point for faster processing and rendering
downsample_factor = 100
df = df.iloc[::downsample_factor].copy().reset_index(drop=True)

print(f"Downsampled to {len(df)} points (every {downsample_factor}th point)")
print(f"Latitude range: {df['lat'].min():.6f} to {df['lat'].max():.6f}")
print(f"Longitude range: {df['lon'].min():.6f} to {df['lon'].max():.6f}")
print(f"AMSL Altitude range: {df['alt_amsl'].min():.2f} to {df['alt_amsl'].max():.2f} m")

# === Calculate Radio Altitude from DEM files ===
print("\nCalculating radio altitude from DEM files...")
print(f"DEM directory: {DEM_DIR}")

# Cache for DEM files and data
dem_cache = {}
dem_data_cache = {}

# Calculate terrain altitude and radio altitude for each point
terrain_alts = []
radio_alts = []

for idx, row in df.iterrows():
    terrain_alt = get_terrain_altitude(row["lat"], row["lon"], dem_cache, dem_data_cache)
    radio_alt = row["alt_amsl"] - terrain_alt
    
    terrain_alts.append(terrain_alt)
    radio_alts.append(radio_alt)
    
    if (idx + 1) % 50 == 0:
        print(f"Processed {idx + 1}/{len(df)} points...")

df["terrain_alt"] = terrain_alts
df["alt"] = radio_alts  # Use radio altitude for visualization

print(f"\nTerrain altitude range: {df['terrain_alt'].min():.2f} to {df['terrain_alt'].max():.2f} m")
print(f"Radio altitude range: {df['alt'].min():.2f} to {df['alt'].max():.2f} m")

# Add point numbers
df["point_number"] = range(1, len(df) + 1)

# === Define layers ===
# Calculate relative altitude from minimum (so the lowest point is at ground level)
min_alt = df["alt"].min()
df["alt_relative"] = df["alt"] - min_alt

print(f"\nAltitude visualization:")
print(f"Minimum altitude: {min_alt:.2f} m")
print(f"Maximum altitude: {df['alt'].max():.2f} m")
print(f"Altitude range: {df['alt'].max() - min_alt:.2f} m")

# Path data for trajectory line
path_data = [{
    "path": df[["lon", "lat", "alt_relative"]].values.tolist(),
    "name": "Helicopter Trajectory (StateData)"
}]

# Create arrow data for direction visualization
# Adjust arrow spacing based on downsampled data (use every 5th point for arrows)
arrow_spacing = 5
arrow_paths = []
for i in range(0, len(df) - 1, arrow_spacing):
    if i + 1 < len(df):
        current = df.iloc[i]
        next_point = df.iloc[i + 1]
        
        dlat = next_point["lat"] - current["lat"]
        dlon = next_point["lon"] - current["lon"]
        dalt = next_point["alt_relative"] - current["alt_relative"]
        
        length = np.sqrt(dlat**2 + dlon**2)
        if length > 0:
            scale = 0.00015 / length
            dlat_scaled = dlat * scale
            dlon_scaled = dlon * scale
            dalt_scaled = dalt * scale * 0.5
            
            base_lat = current["lat"]
            base_lon = current["lon"]
            base_alt = current["alt_relative"]
            
            full_tip_lat = base_lat + dlat_scaled
            full_tip_lon = base_lon + dlon_scaled
            full_tip_alt = base_alt + dalt_scaled
            
            perp_length = np.sqrt(dlat_scaled**2 + dlon_scaled**2)
            if perp_length > 0:
                perp_lat = -dlon_scaled / perp_length
                perp_lon = dlat_scaled / perp_length
                
                arrowhead_size = perp_length * 0.3
                arrowhead_back_offset = 0.2
                
                tip_point = [full_tip_lon, full_tip_lat, full_tip_alt]
                
                base_center_lat = full_tip_lat - dlat_scaled * arrowhead_back_offset
                base_center_lon = full_tip_lon - dlon_scaled * arrowhead_back_offset
                base_center_alt = full_tip_alt - dalt_scaled * arrowhead_back_offset
                
                left_lat = base_center_lat + perp_lat * arrowhead_size
                left_lon = base_center_lon + perp_lon * arrowhead_size
                left_alt = base_center_alt
                
                right_lat = base_center_lat - perp_lat * arrowhead_size
                right_lon = base_center_lon - perp_lon * arrowhead_size
                right_alt = base_center_alt
                
                arrow_paths.append({
                    "polygon": [[
                        tip_point,
                        [left_lon, left_lat, left_alt],
                        [right_lon, right_lat, right_alt],
                        tip_point
                    ]],
                    "type": "arrowhead_triangle",
                    "point_number": current["point_number"],
                    "lat": current["lat"],
                    "lon": current["lon"],
                    "alt": current["alt"],
                    "point_type": ""
                })

print(f"\nCreated {len(arrow_paths)} direction arrows along the trajectory")

arrowhead_triangles = [p for p in arrow_paths if p['type'] == 'arrowhead_triangle']
arrowhead_layer = pdk.Layer(
    "PolygonLayer",
    data=arrowhead_triangles,
    get_polygon="polygon",
    get_fill_color=[0, 200, 0, 220],
    get_line_color=[0, 150, 0],
    line_width_min_pixels=2,
    line_width_max_pixels=4,
    elevation_scale=1,
    pickable=True,
    extruded=False,
    wireframe=False,
)

# Vertical lines (every 10th point from downsampled data) - same as altitude labels
df_vertical_lines = df.iloc[::10].copy()
vertical_lines_data = []
for idx, row in df_vertical_lines.iterrows():
    alt = row["alt_relative"]
    path_points = [
        [row["lon"], row["lat"], 0],
        [row["lon"], row["lat"], alt * 0.25],
        [row["lon"], row["lat"], alt * 0.5],
        [row["lon"], row["lat"], alt * 0.75],
        [row["lon"], row["lat"], alt]
    ]
    vertical_lines_data.append({"path": path_points})

vertical_lines_layer = pdk.Layer(
    "PathLayer",
    data=vertical_lines_data,
    get_path="path",
    get_color=[255, 100, 0],
    width_scale=1,
    width_min_pixels=5,
    get_width=1,
    elevation_scale=1,
    pickable=True,
    billboard=True,
)

ground_points_df = df_vertical_lines.copy()
ground_points_df["alt_relative"] = 0
ground_points_df["point_number"] = ""

ground_points_layer = pdk.Layer(
    "ScatterplotLayer",
    data=ground_points_df,
    get_position='[lon, lat, alt_relative]',
    get_color=[255, 140, 0],
    get_radius=3,
    radius_min_pixels=2,
    radius_max_pixels=10,
    pickable=True,
    elevation_scale=1,
    extruded=False,
)

path_layer = pdk.Layer(
    "PathLayer",
    data=path_data,
    get_path="path",
    get_color=[0, 0, 255],
    width_scale=1,
    width_min_pixels=1,
    get_width=1,
    elevation_scale=1,
    pickable=True,
)

# Handle overlapping points
df_points = df.copy()
np.random.seed(42)
jitter_lat = np.random.uniform(-0.00002, 0.00002, size=len(df_points))
jitter_lon = np.random.uniform(-0.00002, 0.00002, size=len(df_points))

threshold = 0.0001
for i in range(len(df_points)):
    distances = np.sqrt(
        (df_points['lat'] - df_points.iloc[i]['lat'])**2 +
        (df_points['lon'] - df_points.iloc[i]['lon'])**2
    )
    close_mask = (distances < threshold) & (distances > 0)
    
    if close_mask.sum() > 0:
        jitter_lat[i] = np.random.uniform(-0.00005, 0.00005)
        jitter_lon[i] = np.random.uniform(-0.00005, 0.00005)

df_points['lat'] += jitter_lat
df_points['lon'] += jitter_lon
df_points['point_type'] = ''

points_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_points,
    get_position='[lon, lat, alt_relative]',
    get_color=[255, 0, 0, 200],
    get_radius=5,
    radius_min_pixels=2,
    radius_max_pixels=20,
    pickable=True,
    elevation_scale=1,
    extruded=False,
    stroked=True,
    get_line_color=[255, 255, 255],
    line_width_min_pixels=1,
)

text_layer = pdk.Layer(
    "TextLayer",
    data=df_points,
    get_position='[lon, lat, alt_relative]',
    get_text="point_number",
    get_color=[255, 255, 255],
    get_size=16,
    get_alignment_baseline="bottom",
    pickable=True,
)

# Altitude labels (every 10th point from downsampled data)
df_alt_labels = df.iloc[::10].copy()
df_alt_labels["alt_label"] = df_alt_labels["alt_relative"].apply(lambda x: f"{x:.1f} m")

altitude_label_layer = pdk.Layer(
    "TextLayer",
    data=df_alt_labels,
    get_position='[lon, lat, alt_relative]',
    get_text="alt_label",
    get_color=[0, 255, 0],
    get_size=20,
    get_alignment_baseline="bottom",
    pickable=True,
)

start_point = df.iloc[0].copy()
end_point = df.iloc[-1].copy()
radio_alt_50m = 50.0

start_end_df = pd.DataFrame([
    {
        "lon": start_point["lon"],
        "lat": start_point["lat"],
        "alt_relative": radio_alt_50m,
        "label": "START",
        "point_type": "START\n",
        "point_number": "",
        "lat": start_point["lat"],
        "lon": start_point["lon"],
        "alt": 50.0
    },
    {
        "lon": end_point["lon"],
        "lat": end_point["lat"],
        "alt_relative": radio_alt_50m,
        "label": "END",
        "point_type": "END\n",
        "point_number": "",
        "lat": end_point["lat"],
        "lon": end_point["lon"],
        "alt": 50.0
    }
])

start_end_points_layer = pdk.Layer(
    "ScatterplotLayer",
    data=start_end_df,
    get_position='[lon, lat, alt_relative]',
    get_color=[139, 69, 19],
    get_radius=35,
    radius_min_pixels=14,
    radius_max_pixels=140,
    pickable=True,
    elevation_scale=1,
    extruded=False,
    stroked=True,
    get_line_color=[255, 255, 255],
    line_width_min_pixels=2,
)

start_end_label_layer = pdk.Layer(
    "TextLayer",
    data=start_end_df,
    get_position='[lon, lat, alt_relative]',
    get_text="label",
    get_color=[255, 255, 0],
    get_size=24,
    get_alignment_baseline="center",
    pickable=True,
)

view_state = pdk.ViewState(
    latitude=df["lat"].mean(),
    longitude=df["lon"].mean(),
    zoom=17,
    pitch=90,
    bearing=0,
    height=800,
)

layers_with_points = [vertical_lines_layer, path_layer, ground_points_layer, points_layer, text_layer, altitude_label_layer, start_end_points_layer, start_end_label_layer]
layers_with_direction = [vertical_lines_layer, path_layer, arrowhead_layer, ground_points_layer, altitude_label_layer, start_end_points_layer, start_end_label_layer]

# Select layers based on view_mode
if view_mode == "direction":
    selected_layers = layers_with_direction
    print(f"\nGenerating direction view (with arrows)")
else:
    selected_layers = layers_with_points
    print(f"\nGenerating points view (with numbered points)")

r = pdk.Deck(
    layers=selected_layers,
    initial_view_state=view_state,
    map_style="https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
    tooltip={"text": "{point_type}Point #{point_number}\nLat: {lat}\nLon: {lon}\nRadio Alt: {alt} m"},
)

r.to_html(output_html)

# Read and modify HTML file
with open(output_html, "r") as f:
    html_content = f.read()

modified_html = html_content.replace('maxPitch: 60', 'maxPitch: 180')
modified_html = modified_html.replace('maxPitch:90', 'maxPitch:180')
modified_html = modified_html.replace('maxPitch: 90', 'maxPitch: 180')
modified_html = modified_html.replace('maxPitch:60', 'maxPitch:180')
modified_html = modified_html.replace('"maxPitch":60', '"maxPitch":180')
modified_html = modified_html.replace('"maxPitch":90', '"maxPitch":180')
modified_html = modified_html.replace("'maxPitch':60", "'maxPitch':180")
modified_html = modified_html.replace("'maxPitch':90", "'maxPitch':180")

def replace_maxpitch_in_mapcontroller(match):
    content = match.group(1)
    content = re.sub(r'maxPitch:\s*\d+', 'maxPitch: 180', content)
    if 'maxPitch' not in content:
        content = 'maxPitch: 180, ' + content
    return f'new deck.MapController({{{content}}}'

modified_html = re.sub(r'new deck\.MapController\(\{([^}]*)\}', replace_maxpitch_in_mapcontroller, modified_html)

if 'MapController' in modified_html and 'maxPitch: 180' not in modified_html:
    modified_html = re.sub(r'(new deck\.MapController\(\{)', r'\1maxPitch: 180, ', modified_html)

with open(output_html, "w") as f:
    f.write(modified_html)

view_type = "points" if view_mode == "points" else "direction arrows"
print(f"\n✓ 3D map saved to '{output_html}' (with {view_type})")

