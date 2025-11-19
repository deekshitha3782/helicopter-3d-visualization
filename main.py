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

# === Configuration ===
# CSV file path (will be replaced by generate script)
location_csv_path = "/home/iq-sim1/Desktop/Deekshitha/android_logs/main_log/19_Aug_2025_logs/19_Aug_3/Location.csv"
# Output HTML filename (will be replaced by generate script)
output_html = "helicopter_3d_map.html"
# View mode: "points" or "direction" (will be replaced by generate script)
view_mode = "points"

# === Load your helicopter trajectory data ===
df = pd.read_csv(location_csv_path)

# Extract required columns
df["lat"] = df["latitude"]
df["lon"] = df["longitude"]
df["alt_amsl"] = df["altitudeAboveMeanSeaLevel"]  # AMSL altitude

print(f"Loaded {len(df)} data points")
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
    
    if (idx + 1) % 500 == 0:
        print(f"Processed {idx + 1}/{len(df)} points...")

df["terrain_alt"] = terrain_alts
df["alt"] = radio_alts  # Use radio altitude for visualization

print(f"\nTerrain altitude range: {df['terrain_alt'].min():.2f} to {df['terrain_alt'].max():.2f} m")
print(f"Radio altitude range: {df['alt'].min():.2f} to {df['alt'].max():.2f} m")
print(f"First few points:\n{df[['lat', 'lon', 'alt_amsl', 'terrain_alt', 'alt']].head()}")

# Add point numbers (1st, 2nd, 3rd, etc.)
df["point_number"] = range(1, len(df) + 1)

# === Define layers ===
# Use actual altitude values (in meters) for vertical axis
# pydeck expects z-coordinate in meters, but we can use elevation_scale to make it more visible
# Calculate relative altitude from minimum (so the lowest point is at ground level)
min_alt = df["alt"].min()
df["alt_relative"] = df["alt"] - min_alt  # Relative altitude from lowest point

print(f"\nAltitude visualization:")
print(f"Minimum altitude: {min_alt:.2f} m")
print(f"Maximum altitude: {df['alt'].max():.2f} m")
print(f"Altitude range: {df['alt_relative'].max():.2f} m")

# Create path data for connecting lines using actual altitude
path_data = [{
    "path": df[["lon", "lat", "alt_relative"]].values.tolist(),
    "name": "Helicopter Trajectory"
}]

# Create arrow data for direction visualization
# Create directional arrows every 20th point to show movement direction
arrow_spacing = 20
arrow_paths = []
for i in range(0, len(df) - 1, arrow_spacing):
    if i + 1 < len(df):
        current = df.iloc[i]
        next_point = df.iloc[i + 1]
        
        # Calculate direction vector
        dlat = next_point["lat"] - current["lat"]
        dlon = next_point["lon"] - current["lon"]
        dalt = next_point["alt_relative"] - current["alt_relative"]
        
        # Normalize to create arrow of fixed length
        length = np.sqrt(dlat**2 + dlon**2)
        if length > 0:
            # Scale to create visible arrow (about 0.00015 degrees for better visibility)
            scale = 0.00015 / length
            dlat_scaled = dlat * scale
            dlon_scaled = dlon * scale
            dalt_scaled = dalt * scale * 0.5  # Less vertical scaling
            
            # Use current point as base
            base_lat = current["lat"]
            base_lon = current["lon"]
            base_alt = current["alt_relative"]
            
            # Full arrow tip position (for arrowhead triangle only - no shaft/rectangle)
            full_tip_lat = base_lat + dlat_scaled
            full_tip_lon = base_lon + dlon_scaled
            full_tip_alt = base_alt + dalt_scaled
            
            # Arrowhead (isosceles triangle pointing forward) - no rectangle/shaft behind it
            # Calculate perpendicular direction for arrowhead (normalized)
            perp_length = np.sqrt(dlat_scaled**2 + dlon_scaled**2)
            if perp_length > 0:
                # Perpendicular vector (rotated 90 degrees)
                perp_lat = -dlon_scaled / perp_length
                perp_lon = dlat_scaled / perp_length
                
                # Arrowhead size (proportional to arrow length) - reduced base width
                arrowhead_size = perp_length * 0.3  # 30% of arrow length (narrower base)
                arrowhead_back_offset = 0.2  # How far back the base is from the tip
                
                # Triangle tip (pointing forward) - this is the full tip position
                tip_point = [full_tip_lon, full_tip_lat, full_tip_alt]
                
                # Base center point (back from tip)
                base_center_lat = full_tip_lat - dlat_scaled * arrowhead_back_offset
                base_center_lon = full_tip_lon - dlon_scaled * arrowhead_back_offset
                base_center_alt = full_tip_alt - dalt_scaled * arrowhead_back_offset
                
                # Left base point (perpendicular left from base center)
                left_lat = base_center_lat + perp_lat * arrowhead_size
                left_lon = base_center_lon + perp_lon * arrowhead_size
                left_alt = base_center_alt
                
                # Right base point (perpendicular right from base center)
                right_lat = base_center_lat - perp_lat * arrowhead_size
                right_lon = base_center_lon - perp_lon * arrowhead_size
                right_alt = base_center_alt
                
                # Create isosceles triangle: tip -> left base -> right base -> tip (closed polygon)
                # PolygonLayer expects polygons as a list of rings, where each ring is a list of coordinates
                # For a simple triangle, we use a single ring with 3 points (pydeck will close it automatically)
                # Add point data for tooltip (hover information)
                arrow_paths.append({
                    "polygon": [
                        [
                            tip_point,  # Tip (pointing forward)
                            [left_lon, left_lat, left_alt],  # Left base point
                            [right_lon, right_lat, right_alt],  # Right base point
                            tip_point  # Back to tip to close the polygon (explicitly closed)
                        ]
                    ],
                    "type": "arrowhead_triangle",
                    "point_number": current["point_number"],  # Point number for tooltip
                    "lat": current["lat"],  # Latitude for tooltip
                    "lon": current["lon"],  # Longitude for tooltip
                    "alt": current["alt"],  # Radio altitude for tooltip
                    "point_type": ""  # Empty for arrow points
                })

print(f"\nCreated {len([p for p in arrow_paths if p['type'] == 'arrowhead_triangle'])} direction arrows (isosceles triangles only, no shaft) along the trajectory")

# Arrowhead layer (isosceles triangle arrowheads using PolygonLayer) - no rectangle/shaft
arrowhead_triangles = [p for p in arrow_paths if p['type'] == 'arrowhead_triangle']
arrowhead_layer = pdk.Layer(
    "PolygonLayer",
    data=arrowhead_triangles,
    get_polygon="polygon",
    get_fill_color=[0, 200, 0, 220],  # Darker green color with slight transparency
    get_line_color=[0, 150, 0],  # Darker green outline
    line_width_min_pixels=2,
    line_width_max_pixels=4,
    elevation_scale=1,
    pickable=True,
    extruded=False,  # Flat triangles (not extruded)
    wireframe=False,  # Filled triangles
)

# Create vertical lines from ground (altitude 0) to every 50th point
# Get every 50th point (same as altitude labels)
df_vertical_lines = df.iloc[::5].copy()  # Every 50th point

# Create vertical line paths from ground to each 50th point
# Add intermediate points to ensure the line is visible
vertical_lines_data = []
for idx, row in df_vertical_lines.iterrows():
    alt = row["alt_relative"]
    # Create path with multiple points for better rendering
    path_points = [
        [row["lon"], row["lat"], 0],  # Ground level
        [row["lon"], row["lat"], alt * 0.25],  # Quarter height
        [row["lon"], row["lat"], alt * 0.5],   # Half height
        [row["lon"], row["lat"], alt * 0.75],  # Three quarter height
        [row["lon"], row["lat"], alt]  # Point altitude
    ]
    vertical_lines_data.append({
        "path": path_points
    })

# Create ground points at every 50th location (altitude 0, no serial number)
ground_points_df = df_vertical_lines.copy()
ground_points_df["alt_relative"] = 0  # Set altitude to ground level
ground_points_df["point_number"] = ""  # No serial number for ground points

print(f"\nCreated {len(vertical_lines_data)} vertical orange lines (every 50th point)")
print(f"Created {len(ground_points_df)} ground points (every 50th point, no numbers)")
if len(vertical_lines_data) > 0:
    print(f"Sample vertical line path: {vertical_lines_data[0]['path']}")
    print(f"First point altitude: {df_vertical_lines.iloc[0]['alt_relative']:.2f} m")

# Vertical lines layer - orange lines from ground to every 50th point
# Convert to list format that PathLayer expects
vertical_lines_list = []
for item in vertical_lines_data:
    vertical_lines_list.append(item)

vertical_lines_layer = pdk.Layer(
    "PathLayer",
    data=vertical_lines_list,
    get_path="path",
    get_color=[255, 100, 0],  # Bright orange color
    width_scale=1,  # Much larger scale
    width_min_pixels=5,  # Larger minimum
    get_width=1,  # Thick
    elevation_scale=1,
    pickable=True,
    billboard=True,  # Keep width constant regardless of zoom
)

# Ground points layer - points at ground level (every 50th point, no numbers)
ground_points_layer = pdk.Layer(
    "ScatterplotLayer",
    data=ground_points_df,
    get_position='[lon, lat, alt_relative]',  # Ground level (altitude 0)
    get_color=[255, 140, 0],  # Orange color to match the lines
    get_radius=3,  # Small ground points
    radius_min_pixels=2,
    radius_max_pixels=10,
    pickable=True,
    elevation_scale=1,
    extruded=False,
)

# Path layer to connect points with lines
# elevation_scale makes the vertical dimension more visible (1 = actual meters, higher = exaggerated)
path_layer = pdk.Layer(
    "PathLayer",
    data=path_data,
    get_path="path",
    get_color=[0, 0, 255],  # Blue color for the path
    width_scale=1,
    width_min_pixels=1,
    get_width=1,
    elevation_scale=1,  # Use 1 for actual height, increase for more visible vertical differences
    pickable=True,
)

# Handle overlapping points by adding small offsets and making them semi-transparent
# Create a copy of the dataframe for point positioning
df_points = df.copy()

# Add small random offset to points that are very close together (within 0.0001 degrees)
# This helps visualize overlapping points
np.random.seed(42)  # For reproducibility

# Calculate distance between all points (more efficient approach)
# Add small jitter to all points to help visualize overlapping ones
jitter_lat = np.random.uniform(-0.00002, 0.00002, size=len(df_points))
jitter_lon = np.random.uniform(-0.00002, 0.00002, size=len(df_points))

# For points that are very close (within threshold), add larger offset
threshold = 0.0001
for i in range(len(df_points)):
    # Find points very close to this one
    distances = np.sqrt(
        (df_points['lat'] - df_points.iloc[i]['lat'])**2 +
        (df_points['lon'] - df_points.iloc[i]['lon'])**2
    )
    close_mask = (distances < threshold) & (distances > 0)
    
    if close_mask.sum() > 0:
        # Add larger offset for overlapping points
        jitter_lat[i] = np.random.uniform(-0.00005, 0.00005)
        jitter_lon[i] = np.random.uniform(-0.00005, 0.00005)

df_points['lat'] += jitter_lat
df_points['lon'] += jitter_lon

# Add tooltip fields for regular points (point_type will be empty for regular points)
df_points['point_type'] = ''  # Empty for regular points

print(f"\nAdjusted {len(df_points)} points to handle overlapping")

# Points layer with smaller size and semi-transparent for overlapping visibility
# Using actual altitude values for z-coordinate
points_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_points,
    get_position='[lon, lat, alt_relative]',  # Use actual altitude (relative to min)
    get_color=[255, 0, 0, 200],  # Red color with alpha (semi-transparent) for overlapping visibility
    get_radius=5,  # Smaller radius
    radius_min_pixels=2,
    radius_max_pixels=20,
    pickable=True,
    elevation_scale=1,  # Use 1 for actual height in meters
    extruded=False,
    stroked=True,  # Add stroke for better visibility
    get_line_color=[255, 255, 255],  # White outline
    line_width_min_pixels=1,
)

# Text layer to display point numbers (use adjusted points)
text_layer = pdk.Layer(
    "TextLayer",
    data=df_points,  # Use adjusted points
    get_position='[lon, lat, alt_relative]',  # Use actual altitude
    get_text="point_number",
    get_color=[255, 255, 255],  # White color for better visibility
    get_size=16,
    get_alignment_baseline="bottom",
    pickable=True,
)

# Create altitude labels for every 50th point (showing height from ground)
# Get every 50th point (1st, 50th, 100th, 150th, etc.)
df_alt_labels = df.iloc[::50].copy()  # Every 50th point starting from index 0
df_alt_labels["alt_label"] = df_alt_labels["alt_relative"].apply(lambda x: f"{x:.1f} m")

print(f"\nAdding altitude labels at {len(df_alt_labels)} points (every 50th point)")
print(f"Altitude labels show height from ground (relative to minimum altitude)")
print(f"Sample altitude labels:\n{df_alt_labels[['point_number', 'alt_relative', 'alt_label']].head()}")

# Altitude label layer
altitude_label_layer = pdk.Layer(
    "TextLayer",
    data=df_alt_labels,
    get_position='[lon, lat, alt_relative]',  # Use actual altitude
    get_text="alt_label",
    get_color=[0, 255, 0],  # Green color for altitude labels
    get_size=20,
    get_alignment_baseline="bottom",
    pickable=True,
)

# Create START and END points at 50m radio altitude
start_point = df.iloc[0].copy()  # First point
end_point = df.iloc[-1].copy()  # Last point

# Calculate 50m relative altitude (radio altitude = 50m above ground)
# alt_relative is already relative to minimum (ground level), so 50.0 = 50m radio altitude
radio_alt_50m = 50.0

# Create dataframe for start/end points with tooltip data
start_end_df = pd.DataFrame([
    {
        "lon": start_point["lon"],
        "lat": start_point["lat"],
        "alt_relative": radio_alt_50m,  # Fixed at 50m radio altitude
        "label": "START",
        "point_type": "START\n",
        "point_number": "",  # Empty for start/end points
        "lat": start_point["lat"],
        "lon": start_point["lon"],
        "alt": 50.0  # Radio altitude 50m for tooltip
    },
    {
        "lon": end_point["lon"],
        "lat": end_point["lat"],
        "alt_relative": radio_alt_50m,  # Fixed at 50m radio altitude
        "label": "END",
        "point_type": "END\n",
        "point_number": "",  # Empty for start/end points
        "lat": end_point["lat"],
        "lon": end_point["lon"],
        "alt": 50.0  # Radio altitude 50m for tooltip
    }
])

print(f"\nAdding START and END points at 50m radio altitude")
print(f"START point: lat={start_point['lat']:.6f}, lon={start_point['lon']:.6f}")
print(f"END point: lat={end_point['lat']:.6f}, lon={end_point['lon']:.6f}")

# Start/End points layer - brown colored points (7 times bigger than red points)
start_end_points_layer = pdk.Layer(
    "ScatterplotLayer",
    data=start_end_df,
    get_position='[lon, lat, alt_relative]',  # At 50m radio altitude
    get_color=[139, 69, 19],  # Brown color (RGB)
    get_radius=35,  # 7 times bigger than red points (5 * 7 = 35)
    radius_min_pixels=14,  # 7 times bigger minimum (2 * 7 = 14)
    radius_max_pixels=140,  # 7 times bigger maximum (20 * 7 = 140)
    pickable=True,
    elevation_scale=1,
    extruded=False,
    stroked=True,
    get_line_color=[255, 255, 255],  # White outline
    line_width_min_pixels=2,  # Thicker outline for larger points
)

# Start/End label layer (text labels)
start_end_label_layer = pdk.Layer(
    "TextLayer",
    data=start_end_df,
    get_position='[lon, lat, alt_relative]',  # At 50m radio altitude
    get_text="label",
    get_color=[255, 255, 0],  # Yellow color for START/END labels
    get_size=24,  # Larger size for visibility
    get_alignment_baseline="center",
    pickable=True,
)

# === Set initial map view ===
# Set pitch to 90 (horizontal view) - can be rotated interactively beyond 90 degrees
view_state = pdk.ViewState(
    latitude=df["lat"].mean(),
    longitude=df["lon"].mean(),
    zoom=17,
    pitch=90,  # Maximum standard pitch (horizontal view)
    bearing=0,
    height=800,  # Map height in pixels
)

# === Create the 3D map ===
# Map is interactive by default - you can pan, rotate, and zoom
# Create two layer configurations: with points and with direction arrows

# Layers for "with points" view
layers_with_points = [vertical_lines_layer, path_layer, ground_points_layer, points_layer, text_layer, altitude_label_layer, start_end_points_layer, start_end_label_layer]

# Layers for "with direction" view (no points, just path and triangle arrows - no rectangle/shaft)
layers_with_direction = [vertical_lines_layer, path_layer, arrowhead_layer, ground_points_layer, altitude_label_layer, start_end_points_layer, start_end_label_layer]

# Select layers based on view_mode
if view_mode == "direction":
    selected_layers = layers_with_direction
    print(f"\nGenerating direction view (with arrows)")
else:
    selected_layers = layers_with_points
    print(f"\nGenerating points view (with numbered points)")

# Create the deck with selected layers
r = pdk.Deck(
    layers=selected_layers,
    initial_view_state=view_state,
    map_style="https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",  # Voyager style with terrain features
    tooltip={"text": "{point_type}Point #{point_number}\nLat: {lat}\nLon: {lon}\nRadio Alt: {alt} m"},
)

# === Save the interactive map to an HTML file ===
# Generate HTML for the selected view
r.to_html(output_html)

# Read the HTML file to modify it
with open(output_html, "r") as f:
    html_content = f.read()

# Modify HTML to allow pitch beyond 90 degrees (up to 180)
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

# Write the modified HTML back
with open(output_html, "w") as f:
    f.write(modified_html)

view_type = "points" if view_mode == "points" else "direction arrows"
print(f"\n✓ 3D map saved to '{output_html}' (with {view_type})")
