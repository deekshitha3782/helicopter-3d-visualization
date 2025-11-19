#!/usr/bin/env python3
"""
Generate HTML visualizations for multiple Location.csv files
"""

import subprocess
import os
import re
from pathlib import Path

# List of CSV files to process
CSV_FILES = [
    "/home/iq-sim1/Desktop/Deekshitha/android_logs/main_log/19_Aug_2025_logs/19_Aug_1/Location.csv",
    "/home/iq-sim1/Desktop/Deekshitha/android_logs/main_log/19_Aug_2025_logs/19_Aug_2/Location.csv",
    "/home/iq-sim1/Desktop/Deekshitha/android_logs/main_log/19_Aug_2025_logs/19_Aug_3/Location.csv",
    "/home/iq-sim1/Desktop/Deekshitha/android_logs/main_log/22_August_2025_logs/22_August1/Location.csv",
]

# Output HTML filenames (will be created)
HTML_FILES = [
    "aug_19th_1.html",
    "aug_19th_2.html",
    "aug_19th_3.html",
    "aug_22nd_1.html",
]

# Display names for index.html
DISPLAY_NAMES = [
    "August 19th - Flight 1",
    "August 19th - Flight 2",
    "August 19th - Flight 3",
    "August 22nd - Flight 1",
]

# StateDataOut files configuration (includes comparison)
STATEDATA_CONFIGS = [
    {
        "location_csv": "/home/iq-sim1/Desktop/Deekshitha/android_logs/main_log/19_Aug_2025_logs/19_Aug_1/Location.csv",
        "statedata_txt": "/home/iq-sim1/Desktop/Deekshitha/android_logs/main_log/19_Aug_2025_logs/19_Aug_1/DataOut/StateDataOut.txt",
        "html_base": "statedata_19_Aug_1.html",
        "html_comparison": "comparison_19_Aug_1.html",
        "display_name": "August 19th - Flight 1"
    },
    {
        "location_csv": "/home/iq-sim1/Desktop/Deekshitha/android_logs/main_log/19_Aug_2025_logs/19_Aug_2/Location.csv",
        "statedata_txt": "/home/iq-sim1/Desktop/Deekshitha/android_logs/main_log/19_Aug_2025_logs/19_Aug_2/DataOut/StateDataOut.txt",
        "html_base": "statedata_19_Aug_2.html",
        "html_comparison": "comparison_19_Aug_2.html",
        "display_name": "August 19th - Flight 2"
    },
    {
        "location_csv": "/home/iq-sim1/Desktop/Deekshitha/android_logs/main_log/19_Aug_2025_logs/19_Aug_3/Location.csv",
        "statedata_txt": "/home/iq-sim1/Desktop/Deekshitha/android_logs/main_log/19_Aug_2025_logs/19_Aug_3/DataOut/StateDataOut.txt",
        "html_base": "statedata_19_Aug_3.html",
        "html_comparison": "comparison_19_Aug_3.html",
        "display_name": "August 19th - Flight 3"
    },
    {
        "location_csv": "/home/iq-sim1/Desktop/Deekshitha/android_logs/main_log/22_August_2025_logs/22_August1/Location.csv",
        "statedata_txt": "/home/iq-sim1/Desktop/Deekshitha/android_logs/main_log/22_August_2025_logs/22_August1/DataOut/StateDataOut.txt",
        "html_base": "statedata_22_Aug_1.html",
        "html_comparison": "comparison_22_Aug_1.html",
        "display_name": "August 22nd - Flight 1"
    },
]

def generate_visualization(csv_path, html_filename_base):
    """Generate Location.csv HTML visualizations (points and direction separately)"""
    html_points = html_filename_base.replace('.html', '_points.html')
    html_direction = html_filename_base.replace('.html', '_direction.html')
    
    results = []
    
    for view_mode, output_file in [("points", html_points), ("direction", html_direction)]:
        print(f"\n{'='*60}")
        print(f"Processing: {csv_path} ({view_mode} view)")
        print(f"Output: {output_file}")
        print(f"{'='*60}")
        
        # Read the main.py template
        with open("main.py", "r") as f:
            main_code = f.read()
        
        # Replace paths and view_mode
        csv_path_pattern = r'location_csv_path\s*=\s*["\']([^"\']+)["\']'
        main_code = re.sub(csv_path_pattern, f'location_csv_path = "{csv_path}"', main_code)
        
        output_html_pattern = r'output_html\s*=\s*["\']([^"\']+)["\']'
        main_code = re.sub(output_html_pattern, f'output_html = "{output_file}"', main_code)
        
        view_mode_pattern = r'view_mode\s*=\s*["\']([^"\']+)["\']'
        main_code = re.sub(view_mode_pattern, f'view_mode = "{view_mode}"', main_code)
        
        # Write temporary script
        temp_script = f"temp_main_{view_mode}.py"
        with open(temp_script, "w") as f:
            f.write(main_code)
        
        try:
            result = subprocess.run(["python3", temp_script], check=True, capture_output=True, text=True, timeout=300)
            if result.stdout and len(result.stdout.strip()) > 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 5:
                    for line in lines[-5:]:
                        print(line)
                else:
                    print(result.stdout)
            print(f"✓ Generated {output_file}")
            results.append(output_file)
        except subprocess.CalledProcessError as e:
            print(f"✗ Error generating {output_file}")
            if e.stderr:
                print("STDERR:", e.stderr[-300:])
        except subprocess.TimeoutExpired:
            print(f"✗ Timeout generating {output_file}")
        finally:
            if os.path.exists(temp_script):
                os.remove(temp_script)
    
    return len(results) == 2, results

def generate_statedata_visualization(config):
    """Generate StateData HTML visualizations (points and direction separately)"""
    html_base = config['html_base'].replace('.html', '')
    html_points = f"{html_base}_points.html"
    html_direction = f"{html_base}_direction.html"
    
    if not os.path.exists(config['location_csv']):
        print(f"✗ Location CSV not found: {config['location_csv']}")
        return False, []
    
    if not os.path.exists(config['statedata_txt']):
        print(f"✗ StateDataOut.txt not found: {config['statedata_txt']}")
        return False, []
    
    results = []
    
    for view_mode, output_file in [("points", html_points), ("direction", html_direction)]:
        print(f"\n{'='*60}")
        print(f"Processing StateData: {config['display_name']} ({view_mode} view)")
        print(f"Output: {output_file}")
        print(f"{'='*60}")
        
        # Read the main_statedata.py template
        with open("main_statedata.py", "r") as f:
            main_code = f.read()
        
        # Replace paths and view_mode
        main_code = main_code.replace("/home/iq-sim1/Desktop/Deekshitha/android_logs/main_log/19_Aug_2025_logs/19_Aug_1/Location.csv", config['location_csv'])
        main_code = main_code.replace("/home/iq-sim1/Desktop/Deekshitha/android_logs/main_log/19_Aug_2025_logs/19_Aug_1/DataOut/StateDataOut.txt", config['statedata_txt'])
        main_code = main_code.replace("statedata_19_Aug_1.html", output_file)
        
        view_mode_pattern = r'view_mode\s*=\s*["\']([^"\']+)["\']'
        main_code = re.sub(view_mode_pattern, f'view_mode = "{view_mode}"', main_code)
        
        # Write temporary script
        temp_script = f"temp_main_statedata_{view_mode}.py"
        with open(temp_script, "w") as f:
            f.write(main_code)
        
        try:
            result = subprocess.run(["python3", temp_script], check=True, capture_output=True, text=True, timeout=600)
            if result.stdout and len(result.stdout.strip()) > 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 5:
                    for line in lines[-5:]:
                        print(line)
                else:
                    print(result.stdout)
            print(f"✓ Generated {output_file}")
            results.append(output_file)
        except subprocess.CalledProcessError as e:
            print(f"✗ Error generating {output_file}")
            if e.stderr:
                print("STDERR:", e.stderr[-300:])
        except subprocess.TimeoutExpired:
            print(f"✗ Timeout generating {output_file}")
        finally:
            if os.path.exists(temp_script):
                os.remove(temp_script)
    
    return len(results) == 2, results

def generate_comparison_visualization(config):
    """Generate comparison HTML showing both Location.csv and StateData"""
    print(f"\n{'='*60}")
    print(f"Processing Comparison: {config['display_name']}")
    print(f"Location CSV: {config['location_csv']}")
    print(f"StateData TXT: {config['statedata_txt']}")
    print(f"Output: {config['html_comparison']}")
    print(f"{'='*60}")
    
    # Check if files exist
    if not os.path.exists(config['location_csv']):
        print(f"✗ Location CSV not found: {config['location_csv']}")
        return False, None
    
    if not os.path.exists(config['statedata_txt']):
        print(f"✗ StateDataOut.txt not found: {config['statedata_txt']}")
        return False, None
    
    # Read the main_comparison.py template
    with open("main_comparison.py", "r") as f:
        main_code = f.read()
    
    # Replace paths and filenames
    old_location_csv = "/home/iq-sim1/Desktop/Deekshitha/android_logs/main_log/19_Aug_2025_logs/19_Aug_1/Location.csv"
    old_statedata = "/home/iq-sim1/Desktop/Deekshitha/android_logs/main_log/19_Aug_2025_logs/19_Aug_1/DataOut/StateDataOut.txt"
    old_html = "comparison_19_Aug_1.html"
    
    main_code = main_code.replace(old_location_csv, config['location_csv'])
    main_code = main_code.replace(old_statedata, config['statedata_txt'])
    main_code = main_code.replace(old_html, config['html_comparison'])
    
    # Write temporary script
    temp_script = "temp_main_comparison.py"
    with open(temp_script, "w") as f:
        f.write(main_code)
    
    try:
        # Run the script
        result = subprocess.run(["python3", temp_script], check=True, capture_output=True, text=True)
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 10:
                print("... (output truncated)")
                for line in lines[-10:]:
                    print(line)
            else:
                print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:", result.stderr)
        
        print(f"✓ Successfully generated {config['html_comparison']}")
        return True, config['html_comparison']
    except subprocess.CalledProcessError as e:
        print(f"✗ Error generating comparison visualization:")
        if e.stdout:
            print("STDOUT:", e.stdout[-500:])
        if e.stderr:
            print("STDERR:", e.stderr[-500:])
        return False, None
    finally:
        # Clean up temporary script
        if os.path.exists(temp_script):
            os.remove(temp_script)

def create_index_html(location_files_list, location_names, statedata_files_list, statedata_names, comparison_files, comparison_names):
    """Create or update index.html with links to all visualizations (4 files per folder)"""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Helicopter 3D Visualizations</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 40px 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        .content {
            padding: 40px;
        }
        .visualization-list {
            list-style: none;
        }
        .visualization-item {
            background: #f8f9fa;
            margin: 15px 0;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            transition: all 0.3s ease;
        }
        .visualization-item:hover {
            background: #e9ecef;
            transform: translateX(5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .visualization-item h3 {
            color: #667eea;
            font-size: 1.3em;
            font-weight: 600;
            margin-bottom: 10px;
        }
        .view-links {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .view-link {
            text-decoration: none;
            color: white;
            background: #667eea;
            padding: 8px 16px;
            border-radius: 5px;
            font-size: 0.9em;
            transition: all 0.3s ease;
            display: inline-block;
        }
        .view-link:hover {
            background: #764ba2;
            transform: translateY(-2px);
            box-shadow: 0 3px 10px rgba(0,0,0,0.2);
        }
        .view-link.points {
            background: #4CAF50;
        }
        .view-link.points:hover {
            background: #45a049;
        }
        .view-link.direction {
            background: #2196F3;
        }
        .view-link.direction:hover {
            background: #1976D2;
        }
        .description {
            color: #6c757d;
            font-size: 0.95em;
            margin-top: 8px;
        }
        .footer {
            text-align: center;
            padding: 20px;
            color: #6c757d;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚁 Helicopter 3D Trajectory Visualizations</h1>
            <p>Interactive 3D maps showing helicopter flight paths with radio altitude data</p>
        </div>
        <div class="content">
            <ul class="visualization-list">
"""
    
    # Add links for each folder (4 files: Location points, Location direction, StateData points, StateData direction)
    for i, display_name in enumerate(location_names):
        loc_files = location_files_list[i] if i < len(location_files_list) else []
        state_files = statedata_files_list[i] if i < len(statedata_files_list) else []
        
        loc_points = next((f for f in loc_files if '_points.html' in f), None)
        loc_direction = next((f for f in loc_files if '_direction.html' in f), None)
        state_points = next((f for f in state_files if '_points.html' in f), None)
        state_direction = next((f for f in state_files if '_direction.html' in f), None)
        
        html_content += f"""                <li class="visualization-item">
                    <h3>{display_name}</h3>
                    <div class="view-links">
                        <a href="{loc_points or '#'}" class="view-link points">📍 Location (Points)</a>
                        <a href="{loc_direction or '#'}" class="view-link direction">📍 Location (Direction)</a>
                        <a href="{state_points or '#'}" class="view-link points">📊 StateData (Points)</a>
                        <a href="{state_direction or '#'}" class="view-link direction">📊 StateData (Direction)</a>
                    </div>
                    <div class="description">3D visualization with radio altitude, terrain data, and trajectory path</div>
                </li>
"""
    
    html_content += """            </ul>
        </div>
        <div class="footer">
        </div>
    </div>
</body>
</html>"""
    
    with open("index.html", "w") as f:
        f.write(html_content)
    total_folders = len(location_names)
    print(f"\n✓ Created/Updated index.html with {total_folders} folders (4 files each: Location points/direction, StateData points/direction)")

if __name__ == "__main__":
    print("="*60)
    print("Helicopter 3D Visualization Generator")
    print("="*60)
    
    # Check if all CSV files exist
    missing_files = []
    for csv_file in CSV_FILES:
        if not os.path.exists(csv_file):
            missing_files.append(csv_file)
    
    if missing_files:
        print("\n⚠ Warning: Some CSV files not found:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nContinuing with available files...\n")
    
    # Generate visualizations
    location_files_list = []
    location_names_list = []
    statedata_files_list = []
    failed = []
    
    for csv_file, html_file_base, display_name in zip(CSV_FILES, HTML_FILES, DISPLAY_NAMES):
        if os.path.exists(csv_file):
            success, html_files = generate_visualization(csv_file, html_file_base)
            if success:
                location_files_list.append(html_files)
                location_names_list.append(display_name)
            else:
                failed.append((csv_file, html_file_base))
        else:
            print(f"\n⚠ Skipping {csv_file} (file not found)")
    
    # Generate StateData visualizations
    statedata_failed = []
    for config in STATEDATA_CONFIGS:
        success, html_files = generate_statedata_visualization(config)
        if success:
            statedata_files_list.append(html_files)
        else:
            statedata_failed.append((config['statedata_txt'], config['html_base']))
    
    # Create index.html (organize by folder)
    if location_files_list and statedata_files_list:
        create_index_html(location_files_list, location_names_list, statedata_files_list, location_names_list, [], [])
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    total_location = sum(len(files) for files in location_files_list)
    total_statedata = sum(len(files) for files in statedata_files_list)
    print(f"✓ Successfully generated: {total_location} Location.csv visualizations ({len(location_files_list)} folders × 2 views)")
    print(f"✓ Successfully generated: {total_statedata} StateData visualizations ({len(statedata_files_list)} folders × 2 views)")
    
    if failed:
        print(f"✗ Failed Location.csv: {len(failed)} folders")
        for csv_file, html_file_base in failed:
            print(f"  - {html_file_base} (from {os.path.basename(csv_file)})")
    
    if statedata_failed:
        print(f"✗ Failed StateData: {len(statedata_failed)} folders")
        for statedata_file, html_file_base in statedata_failed:
            print(f"  - {html_file_base} (from {os.path.basename(statedata_file)})")
    
    print("\nGenerated HTML files (4 per folder):")
    for i, display_name in enumerate(location_names_list):
        print(f"  {display_name}:")
        if i < len(location_files_list):
            for f in location_files_list[i]:
                print(f"    - {f}")
        if i < len(statedata_files_list):
            for f in statedata_files_list[i]:
                print(f"    - {f}")
    
    print(f"\n✓ index.html created/updated")
    print("\nNext steps:")
    print("1. Upload all HTML files to GitHub")
    print("2. Your visualizations will be accessible via index.html")

