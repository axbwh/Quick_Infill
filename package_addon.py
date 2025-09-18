import os
import zipfile
import re
import argparse
import toml

ADDON_NAME = "Quick_Infill"
SCRIPT_NAME = "package_addon.py"
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), "releases")
EXCLUDE = {"__pycache__", ".git", ".vscode", SCRIPT_NAME, "releases", ".gitignore", "README.md", "scripts"}

# Extract version from blender_manifest.toml
def get_version():
    manifest_file = os.path.join(os.path.dirname(__file__), "blender_manifest.toml")
    try:
        with open(manifest_file, "r") as f:
            manifest = toml.load(f)
        version = manifest.get("version", "1.0.0")
        return f"v{version}"
    except Exception as e:
        print(f"Error reading manifest: {e}")
        return "v1.0.0"

# Increment version in blender_manifest.toml
def increment_version():
    manifest_file = os.path.join(os.path.dirname(__file__), "blender_manifest.toml")
    
    try:
        with open(manifest_file, "r") as f:
            manifest = toml.load(f)
        
        current_version = manifest.get("version", "1.0.0")
        version_parts = current_version.split(".")
        
        # Increment patch version
        if len(version_parts) >= 3:
            major, minor, patch = int(version_parts[0]), int(version_parts[1]), int(version_parts[2])
            patch += 1
        else:
            major, minor, patch = 1, 0, 1
        
        new_version = f"{major}.{minor}.{patch}"
        manifest["version"] = new_version
        
        # Write back to file
        with open(manifest_file, "w") as f:
            toml.dump(manifest, f)
        
        print(f"Version incremented: {current_version} → {new_version}")
        return f"v{new_version}"
        
    except Exception as e:
        print(f"Error incrementing version: {e}")
        return get_version()

def should_include(path):
    """Check if a file/directory should be included in the package."""
    for part in path.split(os.sep):
        if part in EXCLUDE:
            return False
    # Exclude any .pyc files
    if path.endswith('.pyc'):
        return False
    return True

def create_package(increment_ver=False):
    """Create the addon package."""
    
    # Determine version
    if increment_ver:
        version = increment_version()
    else:
        version = get_version()
    
    # Create output directory
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Create zip filename without timestamp
    output_zip = os.path.join(OUTPUT_FOLDER, f"{ADDON_NAME}_{version}.zip")
    
    print(f"Packaging {ADDON_NAME} {version}...")
    print(f"Output: {output_zip}")
    
    addon_root = os.path.dirname(__file__)
    
    try:
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(addon_root):
                # Filter out excluded directories
                dirs[:] = [d for d in dirs if should_include(os.path.join(root, d))]
                
                for file in files:
                    filepath = os.path.join(root, file)
                    
                    if should_include(filepath):
                        # Calculate relative path from addon root
                        arcname = os.path.relpath(filepath, addon_root)
                        
                        # Skip the output zip itself if it's in the same directory
                        if os.path.samefile(filepath, output_zip):
                            continue
                            
                        try:
                            zipf.write(filepath, arcname)
                            print(f"  Added: {arcname}")
                        except Exception as e:
                            print(f"  Error adding {arcname}: {e}")
        
        print(f"\nPackage created successfully: {output_zip}")
        return output_zip
        
    except Exception as e:
        print(f"Error creating package: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package the Quick Infill Blender addon.")
    parser.add_argument("--increment", action="store_true", help="Increment the version number before packaging.")
    parser.add_argument("--no-increment", action="store_true", help="Do not increment the version number before packaging.")
    parser.add_argument("--version", action="store_true", help="Show current version and exit.")
    
    args = parser.parse_args()
    
    if args.version:
        print(f"Current version: {get_version()}")
    elif args.increment:
        create_package(increment_ver=True)
    elif args.no_increment:
        create_package(increment_ver=False)
    else:
        # Interactive mode - ask user
        current_version = get_version()
        print(f"Current version: {current_version}")
        
        while True:
            response = input("Increment version before packaging? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                create_package(increment_ver=True)
                break
            elif response in ['n', 'no']:
                create_package(increment_ver=False)
                break
            else:
                print("Please enter 'y' for yes or 'n' for no.")