# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import bpy
import os
import sys
from . import ui, heal_cavity

# Global flag to track wheel loading
_wheels_loaded = False
_dll_directories_added = False

def _add_meshlib_dll_directories():
	"""Add meshlib DLL directories to search path on Windows BEFORE importing meshlib."""
	global _dll_directories_added
	if _dll_directories_added:
		return
	
	if sys.platform != 'win32':
		_dll_directories_added = True
		return
	
	found_dirs = []
	meshlib_site_packages = None
	
	# Method 1: Search sys.path for site-packages containing meshlib
	for path in sys.path:
		meshlib_dir = os.path.join(path, 'meshlib')
		if os.path.isdir(meshlib_dir):
			meshlib_site_packages = path
			meshlib_libs = os.path.join(path, 'meshlib.libs')
			if os.path.isdir(meshlib_libs):
				found_dirs.append(meshlib_libs)
			break
	
	# Method 2: Check Blender's extension directories
	if not meshlib_site_packages:
		try:
			for ext_path in bpy.utils.script_paths(subdir="extensions"):
				local_site = os.path.join(ext_path, '.local', 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages')
				meshlib_dir = os.path.join(local_site, 'meshlib')
				if os.path.isdir(meshlib_dir):
					meshlib_site_packages = local_site
					meshlib_libs = os.path.join(local_site, 'meshlib.libs')
					if os.path.isdir(meshlib_libs):
						found_dirs.append(meshlib_libs)
					break
		except Exception as e:
			print(f"[Quick Infill] Warning checking extension paths: {e}")
	
	# If meshlib.libs doesn't exist but meshlib does, extract from bundled wheel
	if meshlib_site_packages and not found_dirs:
		meshlib_libs = os.path.join(meshlib_site_packages, 'meshlib.libs')
		if not os.path.isdir(meshlib_libs):
			print(f"[Quick Infill] meshlib.libs not found, extracting from bundled wheel...")
			try:
				_extract_meshlib_libs(meshlib_site_packages)
				if os.path.isdir(meshlib_libs):
					found_dirs.append(meshlib_libs)
					print(f"[Quick Infill] Successfully extracted meshlib.libs")
			except Exception as e:
				print(f"[Quick Infill] Failed to extract meshlib.libs: {e}")
	
	# Add all found directories
	if found_dirs:
		for dll_dir in found_dirs:
			try:
				if hasattr(os, 'add_dll_directory'):
					os.add_dll_directory(dll_dir)
				os.environ['PATH'] = dll_dir + os.pathsep + os.environ.get('PATH', '')
				print(f"[Quick Infill] Added DLL directory: {dll_dir}")
			except Exception as e:
				print(f"[Quick Infill] Warning adding DLL dir {dll_dir}: {e}")
	else:
		print(f"[Quick Infill] Warning: Could not find or create meshlib.libs directory")
	
	_dll_directories_added = True


def _extract_meshlib_libs(target_site_packages):
	"""Extract meshlib.libs from bundled wheel to target site-packages."""
	import zipfile
	
	# Find the bundled wheel
	addon_dir = os.path.dirname(os.path.abspath(__file__))
	wheels_dir = os.path.join(addon_dir, 'wheels')
	
	wheel_path = None
	if os.path.isdir(wheels_dir):
		for f in os.listdir(wheels_dir):
			if f.startswith('meshlib') and f.endswith('.whl'):
				wheel_path = os.path.join(wheels_dir, f)
				break
	
	if not wheel_path or not os.path.isfile(wheel_path):
		raise FileNotFoundError(f"Could not find meshlib wheel in {wheels_dir}")
	
	print(f"[Quick Infill] Extracting from wheel: {wheel_path}")
	
	with zipfile.ZipFile(wheel_path, 'r') as whl:
		# Extract only meshlib.libs/ entries
		for member in whl.namelist():
			if member.startswith('meshlib.libs/'):
				whl.extract(member, target_site_packages)
				print(f"[Quick Infill]   Extracted: {member}")

def ensure_wheels_loaded():
	"""Ensure wheels are loaded before first use."""
	global _wheels_loaded
	if not _wheels_loaded:
		try:
			# Register DLL directories first on Windows - BEFORE importing meshlib
			_add_meshlib_dll_directories()
			# Test wheel availability by importing both modules
			import meshlib.mrmeshpy
			import meshlib.mrcudapy
			_wheels_loaded = True
			print("[Quick Infill] Meshlib wheels loaded successfully")
		except ImportError as e:
			print(f"[Quick Infill] Failed to load meshlib wheels: {e}")
			_wheels_loaded = False
			raise ImportError(f"Quick Infill requires meshlib wheels: {str(e)}")

def get_meshlib():
	"""Get meshlib modules with proper error handling."""
	ensure_wheels_loaded()
	try:
		import meshlib.mrmeshpy as mm
		import meshlib.mrcudapy as mc
		return mm, mc
	except ImportError as e:
		raise ImportError(f"Failed to import meshlib after wheel loading: {str(e)}")

def register():
	# Try to load wheels, but don't crash if they fail
	try:
		ensure_wheels_loaded()
	except ImportError as e:
		print(f"[Quick Infill] Warning: {e}")
		print("[Quick Infill] Addon loaded but meshlib operations will fail until wheels are available")
		# Continue registration so users can see the panel and get error messages
	
	# Register operator first so UI can reference it
	heal_cavity.register()
	ui.register()


def unregister():
	ui.unregister()
	heal_cavity.unregister()
