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

# Flags to prevent redundant operations
_initialized = False


def _setup_meshlib_paths():
	"""
	Ensure meshlib is findable and DLL directories are registered.
	
	Blender's Extension system installs wheels but sometimes fails to add
	the site-packages to sys.path on restart. This function fixes that.
	
	IMPORTANT: On Windows, DLL directories must be registered BEFORE importing meshlib.
	"""
	global _initialized
	if _initialized:
		return
	_initialized = True
	
	# Find meshlib in sys.path or extension site-packages
	meshlib_site = None
	
	# First check existing sys.path
	for path in sys.path:
		if os.path.isdir(os.path.join(path, 'meshlib')):
			meshlib_site = path
			break
	
	# If not found, search extension site-packages
	if not meshlib_site:
		try:
			for ext_path in bpy.utils.script_paths(subdir="extensions"):
				local_site = os.path.join(
					ext_path, '.local', 'lib',
					f'python{sys.version_info.major}.{sys.version_info.minor}',
					'site-packages'
				)
				if os.path.isdir(os.path.join(local_site, 'meshlib')):
					meshlib_site = local_site
					# Add to sys.path
					if meshlib_site not in sys.path:
						sys.path.insert(0, meshlib_site)
						print(f"[Quick Infill] Added to sys.path: {meshlib_site}")
					break
		except Exception as e:
			print(f"[Quick Infill] Error searching extension paths: {e}")
	
	if not meshlib_site:
		print("[Quick Infill] Warning: meshlib not found in extension site-packages")
		print("[Quick Infill] Please install via Edit > Preferences > Extensions > Install from Disk")
		return
	
	# Windows: Register DLL directory BEFORE any import attempt
	if sys.platform == 'win32':
		meshlib_libs = os.path.join(meshlib_site, 'meshlib.libs')
		if os.path.isdir(meshlib_libs):
			try:
				os.add_dll_directory(meshlib_libs)
				os.environ['PATH'] = meshlib_libs + os.pathsep + os.environ.get('PATH', '')
				print(f"[Quick Infill] Added DLL directory: {meshlib_libs}")
			except Exception as e:
				print(f"[Quick Infill] Warning adding DLL path: {e}")
		else:
			print(f"[Quick Infill] Warning: meshlib.libs not found at {meshlib_libs}")


def ensure_wheels_loaded():
	"""Ensure meshlib is importable before first use."""
	_setup_meshlib_paths()
	
	try:
		import meshlib.mrmeshpy
		import meshlib.mrcudapy
		return True
	except ImportError as e:
		raise ImportError(
			f"Quick Infill: Failed to load meshlib - {e}\n"
			"Please reinstall via Edit > Preferences > Extensions > Install from Disk"
		)


def get_meshlib():
	"""Get meshlib modules with proper error handling."""
	ensure_wheels_loaded()
	import meshlib.mrmeshpy as mm
	import meshlib.mrcudapy as mc
	return mm, mc


def register():
	# Set up paths but don't fail registration if meshlib isn't available
	try:
		_setup_meshlib_paths()
	except Exception as e:
		print(f"[Quick Infill] Warning during setup: {e}")
	
	heal_cavity.register()
	ui.register()


def unregister():
	ui.unregister()
	heal_cavity.unregister()
