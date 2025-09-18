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
from . import ui, heal_cavity

# Global flag to track wheel loading
_wheels_loaded = False

def ensure_wheels_loaded():
	"""Ensure wheels are loaded before first use."""
	global _wheels_loaded
	if not _wheels_loaded:
		try:
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
	# Ensure wheels are loaded during registration
	ensure_wheels_loaded()
	
	# Register operator first so UI can reference it
	heal_cavity.register()
	ui.register()


def unregister():
	ui.unregister()
	heal_cavity.unregister()
