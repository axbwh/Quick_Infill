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

from . import settings, tools_operators, support_tools, remove_scoop, heal_cavity, ui
from .meshlib_utils import get_meshlib  # noqa: F401  (kept for external callers)

_modules = (settings, tools_operators, support_tools, remove_scoop, heal_cavity, ui)


def register():
    # meshlib is imported on the first tool use, not here, so Blender starts
    # (and quits) exactly as before for sessions that never use Quick Infill.
    for m in _modules:
        m.register()


def unregister():
    for m in reversed(_modules):
        m.unregister()
