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

from . import settings, tools_operators, support_tools, heal_cavity, ui
from .meshlib_utils import get_meshlib  # noqa: F401  (kept for external callers)

_modules = (settings, tools_operators, support_tools, heal_cavity, ui)


def register():
    for m in _modules:
        m.register()
    # Import meshlib now (not on the first button press) so the first tool
    # run doesn't pay the import cost. A missing install must not block
    # registration, so failures are only printed.
    try:
        get_meshlib()
    except ImportError as e:
        print(f"[Quick Infill] {e}")


def unregister():
    for m in reversed(_modules):
        m.unregister()
