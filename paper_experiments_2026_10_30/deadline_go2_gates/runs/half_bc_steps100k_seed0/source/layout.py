"""Shared metric geometry for physics and previews; no latent task observations."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Config:
    dt: float = .5
    episodes: int = 3
    # Based on the saved physical route checks; not shortest-path bounds.
    deadlines: tuple = (18., 48., 100.)
    base: tuple = (0., 0.)
    goal: tuple = (6., 0.)
    fallback_goal: tuple = (-.35, -2.6)
    fallback_value: float = 1.
    goal_value: float = 4.
    command_limits: tuple = (.6, .4, 0.)
    pickup_radius: float = .45
    delivery_radius: float = .45
    gate_open_probability: float = .5
    sensor_range: float = 2.
    map_extent: float = 4.
    coordinate_scale: float = 8.
    grid_size: int = 64
    voxel_size: float = .125
    voxel_height: float = 1.5
    ground_cutoff: float = .08

    @property
    def stations(self):
        # Shared Genesis adapter draws a non-contact loading-station fixture.
        return (self.fallback_goal,self.goal)


@dataclass(frozen=True)
class Box:
    name: str
    xy: tuple
    size: tuple
    height: float = 1.4


DIVIDER_X = 4.
PROBE_X = DIVIDER_X-.85
EXIT_X = DIVIDER_X+.85

WALLS = (
    Box('boundary_left', (-1.2, -.8), (.25, 11.45)),
    Box('boundary_right', (7.2, -.8), (.25, 11.45)),
    Box('boundary_bottom', (3., -6.4), (8.65, .25)),
    Box('boundary_top', (3., 4.8), (8.65, .25)),
    Box('divider_bottom', (DIVIDER_X, -6.2), (.25, .4)),
    Box('divider_lower', (DIVIDER_X, -2.6), (.25, 3.6)),
    Box('divider_upper', (DIVIDER_X, 1.8), (.25, 2.)),
    Box('divider_top', (DIVIDER_X, 4.6), (.25, .4)),
)
GATES = (Box('middle_gate', (DIVIDER_X, 0.), (.25, 1.6)),
         Box('upper_gate', (DIVIDER_X, 3.6), (.25, 1.6)))
PATTERNS = ((True, True), (True, False), (False, True), (False, False))
# True means open. The lower opening at y=-5.2 is always open.
ROUTES = {
    'fallback': ((-.35,-2.6),),
    'middle': ((PROBE_X, 0.), (EXIT_X, 0.), (6., 0.)),
    'upper': ((PROBE_X, 3.6), (EXIT_X, 3.6), (6., 0.)),
    'lower': ((PROBE_X, -5.2), (EXIT_X, -5.2), (6., 0.)),
}


def pattern_name(doors):
    return ''.join('1' if value else '0' for value in doors)


def route_for(doors):
    """Oracle ONLY for physical feasibility calibration, never a policy input."""
    return 'middle' if doors[0] else 'upper' if doors[1] else 'lower'


def footprint_clearance(position, doors, radius=.40):
    """Approximate 0.40m circle diagnostic, not an exact robot body envelope."""
    boxes = list(WALLS) + [box for box, opened in zip(GATES, doors) if not opened]
    values = []
    for box in boxes:
        d = np.abs(np.asarray(position)[:2] - box.xy) - np.asarray(box.size)/2
        signed = np.linalg.norm(np.maximum(d, 0)) + min(float(d.max()), 0.)
        values.append(signed-radius)
    return float(min(values))
