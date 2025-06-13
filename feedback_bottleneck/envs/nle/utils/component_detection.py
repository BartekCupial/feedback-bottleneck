import itertools
from typing import List, Tuple

import numpy as np
from scipy import ndimage

from feedback_bottleneck.envs.nle.utils.glyph import SS, C, G


def get_revelable_positions(objects, seen, visited, labeled_features):
    """
    Finds walkable tiles that unveil new areas.
    """
    walkable = get_walkable_mask(objects)

    structure = ndimage.generate_binary_structure(2, 2)
    unexplored_edges = np.logical_and(ndimage.binary_dilation(~seen, structure), seen)
    walkable_edges = np.logical_and(unexplored_edges, walkable)
    discovery_potential = np.logical_and(walkable_edges, ~visited)
    feature_unexplored = np.logical_and(labeled_features, discovery_potential)

    return np.argwhere(feature_unexplored)


def get_walkable_mask(objects: np.ndarray) -> np.ndarray:
    # Derive walkable area from the persistent map objects
    return isin(
        objects,
        G.FLOOR,
        G.STAIR_UP,
        G.STAIR_DOWN,
        G.DOOR_OPENED,
        G.TRAPS,
        G.ALTAR,
        G.FOUNTAIN,
        G.SINK,
        G.MONS,
        G.PETS,
        G.BODIES,
        G.OBJECTS,
        G.STATUES,
    )


def label_dungeon_features(objects: np.ndarray):
    structure = ndimage.generate_binary_structure(2, 1)

    # Derive walkable mask on the fly
    walkable = get_walkable_mask(objects)

    # Rooms
    room_floor = frozenset({SS.S_room, SS.S_darkroom})
    rooms = isin(objects, room_floor)
    labeled_rooms, num_rooms = ndimage.label(rooms, structure=structure)

    # Corridors
    corridor_floor = frozenset({SS.S_corr, SS.S_litcorr})
    corridors = isin(objects, corridor_floor)
    labeled_corridors, num_corridors = ndimage.label(corridors, structure=structure)

    # Doors
    doors = isin(objects, frozenset({SS.S_ndoor}), G.DOOR_OPENED)

    # Combine rooms and corridors
    labeled_features = np.zeros_like(objects)
    labeled_features[rooms] = labeled_rooms[rooms]
    labeled_features[corridors] = labeled_corridors[corridors] + num_rooms

    def label_walkable_features(position):
        neighbors = []
        height, width = objects.shape
        for x, y in itertools.product([-1, 0, 1], repeat=2):
            if x == 0 and y == 0:
                continue
            if not (0 <= position[0] + y < height) or not (0 <= position[1] + x < width):
                continue

            neighbor = labeled_features[position[0] + y, position[1] + x]
            if neighbor != 0:
                neighbors.append(neighbor)

        if neighbors:
            if np.all(np.array(neighbors) <= num_rooms):
                rooms[position] = True
            else:
                corridors[position] = True

    # Handle walkable spots that were missed (like player starting position)
    # We use the derived 'walkable' mask here
    for p in np.argwhere(np.logical_and(walkable, labeled_features == 0)):
        label_walkable_features(tuple(p))

    corridors[doors] = True
    rooms[doors] = False

    labeled_rooms, num_rooms = ndimage.label(rooms, structure=structure)
    labeled_corridors, num_corridors = ndimage.label(corridors, structure=structure)
    labeled_features = np.zeros_like(objects)
    labeled_features[rooms] = labeled_rooms[rooms]
    labeled_features[corridors] = labeled_corridors[corridors] + num_rooms

    return labeled_features, num_rooms, num_corridors


def room_detection(objects: np.ndarray) -> Tuple[np.ndarray, int]:
    labeled_features, num_rooms, _ = label_dungeon_features(objects)
    labeled_features[labeled_features > num_rooms] = 0
    return labeled_features, num_rooms


def corridor_detection(objects: np.ndarray) -> Tuple[np.ndarray, int]:
    labeled_features, num_rooms, _ = label_dungeon_features(objects)
    labeled_features[labeled_features <= num_rooms] = 0
    labeled_features -= num_rooms
    labeled_features[labeled_features < 0] = 0
    return labeled_features, _


def isin(array: np.ndarray, *elems) -> np.ndarray:
    """
    Checks if elements of 'array' are present in the flattened 'elems'.
    """
    # Flatten arguments (handles lists, tuples, sets, frozensets, and single values)
    flat_elems = []
    for e in elems:
        if isinstance(e, (list, tuple, set, frozenset, np.ndarray)):
            flat_elems.extend(e)
        else:
            flat_elems.append(e)

    # allow_unknown_types=False assumes we are comparing strictly numerical glyphs
    return np.isin(array, flat_elems)
