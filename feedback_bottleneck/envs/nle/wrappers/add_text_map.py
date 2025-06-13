import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
from scipy import ndimage

from feedback_bottleneck.envs.nle.utils.blstats import BLStats
from feedback_bottleneck.envs.nle.utils.component_detection import (
    corridor_detection,
    get_revelable_positions,
    isin,
    room_detection,
)
from feedback_bottleneck.envs.nle.utils.entity import Entity
from feedback_bottleneck.envs.nle.utils.glyph import SHOP, SS, C, G


class AddTextMap(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Memory structure: (dungeon, level) -> { 'objects': arr, 'visited': arr, 'seen': arr }
        self.dungeon_maps = {}

        self.terrain_features = defaultdict(dict)
        self.shops = defaultdict(list)
        self.map_description = ""

        self.update(obs)

        return self.populate_obs(obs), info

    def get_entity(self, obs, blstats: BLStats, glyphs: np.ndarray) -> Entity:
        """
        Returns:
            Entity object with the player
        """
        position = (blstats.y, blstats.x)
        return Entity(position, glyphs[position])

    def get_entities(self, obs, blstats: BLStats, glyphs: np.ndarray) -> List[Entity]:
        """
        Returns:
            List of Entity objects with the monsters
        """
        monster_mask = isin(glyphs, G.MONS, G.INVISIBLE_MON)
        monster_mask[blstats.y, blstats.x] = 0

        return [Entity(position, glyphs[position]) for position in list(zip(*np.where(monster_mask)))]

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.update(obs)

        return self.populate_obs(obs), reward, terminated, truncated, info

    def populate_obs(self, obs):
        return {**obs, "text_map": str(self)}

    def update(self, obs):
        blstats = BLStats(*obs["blstats"])
        message = obs["text_message"]
        glyphs = obs["glyphs"]
        entity: Entity = self.get_entity(obs, blstats, glyphs)
        entities: List[Entity] = self.get_entities(obs, blstats, glyphs)

        self.update_map(glyphs, blstats)
        self.update_terrain_features(glyphs, blstats)
        self.update_shops(blstats, message, entity, entities)
        self.map_description = self.describe_map(blstats, entity)

    def update_map(self, glyphs: np.ndarray, blstats: BLStats):
        """
        Maintains the persistent map state (Fog of War & X-Ray for monsters).
        Replaces the Level class.
        """
        key = (blstats.dungeon_number, blstats.level_number)

        # Initialize if new level
        if key not in self.dungeon_maps:
            self.dungeon_maps[key] = {
                "objects": np.full((C.SIZE_Y, C.SIZE_X), -1, dtype=np.int16),
                "visited": np.zeros((C.SIZE_Y, C.SIZE_X), dtype=bool),
                "seen": np.zeros((C.SIZE_Y, C.SIZE_X), dtype=bool),
            }

        d_map = self.dungeon_maps[key]

        # Update visited status
        d_map["visited"][blstats.y, blstats.x] = True

        # Update Floor/Walls (Background)
        # We only update 'objects' if we see actual static terrain (not monsters)
        bg_mask = isin(
            glyphs,
            G.FLOOR,
            G.STAIR_UP,
            G.STAIR_DOWN,
            G.DOOR_OPENED,
            G.TRAPS,
            G.ALTAR,
            G.FOUNTAIN,
            G.SINK,
            G.WALL,
            G.DOOR_CLOSED,
            G.BARS,
            G.BOULDER,
            frozenset({SS.S_lava, SS.S_water}),
        )
        d_map["objects"][bg_mask] = glyphs[bg_mask]
        d_map["seen"][bg_mask] = True

        # Mark monsters/items as seen
        fg_mask = isin(glyphs, G.MONS, G.PETS, G.BODIES, G.OBJECTS, G.STATUES)
        d_map["seen"][fg_mask] = True

    def update_terrain_features(self, glyphs, blstats: BLStats):
        current_features = self.get_terrain_features(glyphs)
        past_features = self.terrain_features[(blstats.dungeon_number, blstats.level_number)].get("features", {})

        for key in ("stairs up", "stairs down"):
            past_positions = past_features.get(key)
            curr_positions = current_features.get(key)

            if past_positions is not None and curr_positions is not None:
                merged = np.vstack((past_positions, curr_positions))
                current_features[key] = np.unique(merged, axis=0)
            elif past_positions is not None and curr_positions is None:
                current_features[key] = past_positions

        self.terrain_features[(blstats.dungeon_number, blstats.level_number)]["features"] = current_features

    def get_terrain_features(self, glyphs) -> Dict[str, Any]:
        name_glyph = {
            "stairs down": G.STAIR_DOWN,
            "stairs up": G.STAIR_UP,
            "altar": G.ALTAR,
            "fountain": G.FOUNTAIN,
            "throne": G.THRONE,
            "sink": G.SINK,
            "trap": G.TRAPS,
            "grave": G.GRAVE,
        }
        terrain_features = {}
        for name, glyph in name_glyph.items():
            mask = isin(glyphs, glyph)
            positions = np.argwhere(mask)
            if len(positions) > 0:
                terrain_features[name] = positions
        return terrain_features

    def update_shops(self, blstats, message, entity, entities):
        matches = re.search(f"Welcome( again)? to [a-zA-Z' ]*({'|'.join(SHOP.name2id.keys())})!", message)
        if matches is None:
            return

        shop_name = matches.groups()[1]
        shop_type = SHOP.name2id[shop_name]
        shop_string = SHOP.id2string[shop_type]
        shop_keepers = [ent.position for ent in entities if ent.name == "shopkeeper"]

        if not shop_keepers:
            return

        closest = min(shop_keepers, key=lambda sk: np.linalg.norm(np.array(sk) - np.array(entity.position)))
        self.shops[(blstats.dungeon_number, blstats.level_number)].append(
            {"name": shop_string, "type": shop_type, "position": closest}
        )

    def describe_room(
        self,
        blstats: BLStats,
        entity: Entity,
        room_mask: np.ndarray,
        dilated_corridors: np.ndarray,
        dilated_doors: np.ndarray,
        dilated_bars: np.ndarray,
        revelable_positions: np.ndarray,
        visited_mask: np.ndarray,
    ):
        def direction_to(from_xy, to_xy):
            dy, dx = to_xy[0] - from_xy[0], to_xy[1] - from_xy[1]
            dirs = []
            if dy < 0:
                dirs.append("north")
            elif dy > 0:
                dirs.append("south")
            if dx < 0:
                dirs.append("west")
            elif dx > 0:
                dirs.append("east")
            return " ".join(dirs) if dirs else "here"

        def get_distance_name(distance):
            distance_order = {
                "very far to the": 32,
                "far to the": 16,
                "to the": 8,
                "a short distance to the": 4,
                "immediately": 1,
                "": 0,
            }
            for name, dist in distance_order.items():
                if distance >= dist:
                    return name
            return "unknown"

        room_coords = np.argwhere(room_mask)
        py, px = entity.position

        # Exploration Status
        # Check if the room has tiles that are revelable (next to unseen space)
        is_revelable = False
        if len(revelable_positions) > 0:
            # Check intersection of room_coords and revelable_positions
            # Broadcasting: (N_room, 1, 2) == (1, N_rev, 2) -> (N_room, N_rev)
            matches = np.all(room_coords[:, None] == revelable_positions[None, :], axis=-1)
            is_revelable = np.any(matches)

        if is_revelable:
            if np.any(np.logical_and(visited_mask, room_mask)):
                explored = "Partially explored"
            else:
                explored = "Unexplored"
        else:
            explored = "Explored"

        # Exits
        corridor_exits = np.argwhere(np.logical_and(dilated_corridors, room_mask))
        door_exits = np.argwhere(np.logical_and(dilated_doors, room_mask))
        bar_exits = np.argwhere(np.logical_and(dilated_bars, room_mask))
        num_exits = len(corridor_exits) + len(door_exits) + len(bar_exits)
        num_closed_doors = len(door_exits)
        num_bars = len(bar_exits)

        # Features inside room
        map_features = self.terrain_features[(blstats.dungeon_number, blstats.level_number)].get("features", {})
        room_features = defaultdict(int)
        for feature_name, positions in map_features.items():
            for pos in positions:
                if room_mask[tuple(pos)]:
                    room_features[feature_name] += 1

        name_plural = {
            "stairs down": ("stairs down", "stairs down"),
            "stairs up": ("stairs up", "stairs up"),
            "altar": ("an altar", "altars"),
            "fountain": ("a fountain", "fountains"),
            "throne": ("a throne", "thrones"),
            "sink": ("a sink", "sinks"),
            "trap": ("a trap", "traps"),
            "grave": ("a grave", "graves"),
        }

        features = []
        for feature, count in room_features.items():
            if count == 1:
                features.append(name_plural[feature][0])
            elif count > 1:
                features.append(f"{count} {name_plural[feature][1]}")

        # Shop Name
        shop_name = None
        for shop_info in self.shops[(blstats.dungeon_number, blstats.level_number)]:
            if room_mask[tuple(shop_info["position"])]:
                shop_name = shop_info["name"]
                break

        # Distance & Direction
        # Calculate Manhattan distances from player to every tile in the room
        dists = np.sum(np.abs(room_coords - np.array([py, px])), axis=1)
        min_dist_idx = np.argmin(dists)
        min_dist = dists[min_dist_idx]

        distance_str = get_distance_name(min_dist)

        if min_dist == 0:
            direction_str = "here"
        else:
            direction_str = direction_to((py, px), room_coords[min_dist_idx])

        return {
            "explored": explored,
            "distance": distance_str,
            "direction": direction_str,
            "num_exits": num_exits,
            "num_closed_doors": num_closed_doors,
            "num_bars": num_bars,
            "features": features,
            "shop_name": shop_name,
        }

    def describe_map(self, blstats, entity):
        key = (blstats.dungeon_number, blstats.level_number)
        d_map = self.dungeon_maps[key]

        objects = d_map["objects"]
        visited = d_map["visited"]
        seen = d_map["seen"]

        # Labeling (using raw objects array)
        labeled_rooms, num_rooms = room_detection(objects)
        labeled_corridors, num_corridors = corridor_detection(objects)

        # Determine revelable positions based on edges of 'seen'
        revelable_positions = get_revelable_positions(objects, seen, visited, labeled_rooms)

        # Dilate for adjacency checks
        dilated_corridors = ndimage.binary_dilation(labeled_corridors)
        dilated_doors = ndimage.binary_dilation(isin(objects, G.DOOR_CLOSED))
        dilated_bars = ndimage.binary_dilation(isin(objects, G.BARS))

        desc = []
        for room_id in range(1, num_rooms + 1):
            room_mask = labeled_rooms == room_id

            room_info = self.describe_room(
                blstats, entity, room_mask, dilated_corridors, dilated_doors, dilated_bars, revelable_positions, visited
            )

            # Construct Text
            explored = room_info["explored"]
            distance = room_info["distance"]
            direction = room_info["direction"]
            num_exits = room_info["num_exits"]
            num_closed = room_info["num_closed_doors"]
            num_bars = room_info["num_bars"]
            shop_name = room_info["shop_name"]

            # Feature string
            feat_str = ""
            if room_info["features"]:
                feat_str = "    Objects: " + ", ".join(room_info["features"]) + "."

            # "You are here" logic
            if direction == "here":
                here_text = "<- You are here."
                direction = ""
                punctuation = ":" if feat_str else ""
            else:
                here_text = ""
                punctuation = ":" if feat_str else "."

            # Main Room Description
            if shop_name:
                base_desc = f"{explored} {shop_name}"
            else:
                base_desc = f"{explored} room"

            # Exits Description
            if num_exits > 0:
                exits_text = f"with {num_exits} {'exit' if num_exits == 1 else 'exits'}"

                blocked = []
                if num_closed > 0:
                    blocked.append(f"{num_closed} closed doors")
                if num_bars > 0:
                    blocked.append(f"{num_bars} iron bars")

                if blocked:
                    exits_text += f" ({'and '.join(blocked)})"

                base_desc += " " + exits_text

            # Assemble sentence parts
            # e.g. "Explored room with 2 exits to the north."
            parts = [base_desc, distance, direction, punctuation, here_text]
            # Filter empty strings
            text_line = " ".join([p for p in parts if p])

            # Clean up formatting artifacts
            text_line = text_line.replace(" :", ":").replace(" .", ".")

            desc.append(text_line)
            if feat_str:
                desc.append(feat_str)

        return "\n".join(desc)

    def __str__(self):
        return self.map_description

    def __repr__(self):
        return self.map_description
