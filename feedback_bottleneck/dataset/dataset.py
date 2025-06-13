import logging
import os
import time
from collections import Counter
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from datasets import Dataset, Image

from feedback_bottleneck.dataset.loading import load_single_split_dataset
from feedback_bottleneck.dataset.tokenizer import Tokenizer
from feedback_bottleneck.dataset.utils import load_or_compute

# try to import crafter vid exporter for rendering saved states on the fly
try:
    from crafter import vid_exporter as crafter_vid
except Exception:
    logging.error("Failed to import crafter vid_exporter." "Rendering saved states will not be available.")


def dict_to_torch_float(data):
    return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


class TrajectoryDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path: str,
        embeddings_path: str,
        normalize_inputs: bool = False,
        gamma: float = 0.99,
        horizon: int = 400,
        goal_types: tuple[str, ...] = ("goal_plan_embedding",),
        max_token_length: int = 128,
        tokenizer_id: str = "distilroberta-base",
        use_game_states: bool = False,
        render_size: tuple[int, int] = (64, 64),
    ):
        self.dataset_path = dataset_path
        self.embeddings_path = embeddings_path
        self.normalize_inputs = normalize_inputs
        self.gamma = gamma
        self.horizon = horizon
        self.goal_types = goal_types
        # whether dataset contains full game-state objects (to be rendered on the fly)
        self.use_game_states = use_game_states
        self.render_size = tuple(render_size)

        self.dataset = load_single_split_dataset(dataset_path, split="train")
        self.embeddings = load_single_split_dataset(embeddings_path, split="train")

        if "goal_plan" in self.goal_types:
            self.tokenizer = Tokenizer(max_token_length=max_token_length, tokenizer_id=tokenizer_id)

        cache_dir = Path(self.embeddings.cache_files[0]["filename"]).parent
        self.plan_indices = load_or_compute(cache_dir / "plan_indices.pkl", self.calculate_plan_indices)

        cache_dir = Path(self.dataset.cache_files[0]["filename"]).parent
        self.traj_begins = load_or_compute(cache_dir / "traj_begins.pkl", self.calculate_traj_begins)
        self.traj_lens = load_or_compute(cache_dir / "traj_lens.pkl", self.calculate_traj_lens)
        self.returns = load_or_compute(cache_dir / "returns.pkl", self.calculate_returns)
        self.traj_probs = load_or_compute(cache_dir / "traj_probs.pkl", self.calculate_probs)
        self.num_timesteps = sum(self.traj_lens)

        self.mean, self.std = (
            load_or_compute(cache_dir / "image_stats.pkl", self.calculate_normalization)
            if normalize_inputs
            else (None, None)
        )

        logging.info("=" * 50)
        logging.info(f"Dataset: {dataset_path}")
        logging.info(f"{len(self.traj_lens)} trajectories, {self.num_timesteps} timesteps found")
        logging.info(f"Average return: {np.mean(self.returns):.2f}, std: {np.std(self.returns):.2f}")
        logging.info(f"Max return: {np.max(self.returns):.2f}, min: {np.min(self.returns):.2f}")
        logging.info("=" * 50)

    @property
    def embedding_shape(self):
        return np.array(self.embeddings["embedding"][0]).shape

    def calculate_plan_indices(self):
        logging.info("Calculating plan indices...")

        plan_timesteps = self.embeddings["plan_timestep"]

        # Create a mapping from timestep to plan index.
        plan_indices = {}
        current_plan_timestep = 0
        for i in range(len(self.dataset)):
            if i < len(self.embeddings) and i >= plan_timesteps[current_plan_timestep]:
                current_plan_timestep += 1

            if current_plan_timestep == 0:
                plan_indices[i] = plan_timesteps[0]
            else:
                idx = min(current_plan_timestep - 1, len(plan_timesteps) - 1)
                plan_indices[i] = plan_timesteps[idx]

        return pd.Series(plan_indices)

    def calculate_traj_begins(self):
        logging.info("Calculating trajectory beginnings...")

        # Find the first index of each episode.
        episode_indices = {}
        for idx, episode_index in enumerate(self.dataset["episode_index"]):
            if episode_index not in episode_indices:
                episode_indices[episode_index] = idx

        return pd.Series(episode_indices)

    def calculate_traj_lens(self):
        logging.info("Calculating trajectory lengths...")

        traj_lens = Counter(self.dataset["episode_index"])

        return pd.Series(traj_lens)

    def calculate_returns(self):
        logging.info("Calculating trajectory returns...")

        episode_indices = self.dataset["episode_index"]
        rewards = self.dataset["rewards"]
        df = pd.DataFrame({"episode_index": episode_indices, "rewards": rewards})
        returns = df.groupby("episode_index")["rewards"].sum()

        return returns

    def calculate_probs(self):
        logging.info("Calculating trajectory probabilities...")

        episode_indices = np.array(self.dataset["episode_index"])

        traj_values = np.array([self.traj_lens[idx] - 1 for idx in episode_indices])

        # Find the last timestep index for each trajectory
        traj_ends = {}
        for i, idx in enumerate(episode_indices):
            traj_ends[idx] = i
        traj_values[list(traj_ends.values())] = 0  # last timestep has no value

        return traj_values / traj_values.sum()

    def calculate_normalization(self, batch_size=32):
        logging.info("Calculating image normalization (mean and std)...")

        channel_sum = 0
        channel_sum_squared = 0
        total_pixels = 0

        for start in range(0, self.num_timesteps, batch_size):
            stop = min(start + batch_size, self.num_timesteps)
            images = np.array([img.resize((64, 64)) for img in self.dataset["image_paths"][start:stop]])

            channel_sum += images.sum(axis=(0, 1, 2))
            channel_sum_squared += (images**2).sum(axis=(0, 1, 2))
            total_pixels += np.prod(images.shape[:3])

        mean = channel_sum / total_pixels
        std = np.sqrt((channel_sum_squared / total_pixels - mean) ** 2)

        return mean, std

    def sample_trajectory(self, idx, single_goal=False):
        start = self.traj_begins[idx]
        stop = start + self.traj_lens[idx]

        images = np.array([img.resize((64, 64)) for img in self.dataset["image_paths"][start:stop]])

        actions = self.dataset["actions"][start:stop]

        if self.normalize_inputs:
            images = (images - self.mean) / self.std

        # Permute images to channels-first (T, C, H, W)
        images = np.array(images).transpose(0, 3, 1, 2)

        data = dict(
            state=images,
            action=actions,
        )

        if single_goal:
            if "goal_plan" in self.goal_types:
                tokens = self.tokenizer(self.embeddings["plan"][self.plan_indices[stop - 1].item()])
                data.update(
                    {
                        "input_ids": np.repeat(tokens["input_ids"][None, :], len(images), axis=0),
                        "attention_mask": np.repeat(tokens["attention_mask"][None, :], len(images), axis=0),
                    }
                )

            if "goal_plan_embedding" in self.goal_types:
                data["goal_plan_embedding"] = np.repeat(
                    np.array(self.embeddings["embedding"][self.plan_indices[stop - 1].item()])[None, :],
                    len(images),
                    axis=0,
                )

            if "goal_future_state" in self.goal_types:
                data["goal_future_state"] = np.repeat(images[-1][None, :], len(images), axis=0)

        else:
            if "goal_plan" in self.goal_types:
                input_ids = []
                attention_mask = []
                for i in range(start, stop):
                    tokens = self.tokenizer(self.embeddings["plan"][self.plan_indices[i].item()])
                    input_ids.append(tokens["input_ids"])
                    attention_mask.append(tokens["attention_mask"])
                data["input_ids"] = np.array(input_ids)
                data["attention_mask"] = np.array(attention_mask)

            if "goal_plan_embedding" in self.goal_types:
                data["goal_plan_embedding"] = np.array(self.embeddings["embedding"][self.plan_indices[start:stop]])

            if "goal_future_state" in self.goal_types:
                data["goal_future_state"] = images

        return dict_to_torch_float(data)

    def sample_batch(self, batch_size=256):
        idxs = np.random.choice(self.num_timesteps, size=batch_size, p=self.traj_probs)

        batch = self.dataset[idxs]
        batch = Dataset.from_dict(batch).cast_column("image_paths", Image())

        images = np.array([img.resize((64, 64)) for img in batch["image_paths"]])
        actions = batch["actions"]
        rewards = batch["rewards"]

        if self.normalize_inputs:
            images = (images - self.mean) / self.std

        # Permute images to channels-first (C, H, W)
        images = images.transpose(0, 3, 1, 2)

        discounts = self.gamma ** np.arange(self.horizon)
        discounts = np.tile(discounts, (batch_size, 1))

        to_go = self.traj_lens[batch["episode_index"]] + self.traj_begins[batch["episode_index"]] - idxs
        mask = np.arange(self.horizon) < np.array(to_go)[:, None]

        discounts_masked = discounts * mask
        probs = discounts_masked / discounts_masked.sum(axis=1, keepdims=True)

        goal_idxs = np.array([np.random.choice(self.horizon, p=probs[i]) for i in range(batch_size)])

        goal_timesteps = self.plan_indices[idxs + goal_idxs]
        goal_embeddings = np.array(self.embeddings["embedding"][goal_timesteps])

        batch = dict(
            state=images,
            action=actions,
            reward=rewards,
            goal_plan_embedding=goal_embeddings,
        )
        return dict_to_torch_float(batch)

    def create_weighted_sampler(self):
        samples_weight = torch.from_numpy(self.traj_probs)
        weighted_sampler = torch.utils.data.WeightedRandomSampler(samples_weight.float(), len(samples_weight))

        return weighted_sampler

    def __len__(self):
        return self.num_timesteps

    def sample_future_index(self, idx):
        """
        Sample a future index based on the trajectory probabilities.
        """
        row = self.dataset[idx]

        discounts = self.gamma ** np.arange(self.horizon)
        to_go = self.traj_lens[row["episode_index"]] + self.traj_begins[row["episode_index"]] - idx
        mask = np.arange(self.horizon) < np.array(to_go)

        discounts_masked = discounts * mask
        probs = discounts_masked / discounts_masked.sum()
        goal_idx = np.random.choice(self.horizon, p=probs)

        return goal_idx

    def __getitem__(self, idx):
        row = self.dataset[idx]

        image = row["image_paths"].resize((64, 64))
        action = row["actions"]
        reward = row["rewards"]

        if self.normalize_inputs:
            image = (image - self.mean) / self.std

        image = np.array(image).transpose(2, 0, 1)

        data = dict(
            state=image,
            action=action,
            reward=reward,
        )

        if "goal_plan" in self.goal_types:
            tokens = self.tokenizer(self.embeddings["plan"][self.plan_indices[idx].item()])
            data.update(tokens)

        if "goal_plan_embedding" in self.goal_types:
            goal_plan_embedding = np.array(self.embeddings["embedding"][self.plan_indices[idx].item()])
            data["goal_plan_embedding"] = goal_plan_embedding

        if "goal_future_state" in self.goal_types:
            goal_idx = self.sample_future_index(idx)
            future_row = self.dataset[idx + goal_idx]

            future_image = future_row["image_paths"].resize((64, 64))

            if self.normalize_inputs:
                future_image = (future_image - self.mean) / self.std

            future_image = np.array(future_image).transpose(2, 0, 1)

            data["goal_future_state"] = future_image

        return dict_to_torch_float(data)

    def _render_state(self, state_entry):
        """
        Render a saved game-state entry to an HxW x 3 numpy image.
        state_entry may be a path (str/Path) to a joblib file containing an Env instance,
        or an Env-like object with a .render(size) method.
        """
        # lazy import joblib only when needed
        if isinstance(state_entry, (str, Path)):
            import joblib

            state = joblib.load(str(state_entry))
        elif isinstance(state_entry, (bytes, bytearray)):
            import tempfile

            import joblib

            with tempfile.NamedTemporaryFile(suffix=".joblib", delete=True) as tmp:
                tmp.write(state_entry)
                tmp.flush()
                state = joblib.load(tmp.name)
        else:
            state = state_entry

        if crafter_vid is not None and hasattr(crafter_vid, "render_state"):
            img = crafter_vid.render_state(state, size=self.render_size)
        elif hasattr(state, "render"):
            img = state.render(self.render_size)
        else:
            raise RuntimeError(
                "Cannot render game state: no renderer found on state object and crafter vid_exporter missing"
            )

        return np.array(img)

    def sample_chunks(self, batch_size: int = 16, chunk_len: int = 8):
        """
        Sample a batch of chunks. Each chunk is a sequence of (state, action) pairs of length chunk_len.
        Returns torch tensors on CPU:
            states: torch.float32 tensor shape (B, L, C, H, W)
            actions: tensor shape (B, L, action_shape)
            mask: bool tensor shape (B, L) where True indicates valid timestep
            goal_embeddings: torch.float32 tensor shape (B, L, embedding_dim)
        """
        # for latency debugging uncomment here
        # logging.info(
        #     f"sample_chunks called pid={os.getpid()} batch_size={batch_size} chunk_len={chunk_len} use_game_states={self.use_game_states}"
        # )
        # t_start = time.time()
        render_time = 0.0

        idxs_np = np.random.choice(self.num_timesteps, size=batch_size, p=self.traj_probs)
        idxs = [int(x) for x in idxs_np]  # conversion to Python int

        # detect whether dataset contains saved game-state objects and which column holds them
        state_col = None
        if self.use_game_states:
            preferred = [
                "state",
                "env_state",
                "env",
                "saved_state",
                "game_state",
                "states",
                "state_dumps",
            ]
            col_names = set()
            if hasattr(self.dataset, "column_names"):
                col_names = set(self.dataset.column_names)
            elif hasattr(self.dataset, "features"):
                col_names = set(self.dataset.features.keys())
            state_col = next((c for c in preferred if c in col_names), None)
            if state_col is None:
                raise RuntimeError("use_game_states=True but no suitable state column found in dataset")

        # prepare containers
        sample0 = self.dataset[0]

        # get a sample image to infer shapes, render if using game states
        if self.use_game_states:
            img0 = self._render_state(self.dataset[state_col][0])
        else:
            img0 = (
                np.array(sample0.get("image_paths").resize((self.render_size[0], self.render_size[1])))
                if "image_paths" in sample0
                else np.array(sample0["image"].resize((self.render_size[0], self.render_size[1])))
            )

        H, W, C = img0.shape

        # infer action shape
        act0 = np.array(sample0.get("actions"))
        if act0.ndim == 0:
            action_shape = ()
        else:
            action_shape = act0.shape

        embedding_dim = np.array(self.embeddings["embedding"][0]).shape[0]

        states = np.zeros((batch_size, chunk_len, C, H, W), dtype=np.float32)
        actions = np.zeros((batch_size, chunk_len) + action_shape, dtype=act0.dtype)
        mask = np.zeros((batch_size, chunk_len), dtype=bool)
        goal_embeddings = np.zeros((batch_size, chunk_len, embedding_dim), dtype=np.float32)

        for bi, start_idx in enumerate(idxs):
            # determine episode and limits
            ep_idx = self.dataset["episode_index"][start_idx]
            traj_start = int(self.traj_begins[ep_idx])
            traj_len = int(self.traj_lens[ep_idx])
            traj_stop = traj_start + traj_len

            for t in range(chunk_len):
                idx = start_idx + t
                if idx < traj_stop:
                    if self.use_game_states:
                        # dataset contains saved Env objects
                        state_entry = self.dataset[state_col][idx]
                        t0 = time.time()
                        img_arr = self._render_state(state_entry)
                        render_time += time.time() - t0
                    else:
                        # since images stored as PIL-like objects in image_paths
                        img = self.dataset["image_paths"][idx]
                        # we may want to store PIL.Image or numpy array
                        try:
                            t0 = time.time()
                            img_arr = np.array(img.resize((self.render_size[0], self.render_size[1])))
                            render_time += time.time() - t0
                        except Exception:
                            img_arr = np.array(img)
                            if img_arr.shape[0] != H or img_arr.shape[1] != W:
                                # try resizing as PIL if possible
                                try:
                                    from PIL import Image as PILImage

                                    pil = PILImage.fromarray(img_arr)
                                    pil = pil.resize((self.render_size[0], self.render_size[1]))
                                    img_arr = np.array(pil)
                                    render_time += time.time() - t0
                                except Exception:
                                    logging.error(
                                        f"Failed to resize image at index {idx} with shape {img_arr.shape}. "
                                        "Ensure images are in the expected format."
                                    )

                    if self.normalize_inputs and self.mean is not None and self.std is not None:
                        img_arr = (img_arr - self.mean) / self.std

                    # to channels-first
                    states[bi, t] = img_arr.transpose(2, 0, 1).astype(np.float32)

                    actions[bi, t] = np.array(self.dataset["actions"][idx])

                    # plan embedding for this timestep
                    plan_idx = int(self.plan_indices[idx])
                    goal_embeddings[bi, t] = np.array(self.embeddings["embedding"][plan_idx])

                    mask[bi, t] = True
                else:
                    # leave zero padding
                    mask[bi, t] = False

        # convert to torch tensors on CPU
        states_t = torch.from_numpy(states)
        actions_t = torch.from_numpy(actions)
        mask_t = torch.from_numpy(mask)
        goal_embeddings_t = torch.from_numpy(goal_embeddings)

        # total_time = time.time() - t_start

        # for latency debugging uncomment here
        # logging.info(
        #     f"sample_chunks pid={os.getpid()} total_time={total_time:.3f}s render_time={render_time:.3f}s"
        # )

        return states_t, actions_t, mask_t, goal_embeddings_t

    def get_action_shape(self):
        sample0 = self.dataset[0]
        act0 = np.array(sample0["actions"])
        if act0.ndim == 0:
            return ()
        return act0.shape
