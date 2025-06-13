import re  # Parsing actions
import sys  # Flushing output

import gymnasium as gym


class EnvWrapper(gym.Wrapper):
    """
    A wrapper class for gym environments to standardize interactions across different environments.
    It provides additional functionalities, such as handling specific observation processing,
    managing action validity, retrieving instruction prompts, and tracking failed action candidates.
    """

    def __init__(self, env, env_name, task_name, args=None):
        super().__init__(env)
        self.env_name = env_name
        self.task_name = task_name
        self.failed_candidates = []
        self.args = args  # Store args for debug access

    @property
    def max_steps(self):
        return self.env.max_steps

    @property
    def actions(self):
        # This property should return the list of available actions
        return self.env.actions if hasattr(self.env, "actions") else list(range(len(self.env.action_space)))

    def get_text_action(self, action):
        return self.env.get_text_action(action)

    def get_instruction_prompt(self, instructions=None):
        if hasattr(self.env, "get_instruction_prompt"):
            return self.env.get_instruction_prompt(instructions)

        if self.env_name == "crafter":
            from feedback_bottleneck.envs.crafter.instruction_prompt import get_instruction_prompt

            return get_instruction_prompt(self.task_name)

        elif self.env_name == "nle":
            from feedback_bottleneck.envs.nle.instruction_prompt import get_instruction_prompt

            return get_instruction_prompt(self.task_name)

        elif self.env_name == "math":
            from feedback_bottleneck.envs.omni_math.instruction_prompt import get_instruction_prompt

            return get_instruction_prompt(self.task_name)
        elif self.env_name == "countdown":
            from feedback_bottleneck.envs.countdown.instruction_prompt import get_instruction_prompt

            return get_instruction_prompt(self.task_name)
        elif self.env_name == "bbeh":
            from feedback_bottleneck.envs.bbeh.instruction_prompt import get_instruction_prompt

            return get_instruction_prompt(self.task_name)
        elif self.env_name == "enterprise_ops":
            from feedback_bottleneck.envs.enterprise_ops.instruction_prompt import get_instruction_prompt
             
            return get_instruction_prompt(self.task_name)
        elif self.env_name == "sciknoweval":
            from feedback_bottleneck.envs.sciknoweval.instruction_prompt import get_instruction_prompt

            return get_instruction_prompt(self.task_name)
        elif self.env_name == "code":
            from feedback_bottleneck.envs.omni_code.instruction_prompt import get_instruction_prompt

            return get_instruction_prompt(self.task_name)
        else:
            raise ValueError(f"Unknown environment: {self.env_name}")

    def check_action_validity(self, candidate_action):
        language_action_space = self.env.get_wrapper_attr("language_action_space")

        # Exact match
        if candidate_action in language_action_space:
            return candidate_action, None

        # If exact match fails, try search
        valid_action = self._search_for_valid_action(candidate_action, language_action_space)

        if valid_action is not None:
            # Track successful corrections for statistics
            if candidate_action != valid_action:
                self._track_action_correction(candidate_action, valid_action)
            return valid_action, None
        else:
            # Fallback to default action and track the failure
            self.failed_candidates.append(candidate_action)
            self._track_action_failure(candidate_action)
            feedback = f"The action '{candidate_action}' was invalid and not recognized. A default action was performed instead. Please ensure you use commands from the provided Core Commands list."
            return self.env.default_action, feedback

    def _search_for_valid_action(self, candidate_action, language_action_space):
        """
        Search for a valid action that matches the candidate.
        Returns None if no good match is found.
        """
        if not candidate_action:
            return None

        valid_actions = language_action_space._values
        candidate_lower = candidate_action.lower().strip()

        # case-insensitive exact match
        for valid_action in valid_actions:
            if candidate_lower == valid_action.lower():
                return valid_action

        # Remove common LLM artifacts
        cleaned_candidate = self._clean_action_text(candidate_lower)

        for valid_action in valid_actions:
            if cleaned_candidate == valid_action.lower():
                return valid_action

        # If all words of the valid action appear in the raw text we got a match
        for valid_action in valid_actions:
            action_words = valid_action.lower().split()
            if len(action_words) > 1 and all(word in cleaned_candidate for word in action_words):
                return valid_action

        # Keyword-based matching for common patterns
        keyword_matches = self._match_by_keywords(cleaned_candidate, valid_actions)
        if keyword_matches:
            return keyword_matches

        return None

    def _clean_action_text(self, text):
        """Remove common LLM prefixes/suffixes."""
        # common prefixes
        prefixes = ["final", "action:", "next:", "i will", "i choose", "action", "assistant"]
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()

        # common suffixes
        suffixes = ["action", "now", "next", ".", "!"]
        for suffix in suffixes:
            if text.endswith(suffix):
                text = text[: -len(suffix)].strip()

        # Remove extra whitespace/punctuation
        text = re.sub(r"[^\w\s]", " ", text)
        text = " ".join(text.split())

        return text

    def _match_by_keywords(self, candidate, valid_actions):
        """Match candidate to valid actions using keyword analysis."""
        # Direction mappings
        direction_keywords = {
            "west": ["west", "left"],
            "east": ["east", "right"],
            "north": ["north", "up"],
            "south": ["south", "down"],
        }

        # Action mappings
        action_keywords = {
            "noop": ["noop", "nothing", "wait", "idle"],
            "do": ["do", "interact", "use", "action"],
            "sleep": ["sleep", "rest"],
            "place": ["place", "put", "set"],
            "make": ["make", "craft", "create"],
        }

        # Movement actions
        for direction, keywords in direction_keywords.items():
            if any(keyword in candidate for keyword in keywords):
                move_action = f"Move {direction.title()}"
                if move_action in valid_actions:
                    return move_action

        # Other action types
        for action_type, keywords in action_keywords.items():
            if any(keyword in candidate for keyword in keywords):
                # Find matching valid actions
                for valid_action in valid_actions:
                    if action_type.lower() in valid_action.lower():
                        return valid_action

        return None

    def _track_action_correction(self, original, corrected):
        """Track when an action was successfully corrected."""
        # Debug output if debug flag is enabled
        debug_enabled = self._get_debug_flag()
        if debug_enabled and original != corrected:
            print(f"[ACTION_DEBUG] EnvWrapper corrected: '{original}' -> '{corrected}'")
            sys.stdout.flush()

    def _track_action_failure(self, failed_action):
        """Track when an action completely failed to be recognized."""
        # Debug output if debug flag is enabled
        debug_enabled = self._get_debug_flag()
        if debug_enabled:
            print(f"[ACTION_DEBUG] EnvWrapper failed to recognize: '{failed_action}' (using default)")
            sys.stdout.flush()

    def _get_debug_flag(self):
        try:
            # Check args if available
            if self.args and hasattr(self.args, "debug_action_extraction"):
                return self.args.debug_action_extraction
        except:
            print(f"{sys._getframe().f_code.co_name} failed in {__file__}")
            return False

    def get_stats(self):
        base_stats = self.env.get_wrapper_attr("get_stats")()

        # Add action validation statistics
        base_stats.update(
            {
                "total_failed_candidates": len(self.failed_candidates),
                "recent_failed_examples": self.failed_candidates[-5:] if self.failed_candidates else [],
            }
        )

        return base_stats
