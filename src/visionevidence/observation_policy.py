"""Versioned, independent prompt ablations; no labels or retrieval results enter here."""

from dataclasses import asdict, dataclass
from pathlib import Path
import tomllib


FORMAT_INSTRUCTIONS = '''
Additional format contract:
Use at most 6 objects, 2 visible_text entries and 2 uncertainties. Empty arrays
are allowed; never fill a quota. Each string must be at most 100 characters.
Use at most 60 whitespace-separated words across all strings combined.
Do not repeat an entry within a field (including case or spacing variants).
Do not list parts of parts, alternative names, synonyms or generic vocabulary.
Represent repeated instances together, using a count only when clearly visible.
Select the most salient items before writing. Finish all three arrays and the
JSON object, then stop. Never continue an inventory or repeat to fill space.'''

DETAIL_INSTRUCTIONS = '''
Additional information contract:
Use objects entries as concise visual phrases, not just category names.
For a salient subject, keep its clearly visible action or pose and distinctive
attributes together in the same phrase. Retain the overall visible scene or a
spatial relationship when it distinguishes the image. Preserve a clearly visible
count or legible number exactly; omit counts that cannot be determined.
Prioritize these grounded details over incidental background items. Do not add
synonyms, keyword expansions, inferred intent, location names or reference captions.
Include only confidently legible text in visible_text; put uncertain readings
only in uncertainties, never also in objects or visible_text. Do not write
absence-of-text boilerplate. If a detail is not visible, omit it rather than guess.'''


@dataclass(frozen=True)
class ObservationPolicy:
    format_constraints: bool = False
    retain_visual_details: bool = False

    def __post_init__(self) -> None:
        if type(self.format_constraints) is not bool or type(self.retain_visual_details) is not bool:
            raise ValueError("observation ablation switches must be booleans")

    @property
    def prompt_version(self) -> str:
        if not (self.format_constraints or self.retain_visual_details):
            return "structured_visual_v1"
        suffix = ("format_details" if self.format_constraints and self.retain_visual_details else
                  "format" if self.format_constraints else "details")
        return "structured_visual_v11a_" + suffix

    def prompt(self, legacy_prompt: str) -> str:
        return (legacy_prompt + (FORMAT_INSTRUCTIONS if self.format_constraints else "")
                + (DETAIL_INSTRUCTIONS if self.retain_visual_details else ""))

    def identity(self) -> dict:
        return {"config_version": "observation_policy_v1", **asdict(self),
                "prompt_version": self.prompt_version,
                "schema_version": "visual_observation_v1",
                "query_builder_version": "objects_visible_text_ascii_v1"}

    def validate(self, observation) -> None:
        """Reject violations after the original strict parser; never edit content."""
        if not self.format_constraints:
            return
        words = 0
        for field, maximum in (("objects", 6), ("visible_text", 2), ("uncertainties", 2)):
            values = getattr(observation, field)
            if len(values) > maximum:
                raise ValueError(f"v11 format: {field} must contain at most {maximum} items")
            normalized = [" ".join(value.casefold().split()) for value in values]
            if len(set(normalized)) != len(normalized):
                raise ValueError(f"v11 format: duplicate {field} entry")
            if any(len(value) > 100 for value in values):
                raise ValueError(f"v11 format: {field} strings must be at most 100 characters")
            words += sum(len(value.split()) for value in values)
        if words > 60:
            raise ValueError("v11 format: all strings must contain at most 60 words combined")


V10_POLICY = ObservationPolicy()


def load_observation_policy(path: str | Path | None = None) -> ObservationPolicy:
    if path is None:
        return V10_POLICY
    try:
        with Path(path).open("rb") as stream:
            config = tomllib.load(stream)
        if config.keys() != {"observation"}:
            raise ValueError("expected only an [observation] table")
        values = config["observation"]
        if not isinstance(values, dict) or values.keys() != {
            "config_version", "format_constraints", "retain_visual_details"
        }:
            raise ValueError("expected config_version and both observation ablation switches")
        if values["config_version"] != "observation_policy_v1":
            raise ValueError("unsupported observation config_version")
        return ObservationPolicy(values["format_constraints"], values["retain_visual_details"])
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"invalid observation policy {path}: {exc}") from exc
