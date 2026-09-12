"""Official tokenizer templates own wrappers and reasoning directives."""
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from reasoning_effort.validators import ThinkingFormatValidator, count_tokens


@dataclass(frozen=True)
class ModelProfile:
    family: str
    effort_map: dict[str, str]
    thinking_field: str


QWEN38 = ModelProfile("qwen3_8", {"low": "low", "medium": "medium", "high": "xhigh"}, "reasoning_content")
LLMJP4 = ModelProfile("llm_jp_4", {"low": "low", "medium": "medium", "high": "high"}, "thinking")


class BaseReasoningEffortFormatter:
    profile: ModelProfile

    def __init__(self, tokenizer: Any, tokenizer_name: str, revision: str = "main"):
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self.revision = revision

    def format(self, canonical: dict) -> dict:
        errors = ThinkingFormatValidator().validate(canonical["thinking"])
        if errors:
            raise ValueError(",".join(errors))
        record = deepcopy(canonical)
        effort = self.profile.effort_map[canonical["canonical_reasoning_effort"]]
        messages = [
            {"role": "user", "content": canonical["question"]},
            {"role": "assistant", "content": canonical["answer"],
             self.profile.thinking_field: canonical["thinking"]},
        ]
        kwargs = self.template_kwargs(effort)
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, **kwargs,
        )
        self.validate_rendered(rendered, canonical, effort)
        record.update(
            qa_id=f"{canonical['canonical_record_id']}:{self.profile.family}",
            reasoning_effort=effort,
            thinking_tokens=count_tokens(self.tokenizer, canonical["thinking"]),
            target_model_family=self.profile.family,
            target_tokenizer=self.tokenizer_name,
            target_tokenizer_revision=self.revision,
            messages=messages,
            chat_template_kwargs=kwargs,
            text=rendered,
        )
        return record

    def template_kwargs(self, effort: str) -> dict:
        return {"reasoning_effort": effort}

    def validate_rendered(self, text: str, canonical: dict, effort: str) -> None:
        raise NotImplementedError


class Qwen38ReasoningEffortFormatter(BaseReasoningEffortFormatter):
    profile = QWEN38

    def template_kwargs(self, effort: str) -> dict:
        return {"reasoning_effort": effort, "enable_thinking": True, "preserve_thinking": True}

    def validate_rendered(self, text: str, canonical: dict, effort: str) -> None:
        expected = f"<think>\n{canonical['thinking']}\n</think>\n\n{canonical['answer'].strip()}<|im_end|>"
        if expected not in text:
            raise ValueError("qwen_chat_template_thinking_or_answer_mismatch")
        if effort != "medium" and f"Reasoning effort is set to {effort}." not in text:
            raise ValueError("qwen_reasoning_condition_missing")
        if effort == "medium" and "Reasoning effort is set to medium." in text:
            raise ValueError("qwen_nonofficial_medium_directive")


class LLMJP4ReasoningEffortFormatter(BaseReasoningEffortFormatter):
    profile = LLMJP4

    def format(self, canonical: dict) -> dict:
        record = super().format(canonical)
        # Dataset schema and Transformers' input schema are different upstream.
        # Keep the lossless adapter input alongside the native channel messages.
        record["template_messages"] = record["messages"]
        record["messages"] = [
            {"role": "system", "name": None, "content": [{
                "type": "system_content",
                "model_identity": "You are LLM-jp-4, a large language model trained by LLM-jp.",
                "reasoning_effort": record["reasoning_effort"].capitalize(),
                "conversation_start_date": "2026-09-11", "knowledge_cutoff": "2025-12",
                "channel_config": {"valid_channels": ["analysis", "commentary", "final"],
                                   "channel_required": True},
            }]},
            {"role": "user", "name": None,
             "content": [{"type": "text", "text": canonical["question"]}]},
            {"role": "assistant", "name": None, "channel": "analysis",
             "content": [{"type": "text", "text": canonical["thinking"]}]},
            {"role": "assistant", "name": None, "channel": "final",
             "content": [{"type": "text", "text": canonical["answer"]}]},
        ]
        return record

    def template_kwargs(self, effort: str) -> dict:
        # Fixed metadata avoids changing the serialized training text on resume.
        return {"reasoning_effort": effort, "conversation_start_date": "2026-09-11"}

    def validate_rendered(self, text: str, canonical: dict, effort: str) -> None:
        analysis = f"<|start|>assistant<|channel|>analysis<|message|>{canonical['thinking']}<|end|>"
        final = f"<|start|>assistant<|channel|>final<|message|>{canonical['answer']}<|return|>"
        if analysis + final not in text or f"Reasoning: {effort}\n" not in text:
            raise ValueError("llmjp_harmony_validation_failed")
        if hasattr(self.tokenizer, "parse_harmony_message"):
            parsed = self.tokenizer.parse_harmony_message(
                self.tokenizer.encode(text, add_special_tokens=False))
            assistant = [m for m in parsed if m.role is not None
                         and self.tokenizer.decode(m.role.token_ids) == "assistant"]
            if len(assistant) != 2:
                raise ValueError("llmjp_harmony_assistant_count_mismatch")
            for message, channel, content in zip(
                assistant, ("analysis", "final"), (canonical["thinking"], canonical["answer"])
            ):
                if (message.channel is None or message.content is None
                        or self.tokenizer.decode(message.channel.token_ids) != channel
                        or self.tokenizer.decode(message.content.token_ids) != content):
                    raise ValueError("llmjp_harmony_token_roundtrip_mismatch")


FORMATTERS = {"qwen3_8": Qwen38ReasoningEffortFormatter, "llm_jp_4": LLMJP4ReasoningEffortFormatter}
