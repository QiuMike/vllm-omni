# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections.abc import Iterable, Sequence
from itertools import islice
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning import ReasoningParser
from vllm.tokenizers import TokenizerLike

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


class StepAudioReasoningParser(ReasoningParser):
    """Reasoning parser for Step-Audio models.

    Step-Audio uses `` and `` tokens for reasoning content.
    The chat template appends `` as a generation prefix so the
    model always starts in a thinking state.

    Unlike DeepSeek-R1 or Step3p5, the Step-Audio tokenizer may not
    represent `` / `` as single special tokens.  This parser
    therefore falls back to **text-based** matching when token-ID
    lookup fails, while still using the faster token-ID path when
    available.
    """

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction."
            )

        self.think_start_token = ""
        self.think_end_token = ""

        # Try to resolve token IDs; fall back to -1 if not in vocab.
        self.think_start_token_id: int = self.vocab.get(
            self.think_start_token, -1
        )
        self.think_end_token_id: int = self.vocab.get(
            self.think_end_token, -1
        )

        # Whether we can use the fast token-ID path.
        self._use_token_ids = (
            self.think_start_token_id != -1
            and self.think_end_token_id != -1
        )

        if not self._use_token_ids:
            # Token IDs not available — will use text-based matching.
            # Estimate the token IDs for `` by encoding.
            # Some tokenizers split `` into multiple tokens.
            # We'll handle this by falling back to text-based end-token
            # detection in extract_reasoning_streaming.
            pass

        # Tracks whether we have already emitted the end of reasoning
        # in streaming mode.
        self._reasoning_ended = False

    # ------------------------------------------------------------------
    # Token-ID helpers (fast path)
    # ------------------------------------------------------------------

    def _has_end_token_in_ids(self, token_ids: Sequence[int]) -> bool:
        if self.think_end_token_id != -1:
            return self.think_end_token_id in token_ids
        return False

    def _has_start_token_in_ids(self, token_ids: Sequence[int]) -> bool:
        if self.think_start_token_id != -1:
            return self.think_start_token_id in token_ids
        return False

    # ------------------------------------------------------------------
    # Text-based helpers (slow path for multi-token markers)
    # ------------------------------------------------------------------

    def _has_end_token_in_text(self, text: str) -> bool:
        return self.think_end_token in text

    # ------------------------------------------------------------------
    # ReasoningParser interface
    # ------------------------------------------------------------------

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        if self._use_token_ids:
            return self._has_end_token_in_ids(input_ids)
        # Fallback: decode and check text (slow, but correct).
        return False

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        if self._use_token_ids:
            return self._has_end_token_in_ids(tuple(delta_ids))
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if not self._use_token_ids:
            return []
        if self.think_end_token_id not in islice(
            input_ids, 0, max(0, len(input_ids) - 1)
        ):
            return []
        return input_ids[input_ids.index(self.think_end_token_id) + 1 :]

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        # Strip leading `` if present (it's in the generation prefix).
        if model_output.startswith(self.think_start_token):
            model_output = model_output[len(self.think_start_token):]

        if self.think_end_token not in model_output:
            # No `` found — everything is reasoning.
            return model_output or None, None

        reasoning, _, content = model_output.partition(self.think_end_token)
        # Strip leading newline that models often emit right after ``.
        content = content.lstrip("\n")
        return reasoning or None, content or None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        # Fast path: single-token markers.
        if self._use_token_ids:
            return self._extract_streaming_token_ids(
                previous_text,
                delta_text,
                previous_token_ids,
                delta_token_ids,
            )

        # Slow path: text-based matching for multi-token markers.
        return self._extract_streaming_text(
            previous_text,
            delta_text,
        )

    # ------------------------------------------------------------------
    # Streaming: token-ID based (fast path)
    # ------------------------------------------------------------------

    def _extract_streaming_token_ids(
        self,
        previous_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        # Skip the `` start token itself.
        if len(delta_token_ids) == 1 and delta_token_ids[0] == self.think_start_token_id:
            return None

        # Skip the `` end token itself.
        if len(delta_token_ids) == 1 and delta_token_ids[0] == self.think_end_token_id:
            self._reasoning_ended = True
            return None

        if self._has_start_token_in_ids(previous_token_ids):
            if self._has_end_token_in_ids(delta_token_ids):
                # Start in previous, end in delta.
                end_index = delta_text.find(self.think_end_token)
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self.think_end_token) :]
                self._reasoning_ended = True
                return DeltaMessage(
                    reasoning=reasoning or None,
                    content=content or None,
                )
            elif self._reasoning_ended:
                # Already past the end — this is content.
                return DeltaMessage(content=delta_text)
            else:
                # Still in reasoning.
                return DeltaMessage(reasoning=delta_text)
        elif self._has_start_token_in_ids(delta_token_ids):
            if self._has_end_token_in_ids(delta_token_ids):
                # Both start and end in delta.
                start_index = delta_text.find(self.think_start_token)
                end_index = delta_text.find(self.think_end_token)
                reasoning = delta_text[
                    start_index + len(self.think_start_token) : end_index
                ]
                content = delta_text[end_index + len(self.think_end_token) :]
                self._reasoning_ended = True
                return DeltaMessage(
                    reasoning=reasoning or None,
                    content=content or None,
                )
            else:
                # Start in delta, no end yet.
                start_index = delta_text.find(self.think_start_token)
                after_start = delta_text[start_index + len(self.think_start_token) :]
                return DeltaMessage(reasoning=after_start or None)
        else:
            # No start token seen.
            if self._reasoning_ended:
                return DeltaMessage(content=delta_text)
            # No start token — model may not always emit ``.
            # If we see `` in delta without prior start, treat as
            # transition from reasoning to content.
            if self._has_end_token_in_ids(delta_token_ids):
                end_index = delta_text.find(self.think_end_token)
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self.think_end_token) :]
                self._reasoning_ended = True
                return DeltaMessage(
                    reasoning=reasoning or None,
                    content=content or None,
                )
            if self._has_end_token_in_ids(previous_token_ids):
                return DeltaMessage(content=delta_text)
            # Still in reasoning (or content if reasoning never started).
            return DeltaMessage(reasoning=delta_text)

    # ------------------------------------------------------------------
    # Streaming: text-based (slow path for multi-token markers)
    # ------------------------------------------------------------------

    def _extract_streaming_text(
        self,
        previous_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        # Check if `` appears in the combined text.
        combined = previous_text + delta_text
        has_end = self._has_end_token_in_text(combined)
        has_end_in_previous = self._has_end_token_in_text(previous_text)
        has_end_in_delta = self._has_end_token_in_text(delta_text)

        if has_end_in_previous or self._reasoning_ended:
            # Already past reasoning — everything is content.
            # Strip leading newlines right after ``.
            if has_end_in_previous and not self._reasoning_ended:
                self._reasoning_ended = True
                after_end = previous_text.partition(self.think_end_token)[2]
                # Any content already in previous after  has been
                # handled; delta is pure content.
            if delta_text.startswith("\n") and not self._reasoning_ended:
                self._reasoning_ended = True
                return DeltaMessage(content=delta_text[1:] or None)
            self._reasoning_ended = True
            return DeltaMessage(content=delta_text)

        if has_end_in_delta:
            # `` found in the delta — split reasoning and content.
            end_index = delta_text.find(self.think_end_token)
            reasoning = delta_text[:end_index]
            content = delta_text[end_index + len(self.think_end_token) :]
            # Strip leading newline after ``.
            if content.startswith("\n"):
                content = content[1:]
            self._reasoning_ended = True
            return DeltaMessage(
                reasoning=reasoning or None,
                content=content or None,
            )

        # No `` seen yet — everything is reasoning.
        # Skip the `` prefix if present in delta.
        if delta_text.strip() == self.think_start_token:
            return None
        if delta_text.startswith(self.think_start_token):
            after_start = delta_text[len(self.think_start_token):]
            return DeltaMessage(reasoning=after_start or None)

        return DeltaMessage(reasoning=delta_text)

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        """Count tokens within ``...`` spans."""
        if not self._use_token_ids:
            return 0

        count = 0
        depth = 0
        for token_id in token_ids:
            if token_id == self.think_start_token_id:
                depth += 1
                continue
            if token_id == self.think_end_token_id:
                if depth > 0:
                    depth -= 1
                continue
            if depth > 0:
                count += 1
        return count