# SPDX-License-Identifier: Apache-2.0
"""Unit tests for multi-modal streaming finish_reason behavior.

Verifies that in the /v1/chat/completions streaming endpoint, only the last
modality chunk carries finish_reason="stop", complying with the OpenAI API
spec. Earlier modalities that finish must emit finish_reason=null.
"""

import enum
import json
from unittest.mock import MagicMock

import pytest

# Python 3.10 compat: StrEnum was added in 3.11
if not hasattr(enum, "StrEnum"):

    class _StrEnum(str, enum.Enum):
        """Minimal StrEnum backport for Python 3.10."""

    enum.StrEnum = _StrEnum  # type: ignore[attr-defined]

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.outputs import CompletionOutput, RequestOutput
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_text_omni_output(
    request_id: str = "test-req",
    text: str = "hello",
    token_ids: list[int] | None = None,
    finish_reason: str | None = None,
    index: int = 0,
    num_prompt_tokens: int = 3,
) -> OmniRequestOutput:
    """Build an OmniRequestOutput wrapping a text RequestOutput."""
    if token_ids is None:
        token_ids = [10, 11, 12]
    res = RequestOutput(
        request_id=request_id,
        prompt="test",
        prompt_token_ids=list(range(num_prompt_tokens)),
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=index,
                text=text,
                token_ids=token_ids,
                cumulative_logprob=0.0,
                logprobs=None,
                finish_reason=finish_reason,
                stop_reason=None,
            )
        ],
        finished=finish_reason is not None,
    )
    return OmniRequestOutput(
        request_id=request_id,
        final_output_type="text",
        request_output=res,
        finished=finish_reason is not None,
    )


def _make_audio_omni_output(
    request_id: str = "test-req",
    index: int = 0,
    num_prompt_tokens: int = 3,
) -> OmniRequestOutput:
    """Build an OmniRequestOutput for audio without depending on torch."""
    res = RequestOutput(
        request_id=request_id,
        prompt="test",
        prompt_token_ids=list(range(num_prompt_tokens)),
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=index,
                text="",
                token_ids=[],
                cumulative_logprob=0.0,
                logprobs=None,
                finish_reason="stop",
                stop_reason=None,
            )
        ],
        finished=True,
    )
    return OmniRequestOutput(
        request_id=request_id,
        final_output_type="audio",
        request_output=res,
        finished=True,
    )


def _mock_audio_choices(index: int = 0, role: str = "assistant"):
    """Return a list with one ChatCompletionResponseStreamChoice for audio."""
    from vllm.entrypoints.openai.engine.protocol import DeltaMessage
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionResponseStreamChoice,
    )

    return [
        ChatCompletionResponseStreamChoice(
            index=index,
            delta=DeltaMessage(role=role, content="dGVzdA=="),
            logprobs=None,
            finish_reason="stop",
        )
    ]


def _build_serving_chat():
    """Create a minimal OmniOpenAIServingChat for testing."""
    from vllm.entrypoints.openai.models.serving import OpenAIServingModels
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    mock_engine = MagicMock()
    mock_engine.errored = False

    models = OpenAIServingModels(
        engine_client=mock_engine,
        base_model_paths=[],
    )

    mock_render = MagicMock()

    instance = OmniOpenAIServingChat(
        engine_client=mock_engine,
        models=models,
        response_role="assistant",
        openai_serving_render=mock_render,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )

    # Mock _create_audio_choice to avoid torch/soundfile dependency.
    instance._create_audio_choice = MagicMock(  # type: ignore[attr-defined]
        side_effect=lambda omni_res, role, request, stream=False: _mock_audio_choices(
            index=omni_res.request_output.outputs[0].index,
            role=role,
        )
    )

    return instance


def _make_request(modalities: list[str], n: int = 1) -> ChatCompletionRequest:
    """Create a ChatCompletionRequest with the given modalities."""
    req = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        n=n,
        stream=True,
    )
    req.modalities = modalities  # type: ignore[attr-defined]
    return req


def _parse_sse_chunks(lines: list[str]) -> list[dict]:
    """Parse SSE ' ...' lines into JSON dicts."""
    prefix = "data: "
    chunks = []
    for line in lines:
        line = line.strip()
        if not line.startswith(prefix):
            continue
        payload = line[len(prefix):].strip()
        if payload == "[DONE]":
            continue
        try:
            chunks.append(json.loads(payload))
        except json.JSONDecodeError:
            pass
    return chunks


async def _collect_stream(gen):
    """Collect all SSE strings from an async generator."""
    result = []
    async for item in gen:
        result.append(item)
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_modality_text_finish_reason():
    """Single-modality (text only) streaming: last chunk has finish_reason='stop'."""
    serving_chat = _build_serving_chat()
    request = _make_request(modalities=["text"])

    async def result_generator():
        yield _make_text_omni_output(text="hel", token_ids=[10, 11], finish_reason=None)
        yield _make_text_omni_output(text="lo", token_ids=[12], finish_reason="stop")

    raw_lines = await _collect_stream(
        serving_chat.chat_completion_stream_generator(
            request=request,
            result_generator=result_generator(),
            request_id="test-req",
            model_name="test-model",
            conversation=[],
            tokenizer=MagicMock(),
            request_metadata=MagicMock(),
        )
    )

    chunks = _parse_sse_chunks(raw_lines)

    finish_reasons = [
        c["choices"][0]["finish_reason"]
        for c in chunks
        if c.get("choices")
    ]

    assert finish_reasons[-1] == "stop"
    assert finish_reasons.count("stop") == 1
    for fr in finish_reasons[:-1]:
        assert fr is None


@pytest.mark.asyncio
async def test_multi_modal_text_audio_only_last_finish_reason():
    """Multi-modal (text + audio): only the audio chunk has finish_reason='stop'.

    Text finishes first but must NOT emit finish_reason='stop' because audio
    has not yet completed.
    """
    serving_chat = _build_serving_chat()
    request = _make_request(modalities=["text", "audio"])

    async def result_generator():
        yield _make_text_omni_output(text="hel", token_ids=[10, 11], finish_reason=None)
        yield _make_text_omni_output(text="lo", token_ids=[12], finish_reason="stop")
        yield _make_audio_omni_output()

    raw_lines = await _collect_stream(
        serving_chat.chat_completion_stream_generator(
            request=request,
            result_generator=result_generator(),
            request_id="test-req",
            model_name="test-model",
            conversation=[],
            tokenizer=MagicMock(),
            request_metadata=MagicMock(),
        )
    )

    chunks = _parse_sse_chunks(raw_lines)

    finish_reasons = []
    for c in chunks:
        for choice in c.get("choices", []):
            finish_reasons.append(choice["finish_reason"])

    # Exactly one chunk should have finish_reason="stop"
    assert finish_reasons.count("stop") == 1
    # The "stop" must be on the last chunk with choices (the audio chunk)
    assert finish_reasons[-1] == "stop"
    # The text finish chunk must have finish_reason=None
    text_finish_idx = None
    for idx, c in enumerate(chunks):
        for choice in c.get("choices", []):
            if c.get("modality") == "text" and choice.get("delta", {}).get("content") == "lo":
                text_finish_idx = idx
    if text_finish_idx is not None:
        assert chunks[text_finish_idx]["choices"][0]["finish_reason"] is None


@pytest.mark.asyncio
async def test_multi_modal_n2_per_choice_finish_reason():
    """n=2 with text+audio: each choice independently tracks modality_finished.

    Both choices' text finishes before audio. Each choice should have
    finish_reason=None on text finish and finish_reason="stop" on audio finish.
    """
    serving_chat = _build_serving_chat()
    request = _make_request(modalities=["text", "audio"], n=2)

    async def result_generator():
        yield _make_text_omni_output(text="A", token_ids=[10], finish_reason=None, index=0)
        yield _make_text_omni_output(text="B", token_ids=[20], finish_reason=None, index=1)
        yield _make_text_omni_output(text="", token_ids=[11], finish_reason="stop", index=0)
        yield _make_text_omni_output(text="", token_ids=[21], finish_reason="stop", index=1)
        yield _make_audio_omni_output(index=0)
        yield _make_audio_omni_output(index=1)

    raw_lines = await _collect_stream(
        serving_chat.chat_completion_stream_generator(
            request=request,
            result_generator=result_generator(),
            request_id="test-req",
            model_name="test-model",
            conversation=[],
            tokenizer=MagicMock(),
            request_metadata=MagicMock(),
        )
    )

    chunks = _parse_sse_chunks(raw_lines)

    per_choice_finish_reasons: dict[int, list] = {}
    for c in chunks:
        for choice in c.get("choices", []):
            idx = choice["index"]
            per_choice_finish_reasons.setdefault(idx, []).append(choice["finish_reason"])

    for idx, reasons in per_choice_finish_reasons.items():
        assert reasons.count("stop") == 1, (
            f"Choice {idx} has {reasons.count('stop')} 'stop' finish_reasons, expected 1"
        )
        assert reasons[-1] == "stop", (
            f"Choice {idx} last finish_reason is {reasons[-1]}, expected 'stop'"
        )


@pytest.mark.asyncio
async def test_single_modality_audio_finish_reason():
    """Audio-only streaming: the audio chunk carries finish_reason='stop'."""
    serving_chat = _build_serving_chat()
    request = _make_request(modalities=["audio"])

    async def result_generator():
        yield _make_audio_omni_output()

    raw_lines = await _collect_stream(
        serving_chat.chat_completion_stream_generator(
            request=request,
            result_generator=result_generator(),
            request_id="test-req",
            model_name="test-model",
            conversation=[],
            tokenizer=MagicMock(),
            request_metadata=MagicMock(),
        )
    )

    chunks = _parse_sse_chunks(raw_lines)

    finish_reasons = [
        choice["finish_reason"]
        for c in chunks
        for choice in c.get("choices", [])
    ]

    assert finish_reasons.count("stop") == 1
    assert finish_reasons[-1] == "stop"