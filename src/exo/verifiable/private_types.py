"""Plaintext types that exist only on the requester and authorized ingress rank."""

from exo.shared.types.text_generation import InputMessage, InputMessageContent
from exo.utils.pydantic_ext import FrozenModel


class VerifiablePrivateTaskPayload(FrozenModel):
    input: list[InputMessage]
    instructions: InputMessageContent | None = None
