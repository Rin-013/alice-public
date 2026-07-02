# Alice v4 "Living Mind" — Cognitive Architecture
#
# Mind (Ghost 1.5B on CPU) thinks continuously in background.
# Proposals buffer stages thoughts for Alice's context.
# Post-processor handles memory/avatar/IRIS after Alice responds.

from .mind import Mind
from .proposals_buffer import ProposalsBuffer
from .output_parser import parse_mind_output

__all__ = ['Mind', 'ProposalsBuffer', 'parse_mind_output']
