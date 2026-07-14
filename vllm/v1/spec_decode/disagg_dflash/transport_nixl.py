# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reserved NIXL transport stub for Disagg-DFlash (not implemented)."""

from __future__ import annotations

from vllm.v1.spec_decode.disagg_dflash.transport import DisaggDFlashClientTransport


class NixlClientTransport(DisaggDFlashClientTransport):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "disagg_dflash_transport='nixl' is not implemented yet"
        )

    def handshake(self) -> None:
        raise NotImplementedError

    def speculate(self, request):
        raise NotImplementedError

    def free(self, request) -> None:
        raise NotImplementedError

    def ping(self) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
