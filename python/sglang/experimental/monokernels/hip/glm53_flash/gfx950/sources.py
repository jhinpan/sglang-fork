# SPDX-License-Identifier: Apache-2.0
"""The kernel sources in every build's cache key: an edit to them rebuilds."""

from sglang.experimental.monokernels.hip.common.plan import source_digest

SOURCES = source_digest("hip/common", "hip/glm53_flash/gfx950")
