# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Thin wrapper: 추론 구현은 ossp_router.gbm_router로 이전했습니다."""

from __future__ import annotations

import sys

from ossp_router.gbm_router import *  # noqa: F401,F403
from ossp_router.gbm_router import main

if __name__ == "__main__":
    sys.exit(main())
