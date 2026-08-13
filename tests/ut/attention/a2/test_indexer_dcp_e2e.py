#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2024 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import sys
from unittest.mock import MagicMock

import torch
import torch_npu  # noqa: F401

if "torch_npu._inductor" not in sys.modules:
    sys.modules["torch_npu._inductor"] = MagicMock()

from tests.ut.attention.utils import create_indexer_inputs  # noqa: E402


def test_indexer_dcp_golden_smoke():
    """golden 链路 smoke：输入构造 → 非 C8 indexer 内核 → 输出形状/类型。

    需要 NPU 硬件。对齐生产实际路径（torch_npu.npu_lightning_indexer，
    非 C8 分支——rc1 上 enable_sparse_li_c8 因 layer_name bug 永不生效）。
    默认输入 B=8、index_n_heads=32、head_dim=128、seq_len=131072、
    dtype=bf16。
    """
    inputs = create_indexer_inputs(seed=0)

    indices, values = torch_npu.npu_lightning_indexer(
        query=inputs.q_li,
        key=inputs.indexer_k,
        weights=inputs.weights,
        actual_seq_lengths_query=inputs.actual_seq_lengths_query,
        actual_seq_lengths_key=inputs.actual_seq_lengths_key,
        block_table=inputs.block_table,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=2048,
        sparse_mode=3,
    )

    assert indices.shape == (8, 2048)
    assert indices.dtype == torch.int32
    assert values.shape == (8, 2048)
