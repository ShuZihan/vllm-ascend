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

if "torch_npu._inductor" not in sys.modules:
    sys.modules["torch_npu._inductor"] = MagicMock()

from tests.ut.attention.utils import (  # noqa: E402
    create_c8_indexer_inputs,
    make_sfa_impl_stub,
)
from vllm_ascend.device.device_op import DeviceOperator  # noqa: E402
from vllm_ascend.utils import enable_custom_op  # noqa: E402

enable_custom_op()


def test_indexer_dcp_golden_smoke():
    """golden 链路 smoke：输入构造 → 内核调用 → 输出形状/类型。

    需要 NPU 硬件（C8 内核 npu_lightning_indexer_quant）。
    默认输入 B=8、index_n_heads=32、head_dim=128、seq_len=131072。
    """
    inputs = create_c8_indexer_inputs(seed=0)
    stub = make_sfa_impl_stub()

    out = DeviceOperator.indexer_select_post_process(
        stub,
        inputs.q_li,
        inputs.q_li_scale,
        inputs.q_li_shape_ori,
        inputs.weights,
        inputs.kv_cache,
        inputs.attn_metadata,
        inputs.actual_seq_lengths_query,
        inputs.actual_seq_lengths_key,
        enable_sparse_li_c8=True,
        use_torch_npu_lightning_indexer=True,
    )

    assert out.shape == (8, 2048)
    assert out.dtype == torch.int32
