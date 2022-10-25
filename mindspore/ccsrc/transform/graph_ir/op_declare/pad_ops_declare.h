/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_PAD_OPS_DECLARE_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_PAD_OPS_DECLARE_H_

#include "utils/hash_map.h"
#include "transform/graph_ir/op_declare/op_declare_macro.h"
#include "ops/pad_ops.h"

namespace mindspore::transform {
DECLARE_OP_ADAPTER(PadD)
DECLARE_OP_USE_OUTPUT(PadD)

DECLARE_OP_ADAPTER(Pad)
DECLARE_OP_USE_OUTPUT(Pad)

DECLARE_OP_ADAPTER(BroadcastToD)
DECLARE_OP_USE_OUTPUT(BroadcastToD)

DECLARE_OP_ADAPTER(BroadcastTo)
DECLARE_OP_USE_OUTPUT(BroadcastTo)

DECLARE_OP_ADAPTER(Diag)
DECLARE_OP_USE_OUTPUT(Diag)

DECLARE_OP_ADAPTER(FillD)
DECLARE_OP_USE_OUTPUT(FillD)

DECLARE_OP_ADAPTER(Fill)
DECLARE_OP_USE_OUTPUT(Fill)

DECLARE_OP_ADAPTER(PadV3)
DECLARE_OP_USE_OUTPUT(PadV3)

DECLARE_OP_ADAPTER(PadV2)
DECLARE_OP_USE_OUTPUT(PadV2)
}  // namespace mindspore::transform
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_PAD_OPS_DECLARE_H_
