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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NPU_LOSS_SCALE_OPS_DECLARE_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NPU_LOSS_SCALE_OPS_DECLARE_H_

#include "utils/hash_map.h"
#include "transform/graph_ir/op_declare/op_declare_macro.h"
#include "ops/npu_loss_scale_ops.h"

namespace mindspore::transform {
DECLARE_OP_ADAPTER(NPUGetFloatStatus)
DECLARE_OP_USE_OUTPUT(NPUGetFloatStatus)

DECLARE_OP_ADAPTER(NPUAllocFloatStatus)
DECLARE_OP_USE_OUTPUT(NPUAllocFloatStatus)

DECLARE_OP_ADAPTER(NPUClearFloatStatus)
DECLARE_OP_USE_OUTPUT(NPUClearFloatStatus)

DECLARE_OP_ADAPTER(NPUGetFloatStatusV2)
DECLARE_OP_USE_OUTPUT(NPUGetFloatStatusV2)

DECLARE_OP_ADAPTER(NPUClearFloatStatusV2)
DECLARE_OP_USE_OUTPUT(NPUClearFloatStatusV2)
}  // namespace mindspore::transform
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NPU_LOSS_SCALE_OPS_DECLARE_H_
