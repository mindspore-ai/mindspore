/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NN_OTHER_OPS_DECLARE_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NN_OTHER_OPS_DECLARE_H_

#include "op_proto/inc/nn_other.h"
#include "plugin/ascend/res_manager/op_adapter/op_declare/op_declare_macro.h"

// ApplyRotaryPosEmb
DECLARE_OP_ADAPTER(ApplyRotaryPosEmb)
DECLARE_OP_USE_OUTPUT(ApplyRotaryPosEmb)

// RotaryPositionEmbedding
DECLARE_OP_ADAPTER(RotaryPositionEmbedding)
DECLARE_OP_USE_OUTPUT(RotaryPositionEmbedding)

// RotaryPositionEmbeddingGrad
DECLARE_OP_ADAPTER(RotaryPositionEmbeddingGrad)
DECLARE_OP_USE_OUTPUT(RotaryPositionEmbeddingGrad)
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NN_OTHER_OPS_DECLARE_H_
