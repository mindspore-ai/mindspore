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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NONLINEAR_FUC_OPS_DECLARE_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NONLINEAR_FUC_OPS_DECLARE_H_

#include "utils/hash_map.h"
#include "ops/nonlinear_fuc_ops.h"
#include "transform/graph_ir/op_declare/op_declare_macro.h"

namespace mindspore::transform {
DECLARE_OP_ADAPTER(ReluGrad)
DECLARE_OP_USE_OUTPUT(ReluGrad)

DECLARE_OP_ADAPTER(ReluGradV2)
DECLARE_OP_USE_OUTPUT(ReluGradV2)

DECLARE_OP_ADAPTER(Relu6)
DECLARE_OP_USE_OUTPUT(Relu6)

DECLARE_OP_ADAPTER(Relu6Grad)
DECLARE_OP_USE_OUTPUT(Relu6Grad)

DECLARE_OP_ADAPTER(Softsign)
DECLARE_OP_USE_OUTPUT(Softsign)

DECLARE_OP_ADAPTER(Softplus)
DECLARE_OP_USE_OUTPUT(Softplus)

DECLARE_OP_ADAPTER(SoftplusGrad)
DECLARE_OP_USE_OUTPUT(SoftplusGrad)

DECLARE_OP_ADAPTER(Tanh)
DECLARE_OP_USE_OUTPUT(Tanh)

DECLARE_OP_ADAPTER(TanhGrad)
DECLARE_OP_USE_OUTPUT(TanhGrad)

DECLARE_OP_ADAPTER(Mish)
DECLARE_OP_USE_OUTPUT(Mish)

DECLARE_OP_ADAPTER(Gelu)
DECLARE_OP_USE_OUTPUT(Gelu)

DECLARE_OP_ADAPTER(GeluGrad)
DECLARE_OP_USE_OUTPUT(GeluGrad)

DECLARE_OP_ADAPTER(CeluV2)
DECLARE_OP_USE_OUTPUT(CeluV2)

DECLARE_OP_ADAPTER(FastGelu)
DECLARE_OP_USE_OUTPUT(FastGelu)

DECLARE_OP_ADAPTER(FastGeluGrad)
DECLARE_OP_USE_OUTPUT(FastGeluGrad)

DECLARE_OP_ADAPTER(Relu)
DECLARE_OP_USE_OUTPUT(Relu)

DECLARE_OP_ADAPTER(ReluV2)
DECLARE_OP_USE_OUTPUT(ReluV2)

DECLARE_OP_ADAPTER(PRelu)
DECLARE_OP_USE_OUTPUT(PRelu)

DECLARE_OP_ADAPTER(Elu)
DECLARE_OP_USE_OUTPUT(Elu)

DECLARE_OP_ADAPTER(EluGrad)
DECLARE_OP_USE_OUTPUT(EluGrad)

DECLARE_OP_ADAPTER(PReluGrad)
DECLARE_OP_USE_OUTPUT(PReluGrad)

DECLARE_OP_ADAPTER(Selu)
DECLARE_OP_USE_OUTPUT(Selu)

DECLARE_OP_ADAPTER(Sigmoid)
DECLARE_OP_USE_OUTPUT(Sigmoid)

DECLARE_OP_ADAPTER(Swish)
DECLARE_OP_USE_OUTPUT(Swish)

DECLARE_OP_ADAPTER(HardSwish)
DECLARE_OP_USE_OUTPUT(HardSwish)

DECLARE_OP_ADAPTER(HardSwishGrad)
DECLARE_OP_USE_OUTPUT(HardSwishGrad)

DECLARE_OP_ADAPTER(HardSigmoid)
DECLARE_OP_USE_OUTPUT(HardSigmoid)

DECLARE_OP_ADAPTER(SigmoidGrad)
DECLARE_OP_USE_OUTPUT(SigmoidGrad)

DECLARE_OP_ADAPTER(LeakyRelu)
DECLARE_OP_USE_OUTPUT(LeakyRelu)

DECLARE_OP_ADAPTER(HardShrink)
DECLARE_OP_USE_OUTPUT(HardShrink)

DECLARE_OP_ADAPTER(HardShrinkGrad)
DECLARE_OP_USE_OUTPUT(HardShrinkGrad)

DECLARE_OP_ADAPTER(SoftShrink)
DECLARE_OP_USE_OUTPUT(SoftShrink)

DECLARE_OP_ADAPTER(SoftShrinkGrad)
DECLARE_OP_USE_OUTPUT(SoftShrinkGrad)

DECLARE_OP_ADAPTER(LogSigmoid)
DECLARE_OP_USE_OUTPUT(LogSigmoid)

DECLARE_OP_ADAPTER(HardSigmoidGrad)
DECLARE_OP_USE_OUTPUT(HardSigmoidGrad)
}  // namespace mindspore::transform
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NONLINEAR_FUC_OPS_DECLARE_H_
