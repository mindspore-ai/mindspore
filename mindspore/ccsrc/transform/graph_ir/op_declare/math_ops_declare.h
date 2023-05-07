/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_MATH_OPS_DECLARE_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_MATH_OPS_DECLARE_H_

#include "utils/hash_map.h"
#include "transform/graph_ir/op_declare/op_declare_macro.h"
#include "mindspore/ccsrc/include/common/utils/utils.h"
#include "ops/math_ops.h"

DECLARE_OP_ADAPTER(ActsULQ)
DECLARE_OP_USE_OUTPUT(ActsULQ)

DECLARE_OP_ADAPTER(ActsULQInputGrad)
DECLARE_OP_USE_OUTPUT(ActsULQInputGrad)

DECLARE_OP_ADAPTER(ActULQClampMaxGrad)
DECLARE_OP_USE_OUTPUT(ActULQClampMaxGrad)

DECLARE_OP_ADAPTER(ActULQClampMinGrad)
DECLARE_OP_USE_OUTPUT(ActULQClampMinGrad)

DECLARE_OP_ADAPTER(IFMR)
DECLARE_OP_USE_OUTPUT(IFMR)

DECLARE_OP_ADAPTER(NLLLoss)
DECLARE_OP_USE_OUTPUT(NLLLoss)

DECLARE_OP_ADAPTER(NLLLossGrad)
DECLARE_OP_USE_OUTPUT(NLLLossGrad)

DECLARE_OP_ADAPTER(Erf)
DECLARE_OP_USE_OUTPUT(Erf)

DECLARE_OP_ADAPTER(Erfc)
DECLARE_OP_USE_OUTPUT(Erfc)

DECLARE_OP_ADAPTER(WtsARQ)
DECLARE_OP_USE_OUTPUT(WtsARQ)

DECLARE_OP_ADAPTER(IsFinite)
DECLARE_OP_USE_OUTPUT(IsFinite)

DECLARE_OP_ADAPTER(IsNan)
DECLARE_OP_USE_OUTPUT(IsNan)

DECLARE_OP_ADAPTER(IsInf)
DECLARE_OP_USE_OUTPUT(IsInf)

DECLARE_OP_ADAPTER(LpNorm)
DECLARE_OP_USE_OUTPUT(LpNorm)

DECLARE_OP_ADAPTER(Trunc)
DECLARE_OP_USE_OUTPUT(Trunc)

DECLARE_OP_ADAPTER(HistogramFixedWidth)
DECLARE_OP_USE_OUTPUT(HistogramFixedWidth)

DECLARE_OP_ADAPTER(Pdist)
DECLARE_OP_USE_OUTPUT(Pdist)

DECLARE_OP_ADAPTER(SoftMarginLossGrad)
DECLARE_OP_USE_OUTPUT(SoftMarginLossGrad)

DECLARE_OP_ADAPTER(Cdist)
DECLARE_OP_USE_OUTPUT(Cdist)

DECLARE_OP_ADAPTER(CdistGrad)
DECLARE_OP_USE_OUTPUT(CdistGrad)

DECLARE_OP_ADAPTER(Conj)
DECLARE_OP_USE_OUTPUT(Conj)

DECLARE_OP_ADAPTER(NextAfter)
DECLARE_OP_USE_OUTPUT(NextAfter)
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_MATH_OPS_DECLARE_H_
