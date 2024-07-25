/**
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_RANDOM_OPS_DECLARE_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_RANDOM_OPS_DECLARE_H_

#include "transform/graph_ir/custom_op_proto/cust_random_ops.h"
#include "transform/graph_ir/op_declare/op_declare_macro.h"
#include "utils/hash_map.h"

DECLARE_OP_ADAPTER(DropOutGenMask)
DECLARE_OP_USE_OUTPUT(DropOutGenMask)

DECLARE_OP_ADAPTER(DropOutGenMaskV4)
DECLARE_OP_USE_OUTPUT(DropOutGenMaskV4)

DECLARE_OP_ADAPTER(StatelessDropOutGenMask)
DECLARE_OP_USE_OUTPUT(StatelessDropOutGenMask)

DECLARE_OP_ADAPTER(LinSpace)
DECLARE_OP_USE_OUTPUT(LinSpace)

DECLARE_OP_ADAPTER(TruncatedNormal)
DECLARE_OP_USE_OUTPUT(TruncatedNormal)

DECLARE_OP_ADAPTER(RandomStandardNormal)
DECLARE_OP_USE_OUTPUT(RandomStandardNormal)

DECLARE_OP_ADAPTER(Multinomial)  // for lite build
DECLARE_OP_USE_OUTPUT(Multinomial)
DECLARE_CUST_OP_ADAPTER(Multinomial)
DECLARE_CUST_OP_USE_OUTPUT(Multinomial)

DECLARE_OP_ADAPTER(Dropout)
DECLARE_OP_USE_OUTPUT(Dropout)

DECLARE_OP_ADAPTER(RandomUniform)
DECLARE_OP_USE_OUTPUT(RandomUniform)

DECLARE_CUST_OP_ADAPTER(LogNormalReverse)
DECLARE_CUST_OP_USE_OUTPUT(LogNormalReverse)

DECLARE_OP_ADAPTER(NonDeterministicInts)
DECLARE_OP_USE_OUTPUT(NonDeterministicInts)

DECLARE_CUST_OP_ADAPTER(Dropout2D)
DECLARE_CUST_OP_USE_OUTPUT(Dropout2D)

DECLARE_CUST_OP_ADAPTER(StandardLaplace)
DECLARE_CUST_OP_USE_OUTPUT(StandardLaplace)

DECLARE_OP_ADAPTER(StatelessRandperm)
DECLARE_OP_USE_OUTPUT(StatelessRandperm)

DECLARE_CUST_OP_ADAPTER(Randperm)
DECLARE_CUST_OP_USE_OUTPUT(Randperm)

DECLARE_CUST_OP_ADAPTER(Gamma)
DECLARE_CUST_OP_USE_OUTPUT(Gamma)

DECLARE_CUST_OP_ADAPTER(RandomPoisson)
DECLARE_CUST_OP_USE_OUTPUT(RandomPoisson)

DECLARE_CUST_OP_ADAPTER(RandomCategorical)
DECLARE_CUST_OP_USE_OUTPUT(RandomCategorical)

DECLARE_CUST_OP_ADAPTER(RandomShuffle)
DECLARE_CUST_OP_USE_OUTPUT(RandomShuffle)

DECLARE_CUST_OP_ADAPTER(RandomChoiceWithMask)
DECLARE_CUST_OP_USE_OUTPUT(RandomChoiceWithMask)

DECLARE_OP_ADAPTER(RandomUniformInt)
DECLARE_OP_USE_OUTPUT(RandomUniformInt)

DECLARE_CUST_OP_ADAPTER(Igamma)
DECLARE_CUST_OP_USE_OUTPUT(Igamma)

DECLARE_CUST_OP_ADAPTER(Poisson)
DECLARE_CUST_OP_USE_OUTPUT(Poisson)

DECLARE_CUST_OP_ADAPTER(LogUniformCandidateSampler)
DECLARE_CUST_OP_USE_OUTPUT(LogUniformCandidateSampler)

DECLARE_CUST_OP_ADAPTER(Dropout3D)
DECLARE_CUST_OP_USE_OUTPUT(Dropout3D)

DECLARE_OP_ADAPTER(ShuffleChannel)
DECLARE_OP_USE_OUTPUT(ShuffleChannel)
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_RANDOM_OPS_DECLARE_H_
