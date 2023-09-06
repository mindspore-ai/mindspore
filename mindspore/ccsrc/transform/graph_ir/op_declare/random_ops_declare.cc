/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "transform/graph_ir/op_declare/random_ops_declare.h"

#include "ops/math_op_name.h"
#include "ops/nn_op_name.h"
#include "ops/random_ops.h"

namespace mindspore::transform {
// DropOutGenMask
INPUT_MAP(DropOutGenMask) = {{1, INPUT_DESC(shape)}, {2, INPUT_DESC(prob)}};
ATTR_MAP(DropOutGenMask) = {{"Seed0", ATTR_DESC(seed, AnyTraits<int64_t>())},
                            {"Seed1", ATTR_DESC(seed2, AnyTraits<int64_t>())}};
OUTPUT_MAP(DropOutGenMask) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(DropOutGenMask, kDropoutGenMaskOpName, ADPT_DESC(DropOutGenMask))

// DropOutGenMaskV4
INPUT_MAP(DropOutGenMaskV4) = {{1, INPUT_DESC(shape)}, {2, INPUT_DESC(prob)}};
ATTR_MAP(DropOutGenMaskV4) = {{"Seed0", ATTR_DESC(seed, AnyTraits<int64_t>())},
                              {"Seed1", ATTR_DESC(seed2, AnyTraits<int64_t>())},
                              {"dtype", ATTR_DESC(dtype, AnyTraits<GEType>())}};
OUTPUT_MAP(DropOutGenMaskV4) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(DropOutGenMaskV4, kNameDropOutGenMaskV4, ADPT_DESC(DropOutGenMaskV4))

// StatelessDropOutGenMask
INPUT_MAP(StatelessDropOutGenMask) = {{1, INPUT_DESC(shape)},
                                      {2, INPUT_DESC(prob)},
                                      {3, INPUT_DESC(seed)},
                                      {4, INPUT_DESC(seed1)},
                                      {5, INPUT_DESC(offset)}};
ATTR_MAP(StatelessDropOutGenMask) = EMPTY_ATTR_MAP;
OUTPUT_MAP(StatelessDropOutGenMask) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(StatelessDropOutGenMask, kStatelessDropOutGenMaskOpName, ADPT_DESC(StatelessDropOutGenMask))

// LinSpace
INPUT_MAP(LinSpace) = {{1, INPUT_DESC(start)}, {2, INPUT_DESC(stop)}, {3, INPUT_DESC(num)}};
ATTR_MAP(LinSpace) = EMPTY_ATTR_MAP;
OUTPUT_MAP(LinSpace) = {{0, OUTPUT_DESC(output)}};
REG_ADPT_DESC(LinSpace, kNameLinSpace, ADPT_DESC(LinSpace))
REG_ADPT_DESC(LinSpaceD, kLinSpaceDOpName, ADPT_DESC(LinSpace))

// RandomChoiceWithMask
INPUT_MAP(RandomChoiceWithMask) = {{1, INPUT_DESC(x)}};
ATTR_MAP(RandomChoiceWithMask) = {{"count", ATTR_DESC(count, AnyTraits<int64_t>())},
                                  {"seed", ATTR_DESC(seed, AnyTraits<int64_t>())},
                                  {"seed2", ATTR_DESC(seed2, AnyTraits<int64_t>())}};
OUTPUT_MAP(RandomChoiceWithMask) = {{0, OUTPUT_DESC(y)}, {1, OUTPUT_DESC(mask)}};
REG_ADPT_DESC(RandomChoiceWithMask, kNameRandomChoiceWithMask, ADPT_DESC(RandomChoiceWithMask))

// TruncatedNormal
INPUT_MAP(TruncatedNormal) = {{1, INPUT_DESC(shape)}};
ATTR_MAP(TruncatedNormal) = {{"seed", ATTR_DESC(seed, AnyTraits<int64_t>())},
                             {"seed2", ATTR_DESC(seed2, AnyTraits<int64_t>())}};
OUTPUT_MAP(TruncatedNormal) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(TruncatedNormal, kNameTruncatedNormal, ADPT_DESC(TruncatedNormal))

// RandomStandardNormal
INPUT_MAP(RandomStandardNormal) = {{1, INPUT_DESC(shape)}};
ATTR_MAP(RandomStandardNormal) = {{"dtype", ATTR_DESC(dtype, AnyTraits<GEType>())},
                                  {"seed", ATTR_DESC(seed, AnyTraits<int64_t>())},
                                  {"seed2", ATTR_DESC(seed2, AnyTraits<int64_t>())}};
OUTPUT_MAP(RandomStandardNormal) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(RandomStandardNormal, kNameStandardNormal, ADPT_DESC(RandomStandardNormal))

// Multinomial
INPUT_MAP(Multinomial) = {{1, INPUT_DESC(logits)}, {2, INPUT_DESC(num_samples)}};
ATTR_MAP(Multinomial) = {{"dtype", ATTR_DESC(dtype, AnyTraits<GEType>())},
                         {"seed", ATTR_DESC(seed, AnyTraits<int64_t>())},
                         {"seed2", ATTR_DESC(seed2, AnyTraits<int64_t>())}};
OUTPUT_MAP(Multinomial) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Multinomial, prim::kPrimMultinomial->name(), ADPT_DESC(Multinomial))

// Dropout
INPUT_MAP(Dropout) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Dropout) = {{"dropout_ratio", ATTR_DESC(dropout_ratio, AnyTraits<float>())}};
OUTPUT_MAP(Dropout) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Dropout, kDropoutOpName, ADPT_DESC(Dropout))

// RandomUniformInt
INPUT_MAP(RandomUniformInt) = {{1, INPUT_DESC(shape)}, {2, INPUT_DESC(min)}, {3, INPUT_DESC(max)}};
ATTR_MAP(RandomUniformInt) = {{"seed", ATTR_DESC(seed, AnyTraits<int64_t>())},
                              {"seed2", ATTR_DESC(seed2, AnyTraits<int64_t>())}};
OUTPUT_MAP(RandomUniformInt) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(RandomUniformInt, kUniformIntOpName, ADPT_DESC(RandomUniformInt))

// RandomUniform
INPUT_MAP(RandomUniform) = {{1, INPUT_DESC(shape)}};
ATTR_MAP(RandomUniform) = {{"dtype", ATTR_DESC(dtype, AnyTraits<GEType>())},
                           {"seed", ATTR_DESC(seed, AnyTraits<int64_t>())},
                           {"seed2", ATTR_DESC(seed2, AnyTraits<int64_t>())}};
OUTPUT_MAP(RandomUniform) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(UniformReal, kNameUniformReal, ADPT_DESC(RandomUniform))

// LogNormalReverse
CUST_INPUT_MAP(LogNormalReverse) = {{1, INPUT_DESC(input)}};
CUST_ATTR_MAP(LogNormalReverse) = {{"mean", ATTR_DESC(mean, AnyTraits<float>())},
                                   {"std", ATTR_DESC(std, AnyTraits<float>())}};
CUST_OUTPUT_MAP(LogNormalReverse) = {{0, OUTPUT_DESC(output)}};
REG_ADPT_DESC(LogNormalReverse, kNameLogNormalReverse, CUST_ADPT_DESC(LogNormalReverse));

// Dropout2D
CUST_INPUT_MAP(Dropout2D) = {{1, INPUT_DESC(x)}};
CUST_ATTR_MAP(Dropout2D) = {{"keep_prob", ATTR_DESC(keep_prob, AnyTraits<float>())}};
CUST_OUTPUT_MAP(Dropout2D) = {{0, OUTPUT_DESC(y)}, {1, OUTPUT_DESC(mask)}};
REG_ADPT_DESC(Dropout2D, kNameDropout2D, CUST_ADPT_DESC(Dropout2D))
}  // namespace mindspore::transform
