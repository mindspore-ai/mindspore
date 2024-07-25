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
CUST_INPUT_MAP(RandomChoiceWithMask) = {{1, INPUT_DESC(x)}};
CUST_ATTR_MAP(RandomChoiceWithMask) = {{"count", ATTR_DESC(count, AnyTraits<int64_t>())},
                                       {"seed", ATTR_DESC(seed, AnyTraits<int64_t>())},
                                       {"seed2", ATTR_DESC(seed2, AnyTraits<int64_t>())}};
CUST_OUTPUT_MAP(RandomChoiceWithMask) = {{0, OUTPUT_DESC(index)}, {1, OUTPUT_DESC(mask)}};
REG_ADPT_DESC(RandomChoiceWithMask, kNameRandomChoiceWithMask, CUST_ADPT_DESC(RandomChoiceWithMask))

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

CUST_INPUT_MAP(Multinomial) = {{1, INPUT_DESC(logits)}, {2, INPUT_DESC(num_samples)}};
CUST_ATTR_MAP(Multinomial) = {{"dtype", ATTR_DESC(dtype, AnyTraits<GEType>())},
                              {"seed", ATTR_DESC(seed, AnyTraits<int64_t>())},
                              {"seed2", ATTR_DESC(seed2, AnyTraits<int64_t>())}};
CUST_OUTPUT_MAP(Multinomial) = {{0, OUTPUT_DESC(y)}};
#ifdef BUILD_LITE
REG_ADPT_DESC(Multinomial, prim::kPrimMultinomial->name(), ADPT_DESC(Multinomial))
#else
REG_ADPT_DESC(Multinomial, prim::kPrimMultinomial->name(), CUST_ADPT_DESC(Multinomial))
#endif

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

// StandardLaplace
CUST_INPUT_MAP(StandardLaplace) = {{1, INPUT_DESC(shape)}};
CUST_ATTR_MAP(StandardLaplace) = {{"seed", ATTR_DESC(seed, AnyTraits<int64_t>())},
                                  {"seed2", ATTR_DESC(seed2, AnyTraits<int64_t>())}};
CUST_OUTPUT_MAP(StandardLaplace) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(StandardLaplace, prim::kPrimStandardLaplace->name(), CUST_ADPT_DESC(StandardLaplace));

// RandpermV2
INPUT_MAP(StatelessRandperm) = {{kIndex1, INPUT_DESC(n)}, {kIndex2, INPUT_DESC(seed)}, {kIndex3, INPUT_DESC(offset)}};
ATTR_MAP(StatelessRandperm) = EMPTY_ATTR_MAP;
INPUT_ATTR_MAP(StatelessRandperm) = {{kIndex4, ATTR_DESC(layout, AnyTraits<int64_t>())},
                                     {kIndex5, ATTR_DESC(dtype, AnyTraits<GEType>())}};
OUTPUT_MAP(StatelessRandperm) = {{kIndex0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(StatelessRandperm, prim::kPrimRandpermV2->name(), ADPT_DESC(StatelessRandperm));

// Randperm
CUST_INPUT_MAP(Randperm) = {{1, INPUT_DESC(n)}};
CUST_ATTR_MAP(Randperm) = {{"max_length", ATTR_DESC(max_length, AnyTraits<int64_t>())},
                           {"pad", ATTR_DESC(pad, AnyTraits<int64_t>())},
                           {"dtype", ATTR_DESC(dtype, AnyTraits<GEType>())}};
CUST_OUTPUT_MAP(Randperm) = {{0, OUTPUT_DESC(output)}};
REG_ADPT_DESC(Randperm, prim::kPrimRandperm->name(), CUST_ADPT_DESC(Randperm));

// Gamma
CUST_INPUT_MAP(Gamma) = {
  {1, INPUT_DESC(shape)}, {2, INPUT_DESC(alpha)}, {3, INPUT_DESC(beta)}, {4, INPUT_DESC(seed)}, {5, INPUT_DESC(seed2)}};
CUST_ATTR_MAP(Gamma) = {{"seed", ATTR_DESC(seed, AnyTraits<int64_t>())},
                        {"seed2", ATTR_DESC(seed2, AnyTraits<int64_t>())}};
CUST_OUTPUT_MAP(Gamma) = {{0, OUTPUT_DESC(output)}};
REG_ADPT_DESC(Gamma, kNameGamma, CUST_ADPT_DESC(Gamma));

// RandomPoisson
CUST_INPUT_MAP(RandomPoisson) = {{1, INPUT_DESC(shape)}, {2, INPUT_DESC(rate)}};
CUST_ATTR_MAP(RandomPoisson) = {{"seed", ATTR_DESC(seed, AnyTraits<int64_t>())},
                                {"seed2", ATTR_DESC(seed2, AnyTraits<int64_t>())}};
CUST_OUTPUT_MAP(RandomPoisson) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(RandomPoisson, prim::kPrimRandomPoisson->name(), CUST_ADPT_DESC(RandomPoisson));

// RandomCategorical
CUST_INPUT_MAP(RandomCategorical) = {{1, INPUT_DESC(logits)}, {2, INPUT_DESC(num_samples)}, {3, INPUT_DESC(seed)}};
CUST_ATTR_MAP(RandomCategorical) = {{"dtype", ATTR_DESC(dtype, AnyTraits<GEType>())}};
CUST_OUTPUT_MAP(RandomCategorical) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(RandomCategorical, prim::kPrimRandomCategorical->name(), CUST_ADPT_DESC(RandomCategorical));

CUST_INPUT_MAP(RandomShuffle) = {{1, INPUT_DESC(x)}};
CUST_ATTR_MAP(RandomShuffle) = {{"seed", ATTR_DESC(seed, AnyTraits<int64_t>())},
                                {"seed2", ATTR_DESC(seed2, AnyTraits<int64_t>())}};
CUST_OUTPUT_MAP(RandomShuffle) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(RandomShuffle, prim::kPrimRandomShuffle->name(), CUST_ADPT_DESC(RandomShuffle));

// Igamma
CUST_INPUT_MAP(Igamma) = {{1, INPUT_DESC(a)}, {2, INPUT_DESC(x)}};
CUST_ATTR_MAP(Igamma) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(Igamma) = {{0, OUTPUT_DESC(z)}};
REG_ADPT_DESC(Igamma, kNameIgamma, CUST_ADPT_DESC(Igamma));

// Poisson
CUST_INPUT_MAP(Poisson) = {
  {1, INPUT_DESC(shape)}, {2, INPUT_DESC(mean)}, {3, INPUT_DESC(seed)}, {4, INPUT_DESC(seed2)}};
CUST_ATTR_MAP(Poisson) = {{"seed", ATTR_DESC(seed, AnyTraits<int64_t>())},
                          {"seed2", ATTR_DESC(seed2, AnyTraits<int64_t>())}};
CUST_OUTPUT_MAP(Poisson) = {{0, OUTPUT_DESC(output)}};
REG_ADPT_DESC(Poisson, kNamePoisson, CUST_ADPT_DESC(Poisson));

// LogUniformCandidateSampler
CUST_INPUT_MAP(LogUniformCandidateSampler) = {{1, INPUT_DESC(true_classes)}};
CUST_ATTR_MAP(LogUniformCandidateSampler) = {{"num_true", ATTR_DESC(num_true, AnyTraits<int64_t>())},
                                             {"num_sampled", ATTR_DESC(num_sampled, AnyTraits<int64_t>())},
                                             {"unique", ATTR_DESC(unique, AnyTraits<bool>())},
                                             {"range_max", ATTR_DESC(range_max, AnyTraits<int64_t>())},
                                             {"seed", ATTR_DESC(seed, AnyTraits<int64_t>())}};
CUST_OUTPUT_MAP(LogUniformCandidateSampler) = {{0, OUTPUT_DESC(sampled_candidates)},
                                               {1, OUTPUT_DESC(true_expected_count)},
                                               {2, OUTPUT_DESC(sampled_expected_count)}};
REG_ADPT_DESC(LogUniformCandidateSampler, kNameLogUniformCandidateSampler, CUST_ADPT_DESC(LogUniformCandidateSampler));

// Dropout3D
CUST_INPUT_MAP(Dropout3D) = {{1, INPUT_DESC(x)}};
CUST_ATTR_MAP(Dropout3D) = {{"keep_prob", ATTR_DESC(keep_prob, AnyTraits<float>())}};
CUST_OUTPUT_MAP(Dropout3D) = {{0, OUTPUT_DESC(y)}, {1, OUTPUT_DESC(mask)}};
REG_ADPT_DESC(Dropout3D, kNameDropout3D, CUST_ADPT_DESC(Dropout3D));

// ShuffleChannel
INPUT_MAP(ShuffleChannel) = {{1, INPUT_DESC(x)}};
ATTR_MAP(ShuffleChannel) = {{"group", ATTR_DESC(group, AnyTraits<int64_t>())}};
OUTPUT_MAP(ShuffleChannel) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ShuffleChannel, kNameChannelShuffle, ADPT_DESC(ShuffleChannel));
}  // namespace mindspore::transform
