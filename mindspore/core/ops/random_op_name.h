/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_BASE_RANDOM_OP_NAME_H_
#define MINDSPORE_CORE_BASE_RANDOM_OP_NAME_H_

namespace mindspore {
// Random
constexpr auto kMultinomialOpName = "Multinomial";
constexpr auto kMultinomialWithReplacementOpName = "MultinomialWithReplacement";
constexpr auto kNonDeterministicIntsOpName = "NonDeterministicInts";
constexpr auto kParameterizedTruncatedNormalOpName = "ParameterizedTruncatedNormal";
constexpr auto kRandomCategoricalOpName = "RandomCategorical";
constexpr auto kRandomChoiceWithMaskOpName = "RandomChoiceWithMask";
constexpr auto kRandomGammaGradOpName = "RandomGammaGrad";
constexpr auto kRandomPoissonOpName = "RandomPoisson";
constexpr auto kRandomShuffleOpName = "RandomShuffle";
constexpr auto kStandardNormalOpName = "StandardNormal";
constexpr auto kStandardLaplaceOpName = "StandardLaplace";
constexpr auto kTruncatedNormalOpName = "TruncatedNormal";
constexpr auto kUniformOpName = "Uniform";
constexpr auto kUniformIntOpName = "UniformInt";
constexpr auto kUniformRealOpName = "UniformReal";
constexpr auto kUniformCandidateSamplerOpName = "UniformCandidateSampler";
constexpr auto kLogUniformCandidateSamplerOpName = "LogUniformCandidateSampler";
constexpr auto kGammaOpName = "Gamma";
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_RANDOM_OP_NAME_H_
