/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_BASE_RANDOM_OPS_H_
#define MINDSPORE_CORE_BASE_RANDOM_OPS_H_

#include <memory>
#include "ir/anf.h"
#include "ir/primitive.h"
#include "utils/hash_map.h"
#include "ops/random_op_name.h"

namespace mindspore {
namespace prim {
// Random
GVAR_DEF(PrimitivePtr, kPrimStandardLaplace, std::make_shared<Primitive>("StandardLaplace"));
GVAR_DEF(PrimitivePtr, kPrimStandardNormal, std::make_shared<Primitive>(kStandardNormalOpName));
GVAR_DEF(PrimitivePtr, kPrimParameterizedTruncatedNormal, std::make_shared<Primitive>("ParameterizedTruncatedNormal"));
GVAR_DEF(PrimitivePtr, kPrimRandomNormal, std::make_shared<Primitive>("RandomNormal"));
GVAR_DEF(PrimitivePtr, kPrimNonDeterministicInts, std::make_shared<Primitive>("NonDeterministicInts"));
GVAR_DEF(PrimitivePtr, kPrimTruncatedNormal, std::make_shared<Primitive>("TruncatedNormal"));
GVAR_DEF(PrimitivePtr, kPrimRandomPoisson, std::make_shared<Primitive>("RandomPoisson"));
GVAR_DEF(PrimitivePtr, kPrimRandomGamma, std::make_shared<Primitive>("RandomGamma"));
GVAR_DEF(PrimitivePtr, kPrimRandomShuffle, std::make_shared<Primitive>("RandomShuffle"));
GVAR_DEF(PrimitivePtr, kPrimRandomGammaGrad, std::make_shared<Primitive>("RandomGammaGrad"));
GVAR_DEF(PrimitivePtr, kPrimRandomCategorical, std::make_shared<Primitive>("RandomCategorical"));
GVAR_DEF(PrimitivePtr, kPrimRandperm, std::make_shared<Primitive>("Randperm"));
GVAR_DEF(PrimitivePtr, kPrimRandpermV2, std::make_shared<Primitive>("RandpermV2"));
GVAR_DEF(PrimitivePtr, kPrimUniformCandidateSampler, std::make_shared<Primitive>("UniformCandidateSampler"));
GVAR_DEF(PrimitivePtr, kPrimLogUniformCandidateSampler, std::make_shared<Primitive>("LogUniformCandidateSampler"));
GVAR_DEF(PrimitivePtr, kPrimMultinomial, std::make_shared<Primitive>("Multinomial"));
GVAR_DEF(PrimitivePtr, kPrimMultinomialWithReplacement, std::make_shared<Primitive>("MultinomialWithReplacement"));
GVAR_DEF(PrimitivePtr, kPrimRandomChoiceWithMask, std::make_shared<Primitive>("RandomChoiceWithMask"));
GVAR_DEF(PrimitivePtr, kPrimUniform, std::make_shared<Primitive>(kUniformOpName));
}  // namespace prim
}  // namespace mindspore
#endif  // MINDSPORE_CORE_BASE_RANDOM_OPS_H_
