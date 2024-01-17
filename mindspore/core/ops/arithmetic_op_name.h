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

#ifndef MINDSPORE_CORE_BASE_ARITHMETIC_OP_NAME_H_
#define MINDSPORE_CORE_BASE_ARITHMETIC_OP_NAME_H_

namespace mindspore {
// Arithmetic
constexpr auto kScalarToTensorOpName = "ScalarToTensor";
constexpr auto kScalarTruncOpName = "ScalarTrunc";
constexpr auto kScalarFloorOpName = "ScalarFloor";
constexpr auto kScalarBitwiseAndOpName = "bit_and";
constexpr auto kScalarBitwiseOrOpName = "bit_or";
constexpr auto kAcoshGradOpName = "AcoshGrad";
constexpr auto kTruncOpName = "Trunc";
constexpr auto kEuclideanNormOpName = "EuclideanNorm";
constexpr auto kGerOpName = "Ger";
constexpr auto kZetaOpName = "Zeta";
constexpr auto kLinearSumAssignmentOpName = "LinearSumAssignment";
constexpr auto kTensorToScalarOpName = "TensorToScalar";
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_ARITHMETIC_OP_NAME_H_
