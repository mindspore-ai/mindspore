/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "ops/utils/general_infer_utils.h"

namespace mindspore::ops {
namespace {
InferInfoParam ScalarInt(int64_t value) { return InferInfoParam{{}, kNumberTypeInt64, CreateScalar<int64_t>(value)}; }

void FeedArgs(GeneralInferParamGenerator *generator, const ShapeVector &x_shape, TypeId x_type,
              const ShapeVector &group_shape, TypeId group_type, int64_t dst_type, int64_t blocksize, int64_t scale_alg,
              const std::string &round_mode = "rint") {
  generator->FeedInputArgs({InferInfoParam{x_shape, x_type}, InferInfoParam{group_shape, group_type},
                            InferInfoParam{{}, kObjectTypeString, MakeValue(round_mode)}, ScalarInt(dst_type),
                            ScalarInt(blocksize), ScalarInt(scale_alg),
                            InferInfoParam{{}, kNumberTypeFloat32, CreateScalar<float>(0.0f)}});
}

void AddSuccessCase(GeneralInferParamGenerator *generator, const ShapeVector &x_shape, TypeId x_type,
                    const ShapeVector &group_shape, int64_t dst_type, int64_t scale_alg, const ShapeVector &y_shape,
                    TypeId y_type, const ShapeVector &scale_shape) {
  FeedArgs(generator, x_shape, x_type, group_shape, kNumberTypeInt32, dst_type, 32, scale_alg);
  generator->FeedExpectedOutput({y_shape, scale_shape}, {y_type, kNumberTypeUInt8});
}

void AddFailureCase(GeneralInferParamGenerator *generator, const ShapeVector &x_shape, TypeId x_type,
                    const ShapeVector &group_shape, TypeId group_type, int64_t dst_type, int64_t blocksize,
                    int64_t scale_alg, const std::string &round_mode = "rint") {
  FeedArgs(generator, x_shape, x_type, group_shape, group_type, dst_type, blocksize, scale_alg, round_mode);
  generator->CaseShouldThrow();
}

std::vector<GeneralInferParam> PrepareParams() {
  GeneralInferParamGenerator generator;
  AddSuccessCase(&generator, {128, 64}, kNumberTypeFloat16, {2}, 23, 0, {128, 64}, kNumberTypeFloat8E5M2, {4, 64, 2});
  AddSuccessCase(&generator, {65, 32}, kNumberTypeBFloat16, {3}, 24, 1, {65, 32}, kNumberTypeFloat8E4M3FN, {4, 32, 2});
  AddSuccessCase(&generator, {-1, 64}, kNumberTypeFloat16, {3}, 23, 0, {-1, 64}, kNumberTypeFloat8E5M2, {-1, 64, 2});
  AddSuccessCase(&generator, {128, -1}, kNumberTypeBFloat16, {-1}, 24, 0, {128, -1}, kNumberTypeFloat8E4M3FN,
                 {-1, -1, 2});
  AddSuccessCase(&generator, {abstract::Shape::kShapeRankAny}, kNumberTypeFloat16, {2}, 23, 0,
                 {abstract::Shape::kShapeRankAny}, kNumberTypeFloat8E5M2, {abstract::Shape::kShapeRankAny});

  AddFailureCase(&generator, {64}, kNumberTypeFloat16, {1}, kNumberTypeInt32, 23, 32, 0);
  AddFailureCase(&generator, {64, 32}, kNumberTypeFloat16, {1, 1}, kNumberTypeInt32, 23, 32, 0);
  AddFailureCase(&generator, {64, 32}, kNumberTypeInt8, {1}, kNumberTypeInt32, 23, 32, 0);
  AddFailureCase(&generator, {64, 32}, kNumberTypeFloat16, {1}, kNumberTypeInt64, 23, 32, 0);
  AddFailureCase(&generator, {64, 32}, kNumberTypeFloat16, {1}, kNumberTypeInt32, 23, 16, 0);
  AddFailureCase(&generator, {64, 32}, kNumberTypeFloat16, {1}, kNumberTypeInt32, 23, 32, 2);
  AddFailureCase(&generator, {64, 32}, kNumberTypeFloat16, {1}, kNumberTypeInt32, 295, 32, 0);
  AddFailureCase(&generator, {64, 32}, kNumberTypeFloat16, {1}, kNumberTypeInt32, 291, 32, 0);
  AddFailureCase(&generator, {64, 32}, kNumberTypeFloat16, {1}, kNumberTypeInt32, 296, 32, 0);
  AddFailureCase(&generator, {64, 32}, kNumberTypeFloat16, {1}, kNumberTypeInt32, 23, 32, 0, "floor");
  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(GroupedDynamicMxQuant, GeneralInferTest, testing::ValuesIn(PrepareParams()));
}  // namespace mindspore::ops
