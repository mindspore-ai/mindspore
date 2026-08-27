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
void FeedArgs(GeneralInferParamGenerator *generator, const ShapeVector &input_shape, TypeId input_type,
              int64_t dst_type, int64_t scale_alg, const std::string &round_mode = "rint", float dst_type_max = 0.0f) {
  generator->FeedInputArgs({InferInfoParam{input_shape, input_type},
                            InferInfoParam{{}, kNumberTypeInt64, CreateScalar<int64_t>(dst_type)},
                            InferInfoParam{{}, kObjectTypeString, MakeValue(round_mode)},
                            InferInfoParam{{}, kNumberTypeInt64, CreateScalar<int64_t>(scale_alg)},
                            InferInfoParam{{}, kNumberTypeFloat32, CreateScalar<float>(dst_type_max)}});
}

void AddSuccessCase(GeneralInferParamGenerator *generator, const ShapeVector &input_shape, TypeId input_type,
                    int64_t dst_type, int64_t scale_alg, const ShapeVector &y_shape, TypeId y_type,
                    const ShapeVector &scale1_shape, const ShapeVector &scale2_shape) {
  FeedArgs(generator, input_shape, input_type, dst_type, scale_alg);
  generator->FeedExpectedOutput({y_shape, scale1_shape, y_shape, scale2_shape},
                                {y_type, kNumberTypeUInt8, y_type, kNumberTypeUInt8});
}

void AddFailureCase(GeneralInferParamGenerator *generator, const ShapeVector &input_shape, TypeId input_type,
                    int64_t dst_type, int64_t scale_alg, const std::string &round_mode = "rint") {
  FeedArgs(generator, input_shape, input_type, dst_type, scale_alg, round_mode);
  generator->CaseShouldThrow();
}

std::vector<GeneralInferParam> PrepareParams() {
  GeneralInferParamGenerator generator;
  AddSuccessCase(&generator, {2, 65, 96}, kNumberTypeFloat16, 24, 0, {2, 65, 96}, kNumberTypeFloat8E4M3FN,
                 {2, 65, 2, 2}, {2, 2, 96, 2});
  AddSuccessCase(&generator, {64, 96}, kNumberTypeBFloat16, 23, 1, {64, 96}, kNumberTypeFloat8E5M2, {64, 2, 2},
                 {1, 96, 2});
  AddSuccessCase(&generator, {64, 96}, kNumberTypeBFloat16, 291, 1, {64, 96}, kNumberTypeFloat8E5M2, {64, 2, 2},
                 {1, 96, 2});
  AddSuccessCase(&generator, {64, 96}, kNumberTypeBFloat16, 292, 1, {64, 96}, kNumberTypeFloat8E4M3FN, {64, 2, 2},
                 {1, 96, 2});
  FeedArgs(&generator, {2, 64, 96}, kNumberTypeFloat16, 24, 0, "rint", 8.0f);
  generator.FeedExpectedOutput({{2, 64, 96}, {2, 64, 2, 2}, {2, 64, 96}, {2, 1, 96, 2}},
                               {kNumberTypeFloat8E4M3FN, kNumberTypeUInt8, kNumberTypeFloat8E4M3FN, kNumberTypeUInt8});
  AddSuccessCase(&generator, {2, -1, 96}, kNumberTypeFloat16, 24, 0, {2, -1, 96}, kNumberTypeFloat8E4M3FN,
                 {2, -1, 2, 2}, {2, -1, 96, 2});
  AddSuccessCase(&generator, {2, 64, -1}, kNumberTypeFloat16, 23, 0, {2, 64, -1}, kNumberTypeFloat8E5M2, {2, 64, -1, 2},
                 {2, 1, -1, 2});
  AddSuccessCase(&generator, {abstract::Shape::kShapeRankAny}, kNumberTypeBFloat16, 24, 0,
                 {abstract::Shape::kShapeRankAny}, kNumberTypeFloat8E4M3FN, {abstract::Shape::kShapeRankAny},
                 {abstract::Shape::kShapeRankAny});

  AddFailureCase(&generator, {64, 96}, kNumberTypeFloat32, 23, 0);
  AddFailureCase(&generator, {96}, kNumberTypeFloat16, 23, 0);
  AddFailureCase(&generator, {1, 1, 1, 1, 1, 1, 1, 2}, kNumberTypeFloat16, 23, 0);
  AddFailureCase(&generator, {64, 96}, kNumberTypeFloat16, 295, 0);
  AddFailureCase(&generator, {64, 96}, kNumberTypeFloat16, 296, 0);
  AddFailureCase(&generator, {64, 96}, kNumberTypeFloat16, 297, 0);
  AddFailureCase(&generator, {64, 96}, kNumberTypeFloat16, 23, -1);
  AddFailureCase(&generator, {64, 96}, kNumberTypeFloat16, 23, 2);
  AddFailureCase(&generator, {64, 96}, kNumberTypeFloat16, 23, 0, "round");
  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(DynamicMxQuantWithDualAxis, GeneralInferTest, testing::ValuesIn(PrepareParams()));
}  // namespace mindspore::ops
