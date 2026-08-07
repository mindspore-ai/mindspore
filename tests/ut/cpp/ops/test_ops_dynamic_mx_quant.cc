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
              const ValuePtr &axis, int64_t dst_type, const ValuePtr &block_size, int64_t scale_alg,
              const std::string &round_mode = "rint") {
  generator->FeedInputArgs(
    {InferInfoParam{input_shape, input_type}, InferInfoParam{{}, kNumberTypeInt64, CreateScalar<int64_t>(dst_type)},
     InferInfoParam{{}, kNumberTypeInt64, axis}, InferInfoParam{{}, kObjectTypeString, MakeValue(round_mode)},
     InferInfoParam{{}, kNumberTypeInt64, block_size},
     InferInfoParam{{}, kNumberTypeInt64, CreateScalar<int64_t>(scale_alg)},
     InferInfoParam{{}, kNumberTypeFloat32, CreateScalar<float>(0.0f)},
     InferInfoParam{{}, kNumberTypeFloat32, CreateScalar<float>(0.0f)}});
}

void AddSuccessCase(GeneralInferParamGenerator *generator, const ShapeVector &input_shape, TypeId input_type,
                    const ValuePtr &axis, int64_t dst_type, const ValuePtr &block_size, int64_t scale_alg,
                    const ShapeVector &y_shape, TypeId y_type, const ShapeVector &scale_shape) {
  FeedArgs(generator, input_shape, input_type, axis, dst_type, block_size, scale_alg);
  generator->FeedExpectedOutput({y_shape, scale_shape}, {y_type, kNumberTypeUInt8});
}

void AddFailureCase(GeneralInferParamGenerator *generator, const ShapeVector &input_shape, TypeId input_type,
                    int64_t axis, int64_t dst_type, int64_t block_size, int64_t scale_alg,
                    const std::string &round_mode = "rint") {
  FeedArgs(generator, input_shape, input_type, CreateScalar<int64_t>(axis), dst_type, CreateScalar<int64_t>(block_size),
           scale_alg, round_mode);
  generator->CaseShouldThrow();
}

std::vector<GeneralInferParam> PrepareParams() {
  GeneralInferParamGenerator generator;
  AddSuccessCase(&generator, {2, 65, 64}, kNumberTypeFloat16, CreateScalar<int64_t>(1), 23, CreateScalar<int64_t>(32),
                 0, {2, 65, 64}, kNumberTypeFloat8E5M2, {2, 2, 64, 2});
  AddSuccessCase(&generator, {3, 32}, kNumberTypeBFloat16, CreateScalar<int64_t>(-1), 24, CreateScalar<int64_t>(32), 1,
                 {3, 32}, kNumberTypeFloat8E4M3FN, {3, 1, 2});
  AddSuccessCase(&generator, {3, 32}, kNumberTypeBFloat16, CreateScalar<int64_t>(-1), 291, CreateScalar<int64_t>(32), 1,
                 {3, 32}, kNumberTypeFloat8E5M2, {3, 1, 2});
  AddSuccessCase(&generator, {3, 32}, kNumberTypeBFloat16, CreateScalar<int64_t>(-1), 292, CreateScalar<int64_t>(32), 1,
                 {3, 32}, kNumberTypeFloat8E4M3FN, {3, 1, 2});
  AddSuccessCase(&generator, {128}, kNumberTypeBFloat16, CreateScalar<int64_t>(0), 24, CreateScalar<int64_t>(32), 2,
                 {128}, kNumberTypeFloat8E4M3FN, {2, 2});
  AddSuccessCase(&generator, {2, -1, 64}, kNumberTypeFloat16, CreateScalar<int64_t>(1), 23, CreateScalar<int64_t>(32),
                 0, {2, -1, 64}, kNumberTypeFloat8E5M2, {2, -1, 64, 2});
  AddSuccessCase(&generator, {abstract::Shape::kShapeRankAny}, kNumberTypeFloat16, CreateScalar<int64_t>(-1), 24,
                 CreateScalar<int64_t>(32), 0, {abstract::Shape::kShapeRankAny}, kNumberTypeFloat8E4M3FN,
                 {abstract::Shape::kShapeRankAny});
  AddSuccessCase(&generator, {2, 64, 96}, kNumberTypeFloat16, kValueAny, 23, CreateScalar<int64_t>(32), 0, {2, 64, 96},
                 kNumberTypeFloat8E5M2, {-1, -1, -1, 2});
  AddSuccessCase(&generator, {2, 64, 96}, kNumberTypeFloat16, CreateScalar<int64_t>(1), 23, kValueAny, 0, {2, 64, 96},
                 kNumberTypeFloat8E5M2, {2, -1, 96, 2});

  AddFailureCase(&generator, {2, 64}, kNumberTypeInt8, -1, 23, 32, 0);
  AddFailureCase(&generator, {}, kNumberTypeFloat16, -1, 23, 32, 0);
  AddFailureCase(&generator, {1, 1, 1, 1, 1, 1, 1, 2}, kNumberTypeFloat16, -1, 23, 32, 0);
  AddFailureCase(&generator, {2, 64}, kNumberTypeFloat16, 2, 23, 32, 0);
  AddFailureCase(&generator, {2, 64}, kNumberTypeFloat16, -1, 23, 16, 0);
  AddFailureCase(&generator, {2, 64}, kNumberTypeFloat16, -1, 23, 64, 0);
  AddFailureCase(&generator, {2, 64}, kNumberTypeFloat16, -1, 23, 32, 3);
  AddFailureCase(&generator, {2, 64}, kNumberTypeFloat16, -1, 295, 32, 0);
  AddFailureCase(&generator, {2, 64}, kNumberTypeFloat16, -1, 296, 32, 0);
  AddFailureCase(&generator, {2, 64}, kNumberTypeFloat16, -1, 297, 32, 0);
  AddFailureCase(&generator, {2, 64}, kNumberTypeFloat16, -1, 23, 32, 0, "floor");
  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(DynamicMxQuant, GeneralInferTest, testing::ValuesIn(PrepareParams()));
}  // namespace mindspore::ops
