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
InferInfoParam NoneInput() { return InferInfoParam{{}, kMetaTypeNone, mindspore::kNone}; }

InferInfoParam ScalarInt(int64_t value) { return InferInfoParam{{}, kNumberTypeInt64, CreateScalar<int64_t>(value)}; }

void FeedArgs(GeneralInferParamGenerator *generator, const InferInfoParam &x, const InferInfoParam &weight,
              const InferInfoParam &group_list, int64_t split_item, int64_t group_type, int64_t group_list_type,
              const InferInfoParam &scale = NoneInput(), const InferInfoParam &output_dtype = NoneInput(),
              const InferInfoParam &scale_dtype = NoneInput(),
              const InferInfoParam &per_token_scale_dtype = NoneInput()) {
  generator->FeedInputArgs({x,
                            weight,
                            NoneInput(),
                            scale,
                            NoneInput(),
                            NoneInput(),
                            NoneInput(),
                            NoneInput(),
                            group_list,
                            NoneInput(),
                            NoneInput(),
                            NoneInput(),
                            ScalarInt(split_item),
                            ScalarInt(group_type),
                            ScalarInt(group_list_type),
                            ScalarInt(0),
                            NoneInput(),
                            output_dtype,
                            scale_dtype,
                            per_token_scale_dtype});
}

void AddSuccessCase(GeneralInferParamGenerator *generator, const InferInfoParam &x, const InferInfoParam &weight,
                    const InferInfoParam &group_list, int64_t split_item, int64_t group_type, int64_t group_list_type,
                    const ShapeArray &output_shapes, TypeId output_type, const InferInfoParam &scale = NoneInput(),
                    const InferInfoParam &output_dtype = NoneInput(), const InferInfoParam &scale_dtype = NoneInput(),
                    const InferInfoParam &per_token_scale_dtype = NoneInput()) {
  FeedArgs(generator, x, weight, group_list, split_item, group_type, group_list_type, scale, output_dtype, scale_dtype,
           per_token_scale_dtype);
  generator->FeedExpectedOutput(output_shapes, TypeIdList(output_shapes.size(), output_type));
}

void AddFailureCase(GeneralInferParamGenerator *generator, const InferInfoParam &x, const InferInfoParam &weight,
                    const InferInfoParam &group_list, int64_t split_item, int64_t group_type, int64_t group_list_type,
                    const InferInfoParam &scale_dtype = NoneInput()) {
  FeedArgs(generator, x, weight, group_list, split_item, group_type, group_list_type, NoneInput(), NoneInput(),
           scale_dtype);
  generator->CaseShouldThrow();
}

std::vector<GeneralInferParam> PrepareParams() {
  GeneralInferParamGenerator generator;
  const auto no_group_list = NoneInput();

  AddSuccessCase(&generator, InferInfoParam{ShapeArray{{2, 4, 8}, {3, 5}}, kNumberTypeFloat16},
                 InferInfoParam{ShapeArray{{8, 10}, {5, 7}}, kNumberTypeFloat16}, no_group_list, 0, -1, 0,
                 {{2, 4, 10}, {3, 7}}, kNumberTypeFloat16);
  AddSuccessCase(&generator, InferInfoParam{ShapeArray{{4, 8}}, kNumberTypeFloat16},
                 InferInfoParam{ShapeArray{{8, 16}}, kNumberTypeFloat16}, no_group_list, 0, -1, 0, {{4, 16}},
                 kNumberTypeBFloat16, NoneInput(), ScalarInt(kNumberTypeBFloat16));
  AddSuccessCase(&generator, InferInfoParam{ShapeArray{{5, 8}}, kNumberTypeFloat16},
                 InferInfoParam{ShapeArray{{8, 10}, {8, 12}}, kNumberTypeFloat16},
                 InferInfoParam{ShapeVector{2}, kNumberTypeInt64, CreatePyIntTuple({2, 5})}, 1, 0, 0,
                 {{2, 10}, {3, 12}}, kNumberTypeFloat16);
  AddSuccessCase(&generator, InferInfoParam{ShapeArray{{5, 8}}, kNumberTypeFloat16},
                 InferInfoParam{ShapeArray{{8, 10}, {8, 12}}, kNumberTypeFloat16},
                 InferInfoParam{ShapeVector{2}, kNumberTypeInt64, CreatePyIntTuple({2, 3})}, 1, 0, 1,
                 {{2, 10}, {3, 12}}, kNumberTypeFloat16);
  AddSuccessCase(&generator, InferInfoParam{ShapeArray{{5, 8}}, kNumberTypeFloat16},
                 InferInfoParam{ShapeArray{{8, 10}, {8, 12}}, kNumberTypeFloat16},
                 InferInfoParam{ShapeVector{2, 2}, kNumberTypeInt64, CreatePyIntTuple({0, 2, 3, 3})}, 1, 0, 2,
                 {{2, 10}, {3, 12}}, kNumberTypeFloat16);
  AddSuccessCase(&generator, InferInfoParam{ShapeArray{{2, 8}, {3, 8}}, kNumberTypeFloat32},
                 InferInfoParam{ShapeArray{{8, 10}, {8, 10}}, kNumberTypeFloat32}, no_group_list, 2, -1, 0, {{5, 10}},
                 kNumberTypeFloat32);
  AddSuccessCase(&generator, InferInfoParam{ShapeArray{{5, 8}}, kNumberTypeFloat16},
                 InferInfoParam{ShapeArray{{2, 8, 10}}, kNumberTypeFloat16},
                 InferInfoParam{ShapeVector{2}, kNumberTypeInt64, CreatePyIntTuple({2, 5})}, 3, 0, 0, {{5, 10}},
                 kNumberTypeFloat16);
  AddSuccessCase(&generator, InferInfoParam{ShapeArray{{5, 8}}, kNumberTypeFloat16},
                 InferInfoParam{ShapeArray{{8, 10}}, kNumberTypeFloat16},
                 InferInfoParam{ShapeVector{4}, kNumberTypeInt64, CreatePyIntTuple({1, 1, 2, 1})}, 3, 2, 1,
                 {{4, 5, 10}}, kNumberTypeFloat16);
  AddSuccessCase(&generator, InferInfoParam{ShapeArray{{abstract::Shape::kShapeRankAny}, {-1, 8}}, kNumberTypeFloat32},
                 InferInfoParam{ShapeArray{{8, 10}, {8, -1}}, kNumberTypeFloat32}, no_group_list, 0, -1, 0,
                 {{abstract::Shape::kShapeRankAny}, {-1, -1}}, kNumberTypeFloat32);
  AddSuccessCase(&generator, InferInfoParam{ShapeArray{{5, 8}}, kNumberTypeFloat16},
                 InferInfoParam{ShapeArray{{8, 10}}, kNumberTypeFloat16}, no_group_list, 0, -1, 0, {{5, 10}},
                 kNumberTypeFloat16, InferInfoParam{ShapeArray{{5, 1}}, kNumberTypeUInt8}, NoneInput(),
                 ScalarInt(kNumberTypeFloat8E8M0FNU), ScalarInt(kNumberTypeFloat8E8M0FNU));

  AddFailureCase(&generator, InferInfoParam{ShapeArray{}, TypeIdList{}, ValuePtrList{}},
                 InferInfoParam{ShapeArray{{8, 10}}, kNumberTypeFloat16}, no_group_list, 0, -1, 0);
  AddFailureCase(&generator, InferInfoParam{ShapeArray{{5, 8}}, kNumberTypeFloat16},
                 InferInfoParam{ShapeArray{{8, 10}}, kNumberTypeFloat16}, no_group_list, 4, -1, 0);
  AddFailureCase(&generator, InferInfoParam{ShapeArray{{5, 8}}, kNumberTypeFloat16},
                 InferInfoParam{ShapeArray{{8, 10}}, kNumberTypeFloat16}, no_group_list, 0, 1, 0);
  AddFailureCase(&generator, InferInfoParam{ShapeArray{{5, 8}}, kNumberTypeFloat16},
                 InferInfoParam{ShapeArray{{8, 10}}, kNumberTypeFloat16}, no_group_list, 0, -1, 3);
  AddFailureCase(&generator, InferInfoParam{ShapeArray{{5, 8}}, kNumberTypeFloat16},
                 InferInfoParam{ShapeArray{{8, 10}}, kNumberTypeFloat16}, no_group_list, 3, 0, 0);
  AddFailureCase(&generator, InferInfoParam{ShapeArray{{5, 8}}, kNumberTypeFloat16},
                 InferInfoParam{ShapeArray{{8, 10}}, kNumberTypeFloat16},
                 InferInfoParam{ShapeVector{2, 1}, kNumberTypeInt64}, 3, 0, 0);
  AddFailureCase(&generator, InferInfoParam{ShapeArray{{5, 8}}, kNumberTypeFloat16},
                 InferInfoParam{ShapeArray{{8, 10}}, kNumberTypeFloat16},
                 InferInfoParam{ShapeVector{2}, kNumberTypeInt32}, 3, 0, 0);
  AddFailureCase(&generator, InferInfoParam{ShapeArray{{5, 8}}, kNumberTypeFloat16},
                 InferInfoParam{ShapeArray{{8, 10}}, kNumberTypeFloat16}, no_group_list, 0, -1, 0,
                 ScalarInt(kNumberTypeFloat16));
  AddFailureCase(&generator, InferInfoParam{ShapeArray{{8}}, kNumberTypeFloat16},
                 InferInfoParam{ShapeArray{{8, 10}}, kNumberTypeFloat16}, no_group_list, 0, -1, 0);

  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(GroupedMatmulV5, GeneralInferTest, testing::ValuesIn(PrepareParams()));
}  // namespace mindspore::ops
