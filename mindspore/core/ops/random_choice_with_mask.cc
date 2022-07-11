/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <set>
#include <memory>
#include <vector>
#include "ops/random_choice_with_mask.h"
#include "ops/op_name.h"
#include "mindapi/ir/value.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void RandomChoiceWithMask::Init(const int64_t count, const int64_t seed, const int64_t seed2) {
  this->set_count(count);
  this->set_seed(seed);
  this->set_seed2(seed2);
}

void RandomChoiceWithMask::set_count(const int64_t count) { this->AddAttr(kCount, api::MakeValue(count)); }

int64_t RandomChoiceWithMask::get_count() {
  auto value_ptr = GetAttr(kCount);
  return GetValue<int64_t>(value_ptr);
}

void RandomChoiceWithMask::set_seed(const int64_t count) { this->AddAttr(kSeed, api::MakeValue(count)); }

int64_t RandomChoiceWithMask::get_seed() {
  auto value_ptr = GetAttr(kSeed);
  return GetValue<int64_t>(value_ptr);
}

void RandomChoiceWithMask::set_seed2(const int64_t seed2) { this->AddAttr(kSeed2, api::MakeValue(seed2)); }

int64_t RandomChoiceWithMask::get_seed2() {
  auto value_ptr = GetAttr(kSeed2);
  return GetValue<int64_t>(value_ptr);
}

namespace {
constexpr size_t kInputMaxRank = 5;
constexpr size_t kInputMinRank = 1;

abstract::TupleShapePtr RandomChoiceWithMaskInferShape(const PrimitivePtr &primitive,
                                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto shape_ptr = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, 0);
  MS_EXCEPTION_IF_NULL(shape_ptr);
  auto input_shape = shape_ptr->shape();

  auto input_rank = SizeToLong(input_shape.size());
  auto input_type = input_args[0]->BuildType();
  const std::set<TypePtr> input_valid_types = {kBool};
  (void)CheckAndConvertUtils::CheckTypeValid("input_x_type", input_type, input_valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("input_rank", input_rank, kGreaterEqual, kInputMinRank, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("input_rank", input_rank, kLessEqual, kInputMaxRank, prim_name);
  auto count_value = primitive->GetAttr(kCount);
  auto count = GetValue<int64_t>(count_value);
  (void)CheckAndConvertUtils::CheckInteger(kCount, count, kGreaterThan, 0, prim_name);
  ShapeVector mask_shape{count};
  ShapeVector index_shape{count, input_rank};
  auto abs_mask_shape = std::make_shared<abstract::Shape>(mask_shape, mask_shape, mask_shape);
  auto abs_index_shape = std::make_shared<abstract::Shape>(index_shape, index_shape, index_shape);
  std::vector<abstract::BaseShapePtr> shape_tuple;
  shape_tuple.push_back(abs_index_shape);
  shape_tuple.push_back(abs_mask_shape);
  return std::make_shared<abstract::TupleShape>(shape_tuple);
}

TuplePtr RandomChoiceWithMaskInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  std::vector<TypePtr> type_tuple;
  type_tuple.push_back(kInt32);
  type_tuple.push_back(kBool);
  return std::make_shared<Tuple>(type_tuple);
}
}  // namespace

MIND_API_BASE_IMPL(RandomChoiceWithMask, PrimitiveC, BaseOperator);
abstract::AbstractBasePtr RandomChoiceWithMaskInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                    const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto infer_type = RandomChoiceWithMaskInferType(primitive, input_args);
  auto infer_shape = RandomChoiceWithMaskInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

REGISTER_PRIMITIVE_EVAL_IMPL(RandomChoiceWithMask, prim::kPrimRandomChoiceWithMask, RandomChoiceWithMaskInfer, nullptr,
                             true);
}  // namespace ops
}  // namespace mindspore
