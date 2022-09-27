/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "ops/sort.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr SortInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 1;
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kGreaterEqual, input_num,
                                           prim_name);
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto x_rank = SizeToLong(x_shape.size());
  auto axis = GetValue<int64_t>(primitive->GetAttr("axis"));
  CheckAndConvertUtils::CheckInRange<int64_t>("axis", axis, kIncludeBoth, {-x_rank, x_rank - 1}, prim_name);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 0);
  auto x = input_args[0]->BuildShape();
  MS_EXCEPTION_IF_NULL(x);
  auto shape_element = x->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_element);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{shape_element, shape_element});
}

TuplePtr SortInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto infer_type = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(infer_type);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  auto type = CheckAndConvertUtils::CheckTensorTypeValid("inputx", infer_type, valid_types, prim->name());
  std::vector<TypePtr> type_tuple;
  type_tuple.push_back(type);
  type_tuple.push_back(kInt32);
  return std::make_shared<Tuple>(type_tuple);
}
}  // namespace

MIND_API_OPERATOR_IMPL(Sort, BaseOperator);
AbstractBasePtr SortInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  auto infertype = SortInferType(primitive, input_args);
  auto infershape = SortInferShape(primitive, input_args);
  return abstract::MakeAbstract(infershape, infertype);
}

int64_t Sort::get_axis() const {
  auto value_ptr = this->GetAttr(kAxis);
  return GetValue<int64_t>(value_ptr);
}

bool Sort::get_descending() const {
  auto value_ptr = this->GetAttr(kDescending);
  return GetValue<bool>(value_ptr);
}

REGISTER_PRIMITIVE_EVAL_IMPL(Sort, prim::kPrimSort, SortInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
