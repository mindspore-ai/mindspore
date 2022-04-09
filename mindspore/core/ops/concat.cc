/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include <map>
#include <string>
#include "ops/concat.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr ConcatInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t kOneNum = 1;
  auto x_shape_ptr = input_args[0]->isa<abstract::AbstractTuple>()
                       ? input_args[0]->cast<abstract::AbstractTuplePtr>()->BuildShape()
                       : input_args[0]->cast<abstract::AbstractListPtr>()->BuildShape();
  auto elements = input_args[0]->isa<abstract::AbstractTuple>()
                    ? input_args[0]->cast<abstract::AbstractTuplePtr>()->elements()
                    : input_args[0]->cast<abstract::AbstractListPtr>()->elements();
  (void)CheckAndConvertUtils::CheckInteger("concat element num", SizeToLong(elements.size()), kGreaterEqual, kOneNum,
                                           prim_name);
  (void)primitive->AddAttr("N", MakeValue(SizeToLong(elements.size())));
  (void)primitive->AddAttr("inputNums", MakeValue(SizeToLong(elements.size())));
  auto element0 = elements[0]->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(element0);
  auto element0_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(element0->BuildShape())[kShape];
  auto element0_rank = element0_shape.size();
  auto axis_temp = GetValue<int64_t>(primitive->GetAttr(kAxis));
  CheckAndConvertUtils::CheckInRange<int64_t>("Concat axis", axis_temp, kIncludeBoth,
                                              {-SizeToLong(element0_rank), SizeToLong(element0_rank) - kOneNum},
                                              prim_name);
  auto axis = axis_temp < 0 ? LongToSize(axis_temp + element0_rank) : LongToSize(axis_temp);
  int64_t all_shp = element0_shape[axis];
  for (size_t i = 1; i < elements.size(); ++i) {
    std::string elementi = "element" + std::to_string(i);
    auto elementi_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(elements[i]->BuildShape())[kShape];
    (void)CheckAndConvertUtils::CheckInteger(elementi + " shape rank", SizeToLong(elementi_shape.size()), kEqual,
                                             SizeToLong(element0_shape.size()), prim_name);
    for (size_t j = 0; j < element0_rank; ++j) {
      if (j != axis && elementi_shape[j] != element0_shape[j]) {
        MS_LOG(EXCEPTION) << "For '" << prim_name << "', element " << i
                          << " shape in input should concat with first element, but it can not.";
      }
    }
    all_shp = all_shp == -1 || elementi_shape[axis] == -1 ? -1 : all_shp + elementi_shape[axis];
  }
  auto ret_shape = element0_shape;
  ret_shape[axis] = all_shp;
  if (x_shape_ptr->IsDynamic()) {
    auto element0_max_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(element0->BuildShape())[kMaxShape];
    auto element0_min_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(element0->BuildShape())[kMinShape];
    auto ret_max_shape = element0_max_shape;
    auto ret_min_shape = element0_min_shape;
    for (size_t i = 1; i < elements.size(); ++i) {
      auto elementi_max_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(elements[i]->BuildShape())[kMaxShape];
      auto elementi_min_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(elements[i]->BuildShape())[kMinShape];
      ret_max_shape[axis] += elementi_max_shape[axis];
      ret_min_shape[axis] += elementi_min_shape[axis];
    }
    return std::make_shared<abstract::Shape>(ret_shape, ret_min_shape, ret_max_shape);
  } else {
    return std::make_shared<abstract::Shape>(ret_shape);
  }
}

TypePtr ConcatInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  if (!input_args[0]->isa<abstract::AbstractTuple>() && !input_args[0]->isa<abstract::AbstractList>()) {
    MS_EXCEPTION(TypeError) << "The input of Concat must be list or tuple of tensors.";
  }
  auto elements = input_args[0]->isa<abstract::AbstractTuple>()
                    ? input_args[0]->cast<abstract::AbstractTuplePtr>()->elements()
                    : input_args[0]->cast<abstract::AbstractListPtr>()->elements();
  std::map<std::string, TypePtr> types;
  for (size_t i = 0; i < elements.size(); ++i) {
    std::string elementi = "element" + std::to_string(i);
    (void)types.emplace(elementi, elements[i]->BuildType());
  }
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, common_valid_types_with_complex_and_bool, prim_name);
  return elements[0]->BuildType();
}
}  // namespace

void Concat::Init(const int64_t axis) { this->set_axis(axis); }
int64_t Concat::get_axis() const {
  auto value_ptr = this->GetAttr(kAxis);
  return GetValue<int64_t>(value_ptr);
}
void Concat::set_axis(const int64_t axis) { (void)this->AddAttr(kAxis, api::MakeValue(axis)); }

MIND_API_OPERATOR_IMPL(Concat, BaseOperator);
AbstractBasePtr ConcatInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  const int64_t kInputNum = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, primitive->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto infer_type = ConcatInferType(primitive, input_args);
  auto infer_shape = ConcatInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(Concat, prim::kPrimConcat, ConcatInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
