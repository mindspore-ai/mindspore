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

#include "ops/stack.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void Stack::set_axis(const int64_t axis) { (void)AddAttr(kAxis, api::MakeValue(axis)); }

int64_t Stack::get_axis() const { return GetValue<int64_t>(GetAttr(kAxis)); }

void Stack::Init(const int64_t axis) { this->set_axis(axis); }
namespace {
abstract::ShapePtr StackInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  if (input_args.size() != 1) {
    MS_LOG(ERROR) << "Invalid output size:" << input_args.size();
  }
  if (input_args.size() < 1) {
    MS_LOG(ERROR) << "Invalid input size " << input_args.size();
  }
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  if (!input_args[0]->isa<abstract::AbstractTuple>() && !input_args[0]->isa<abstract::AbstractList>() &&
      !input_args[0]->isa<abstract::AbstractTensor>()) {
    MS_EXCEPTION(TypeError) << "The input of Stack must be list or tuple of tensors.";
  }
  auto elements =
    input_args[0]->isa<abstract::AbstractTensor>()
      ? input_args
      : (input_args[0]->isa<abstract::AbstractTuple>() ? input_args[0]->cast<abstract::AbstractTuplePtr>()->elements()
                                                       : input_args[0]->cast<abstract::AbstractListPtr>()->elements());
  const int64_t kOneNum = 1;
  (void)CheckAndConvertUtils::CheckInteger("stack element num", SizeToLong(elements.size()), kGreaterEqual, kOneNum,
                                           primitive->name());
  auto element0 = elements[0]->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(element0);
  auto element0_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(element0->BuildShape())[kShape];
  auto element0_rank = element0_shape.size();
  auto shape_ptr = std::make_shared<abstract::Shape>(
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(elements[0]->BuildShape())[kShape]);
  auto shape_v = shape_ptr->shape();
  bool isDynamic = true;
  if (find(shape_v.begin(), shape_v.end(), -1) == shape_v.end()) isDynamic = false;
  if (!isDynamic) {
    auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(element0->BuildShape())[kShape];
    for (size_t i = 1; i < elements.size(); ++i) {
      auto input_shape_tmp = CheckAndConvertUtils::ConvertShapePtrToShapeMap(elements[i]->BuildShape())[kShape];
      if (input_shape_tmp.size() != input_shape.size()) {
        MS_EXCEPTION(ValueError) << "All input shape size should be the same!";
      }
      for (size_t j = 0; j < input_shape.size(); ++j) {
        if (input_shape_tmp.at(j) != input_shape.at(j))
          MS_EXCEPTION(ValueError) << "All input shape should be the same! " << input_shape_tmp << " And "
                                   << input_shape;
      }
    }
    std::vector<int64_t> infer_shape = input_shape;
    auto axis_temp = GetValue<int64_t>(primitive->GetAttr(kAxis));
    CheckAndConvertUtils::CheckInRange<int64_t>("Stack axis", axis_temp, kIncludeBoth,
                                                {-SizeToLong(element0_rank) - kOneNum, SizeToLong(element0_rank)},
                                                primitive->name());
    auto axis = axis_temp < 0 ? static_cast<size_t>(axis_temp) + element0_rank + 1 : LongToSize(axis_temp);
    (void)infer_shape.insert(infer_shape.begin() + axis, elements.size());
    return std::make_shared<abstract::Shape>(infer_shape);
  }
  std::vector<int64_t> output_shape;
  for (int i = 0; i < SizeToLong(shape_v.size()); i++) {
    output_shape.push_back(-1);
  }
  auto min_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(element0->BuildShape())[kMinShape];
  auto max_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(element0->BuildShape())[kMaxShape];
  if (!min_shape.size() || !min_shape.size()) {
    MS_EXCEPTION(ValueError) << "For Stack, inputs['shape'] min or max value is empty.";
  }
  return std::make_shared<abstract::Shape>(output_shape, min_shape, max_shape);
}

TypePtr StackInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  if (!input_args[0]->isa<abstract::AbstractTuple>() && !input_args[0]->isa<abstract::AbstractList>() &&
      !input_args[0]->isa<abstract::AbstractTensor>()) {
    MS_EXCEPTION(TypeError) << "The input of Stack must be list or tuple of tensors.";
  }
  auto elements =
    input_args[0]->isa<abstract::AbstractTensor>()
      ? input_args
      : (input_args[0]->isa<abstract::AbstractTuple>() ? input_args[0]->cast<abstract::AbstractTuplePtr>()->elements()
                                                       : input_args[0]->cast<abstract::AbstractListPtr>()->elements());
  const int64_t kOneNum = 1;
  (void)CheckAndConvertUtils::CheckInteger("stack element num", SizeToLong(elements.size()), kGreaterEqual, kOneNum,
                                           primitive->name());
  auto element0 = elements[0]->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(element0);
  auto infer_type0 = element0->BuildType();
  MS_EXCEPTION_IF_NULL(infer_type0);
  for (size_t i = 1; i < elements.size(); i++) {
    auto elementi = elements[i]->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(elementi);
    auto infer_typei = elementi->BuildType();
    MS_EXCEPTION_IF_NULL(infer_typei);
    if (infer_typei == infer_type0) {
      MS_EXCEPTION(TypeError) << "All input should have the same data type!input[" << i
                              << "] data type = " << infer_typei << "infer_type0= " << infer_type0;
    }
  }
  return infer_type0;
}
}  // namespace

MIND_API_OPERATOR_IMPL(Stack, BaseOperator);
AbstractBasePtr StackInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  auto infer_shape = StackInferShape(primitive, input_args);
  auto infer_type = StackInferType(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_C(kNameStack, Stack);
}  // namespace ops
}  // namespace mindspore
