/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "ops/transpose.h"
#include <vector>
#include <memory>
#include <algorithm>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto x_min_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kMinShape];
  auto x_max_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kMaxShape];
  ShapeVector p_value;
  if (input_args.size() == 1) {
    ValuePtr perm = primitive->GetAttr("perm");
    auto perm_val = perm->cast<ValueTuplePtr>();
    MS_EXCEPTION_IF_NULL(perm_val);
    auto perm_val_data = perm_val->value();
    (void)std::transform(std::begin(perm_val_data), std::end(perm_val_data), std::back_inserter(p_value),
                         [](const ValuePtr &e) -> int64_t { return GetValue<int64_t>(e); });
  } else {
    auto perm_value = input_args[1]->BuildValue();
    MS_EXCEPTION_IF_NULL(perm_value);
    if (perm_value->isa<tensor::Tensor>()) {
      p_value = CheckAndConvertUtils::CheckTensorIntValue("perm value", perm_value, op_name);
    } else {
      p_value = CheckAndConvertUtils::CheckAttrTupleInt("perm value", perm_value, op_name);
    }
  }
  if (x_shape.size() != p_value.size()) {
    MS_EXCEPTION(ValueError) << "The dimension of x " << x_shape.size() << " and perm " << p_value.size()
                             << " must be equal.";
  }
  for (auto i : p_value) {
    (void)CheckAndConvertUtils::CheckInteger("perm element", i, kGreaterEqual, 0, op_name);
    (void)CheckAndConvertUtils::CheckInteger("perm element", i, kLessThan, SizeToLong(p_value.size()), op_name);
  }
  std::vector<int64_t> tmp(p_value);
  for (auto it = tmp.begin(); it != tmp.end();) {
    auto dim = *it;
    if (!tmp.empty()) {
      it = tmp.erase(it);
    }
    if (std::find(tmp.begin(), tmp.end(), dim) != tmp.end()) {
      MS_EXCEPTION(ValueError) << "The value of perm is wrong";
    }
  }
  std::vector<int64_t> in_shape(p_value);
  (void)std::transform(in_shape.begin(), in_shape.end(), in_shape.begin(), [x_shape](size_t i) { return x_shape[i]; });
  if (!x_min_shape.empty() && !x_max_shape.empty()) {
    std::vector<int64_t> min_shape;
    std::vector<int64_t> max_shape;
    for (auto i : p_value) {
      min_shape.push_back(x_min_shape[LongToSize(i)]);
      max_shape.push_back(x_max_shape[LongToSize(i)]);
    }
    return std::make_shared<abstract::Shape>(in_shape, min_shape, max_shape);
  } else {
    return std::make_shared<abstract::Shape>(in_shape);
  }
}

TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  return CheckAndConvertUtils::CheckSubClass("x", input_args[0]->BuildType(), {kTensorType}, prim->name());
}
}  // namespace

AbstractBasePtr TransposeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  (void)CheckAndConvertUtils::CheckInteger("Transpose infer", SizeToLong(input_args.size()), kGreaterEqual, 1,
                                           primitive->name());
  auto type = InferType(primitive, input_args);
  auto shape = InferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(Transpose, prim::kPrimTranspose, TransposeInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
