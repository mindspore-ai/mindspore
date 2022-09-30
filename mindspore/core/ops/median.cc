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

#include "ops/median.h"
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
void Median::Init(const bool global_median, const int64_t axis, const bool keep_dims) {
  this->set_global_median(global_median);
  this->set_axis(axis);
  this->set_keep_dims(keep_dims);
}

void Median::set_global_median(const bool global_median) {
  (void)this->AddAttr(kGlobalMedian, api::MakeValue(global_median));
}

void Median::set_keep_dims(const bool keep_dims) { (void)this->AddAttr(kKeepDims, api::MakeValue(keep_dims)); }

void Median::set_axis(const int64_t &axis) {
  int64_t f = axis;
  (void)this->AddAttr(kAxis, api::MakeValue(f));
}

bool Median::get_global_median() const {
  auto value_ptr = GetAttr(kGlobalMedian);
  return GetValue<bool>(value_ptr);
}

bool Median::get_keep_dims() const {
  auto value_ptr = GetAttr(kKeepDims);
  return GetValue<bool>(value_ptr);
}

int64_t Median::get_axis() const {
  auto value_ptr = GetAttr(kAxis);
  return GetValue<int64_t>(value_ptr);
}

namespace {
abstract::TupleShapePtr MedianInferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  int64_t x_size = static_cast<int64_t>(x_shape.size());
  std::vector<int64_t> out;
  auto check_global_median = primitive->GetAttr(kGlobalMedian);
  MS_EXCEPTION_IF_NULL(check_global_median);
  bool global_median = GetValue<bool>(check_global_median);
  if (!global_median) {
    auto check_axis = primitive->GetAttr(kAxis);
    auto axis = GetValue<int64_t>(check_axis);
    auto check_keepdim = primitive->GetAttr(kKeepDims);
    bool keepdim = GetValue<bool>(check_keepdim);
    if (x_size == 0) {
      CheckAndConvertUtils::CheckInRange(kAxis, axis, kIncludeLeft, {-1, 1}, "Median");
    } else {
      CheckAndConvertUtils::CheckInRange(kAxis, axis, kIncludeLeft, {-x_size, x_size}, "Median");
    }
    if (axis < 0) {
      axis += x_size;
    }
    for (int64_t i = 0; i < x_size; i++) {
      if (i == axis) {
        if (keepdim) {
          out.push_back(1);
        }
      } else {
        out.push_back(static_cast<int64_t>(x_shape[LongToSize(i)]));
      }
    }
  }
  abstract::ShapePtr out_shape = std::make_shared<abstract::Shape>(out);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{out_shape, out_shape});
}

TuplePtr MedianInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  if (std::any_of(input_args.begin(), input_args.end(), [](const AbstractBasePtr &a) { return a == nullptr; })) {
    MS_LOG(EXCEPTION) << "nullptr";
  }
  const std::set<TypePtr> valid_types = {kInt16, kInt32, kInt64, kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[0]->BuildType(), valid_types, primitive->name());
  return std::make_shared<Tuple>(
    std::vector<TypePtr>{input_args[0]->BuildType(), std::make_shared<TensorType>(kInt64)});
}
}  // namespace

MIND_API_OPERATOR_IMPL(Median, BaseOperator);
AbstractBasePtr MedianInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = MedianInferType(primitive, input_args);
  auto infer_shape = MedianInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(Median, prim::kPrimMedian, MedianInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
