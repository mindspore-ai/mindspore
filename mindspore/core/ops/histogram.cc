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

#include "ops/histogram.h"
#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr HistogramInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto bins_ptr = primitive->GetAttr(kBins);
  MS_EXCEPTION_IF_NULL(bins_ptr);
  auto min_ptr = primitive->GetAttr(kMin);
  MS_EXCEPTION_IF_NULL(min_ptr);
  auto min_attr = GetValue<float>(min_ptr);
  auto max_ptr = primitive->GetAttr(kMax);
  MS_EXCEPTION_IF_NULL(max_ptr);
  auto max_attr = GetValue<float>(max_ptr);
  if (min_attr > max_attr) {
    MS_EXCEPTION(ValueError) << "For Histogram, attr 'min' value should not greater than attr 'max'. "
                             << "but get attr min = " << min_attr << ", attr max = " << max_attr << ". ";
  }
  auto y_size = GetValue<int64_t>(bins_ptr);
  if (y_size <= 0) {
    MS_EXCEPTION(ValueError) << "For Histogram, attr 'bins' value should greater than 0. but get " << y_size;
  }
  ShapeVector y_shape = {y_size};
  return std::make_shared<abstract::Shape>(y_shape);
}

TypePtr HistogramInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kInt32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[kInputIndex0]->BuildType(), valid_types, op_name);
  return kInt32;
}
}  // namespace

void Histogram::set_bins(const int64_t bins) { (void)this->AddAttr(kBins, api::MakeValue(bins)); }
int64_t Histogram::get_bins() const { return GetValue<int64_t>(GetAttr(kBins)); }

void Histogram::set_min(const float min) { (void)this->AddAttr(kMin, api::MakeValue(min)); }
float Histogram::get_min() const { return GetValue<float>(GetAttr(kMin)); }

void Histogram::set_max(const float max) { (void)this->AddAttr(kMax, api::MakeValue(max)); }
float Histogram::get_max() const { return GetValue<float>(GetAttr(kMax)); }

AbstractBasePtr HistogramInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  constexpr int64_t input_num = 1;
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, input_num, op_name);
  auto types = HistogramInferType(primitive, input_args);
  auto shapes = HistogramInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

MIND_API_BASE_IMPL(Histogram, PrimitiveC, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(Histogram, prim::kPrimHistogram, HistogramInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
