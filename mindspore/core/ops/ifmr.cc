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

#include "ops/ifmr.h"
#include <memory>
#include <set>
#include <vector>
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kIFMRInputSize = 4;
abstract::TupleShapePtr IFMRInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();

  (void)CheckAndConvertUtils::CheckInteger("[input number]", static_cast<int64_t>(input_args.size()), kEqual,
                                           kIFMRInputSize, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }

  MS_EXCEPTION_IF_CHECK_FAIL(input_args[kInputIndex1]->BuildShape()->isa<abstract::Shape>(), "data_min's shape wrong.");
  auto data_min_shape_element = input_args[kInputIndex1]->BuildShape()->cast<abstract::ShapePtr>();
  auto data_min_shape = data_min_shape_element->shape();
  (void)CheckAndConvertUtils::CheckInteger("data_min's rank", SizeToLong(data_min_shape.size()), kEqual, 1, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("data_min's first-dim", data_min_shape.front(), kEqual, 1, prim_name);
  MS_EXCEPTION_IF_CHECK_FAIL(input_args[kInputIndex2]->BuildShape()->isa<abstract::Shape>(), "data_max's shape wrong.");
  auto data_max_shape_element = input_args[kInputIndex2]->BuildShape()->cast<abstract::ShapePtr>();
  auto data_max_shape = data_max_shape_element->shape();
  (void)CheckAndConvertUtils::CheckInteger("data_max's rank", SizeToLong(data_max_shape.size()), kEqual, 1, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("data_max's first-dim", data_max_shape.front(), kEqual, 1, prim_name);
  MS_EXCEPTION_IF_CHECK_FAIL(input_args[kInputIndex3]->BuildShape()->isa<abstract::Shape>(), "cumsum's shape wrong.");
  auto cumsum_shape_element = input_args[kInputIndex3]->BuildShape()->cast<abstract::ShapePtr>();
  auto cumsum_shape = cumsum_shape_element->shape();
  (void)CheckAndConvertUtils::CheckInteger("cumsum's rank", SizeToLong(cumsum_shape.size()), kEqual, 1, prim_name);

  ShapeVector out_shape{1};
  auto out_shape_ptr = std::make_shared<abstract::Shape>(out_shape, out_shape, out_shape);
  abstract::BaseShapePtrList out_shape_list = {out_shape_ptr, out_shape_ptr};
  return std::make_shared<abstract::TupleShape>(out_shape_list);
}

TuplePtr IFMRInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();

  (void)CheckAndConvertUtils::CheckInteger("[input number]", static_cast<int64_t>(input_args.size()), kEqual,
                                           kIFMRInputSize, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  std::set<TypePtr> valid_type = {kFloat16, kFloat32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("data", input_args[kInputIndex0]->BuildType(), valid_type,
                                                   prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("data_min", input_args[kInputIndex1]->BuildType(), valid_type,
                                                   prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("data_max", input_args[kInputIndex2]->BuildType(), valid_type,
                                                   prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("cumsum", input_args[kInputIndex3]->BuildType(), {kInt32},
                                                   prim_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{kFloat32, kFloat32});
}
}  // namespace

void IFMR::Init(const float min_percentile, const float max_percentile, const std::vector<float> &search_range,
                const float search_step, const bool with_offset) {
  this->set_min_percentile(min_percentile);
  this->set_max_percentile(max_percentile);
  this->set_search_range(search_range);
  this->set_search_step(search_step);
  this->set_with_offset(with_offset);
}

void IFMR::set_min_percentile(const float min_percentile) {
  (void)this->AddAttr(kMinPercentile, api::MakeValue(min_percentile));
}

void IFMR::set_max_percentile(const float max_percentile) {
  (void)this->AddAttr(kMaxPercentile, api::MakeValue(max_percentile));
}

void IFMR::set_search_range(const std::vector<float> &search_range) {
  (void)this->AddAttr(kSearchRange, api::MakeValue(search_range));
}

void IFMR::set_search_step(const float search_step) { (void)this->AddAttr(kSearchStep, api::MakeValue(search_step)); }

void IFMR::set_with_offset(const bool with_offset) { (void)this->AddAttr(kWithOffset, api::MakeValue(with_offset)); }

float IFMR::get_min_percentile() const { return GetValue<float>(GetAttr(kMinPercentile)); }

float IFMR::get_max_percentile() const { return GetValue<float>(GetAttr(kMaxPercentile)); }

std::vector<float> IFMR::get_search_range() const { return GetValue<std::vector<float>>(GetAttr(kSearchRange)); }

float IFMR::get_search_step() const { return GetValue<float>(GetAttr(kSearchStep)); }

bool IFMR::get_with_offset() const { return GetValue<bool>(GetAttr(kWithOffset)); }

AbstractBasePtr IFMRInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  auto infer_type = IFMRInferType(primitive, input_args);
  auto infer_shape = IFMRInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
MIND_API_OPERATOR_IMPL(IFMR, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(IFMR, prim::kPrimIFMR, IFMRInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
