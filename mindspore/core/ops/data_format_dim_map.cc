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

#include "ops/data_format_dim_map.h"

#include <algorithm>
#include <set>

#include "ops/op_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "utils/tensor_construct_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr DataFormatDimMapInferShape(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual, 1, prim_name);
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto x_shape = shape_map[kShape];
  return std::make_shared<abstract::Shape>(x_shape);
}

TypePtr DataFormatDimMapInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual, 1, prim_name);
  auto x_type = input_args[0]->BuildType();
  const std::set<TypePtr> valid_types = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input type", x_type, valid_types, prim_name);
  return x_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(DataFormatDimMap, BaseOperator);
AbstractBasePtr DataFormatDimMapInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto infer_type = DataFormatDimMapInferType(primitive, input_args);
  auto infer_shape = DataFormatDimMapInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

void DataFormatDimMap::Init(const std::string &src_format, const std::string &dst_format) {
  this->set_src_format(src_format);
  this->set_dst_format(dst_format);
}

void DataFormatDimMap::set_src_format(const std::string &src_format) {
  CheckAndConvertUtils::CheckString(kSrcFormat, src_format, {"NHWC", "NCHW"}, this->name());
  (void)this->AddAttr(kSrcFormat, api::MakeValue(src_format));
}

std::string DataFormatDimMap::get_src_format() const {
  auto value_ptr = this->GetAttr(kSrcFormat);
  return GetValue<std::string>(value_ptr);
}

void DataFormatDimMap::set_dst_format(const std::string &dst_format) {
  CheckAndConvertUtils::CheckString(kSrcFormat, dst_format, {"NHWC", "NCHW"}, this->name());
  (void)this->AddAttr(kDstFormat, api::MakeValue(dst_format));
}

std::string DataFormatDimMap::get_dst_format() const {
  auto value_ptr = this->GetAttr(kDstFormat);
  return GetValue<std::string>(value_ptr);
}

REGISTER_PRIMITIVE_EVAL_IMPL(DataFormatDimMap, prim::kPrimDataFormatDimMap, DataFormatDimMapInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
