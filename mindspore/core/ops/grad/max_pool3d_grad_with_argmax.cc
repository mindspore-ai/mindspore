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

#include "ops/grad/max_pool3d_grad_with_argmax.h"

#include <map>
#include <algorithm>
#include <set>
#include <utility>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void MaxPool3DGradWithArgmax::Init(const std::vector<int64_t> &kernel_size, const std::vector<int64_t> &strides,
                                   const std::vector<int64_t> &pads, const std::vector<int64_t> &dialtion,
                                   bool ceil_mode, const Format &format) {
  set_kernel_size(kernel_size);
  set_strides(strides);
  set_pads(pads);
  set_dilation(dialtion);
  set_ceil_mode(ceil_mode);
  set_format(format);
}

void MaxPool3DGradWithArgmax::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  (void)AddAttr(kKSize, api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kKSize, kernel_size, name())));
}

void MaxPool3DGradWithArgmax::set_strides(const std::vector<int64_t> &strides) {
  (void)AddAttr(kStrides, api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kStrides, strides, name())));
}

void MaxPool3DGradWithArgmax::set_pads(const std::vector<int64_t> &pads) { (void)AddAttr(kPads, api::MakeValue(pads)); }

void MaxPool3DGradWithArgmax::set_dilation(const std::vector<int64_t> &dilation) {
  int64_t kMinDilationSize = 3;
  int64_t size = SizeToLong(dilation.size());
  (void)CheckAndConvertUtils::CheckInteger("dilation_shape", size, kGreaterThan, kMinDilationSize, name());
  std::vector<int64_t> d;
  for (int64_t i = size - kMinDilationSize; i < size; i++) {
    d.push_back(dilation[i]);
  }
  (void)AddAttr(kDilation, api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kDilation, d, name())));
}

void MaxPool3DGradWithArgmax::set_ceil_mode(bool ceil_mode) { (void)AddAttr(kCeilMode, api::MakeValue(ceil_mode)); }

void MaxPool3DGradWithArgmax::set_format(const Format &format) {
  int64_t f = format;
  (void)AddAttr(kFormat, api::MakeValue(f));
}

std::vector<int64_t> MaxPool3DGradWithArgmax::get_kernel_size() const {
  return GetValue<std::vector<int64_t>>(GetAttr(kKSize));
}

std::vector<int64_t> MaxPool3DGradWithArgmax::get_strides() const {
  return GetValue<std::vector<int64_t>>(GetAttr(kStrides));
}

std::vector<int64_t> MaxPool3DGradWithArgmax::get_pads() const {
  return GetValue<std::vector<int64_t>>(GetAttr(kPads));
}

std::vector<int64_t> MaxPool3DGradWithArgmax::get_dilation() const {
  return GetValue<std::vector<int64_t>>(GetAttr(kDilation));
}

bool MaxPool3DGradWithArgmax::get_ceil_mode() const { return GetValue<bool>(GetAttr(kCeilMode)); }

Format MaxPool3DGradWithArgmax::get_format() const {
  auto value_ptr = GetAttr(kFormat);
  MS_EXCEPTION_IF_NULL(value_ptr);
  if (!value_ptr->isa<mindspore::api::StringImm>()) {
    return Format(GetValue<int64_t>(value_ptr));
  }
  static const std::map<std::string, int64_t> valid_dataformat = {
    {"NCDHW", Format::NCDHW},
  };
  auto attr_value_str = GetValue<std::string>(value_ptr);
  (void)std::transform(attr_value_str.begin(), attr_value_str.end(), attr_value_str.begin(), toupper);
  auto iter = valid_dataformat.find(attr_value_str);
  if (iter == valid_dataformat.end()) {
    MS_LOG(EXCEPTION) << "Invalid format " << attr_value_str << ", use NCDHW";
  }
  return Format(iter->second);
}

namespace {
TypePtr MaxPool3DGradWithArgmaxInferType(const PrimitivePtr &prim,
                                         const std::vector<abstract::AbstractBasePtr> &input_args) {
  (void)CheckAndConvertUtils::CheckTensorTypeValid("argmax", input_args[kInputIndex2]->BuildType(), {kInt64, kInt32},
                                                   prim->name());
  const std::set<TypePtr> valid_types = {kInt8,   kInt16,  kInt32,   kInt64,   kUInt8,  kUInt16,
                                         kUInt32, kUInt64, kFloat16, kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[kInputIndex0]->BuildType(), valid_types,
                                                   prim->name());
  return input_args[0]->BuildType();
}

abstract::ShapePtr MaxPool3DGradWithArgmaxInferShape(const PrimitivePtr &prim,
                                                     const std::vector<abstract::AbstractBasePtr> &input_args) {
  const int64_t kInputDims = 5;
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto argmax_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  if (IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::Shape>(x_shape);
  }
  (void)CheckAndConvertUtils::CheckInteger("x_shape", x_shape.size(), kEqual, kInputDims, prim->name());
  (void)CheckAndConvertUtils::CheckInteger("argmax_shape", argmax_shape.size(), kEqual, kInputDims, prim->name());
  return std::make_shared<abstract::Shape>(x_shape);
}
}  // namespace

abstract::AbstractBasePtr MaxPool3DGradWithArgmaxInfer(const abstract::AnalysisEnginePtr &,
                                                       const PrimitivePtr &primitive,
                                                       const std::vector<abstract::AbstractBasePtr> &input_args) {
  const int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = MaxPool3DGradWithArgmaxInferType(primitive, input_args);
  auto infer_shape = MaxPool3DGradWithArgmaxInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(MaxPool3DGradWithArgmax, BaseOperator);

// AG means auto generated
class MIND_API AGMaxPool3DGradWithArgmaxInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return MaxPool3DGradWithArgmaxInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return MaxPool3DGradWithArgmaxInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return MaxPool3DGradWithArgmaxInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(MaxPool3DGradWithArgmax, prim::kPrimMaxPool3DGradWithArgmax,
                                 AGMaxPool3DGradWithArgmaxInfer, false);
}  // namespace ops
}  // namespace mindspore
