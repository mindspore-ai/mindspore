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

#include "ops/grad/max_pool_grad_with_argmax.h"

#include <algorithm>
#include <string>
#include <memory>
#include <set>
#include <vector>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void MaxPoolGradWithArgmax::set_pad_mode(const PadMode &pad_mode) {
  int64_t swi = pad_mode;
  (void)this->AddAttr(kPadMode, api::MakeValue(swi));
}

PadMode MaxPoolGradWithArgmax::get_pad_mode() const {
  auto value_ptr = GetAttr(kPadMode);
  MS_EXCEPTION_IF_NULL(value_ptr);
  auto mode_str = GetValue<std::string>(value_ptr);
  (void)std::transform(mode_str.begin(), mode_str.end(), mode_str.begin(), ::toupper);
  MS_EXCEPTION_IF_CHECK_FAIL((mode_str == "SAME" || mode_str == "VALID"),
                             "MaxPoolGradWithArgmax only supports pad mode 'SAME' or 'VALID', but get " + mode_str);
  return mode_str == "SAME" ? PadMode::SAME : PadMode::VALID;
}

void MaxPoolGradWithArgmax::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  (void)this->AddAttr(
    kKernelSize, api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kKernelSize, kernel_size, this->name())));
}
std::vector<int64_t> MaxPoolGradWithArgmax::get_kernel_size() const {
  return GetValue<std::vector<int64_t>>(GetAttr(kKernelSize));
}

void MaxPoolGradWithArgmax::set_strides(const std::vector<int64_t> &strides) {
  (void)this->AddAttr(kStrides,
                      api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kStrides, strides, this->name())));
}
std::vector<int64_t> MaxPoolGradWithArgmax::get_strides() const {
  return GetValue<std::vector<int64_t>>(GetAttr(kStrides));
}

void MaxPoolGradWithArgmax::Init(const std::vector<int64_t> &kernel_size, const std::vector<int64_t> &stride,
                                 const PadMode &pad_mode, const Format &format) {
  this->set_pad_mode(pad_mode);
  this->set_kernel_size(kernel_size);
  this->set_strides(stride);
}

namespace {
abstract::ShapePtr MaxPoolGradWithArgmaxInferShape(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) {
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kDim0]->BuildShape())[kShape];
  // ToSupport Dynamic rank
  if (IsDynamicRank(x_shape)) {
    // The input tensor of Primitive MaxPoolGradWithArgmax must be a 4-D tensor and the data format is NCHW/NHWC.
    // So DynamicRank can transfer to 4-D dynamic shape
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{-1, -1, -1, -1});
  }
  constexpr int64_t kXRank = 4;
  CheckAndConvertUtils::CheckInteger("x_rank", SizeToLong(x_shape.size()), kEqual, kXRank, kNameMaxPoolGradWithArgmax);
  return std::make_shared<abstract::Shape>(x_shape);
}

TypePtr MaxPoolGradWithArgmaxInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto name = prim->name();
  const std::set<TypePtr> valid_grad_types = {kInt8,   kInt16,  kInt64,   kUInt8,   kUInt16,
                                              kUInt32, kUInt64, kFloat16, kFloat32, kFloat64};
  auto grad_type = input_args[kDim1]->BuildType();
  auto inferred_type = CheckAndConvertUtils::CheckTensorTypeValid("x", grad_type, valid_grad_types, name);
  return inferred_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(MaxPoolGradWithArgmax, BaseOperator);

AbstractBasePtr MaxPoolGradWithArgmaxInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto maxpool_with_argmax_infer_type = MaxPoolGradWithArgmaxInferType(primitive, input_args);
  auto maxpool_with_argmax_infer_shape = MaxPoolGradWithArgmaxInferShape(primitive, input_args);
  return std::make_shared<abstract::AbstractTensor>(maxpool_with_argmax_infer_type, maxpool_with_argmax_infer_shape);
}

// AG means auto generated
class MIND_API AGMaxPoolGradWithArgmaxInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return MaxPoolGradWithArgmaxInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return MaxPoolGradWithArgmaxInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return MaxPoolGradWithArgmaxInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(MaxPoolGradWithArgmax, prim::kPrimMaxPoolGradWithArgmax, AGMaxPoolGradWithArgmaxInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
