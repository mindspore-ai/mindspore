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

#include "ops/avg_pool.h"

#include <string>
#include <algorithm>
#include <memory>
#include <vector>
#include <cmath>
#include <map>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ir/value.h"
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
void AvgPool::set_pad_mode(const PadMode &pad_mode) {
  int64_t swi = pad_mode;
  (void)this->AddAttr(kPadMode, api::MakeValue(swi));
}

PadMode AvgPool::get_pad_mode() const { return PadMode(GetValue<int64_t>(GetAttr(kPadMode))); }
void AvgPool::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  (void)this->AddAttr(
    kKernelSize, api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kKernelSize, kernel_size, this->name())));
}

std::vector<int64_t> AvgPool::get_kernel_size() const { return GetValue<std::vector<int64_t>>(GetAttr(kKernelSize)); }
void AvgPool::set_strides(const std::vector<int64_t> &strides) {
  (void)this->AddAttr(kStrides,
                      api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kStrides, strides, this->name())));
}

std::vector<int64_t> AvgPool::get_strides() const { return GetValue<std::vector<int64_t>>(GetAttr(kStrides)); }

void AvgPool::set_format(const Format &format) {
  int64_t f = format;
  (void)this->AddAttr(kFormat, api::MakeValue(f));
}

Format AvgPool::get_format() const { return Format(GetValue<int64_t>(GetAttr(kFormat))); }

void AvgPool::set_pad(const std::vector<int64_t> &pad) { (void)this->AddAttr(kPad, api::MakeValue(pad)); }

std::vector<int64_t> AvgPool::get_pad() const {
  auto value_ptr = GetAttr(kPad);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void AvgPool::set_round_mode(const RoundMode &round_mode) {
  int64_t swi = round_mode;
  (void)this->AddAttr(kRoundMode, api::MakeValue(swi));
}

RoundMode AvgPool::get_round_mode() const {
  auto value_ptr = GetAttr(kRoundMode);
  return RoundMode(GetValue<int64_t>(value_ptr));
}

void AvgPool::Init(const std::vector<int64_t> &kernel_size, const std::vector<int64_t> &stride, const PadMode &pad_mode,
                   const Format &format, const std::vector<int64_t> &pad, const RoundMode &round_mode) {
  this->set_pad_mode(pad_mode);
  this->set_kernel_size(kernel_size);
  this->set_strides(stride);
  this->set_format(format);
  this->set_pad(pad);
  this->set_round_mode(round_mode);
}

namespace {
abstract::ShapePtr AvgPoolInferShape(const PrimitivePtr &primitive,
                                     const std::vector<abstract::AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto in_shape = shape_map[kShape];
  if (IsDynamicRank(in_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
  }
  int64_t format = CheckAndConvertUtils::GetAndCheckFormat(primitive->GetAttr("format"));
  mindspore::Format format_enum = static_cast<mindspore::Format>(format);
  const int64_t x_size = 4;
  const int64_t attr_size = 4;
  (void)CheckAndConvertUtils::CheckInteger("x_rank", SizeToLong(in_shape.size()), kEqual, x_size, op_name);
  if (format_enum == NHWC) {
    in_shape = {in_shape[0], in_shape[3], in_shape[1], in_shape[2]};
  }

  auto kernel_size = GetValue<std::vector<int64_t>>(primitive->GetAttr(kKernelSize));
  int64_t pad_mode;
  CheckAndConvertUtils::GetPadModEnumValue(primitive->GetAttr(kPadMode), &pad_mode, true);
  mindspore::PadMode pad_mode_enum = static_cast<mindspore::PadMode>(pad_mode);

  auto batch = in_shape[0];
  auto channel = in_shape[1];
  auto in_h = in_shape[2];
  auto in_w = in_shape[3];

  auto strides = GetValue<std::vector<int64_t>>(primitive->GetAttr(kStrides));
  (void)CheckAndConvertUtils::CheckInteger("kernel", SizeToLong(kernel_size.size()), kEqual, attr_size, op_name);
  (void)CheckAndConvertUtils::CheckInteger("strides", SizeToLong(strides.size()), kEqual, attr_size, op_name);
  if (std::any_of(strides.begin(), strides.end(), [](int64_t stride) { return stride <= 0; })) {
    MS_LOG(EXCEPTION) << "For '" << op_name << "', strides must be positive, but it's " << strides << ".";
  }
  if (std::any_of(kernel_size.begin(), kernel_size.end(), [](int64_t size) { return size <= 0; })) {
    MS_LOG(EXCEPTION) << "For '" << op_name << "', Kernel size must be positive, but it's " << kernel_size << ".";
  }

  auto kernel_h = kernel_size[2];
  auto kernel_w = kernel_size[3];
  auto stride_h = strides[2];
  auto stride_w = strides[3];
  int64_t out_h = abstract::Shape::kShapeDimAny;
  int64_t out_w = abstract::Shape::kShapeDimAny;

  if (pad_mode_enum == VALID) {
    out_h = in_h == abstract::Shape::kShapeDimAny
              ? abstract::Shape::kShapeDimAny
              : (static_cast<int64_t>(std::ceil((in_h - (kernel_h - 1)) / static_cast<float>(stride_h))));
    out_w = in_w == abstract::Shape::kShapeDimAny
              ? abstract::Shape::kShapeDimAny
              : (static_cast<int64_t>(std::ceil((in_w - (kernel_w - 1)) / static_cast<float>(stride_w))));
  } else if (pad_mode_enum == SAME) {
    out_h = in_h == abstract::Shape::kShapeDimAny
              ? abstract::Shape::kShapeDimAny
              : (static_cast<int64_t>(std::ceil(in_h / static_cast<float>(stride_h))));
    out_w = in_w == abstract::Shape::kShapeDimAny
              ? abstract::Shape::kShapeDimAny
              : (static_cast<int64_t>(std::ceil(in_w / static_cast<float>(stride_w))));
  }
  std::vector<int64_t> out_shape = {batch, channel, out_h, out_w};

  if (format_enum == NHWC) {
    out_shape = {batch, out_h, out_w, channel};
  }

  auto var_shape = input_args[kInputIndex0]->BuildShape();
  auto var_shape_ptr = var_shape->cast<abstract::ShapePtr>();
  if (!var_shape_ptr->IsDynamic()) {
    if (std::any_of(out_shape.begin(), out_shape.end(), [](int64_t a) { return a <= 0; })) {
      MS_LOG(EXCEPTION) << "For Out_Shape values must be positive, but it's " << out_shape << ".";
    }
  }

  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr AvgPoolInferType(const PrimitivePtr &prim, const std::vector<abstract::AbstractBasePtr> &input_args) {
  if (std::any_of(input_args.begin(), input_args.end(),
                  [](const abstract::AbstractBasePtr arg) { return arg == nullptr; })) {
    MS_LOG(EXCEPTION) << "For '" << prim->name()
                      << "', the input args userd for infer shape and type is necessary, but got missing it.";
  }
  auto name = prim->name();
  auto input_type = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(input_type);
  auto input_tensor_type = input_type->cast<TensorTypePtr>();
  if (input_tensor_type == nullptr) {
    MS_LOG_EXCEPTION << "For '" << name << "', the input must be a tensor but got " << input_type->ToString() << ".";
  }
  return input_tensor_type->element();
}
}  // namespace

abstract::AbstractBasePtr AvgPoolInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, input_num, primitive->name());
  auto infer_type = AvgPoolInferType(primitive, input_args);
  auto infer_shape = AvgPoolInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
MIND_API_OPERATOR_IMPL(AvgPool, BaseOperator);

// AG means auto generated
class MIND_API AGAvgPoolInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return AvgPoolInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return AvgPoolInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return AvgPoolInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(AvgPool, prim::kPrimAvgPool, AGAvgPoolInfer, false);
}  // namespace ops
}  // namespace mindspore
