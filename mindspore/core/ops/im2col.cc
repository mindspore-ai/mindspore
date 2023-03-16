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

#include "ops/im2col.h"
#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "abstract/abstract_value.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void Im2Col::set_ksizes(const std::vector<int64_t> &ksizes) { (void)this->AddAttr(kKsizes, api::MakeValue(ksizes)); }

std::vector<int64_t> Im2Col::get_ksizes() const {
  auto value_ptr = GetAttr(kKsizes);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void Im2Col::set_strides(const std::vector<int64_t> &strides) {
  (void)this->AddAttr(kStrides, api::MakeValue(strides));
}

std::vector<int64_t> Im2Col::get_strides() const {
  auto value_ptr = GetAttr(kStrides);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void Im2Col::set_dilations(const std::vector<int64_t> &dilations) {
  (void)this->AddAttr(kDilations, api::MakeValue(dilations));
}

std::vector<int64_t> Im2Col::get_dilations() const {
  auto value_ptr = GetAttr(kDilations);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void Im2Col::set_pads(const std::vector<int64_t> &pads) { (void)this->AddAttr(kPads, api::MakeValue(pads)); }

std::vector<int64_t> Im2Col::get_pads() const {
  auto value_ptr = GetAttr(kPads);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

namespace {
abstract::ShapePtr Im2ColInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  constexpr size_t size_2 = 2;
  constexpr size_t size_4 = 4;
  auto in_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  if (IsDynamic(in_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{size_4, abstract::Shape::kShapeDimAny});
  }

  (void)CheckAndConvertUtils::CheckInteger("dimension of input x", SizeToLong(in_shape.size()), kEqual,
                                           SizeToLong(size_4), op_name);
  (void)CheckAndConvertUtils::CheckPositiveVectorExcludeZero("spatial size of input", in_shape, op_name);

  auto ksizes_ptr = primitive->GetAttr(kKsizes);
  MS_EXCEPTION_IF_NULL(ksizes_ptr);
  auto ksizes = GetValue<std::vector<int64_t>>(ksizes_ptr);

  auto strides_ptr = primitive->GetAttr(kStrides);
  MS_EXCEPTION_IF_NULL(strides_ptr);
  auto strides = GetValue<std::vector<int64_t>>(strides_ptr);

  auto dilations_ptr = primitive->GetAttr(kDilations);
  MS_EXCEPTION_IF_NULL(dilations_ptr);
  auto dilations = GetValue<std::vector<int64_t>>(dilations_ptr);

  auto pads_ptr = primitive->GetAttr(kPads);
  MS_EXCEPTION_IF_NULL(pads_ptr);
  auto pads = GetValue<std::vector<int64_t>>(pads_ptr);

  if (ksizes.empty() || ksizes.size() > size_2) {
    MS_EXCEPTION(ValueError)
      << "For Im2Col, the element number of ksizes must be 1 or 2 when x_format only support NCHW, but get "
      << ksizes.size() << " elements in ksizes.";
  }
  if (strides.empty() || strides.size() > size_2) {
    MS_EXCEPTION(ValueError)
      << "For Im2Col, the element number of strides must be 1 or 2 when x_format only support NCHW, but get "
      << strides.size() << " elements in strides.";
  }
  if (dilations.empty() || dilations.size() > size_2) {
    MS_EXCEPTION(ValueError)
      << "For Im2Col, the element number of dilations must be 1 or 2 when x_format only support NCHW, but get "
      << dilations.size() << " elements in dilations.";
  }
  (void)CheckAndConvertUtils::CheckPositiveVectorExcludeZero(kKsizes, ksizes, op_name);
  (void)CheckAndConvertUtils::CheckPositiveVectorExcludeZero(kDilations, dilations, op_name);
  (void)CheckAndConvertUtils::CheckPositiveVectorExcludeZero(kStrides, strides, op_name);

  const int64_t in_n = in_shape[kInputIndex0];
  const int64_t in_c = in_shape[kInputIndex1];
  const int64_t in_h = in_shape[kInputIndex2];
  const int64_t in_w = in_shape[kInputIndex3];

  int64_t filter_h = ksizes.front();
  int64_t filter_w = ksizes.back();
  int64_t dilation_h = dilations.front();
  int64_t dilation_w = dilations.back();
  int64_t stride_h = strides.front();
  MS_EXCEPTION_IF_ZERO("stride_h", stride_h);
  int64_t stride_w = strides.back();
  MS_EXCEPTION_IF_ZERO("stride_w", stride_w);

  int64_t out_h{0}, out_w{0}, total_block{0}, kernel_product{0};
  int64_t pad_h{0}, pad_w{0};

  (void)CheckAndConvertUtils::CheckPositiveVector(kPads, pads, op_name);
  if (!pads.empty() && (pads.size() <= size_2 || pads.size() == size_4)) {
    pad_h = pads.front();
    pad_w = pads.back();
  } else {
    MS_EXCEPTION(ValueError) << "For Im2Col, the size of pads must be 1 or 2, but get " << pads.size()
                             << "elements in pads.";
  }
  if (pads.size() == size_4 && (pads[kInputIndex0] != pads[kInputIndex1] || pads[kInputIndex2] != pads[kInputIndex3])) {
    MS_EXCEPTION(ValueError) << "For Im2Col, the 1st and 2nd / 3rd and 4th padding value should be same, but got "
                             << pads;
  }
  out_h = (in_h + pad_h + pad_h - (dilation_h * (filter_h - 1) + 1)) / stride_h + 1;
  out_w = (in_w + pad_w + pad_w - (dilation_w * (filter_w - 1) + 1)) / stride_w + 1;

  kernel_product = filter_h * filter_w;
  total_block = out_h * out_w;
  if (out_h < 1 || out_w < 1) {
    MS_EXCEPTION(ValueError) << "For Im2Col, given input with spatial size (" << in_n << ", " << in_c << ", " << in_h
                             << ", " << in_w << "), ksizes=(" << filter_h << ", " << filter_w << "), dilation=("
                             << dilation_h << ", " << dilation_w << ", pads=(" << pads
                             << "), calculated shape of output as (" << in_h << ", " << in_c << ", " << kernel_product
                             << ", " << total_block << "), which is too small (non-positive).";
  }

  // current only support NCHW
  std::vector<int64_t> out_shape = {in_n, in_c, kernel_product, total_block};
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr Im2ColInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  return CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[kInputIndex0]->BuildType(),
                                                    common_valid_types_with_complex, primitive->name());
}
}  // namespace

AbstractBasePtr Im2ColInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  constexpr int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);
  auto infer_type = Im2ColInferType(primitive, input_args);
  auto infer_shape = Im2ColInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(Im2Col, BaseOperator);

// AG means auto generated
class MIND_API AGIm2ColInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return Im2ColInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return Im2ColInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return Im2ColInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Im2Col, prim::kPrimIm2Col, AGIm2ColInfer, false);
}  // namespace ops
}  // namespace mindspore
