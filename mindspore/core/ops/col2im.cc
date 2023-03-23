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

#include "ops/col2im.h"

#include <memory>
#include <set>
#include <vector>

#include "abstract/abstract_value.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
template <typename T>
inline T DivRtn(T x, T y) {
  if (y == 0) {
    return 0;
  }
  T q = x / y;
  T r = x % y;
  if ((r != 0) && ((r < 0) != (y < 0))) {
    --q;
  }
  return q;
}

void Col2ImShapeCheck(const ShapeVector &x_shape, const std::vector<int64_t> &kernel_size,
                      const std::vector<int64_t> &dilation, const std::vector<int64_t> &padding,
                      const std::vector<int64_t> &strides, const int64_t output_height, const int64_t output_width) {
  const int64_t kernel_height = kernel_size.front();
  const int64_t kernel_width = kernel_size.back();

  const int64_t dilation_height = dilation.front();
  const int64_t dilation_width = dilation.back();

  const int64_t pad_height = padding.front();
  const int64_t pad_width = padding.back();

  const int64_t stride_height = strides.front();
  const int64_t stride_width = strides.back();

  const int64_t n_input_plane = x_shape[kInputIndex2];
  if (n_input_plane != (kernel_width * kernel_height)) {
    MS_EXCEPTION(ValueError) << "For Col2Im, expected size of input's dimension 2 to be divisible by the product of "
                                "kernel_size, but got input.size(2)="
                             << n_input_plane << " and kernel_size=(" << kernel_height << ", " << kernel_width << ").";
  }

  const int64_t input_length = x_shape[kInputIndex3];
  constexpr int64_t kInt64Number2 = 2;
  const int64_t n_blocks_height =
    DivRtn<int64_t>(output_height + kInt64Number2 * pad_height - dilation_height * (kernel_height - 1) - 1,
                    stride_height) +
    1;
  const int64_t n_blocks_width =
    DivRtn<int64_t>(output_width + kInt64Number2 * pad_width - dilation_width * (kernel_width - 1) - 1, stride_width) +
    1;
  if (input_length != (n_blocks_height * n_blocks_width)) {
    MS_EXCEPTION(ValueError) << "For 'Col2Im', size of input's 4th dimension must be equal to calculated number of "
                                "sliding blocks, but got input.size["
                             << kInputIndex3 << "]: " << input_length
                             << ", calculated number of sliding blocks: " << n_blocks_height * n_blocks_width
                             << ". Please refer to Mindspore official website API docs for details about how the "
                                "number of sliding blocks is calculated.";
  }
  if (n_blocks_height <= 0 || n_blocks_width <= 0) {
    MS_EXCEPTION(ValueError) << "For Col2Im, given output_size=(" << output_height << ", " << output_width
                             << "), kernel_size=(" << kernel_height << ", " << kernel_width << "), dilation=("
                             << dilation_height << ", " << dilation_width << "), padding=(" << pad_height << ", "
                             << pad_width << "), stride=(" << stride_height << ", " << stride_width
                             << "), calculated shape of the array of sliding blocks as (" << n_blocks_height << ", "
                             << n_blocks_width << "), which is too small (non-positive)";
  }
}

abstract::ShapePtr Col2ImInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto output_size_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];

  constexpr int64_t output_size_dim0 = 2;
  constexpr int64_t x_size = 4;

  if (output_size_shape.size() != 1) {
    MS_EXCEPTION(ValueError) << "For 'Col2Im', 'output_size' must be a 1D Tensor, but got a "
                             << output_size_shape.size() << "-D Tensor.";
  }
  if (!IsDynamic(output_size_shape) && output_size_shape[0] != output_size_dim0) {
    MS_EXCEPTION(ValueError)
      << "For 'Col2Im', 'output_size' must be a 1D Tensor with 2 elements, but got a 1-D Tensor with "
      << output_size_shape[0] << " elements.";
  }
  if (!IsDynamicRank(x_shape)) {
    (void)CheckAndConvertUtils::CheckInteger("x dimension", SizeToLong(x_shape.size()), kEqual, x_size, op_name);
  }

  constexpr int64_t attr_size = 2;

  auto kernel_size_ptr = primitive->GetAttr(kKernelSize);
  MS_EXCEPTION_IF_NULL(kernel_size_ptr);
  auto kernel_size = GetValue<std::vector<int64_t>>(kernel_size_ptr);

  auto dialtion_ptr = primitive->GetAttr(kDilation);
  MS_EXCEPTION_IF_NULL(dialtion_ptr);
  auto dilation = GetValue<std::vector<int64_t>>(dialtion_ptr);

  auto padding_ptr = primitive->GetAttr(kPadding);
  MS_EXCEPTION_IF_NULL(padding_ptr);
  auto padding = GetValue<std::vector<int64_t>>(padding_ptr);

  auto stride_ptr = primitive->GetAttr(kStride);
  MS_EXCEPTION_IF_NULL(stride_ptr);
  auto stride = GetValue<std::vector<int64_t>>(stride_ptr);

  (void)CheckAndConvertUtils::CheckInteger(kKernelSize, SizeToLong(kernel_size.size()), kEqual, attr_size, op_name);
  (void)CheckAndConvertUtils::CheckInteger(kDilation, SizeToLong(dilation.size()), kEqual, attr_size, op_name);
  (void)CheckAndConvertUtils::CheckInteger(kPadding, SizeToLong(padding.size()), kEqual, attr_size, op_name);
  (void)CheckAndConvertUtils::CheckInteger(kStride, SizeToLong(stride.size()), kEqual, attr_size, op_name);

  (void)CheckAndConvertUtils::CheckPositiveVectorExcludeZero(kKernelSize, kernel_size, op_name);
  (void)CheckAndConvertUtils::CheckPositiveVectorExcludeZero(kDilation, dilation, op_name);
  (void)CheckAndConvertUtils::CheckPositiveVector(kPadding, padding, op_name);
  (void)CheckAndConvertUtils::CheckPositiveVectorExcludeZero(kStride, stride, op_name);

  auto output_size_ptr = input_args[kInputIndex1];
  MS_EXCEPTION_IF_NULL(output_size_ptr);
  auto output_size_value = GetShapeValue(primitive, output_size_ptr);

  auto is_dynamic_rank = IsDynamicRank(x_shape) || IsDynamicRank(output_size_value);
  if (is_dynamic_rank) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>(x_size, abstract::Shape::kShapeDimAny));
  }

  if (!(IsDynamic(x_shape) || !(IsValueKnown(input_args[1]->BuildValue())))) {
    Col2ImShapeCheck(x_shape, kernel_size, dilation, padding, stride, output_size_value[kInputIndex0],
                     output_size_value[kInputIndex1]);
  }

  ShapeVector y_shape = {x_shape[0], x_shape[1], output_size_value[0], output_size_value[1]};
  return std::make_shared<abstract::Shape>(y_shape);
}

TypePtr Col2ImInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  return CheckAndConvertUtils::GetTensorInputType(kNameCol2Im, input_args, kInputIndex0);
}
}  // namespace

AbstractBasePtr Col2ImInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  constexpr int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);
  auto types = Col2ImInferType(primitive, input_args);
  auto shapes = Col2ImInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

MIND_API_OPERATOR_IMPL(Col2Im, BaseOperator);

// AG means auto generated
class MIND_API AGCol2ImInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return Col2ImInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return Col2ImInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return Col2ImInfer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {1}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Col2Im, prim::kPrimCol2Im, AGCol2ImInfer, false);
}  // namespace ops
}  // namespace mindspore
