/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "ops/ops_func_impl/extract_image_patches.h"
#include <string>
#include <memory>
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "mindapi/base/types.h"
#include "abstract/dshape.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr auto kExtractImagePatchesInputDims = 4;
constexpr auto kExtractImagePatchesArgDims = 2;
}  // namespace
std::vector<int64_t> CheckAndGetExtractImagePatchesList(string prim_name, string param_name,
                                                        AbstractBasePtr input_arg) {
  std::vector<int64_t> out(2, abstract::Shape::kShapeDimAny);
  auto arg_value = GetArrayValue<int64_t>(input_arg->GetValue());
  if (!arg_value.has_value()) {
    return out;
  }

  MS_CHECK_VALUE(arg_value->size() == kExtractImagePatchesArgDims,
                 "For " + prim_name + "the rank of " + param_name + " must be 2.");

  auto arg_v = arg_value.value();
  for (size_t i = 0; i < kExtractImagePatchesArgDims; i++) {
    if (!arg_v.IsValueUnknown(i)) {
      out[i] = arg_v[i];

      auto param_name_v = param_name + (i == 0 ? "_row" : "_col");
      MS_CHECK_VALUE(out[i] > 0, "For " + prim_name + "the " + param_name_v +
                                   " must be positive integer number, bot got " + scalar_to_string(out[i]));
    }
  }

  return out;
}

void CalculateOutputRowCol(std::vector<int64_t> *out_shape, const std::vector<int64_t> &in_shape, PadMode pad_mode,
                           int64_t ksize_row, int64_t ksize_col, int64_t stride_row, int64_t stride_col,
                           int64_t rate_row, int64_t rate_col) {
  if (pad_mode == PadMode::VALID) {
    if (in_shape[kIndex2] != abstract::Shape::kShapeDimAny && ksize_row != abstract::Shape::kShapeDimAny &&
        rate_row != abstract::Shape::kShapeDimAny && stride_row != abstract::Shape::kShapeDimAny) {
      out_shape->at(kIndex2) = (in_shape[kIndex2] - (ksize_row + (ksize_row - 1) * (rate_row - 1))) / stride_row + 1;
    }
    if (in_shape[kIndex3] != abstract::Shape::kShapeDimAny && ksize_col != abstract::Shape::kShapeDimAny &&
        rate_col != abstract::Shape::kShapeDimAny && stride_col != abstract::Shape::kShapeDimAny) {
      out_shape->at(kIndex3) = (in_shape[kIndex3] - (ksize_col + (ksize_col - 1) * (rate_col - 1))) / stride_col + 1;
    }
  }

  if (pad_mode == PadMode::SAME) {
    if (in_shape[kIndex2] != abstract::Shape::kShapeDimAny && stride_row != abstract::Shape::kShapeDimAny) {
      out_shape->at(kIndex2) = (in_shape[kIndex2] - 1) / stride_row + 1;
    }
    if (in_shape[kIndex3] != abstract::Shape::kShapeDimAny && stride_col != abstract::Shape::kShapeDimAny) {
      out_shape->at(kIndex3) = (in_shape[kIndex3] - 1) / stride_col + 1;
    }
  }
  return;
}

void CheckOutputShapeValid(string prim_name, const std::vector<int64_t> &out_shape) {
  for (size_t idx = 0; idx < out_shape.size(); idx++) {
    if (out_shape[idx] == abstract::Shape::kShapeDimAny) {
      continue;
    }
    MS_CHECK_VALUE(out_shape[idx] > 0,
                   "For " + prim_name + "the output shape should greater than 0, but got " + ToString(out_shape));
  }
  return;
}

BaseShapePtr ExtractImagePatchesFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                     const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  ShapeVector out_shape{abstract::TensorShape::kShapeDimAny, abstract::TensorShape::kShapeDimAny,
                        abstract::TensorShape::kShapeDimAny, abstract::TensorShape::kShapeDimAny};

  auto x_shape = input_args[kInputIndex0]->GetShape()->GetShapeVector();
  if (IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::TensorShape>(out_shape);
  }
  MS_CHECK_VALUE(x_shape.size() == kExtractImagePatchesInputDims,
                 "For " + prim_name + "the rank of input_x must be 4.");
  out_shape[kIndex0] = x_shape[kIndex0];

  auto ksizes = CheckAndGetExtractImagePatchesList(prim_name, "ksizes", input_args[kInputIndex1]);
  int64_t ksize_row = ksizes[kIndex0];
  int64_t ksize_col = ksizes[kIndex1];

  if (x_shape[kIndex1] != abstract::Shape::kShapeDimAny && ksize_row != abstract::Shape::kShapeDimAny &&
      ksize_col != abstract::Shape::kShapeDimAny) {
    out_shape[kIndex1] = x_shape[kIndex1] * ksize_row * ksize_col;
  }
  auto padding_v = GetScalarValue<int64_t>(input_args[kInputIndex4]->GetValue());
  if (!padding_v.has_value()) {
    return std::make_shared<abstract::TensorShape>(out_shape);
  }
  auto padding = (PadMode)padding_v.value();
  MS_CHECK_VALUE(padding == PadMode::VALID || padding == PadMode::SAME,
                 "For " + prim_name + "the padding support 'VALID' 'SAME'");

  auto strides = CheckAndGetExtractImagePatchesList(prim_name, "strides", input_args[kInputIndex2]);
  auto rates = CheckAndGetExtractImagePatchesList(prim_name, "rates", input_args[kInputIndex3]);

  CalculateOutputRowCol(&out_shape, x_shape, padding, ksize_row, ksize_col, strides[kIndex0], strides[kIndex1],
                        rates[kIndex0], rates[kIndex1]);

  CheckOutputShapeValid(prim_name, out_shape);

  return std::make_shared<abstract::TensorShape>(out_shape);
}

TypePtr ExtractImagePatchesFuncImpl::InferType(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[0]->GetType()->Clone();
}
}  // namespace ops
}  // namespace mindspore
