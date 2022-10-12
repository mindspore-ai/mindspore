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

#include "ops/adaptive_max_pool2d.h"
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
std::vector<int64_t> AdaptiveMaxPool2D::output_size() const {
  auto value_ptr = GetAttr("output_size");
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

bool AdaptiveMaxPool2D::return_indices() const {
  auto value_ptr = GetAttr("return_indices");
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<bool>(value_ptr);
}

namespace {
abstract::BaseShapePtr AdaptiveMaxPool2DInferShape(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  if (input_args.size() != 1) {
    MS_EXCEPTION(ValueError) << "For primitive[AdaptiveMaxPool2D], the num of input args should be 1, but got "
                             << input_args.size();
  }
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  if (shape_map.empty()) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>());
  }

  auto in_shape_vector = shape_map[kShape];
  const auto &output_size_ptr = primitive->GetAttr("output_size");
  MS_EXCEPTION_IF_NULL(output_size_ptr);
  const auto &output_size = GetValue<std::vector<int64_t>>(output_size_ptr);
  if (in_shape_vector.size() == 1) {
    if (in_shape_vector[0] != kDynamicRankValue) {
      MS_EXCEPTION(ValueError)
        << "For primitive[AdaptiveMaxPool2D], the shape size of input argument[input_x] must be 3 "
           "or 4, but got shape size is 1.";
    }
  } else if ((in_shape_vector.size() != kFormatCHWShapeSize && in_shape_vector.size() != kFormatNCHWShapeSize) ||
             output_size.size() != kOutputSizeAttrSize) {
    MS_EXCEPTION(ValueError) << "For primitive[AdaptiveMaxPool2D], the shape size of input argument[input_x] must be 3 "
                                "or 4 and the size of attr[output_size] must be 2, but got shape size:"
                             << in_shape_vector.size() << " and output_size size:" << output_size.size();
  }

  // Update the output shape by output size and input shape.
  if (in_shape_vector.size() != 1) {
    auto input_size_iter = in_shape_vector.rbegin();
    auto output_size_iter = output_size.rbegin();
    for (; output_size_iter != output_size.rend(); ++output_size_iter, ++input_size_iter) {
      // If output size is none, the input shape should be used.
      if (*output_size_iter != kPyValueNone) {
        *input_size_iter = *output_size_iter;
      }
    }
  }

  const auto &return_indices_ptr = primitive->GetAttr("return_indices");
  MS_EXCEPTION_IF_NULL(return_indices_ptr);
  const auto &return_indices = GetValue<bool>(return_indices_ptr);
  auto in_shape = std::make_shared<abstract::Shape>(in_shape_vector);

  // If return indices is true, need to output the indices corresponding to the max value, whose shape is the same
  // as the max value.
  if (return_indices) {
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{in_shape, in_shape});
  }
  return in_shape;
}

TypePtr AdaptiveMaxPool2DInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  if (input_args.size() != 1) {
    MS_EXCEPTION(ValueError) << "For primitive[AdaptiveMaxPool2D], the num of input args should be 1, but got "
                             << input_args.size();
  }
  MS_EXCEPTION_IF_NULL(input_args[0]);

  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  auto input_type =
    CheckAndConvertUtils::CheckTensorTypeValid("input_x", input_args[0]->BuildType(), valid_types, prim->name());

  const auto &return_indices_ptr = prim->GetAttr("return_indices");
  MS_EXCEPTION_IF_NULL(return_indices_ptr);
  const auto &return_indices = GetValue<bool>(return_indices_ptr);

  // If return indices is true, need to output the indices corresponding to the max value, whose shape is the same
  // as the max value.
  if (return_indices) {
    auto indices_type = kInt64;
    return std::make_shared<Tuple>(std::vector<TypePtr>{input_type, indices_type});
  }
  return input_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(AdaptiveMaxPool2D, BaseOperator);
AbstractBasePtr AdaptiveMaxPool2DInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) {
  return abstract::MakeAbstract(AdaptiveMaxPool2DInferShape(primitive, input_args),
                                AdaptiveMaxPool2DInferType(primitive, input_args));
}
REGISTER_PRIMITIVE_EVAL_IMPL(AdaptiveMaxPool2D, prim::kPrimAdaptiveMaxPool2D, AdaptiveMaxPool2DInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
