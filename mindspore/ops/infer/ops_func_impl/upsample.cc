/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/upsample.h"

#include <cstdint>
#include <numeric>
#include <tuple>
#include <string>
#include <functional>
#include <utility>
#include <vector>

#include "ops/ops_func_impl/op_func_impl.h"
#include "abstract/dshape.h"
#include "ops_utils/op_constants.h"
#include "ir/anf.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "mindapi/base/types.h"
#include "view/view_strides_calculator.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "utils/value_utils.h"

namespace mindspore {
namespace ops {
namespace {
void CheckIfSizeAndScalesDefined(const std::string &prim_name, bool is_output_size_none, bool is_scales_none) {
  if (MS_UNLIKELY(is_output_size_none == is_scales_none)) {
    if (is_output_size_none) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', either output_size or scales should be defined.";
    } else {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', only one of output_size or scales should be defined.";
    }
  }
}

template <typename T>
T GetElemFromArray(const PrimitivePtr &primitive, const ArrayValue<T> &array_value, const size_t i,
                   const std::string &arg_name) {
  T elem_value = abstract::TensorShape::kShapeDimAny;
  if (i < array_value.size() && !array_value.IsValueUnknown(i)) {
    elem_value = array_value[i];
    const T zero = 0;
    if (elem_value <= zero) {
      MS_EXCEPTION(ValueError) << "For " << primitive->name() << ", " << arg_name
                               << "'s value should be greater than 0, but got " << elem_value;
    }
  }
  return elem_value;
}

void InferShapeFromSizeWithControlFlow(const PrimitivePtr &primitive, const ArrayValue<int64_t> &size_array,
                                       std::vector<int64_t> *const output_shape, const size_t ele_num) {
  for (size_t i = 0; i < ele_num; ++i) {
    (*output_shape)[i + kDim2] = GetElemFromArray<int64_t>(primitive, size_array, i, "size");
  }
}

void InferShapeFromSize(const PrimitivePtr &primitive, const ArrayValue<int64_t> &size_array,
                        std::vector<int64_t> *const output_shape, const size_t ele_num) {
  MS_CHECK_VALUE(size_array.size() == ele_num,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("number of size", SizeToLong(size_array.size()), kEqual,
                                                             SizeToLong(ele_num), primitive));
  for (size_t i = 0; i < ele_num; ++i) {
    if (MS_UNLIKELY(size_array.IsValueUnknown(i))) {
      continue;
    }
    MS_CHECK_VALUE(size_array[i] > 0, CheckAndConvertUtils::FormatCheckIntegerMsg("size value", size_array[i],
                                                                                  kGreaterThan, 0, primitive));
    (*output_shape)[i + kDim2] = size_array[i];
  }
}

void InferShapeFromSize(const PrimitivePtr &primitive, const AbstractBasePtr &input_arg,
                        std::vector<int64_t> *const output_shape, const size_t ele_num,
                        const bool is_dyn_rank_with_control_flow) {
  auto size_array_opt = GetArrayValue<int64_t>(input_arg);
  if (MS_UNLIKELY(!size_array_opt.has_value())) {
    return;
  }

  auto size_array = size_array_opt.value();
  if (MS_LIKELY(!is_dyn_rank_with_control_flow)) {
    InferShapeFromSize(primitive, size_array, output_shape, ele_num);
  } else {
    InferShapeFromSizeWithControlFlow(primitive, size_array, output_shape, ele_num);
  }
}

void InferShapeFromScalesWithControlFlow(const PrimitivePtr &primitive, const ArrayValue<pyfloat> &scales_array,
                                         std::vector<int64_t> *const output_shape, const ShapeVector &input_shape,
                                         const size_t ele_num) {
  for (size_t i = 0; i < ele_num; ++i) {
    auto scale = GetElemFromArray<pyfloat>(primitive, scales_array, i, "scales");
    if (input_shape[i + kDim2] != abstract::Shape::kShapeDimAny &&
        static_cast<int64_t>(scale) != abstract::Shape::kShapeDimAny) {
      (*output_shape)[i + kDim2] = static_cast<int64_t>(floor(input_shape[i + kDim2] * scale));
    }
  }
}

void InferShapeFromScales(const PrimitivePtr &primitive, const ArrayValue<pyfloat> &scales_array,
                          std::vector<int64_t> *const output_shape, const ShapeVector &input_shape,
                          const size_t ele_num) {
  MS_CHECK_VALUE(scales_array.size() == ele_num,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("number of scales", SizeToLong(scales_array.size()),
                                                             kEqual, SizeToLong(ele_num), primitive));
  for (size_t i = 0; i < ele_num; ++i) {
    if (MS_UNLIKELY(scales_array.IsValueUnknown(i))) {
      continue;
    }
    MS_CHECK_VALUE(scales_array[i] > 0, CheckAndConvertUtils::FormatCheckIntegerMsg("size value", scales_array[i],
                                                                                    kGreaterThan, 0, primitive));
    if (input_shape[i + kDim2] != abstract::Shape::kShapeDimAny) {
      (*output_shape)[i + kDim2] = static_cast<int64_t>(floor(input_shape[i + kDim2] * scales_array[i]));
    }
  }
}

void InferShapeFromScales(const PrimitivePtr &primitive, const AbstractBasePtr &input_arg,
                          std::vector<int64_t> *const output_shape, const ShapeVector &input_shape,
                          const size_t ele_num, const bool is_dyn_rank_with_control_flow) {
  if (MS_UNLIKELY(IsDynamicRank(input_shape))) {
    return;
  }

  auto scales_array_opt = GetArrayValue<pyfloat>(input_arg);
  if (MS_UNLIKELY(!scales_array_opt.has_value())) {
    return;
  }

  auto scales_array = scales_array_opt.value();
  if (MS_LIKELY(!is_dyn_rank_with_control_flow)) {
    InferShapeFromScales(primitive, scales_array, output_shape, input_shape, ele_num);
  } else {
    InferShapeFromScalesWithControlFlow(primitive, scales_array, output_shape, input_shape, ele_num);
  }
}

std::vector<int64_t> InferShapeWithNone(const PrimitivePtr &primitive, const AbstractBasePtr &size_arg,
                                        const AbstractBasePtr &scale_arg, const std::vector<int64_t> &input_shape,
                                        const size_t image_rank) {
  const auto &prim_name = primitive->name();
  auto is_output_size_none = size_arg->GetType()->type_id() == kMetaTypeNone;
  auto is_scales_none = scale_arg->GetType()->type_id() == kMetaTypeNone;
  CheckIfSizeAndScalesDefined(prim_name, is_output_size_none, is_scales_none);

  static std::set<std::string> prim_with_control_flow_list{"UpsampleNearest1D", "UpsampleNearest2D",
                                                           "UpsampleNearest3D"};
  bool is_dyn_rank_with_control_flow =
    prim_with_control_flow_list.find(prim_name) != prim_with_control_flow_list.end() && IsDynamicRank(input_shape);

  std::vector<int64_t> output_shape(image_rank, abstract::Shape::kShapeDimAny);
  if (MS_LIKELY(!IsDynamicRank(input_shape))) {
    output_shape[kDim0] = input_shape[kDim0];
    output_shape[kDim1] = input_shape[kDim1];
  }

  const size_t ele_num = image_rank - kDim2;
  if (is_output_size_none) {
    InferShapeFromScales(primitive, scale_arg, &output_shape, input_shape, ele_num, is_dyn_rank_with_control_flow);
  } else {
    InferShapeFromSize(primitive, size_arg, &output_shape, ele_num, is_dyn_rank_with_control_flow);
  }

  return output_shape;
}

std::vector<int64_t> InferShapeWithNone(const PrimitivePtr &primitive, const ValuePtr &size_value,
                                        const ValuePtr &scale_value, const std::vector<int64_t> &input_shape,
                                        const size_t image_rank) {
  const auto &prim_name = primitive->name();
  auto is_output_size_none = size_value == mindspore::kNone;
  auto is_scales_none = scale_value == mindspore::kNone;
  CheckIfSizeAndScalesDefined(prim_name, is_output_size_none, is_scales_none);

  std::vector<int64_t> output_shape{input_shape[kDim0], input_shape[kDim1]};
  const size_t ele_num = image_rank - kDim2;
  output_shape.insert(output_shape.end(), ele_num, abstract::Shape::kShapeDimAny);
  if (is_output_size_none) {
    auto scales_array = GetArrayValue<pyfloat>(scale_value);
    MS_ASSERT(scales_array.has_value());
    InferShapeFromScales(primitive, scales_array.value(), &output_shape, input_shape, ele_num);
  } else {
    auto size_array = GetArrayValue<int64_t>(size_value);
    MS_ASSERT(size_array.has_value());
    InferShapeFromSize(primitive, size_array.value(), &output_shape, ele_num);
  }

  return output_shape;
}

std::vector<int64_t> InferShapeFromOriginSize(const PrimitivePtr &primitive,
                                              const ArrayValue<int64_t> &input_size_array, const size_t image_rank) {
  MS_CHECK_VALUE(input_size_array.size() == image_rank, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                                          "number of input_size", SizeToLong(input_size_array.size()),
                                                          kEqual, SizeToLong(image_rank), primitive));
  std::vector<int64_t> input_shape(image_rank, abstract::Shape::kShapeDimAny);
  for (size_t i = 0; i < input_size_array.size(); ++i) {
    if (MS_UNLIKELY(input_size_array.IsValueUnknown(i))) {
      continue;
    }
    MS_CHECK_VALUE(input_size_array[i] > 0, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                              "size value", input_size_array[i], kGreaterThan, 0, primitive));
    input_shape[i] = input_size_array[i];
  }
  return input_shape;
}

std::vector<int64_t> InferShapeFromOriginSizeArg(const PrimitivePtr &primitive, const AbstractBasePtr &origin_size_arg,
                                                 const size_t image_rank) {
  auto input_size_array_opt = GetArrayValue<int64_t>(origin_size_arg);
  if (MS_UNLIKELY(!input_size_array_opt.has_value())) {
    std::vector<int64_t> input_shape(image_rank, abstract::TensorShape::kShapeDimAny);
    return input_shape;
  }
  return InferShapeFromOriginSize(primitive, input_size_array_opt.value(), image_rank);
}

std::vector<int64_t> InferShapeFromOriginSizeArg(const PrimitivePtr &primitive, const ValuePtr &origin_size_value,
                                                 const size_t image_rank) {
  auto input_size_array = GetArrayValue<int64_t>(origin_size_value);
  MS_ASSERT(input_size_array.has_value());
  return InferShapeFromOriginSize(primitive, input_size_array.value(), image_rank);
}

void CheckUpsampleGradAndOutputShapes(const PrimitivePtr &primitive, const std::vector<int64_t> &dout_shape,
                                      const std::vector<int64_t> &output_shape, const size_t image_rank) {
  MS_CHECK_VALUE(dout_shape.size() == image_rank,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("grad rank", SizeToLong(dout_shape.size()), kEqual,
                                                             SizeToLong(image_rank), primitive));
  const auto &prim_name = primitive->name();
  if (MS_UNLIKELY(output_shape != dout_shape)) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', The shape of grad, which should the same as that of output, is " << dout_shape
                             << ", but the shape of output is (" << output_shape << ".";
  }
}

void UpsampleCheckInputShape(const PrimitivePtr &primitive, const std::vector<int64_t> &input_shape,
                             const size_t image_rank) {
  if (MS_UNLIKELY(IsDynamicRank(input_shape))) {
    return;
  }
  MS_CHECK_VALUE(input_shape.size() == image_rank,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("image rank", SizeToLong(input_shape.size()), kEqual,
                                                             SizeToLong(image_rank), primitive));
  if (MS_LIKELY(!IsDynamic(input_shape))) {
    auto input_num =
      std::accumulate(input_shape.begin() + kIndex1, input_shape.end(), int64_t(1), std::multiplies<int64_t>());
    if (input_num <= 0) {
      MS_EXCEPTION(RuntimeError) << "For " << primitive->name() << ", non-empty " << image_rank
                                 << "D data tensor expected but got a tensor with shape " << input_shape;
    }
  }
}
}  // namespace
BaseShapePtr UpsampleForwardInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                                       const size_t image_rank) {
  const auto &input_shape = input_args.at(0)->GetShape()->GetShapeVector();
  UpsampleCheckInputShape(primitive, input_shape, image_rank);
  auto output_shape = InferShapeWithNone(primitive, input_args[kIndex1], input_args[kIndex2], input_shape, image_rank);
  return std::make_shared<abstract::TensorShape>(std::move(output_shape));
}

ShapeArray UpsampleForwardInferShape(const PrimitivePtr &primitive, const std::vector<ValuePtr> &input_values,
                                     const size_t image_rank) {
  const auto &input = input_values[kIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(input);
  const auto &input_shape = input->shape();
  UpsampleCheckInputShape(primitive, input_shape, image_rank);
  auto output_shape =
    InferShapeWithNone(primitive, input_values.at(kIndex1), input_values.at(kIndex2), input_shape, image_rank);
  return {std::move(output_shape)};
}

BaseShapePtr UpsampleBackwardInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                                        const size_t image_rank) {
  auto input_shape = InferShapeFromOriginSizeArg(primitive, input_args[kIndex1], image_rank);
  return std::make_shared<abstract::TensorShape>(std::move(input_shape));
}

ShapeArray UpsampleBackwardInferShape(const PrimitivePtr &primitive, const std::vector<ValuePtr> &input_values,
                                      const size_t image_rank) {
  const auto &dout = input_values[kIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(dout);
  const auto &dout_shape = dout->shape();
  auto input_shape = InferShapeFromOriginSizeArg(primitive, input_values.at(kIndex1), image_rank);
  auto output_shape =
    InferShapeWithNone(primitive, input_values.at(kIndex2), input_values.at(kIndex3), input_shape, image_rank);
  CheckUpsampleGradAndOutputShapes(primitive, dout_shape, output_shape, image_rank);
  return {std::move(input_shape)};
}

int32_t UpsampleBackwardCheck(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                              const size_t image_rank) {
  const auto &dout_shape = input_args[kIndex0]->GetShape()->GetShapeVector();
  if (MS_UNLIKELY(IsDynamic(dout_shape))) {
    return OP_CHECK_RETRY;
  }
  auto input_shape = InferShapeFromOriginSizeArg(primitive, input_args[kIndex1], image_rank);
  if (MS_UNLIKELY(IsDynamic(input_shape))) {
    return OP_CHECK_RETRY;
  }
  auto output_shape = InferShapeWithNone(primitive, input_args[kIndex2], input_args[kIndex3], input_shape, image_rank);
  if (MS_UNLIKELY(IsDynamic(output_shape))) {
    return OP_CHECK_RETRY;
  }
  CheckUpsampleGradAndOutputShapes(primitive, dout_shape, output_shape, image_rank);
  return OP_CHECK_SUCCESS;
}
}  // namespace ops
}  // namespace mindspore
