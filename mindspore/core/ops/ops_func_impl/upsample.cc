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

#include "ops/ops_func_impl/upsample.h"

#include <tuple>
#include <string>
#include <functional>
#include <utility>

#include "ops/ops_func_impl/op_func_impl.h"
#include "ops/op_utils.h"
#include "mindapi/base/types.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
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

void InferShapeFromSize(const PrimitivePtr &primitive, const AbstractBasePtr &input_arg,
                        std::vector<int64_t> *const y_shape, const size_t ele_num, const bool is_skip) {
  auto size_array_opt = GetArrayValue<int64_t>(input_arg);
  if (MS_UNLIKELY(!size_array_opt.has_value())) {
    return;
  }

  auto size_array = size_array_opt.value();
  if (MS_LIKELY(!is_skip)) {
    MS_CHECK_VALUE(size_array.size() == ele_num,
                   CheckAndConvertUtils::FormatCheckIntegerMsg("number of size", SizeToLong(size_array.size()), kEqual,
                                                               SizeToLong(ele_num), primitive));
    for (size_t i = 0; i < ele_num; ++i) {
      if (MS_UNLIKELY(size_array.IsValueUnknown(i))) {
        continue;
      }
      MS_CHECK_VALUE(size_array[i] > 0, CheckAndConvertUtils::FormatCheckIntegerMsg("size value", size_array[i],
                                                                                    kGreaterThan, 0, primitive));
      (*y_shape)[i + kDim2] = size_array[i];
    }
  } else {
    for (size_t i = 0; i < ele_num; ++i) {
      (*y_shape)[i + kDim2] = GetElemFromArray<int64_t>(primitive, size_array, i, "size");
    }
  }
}

void InferShapeFromScales(const PrimitivePtr &primitive, const AbstractBasePtr &input_arg,
                          std::vector<int64_t> *const y_shape, const ShapeVector &x_shape, const size_t ele_num,
                          const bool is_skip) {
  if (MS_UNLIKELY(IsDynamicRank(x_shape))) {
    return;
  }

  auto scales_array_opt = GetArrayValue<pyfloat>(input_arg);
  if (MS_UNLIKELY(!scales_array_opt.has_value())) {
    return;
  }

  auto scales_array = scales_array_opt.value();
  if (MS_LIKELY(!is_skip)) {
    MS_CHECK_VALUE(scales_array.size() == ele_num,
                   CheckAndConvertUtils::FormatCheckIntegerMsg("number of scales", SizeToLong(scales_array.size()),
                                                               kEqual, SizeToLong(ele_num), primitive));
    for (size_t i = 0; i < ele_num; ++i) {
      if (MS_UNLIKELY(scales_array.IsValueUnknown(i))) {
        continue;
      }
      MS_CHECK_VALUE(scales_array[i] > 0, CheckAndConvertUtils::FormatCheckIntegerMsg("size value", scales_array[i],
                                                                                      kGreaterThan, 0, primitive));
      if (x_shape[i + kDim2] != abstract::Shape::kShapeDimAny) {
        (*y_shape)[i + kDim2] = static_cast<int64_t>(floor(x_shape[i + kDim2] * scales_array[i]));
      }
    }
  } else {
    for (size_t i = 0; i < ele_num; ++i) {
      auto scale = GetElemFromArray<pyfloat>(primitive, scales_array, i, "scales");
      if (x_shape[i + kDim2] != abstract::Shape::kShapeDimAny &&
          static_cast<int64_t>(scale) != abstract::Shape::kShapeDimAny) {
        (*y_shape)[i + kDim2] = static_cast<int64_t>(floor(x_shape[i + kDim2] * scale));
      }
    }
  }
}

std::vector<int64_t> InferShapeWithNone(const PrimitivePtr &primitive,
                                        const std::tuple<AbstractBasePtr, AbstractBasePtr> &input_args,
                                        const std::vector<int64_t> &x_shape, const size_t image_rank) {
  const auto &prim_name = primitive->name();

  auto &[size_arg, scale_arg] = input_args;
  auto is_output_size_none = size_arg->GetType()->type_id() == kMetaTypeNone;
  auto is_scales_none = scale_arg->GetType()->type_id() == kMetaTypeNone;
  if (MS_UNLIKELY(is_output_size_none == is_scales_none)) {
    if (is_output_size_none) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', either output_size or scales should be defined.";
    } else {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', only one of output_size or scales should be defined.";
    }
  }

  static std::set<std::string> black_list{"UpsampleNearest1D", "UpsampleNearest2D", "UpsampleNearest3D"};
  bool is_skip = black_list.find(prim_name) != black_list.end() && IsDynamicRank(x_shape);

  std::vector<int64_t> y_shape(image_rank, abstract::Shape::kShapeDimAny);
  if (MS_LIKELY(!IsDynamicRank(x_shape))) {
    y_shape[kDim0] = x_shape[kDim0];
    y_shape[kDim1] = x_shape[kDim1];
  }

  const size_t ele_num = image_rank - kDim2;
  if (is_output_size_none) {
    InferShapeFromScales(primitive, scale_arg, &y_shape, x_shape, ele_num, is_skip);
  } else {
    InferShapeFromSize(primitive, size_arg, &y_shape, ele_num, is_skip);
  }

  return y_shape;
}

std::vector<int64_t> InferShapeFromOriginSizeArg(const PrimitivePtr &primitive, const AbstractBasePtr &origin_size_arg,
                                                 const size_t image_rank) {
  std::vector<int64_t> x_shape(image_rank, abstract::TensorShape::kShapeDimAny);

  auto input_size_array_opt = GetArrayValue<int64_t>(origin_size_arg);
  if (MS_UNLIKELY(!input_size_array_opt.has_value())) {
    return x_shape;
  }

  auto input_size_array = input_size_array_opt.value();
  MS_CHECK_VALUE(input_size_array.size() == image_rank, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                                          "number of input_size", SizeToLong(input_size_array.size()),
                                                          kEqual, SizeToLong(image_rank), primitive));

  for (size_t i = 0; i < input_size_array.size(); ++i) {
    if (MS_UNLIKELY(input_size_array.IsValueUnknown(i))) {
      continue;
    }
    MS_CHECK_VALUE(input_size_array[i] > 0, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                              "size value", input_size_array[i], kGreaterThan, 0, primitive));
    x_shape[i] = input_size_array[i];
  }

  return x_shape;
}
}  // namespace

BaseShapePtr UpsampleForwardInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                                       const size_t image_rank) {
  const auto &x_shape = input_args.at(0)->GetShape()->GetShapeVector();
  if (MS_LIKELY(!IsDynamicRank(x_shape))) {
    MS_CHECK_VALUE(x_shape.size() == image_rank,
                   CheckAndConvertUtils::FormatCheckIntegerMsg("image rank", SizeToLong(x_shape.size()), kEqual,
                                                               SizeToLong(image_rank), primitive));
    if (MS_LIKELY(!IsDynamic(x_shape))) {
      auto input_num = std::accumulate(x_shape.begin(), x_shape.end(), int64_t(1), std::multiplies<int64_t>());
      if (input_num <= 0) {
        MS_EXCEPTION(RuntimeError) << "For " << primitive->name()
                                   << ", input sizes should be greater than 0, but got input.shape: " << x_shape << ".";
      }
    }
  }

  auto y_shape = InferShapeWithNone(primitive, std::make_tuple(input_args[1], input_args[2]), x_shape, image_rank);

  return std::make_shared<abstract::TensorShape>(std::move(y_shape));
}

BaseShapePtr UpsampleBackwardInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                                        const size_t image_rank) {
  const size_t input_size_idx = 1;
  auto x_shape = InferShapeFromOriginSizeArg(primitive, input_args[input_size_idx], image_rank);
  return std::make_shared<abstract::TensorShape>(std::move(x_shape));
}

int32_t UpsampleBackwardCheck(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                              const size_t image_rank) {
  const auto &dout_shape = input_args[0]->GetShape()->GetShapeVector();
  if (MS_UNLIKELY(IsDynamicRank(dout_shape))) {
    return OP_CHECK_RETRY;
  }

  MS_CHECK_VALUE(dout_shape.size() == image_rank,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("grad rank", SizeToLong(dout_shape.size()), kEqual,
                                                             SizeToLong(image_rank), primitive));

  auto x_shape = InferShapeFromOriginSizeArg(primitive, input_args[1], image_rank);
  auto y_shape = InferShapeWithNone(primitive, std::make_tuple(input_args[2], input_args[3]), x_shape, image_rank);

  const auto &prim_name = primitive->name();
  for (size_t i = 0; i < image_rank; ++i) {
    if (y_shape[i] != abstract::Shape::kShapeDimAny && dout_shape[i] != abstract::Shape::kShapeDimAny &&
        y_shape[i] != dout_shape[i]) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "', The shape of grad, which should the same as that of output, is " << dout_shape
                               << ", but the shape of output is (" << y_shape << ".";
    }
  }

  return OP_CHECK_SUCCESS;
}
}  // namespace ops
}  // namespace mindspore
