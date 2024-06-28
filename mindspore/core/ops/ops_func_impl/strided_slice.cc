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

#include "ops/ops_func_impl/strided_slice.h"

#include <algorithm>
#include <bitset>
#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "include/common/utils/utils.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ir/tensor.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/array_ops.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
namespace {
struct SliceInfo {
  std::vector<int64_t> slice_value;
  size_t length;
  bool is_value_unknown{false};
  bool is_rank_unknown{false};
  // when is_value_unknown is true, the valid_value_info is valid.
  // represent the situation where the value is unknown. e.g. value=(1, None, 1), valid_value_info=(true, false, true).
  // when valid_value_info[i] is false, value[i] is INT64_MAX ==> value[i] is invalid.
  std::vector<bool> valid_value_info;
};

struct MaskInfo {
  std::vector<int64_t> begin_pos;
  std::vector<int64_t> end_pos;
  std::vector<int64_t> ellipsis_pos;
  std::vector<int64_t> new_axis_pos;
  std::vector<int64_t> shrink_axis_pos;
};

std::vector<int64_t> TenToTwo(int64_t num, const size_t &length) {
  std::vector<int64_t> output;
  const int64_t factor = 2;
  size_t i = 0;
  while (num) {
    ++i;
    output.push_back(num % factor);
    num /= factor;
  }
  while (i < length) {
    output.push_back(0);
    ++i;
  }
  return output;
}

int64_t GetSlicingLengthForPositiveStrides(int64_t start_pos, int64_t end_pos, int64_t strides, int64_t x_dim) {
  int64_t slicing_length = 0;
  if ((start_pos < x_dim) && end_pos >= -x_dim) {
    if (-x_dim <= start_pos && start_pos < 0) {
      start_pos += x_dim;
    }
    if (start_pos < -x_dim) {
      start_pos = 0;
    }
    if (-x_dim <= end_pos && end_pos < 0) {
      end_pos += x_dim;
    }
    if (end_pos > x_dim) {
      end_pos = x_dim;
    }
    if (start_pos > end_pos) {
      slicing_length = 0;
    } else {
      slicing_length = 1 + (end_pos - 1 - start_pos) / strides;
    }
  }
  return slicing_length;
}

int64_t GetSlicingLengthForNegativeStrides(int64_t start_pos, int64_t end_pos, int64_t strides, int64_t x_dim) {
  int64_t slicing_length = 0;
  if (start_pos >= -x_dim && end_pos < x_dim) {
    if (start_pos >= 0 && start_pos < x_dim) {
      start_pos += -x_dim;
    }
    if (start_pos >= x_dim) {
      start_pos = -1;
    }
    if (end_pos >= 0 && end_pos < x_dim) {
      end_pos += -x_dim;
    }
    if (end_pos < -x_dim - 1) {
      end_pos = -x_dim - 1;
    }
    if (start_pos <= end_pos) {
      slicing_length = 0;
    } else {
      slicing_length = 1 + (end_pos + 1 - start_pos) / strides;
    }
  }
  return slicing_length;
}

int64_t ComputeSlicingLength(int64_t start_pos, int64_t end_pos, int64_t strides, int64_t x_dim) {
  int64_t slicing_length = 0;
  if (strides == 0) {
    MS_EXCEPTION(ValueError) << "For 'StridedSlice', input 'strides' can not contain 0.";
  }
  if (x_dim == abstract::Shape::kShapeDimAny) {
    return abstract::Shape::kShapeDimAny;
  }
  if (strides > 0) {
    slicing_length = GetSlicingLengthForPositiveStrides(start_pos, end_pos, strides, x_dim);
  } else {
    slicing_length = GetSlicingLengthForNegativeStrides(start_pos, end_pos, strides, x_dim);
  }
  return slicing_length;
}

SliceInfo GetSliceInfo(const AbstractBasePtr &input_arg, const std::string &arg_name) {
  SliceInfo slice_info;
  MS_EXCEPTION_IF_NULL(input_arg);
  auto slice_shape = input_arg->GetShape();
  MS_EXCEPTION_IF_NULL(slice_shape);
  if (slice_shape->isa<abstract::DynamicSequenceShape>()) {
    slice_info.is_rank_unknown = true;
    return slice_info;
  }

  auto slice_array_opt = GetArrayValue<int64_t>(input_arg);
  if (!slice_array_opt.has_value()) {
    if (slice_shape->isa<abstract::SequenceShape>()) {
      auto seq_shape = slice_shape->cast<abstract::SequenceShapePtr>();
      MS_EXCEPTION_IF_NULL(seq_shape);
      size_t slice_size = seq_shape->size();
      slice_info.is_value_unknown = true;
      slice_info.length = slice_size;
      // represent slice value is (None, None, ...)
      std::vector<bool> valid_info(slice_size, false);
      std::vector<int64_t> slice_v(slice_size, INT64_MAX);
      slice_info.valid_value_info = valid_info;
      slice_info.slice_value = slice_v;
      return slice_info;
    }
    slice_info.is_rank_unknown = true;
    return slice_info;
  }

  auto slice_array = slice_array_opt.value();
  if (!slice_array.HasUnknownValue()) {
    slice_info.slice_value = slice_array.ToVector();
    slice_info.length = slice_info.slice_value.size();
    std::vector<bool> valid_info(slice_info.length, true);
    slice_info.valid_value_info = valid_info;
    if (arg_name == "strides") {
      if (std::any_of(slice_info.slice_value.begin(), slice_info.slice_value.end(),
                      [](int64_t stride_value) { return stride_value == 0; })) {
        MS_EXCEPTION(ValueError) << "For 'StridedSlice', input 'strides' can not contain 0.";
      }
    }
    return slice_info;
  }

  slice_info.is_value_unknown = true;
  slice_info.length = slice_array.size();
  for (size_t i = 0; i < slice_array.size(); i++) {
    if (slice_array.IsValueUnknown(i)) {
      slice_info.valid_value_info.push_back(false);
      slice_info.slice_value.push_back(INT64_MAX);  // placeholder, invalid value
    } else {
      slice_info.valid_value_info.push_back(true);
      slice_info.slice_value.push_back(slice_array[i]);
      if (arg_name == "strides" && slice_array[i] == 0) {
        MS_EXCEPTION(ValueError) << "For 'StridedSlice', input 'strides' can not contain 0.";
      }
    }
  }
  return slice_info;
}

int64_t GetMaskValue(const AbstractBasePtr &input_arg, const std::string &arg_name) {
  MS_EXCEPTION_IF_NULL(input_arg);
  auto mask_opt = GetScalarValue<int64_t>(input_arg->GetValue());
  if (MS_UNLIKELY(!mask_opt.has_value())) {
    MS_EXCEPTION(ValueError) << "For 'StridedSlice'," << arg_name << " mask must be constant value.";
  }
  if (mask_opt.value() < 0) {
    MS_EXCEPTION(ValueError) << "For 'StridedSlice'," << arg_name << " mask must be greater equal 0, but got "
                             << mask_opt.value();
  }
  return mask_opt.value();
}

ShapeVector DynamicComputeInferShape(const PrimitivePtr &primitive, const SliceInfo &begin_info,
                                     const SliceInfo &end_info, const SliceInfo &strides_info,
                                     const ShapeVector &x_shape, const MaskInfo &mask_info) {
  size_t slice_len = begin_info.length;
  size_t i = 0;
  size_t j = 0;
  ShapeVector infer_shape;
  size_t x_rank = x_shape.size();

  auto iter = std::find(mask_info.ellipsis_pos.begin(), mask_info.ellipsis_pos.end(), static_cast<int64_t>(1));
  if (iter != mask_info.ellipsis_pos.end()) {
    MS_LOG(DEBUG) << "For 'StridedSlice', ellipsis mask is currently not support precise infer in dynamic shape.";
    return {abstract::Shape::kShapeRankAny};
  }

  while (i < x_rank || j < slice_len) {
    int64_t slicing_length = -1;
    int64_t x_dim_size = x_shape[i];
    int64_t begin;
    int64_t end;
    int64_t stride;
    begin = begin_info.slice_value[j];
    end = end_info.slice_value[j];
    stride = strides_info.slice_value[j];
    if (j < slice_len) {
      if (j < mask_info.begin_pos.size() && mask_info.begin_pos[j] == 1) {
        begin = stride < 0 ? -1 : 0;
      }
      if (j < mask_info.end_pos.size() && mask_info.end_pos[j] == 1) {
        end = stride < 0 ? -(x_dim_size + 1) : x_dim_size;
      }
      if (j < mask_info.new_axis_pos.size() && mask_info.new_axis_pos[j] == 1) {
        infer_shape.push_back(1);
        j += 1;
        continue;
      }
      if (j < mask_info.shrink_axis_pos.size() && mask_info.shrink_axis_pos[j] == 1) {
        j += 1;
        i += 1;
        continue;
      }
      bool is_slice_valid_value =
        begin_info.valid_value_info[j] && end_info.valid_value_info[j] && strides_info.valid_value_info[j];
      if (!is_slice_valid_value) {
        slicing_length = -1;
      } else {
        slicing_length = ComputeSlicingLength(begin, end, stride, x_dim_size);
      }
    } else {
      begin = 0;
      end = x_shape[i];
      stride = 1;
      slicing_length = ComputeSlicingLength(begin, end, stride, x_dim_size);
    }
    infer_shape.push_back(slicing_length);
    i += 1;
    j += 1;
  }
  return infer_shape;
}

void EllipsisInferShape(const PrimitivePtr &primitive, const ShapeVector &x_shape, const ShapeVector &begin_v,
                        const ShapeVector &end_v, const ShapeVector &strides_v, ShapeVector *infer_shape, size_t i,
                        size_t j, bool has_ellipsis, const MaskInfo &mask_info) {
  if (!has_ellipsis) {
    return;
  }
  MS_EXCEPTION_IF_NULL(primitive);
  size_t x_rank = x_shape.size();
  size_t slice_len = begin_v.size();
  (void)CheckAndConvertUtils::CheckInteger("infer", SizeToLong(mask_info.new_axis_pos.size()), kGreaterEqual,
                                           SizeToLong(slice_len), primitive->name());

  size_t num = 0;
  for (size_t n = j + 1; n < slice_len; n++) {
    if (mask_info.new_axis_pos[n] == 1) {
      num++;
    }
  }

  size_t ellipsis_occupied_dims = x_rank - i - (slice_len - (j + 1)) + num;
  MS_EXCEPTION_IF_NULL(infer_shape);
  (void)infer_shape->insert(infer_shape->end(), x_shape.begin() + LongToSize(i),
                            x_shape.begin() + SizeToLong(i + ellipsis_occupied_dims));
  j += 1;
  i += ellipsis_occupied_dims;

  while (i < x_rank || j < slice_len) {
    int64_t x_dim_size = x_shape[i];
    int64_t start = begin_v[j];
    int64_t finish = end_v[j];
    int64_t strides = strides_v[j];
    if (j < mask_info.begin_pos.size() && mask_info.begin_pos[j] == 1) {
      start = strides_v[j] < 0 ? -1 : 0;
    }
    if (j < mask_info.end_pos.size() && mask_info.end_pos[j] == 1) {
      finish = strides_v[j] < 0 ? -(x_shape[i] + 1) : x_shape[i];
    }
    if (j < mask_info.new_axis_pos.size() && mask_info.new_axis_pos[j] == 1) {
      infer_shape->push_back(1);
      j += 1;
      continue;
    }
    if (j < mask_info.shrink_axis_pos.size() && mask_info.shrink_axis_pos[j] == 1) {
      if (!(-x_shape[i] <= start && start < x_shape[i]) || strides < 0) {
        MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the 'strides[" << j << "]' cannot be "
                                 << "negative number and 'begin[" << j << "]' must be in [-" << x_shape[i] << ", "
                                 << x_shape[i] << ") when 'shrink_axis_mask' is greater than 0, but got 'strides[" << j
                                 << "]': " << strides << ", 'begin[" << j << "]': " << start << ".";
      }
      j += 1;
      i += 1;
      continue;
    }
    int64_t slicing_length = ComputeSlicingLength(start, finish, strides, x_dim_size);
    infer_shape->push_back(slicing_length);
    i += 1;
    j += 1;
  }
  return;
}

ShapeVector ComputeInferShape(const PrimitivePtr &primitive, const ShapeVector &begin_v, const ShapeVector &end_v,
                              const ShapeVector &strides_v, const ShapeVector &x_shape, const MaskInfo &mask_info) {
  auto slice_len = begin_v.size();
  size_t i = 0;
  size_t j = 0;
  int64_t start;
  int64_t finish;
  int64_t strides;
  int64_t slicing_length;
  bool has_ellipsis = false;
  ShapeVector infer_shape;
  size_t x_rank = x_shape.size();
  while (i < x_rank || j < slice_len) {
    int64_t x_dim_size = x_shape[i];
    if (j < slice_len) {
      start = begin_v[j];
      finish = end_v[j];
      strides = strides_v[j];
      if (j < mask_info.ellipsis_pos.size() && mask_info.ellipsis_pos[j] == 1) {
        has_ellipsis = true;
        break;
      }
      if (j < mask_info.begin_pos.size() && mask_info.begin_pos[j] == 1) {
        start = strides_v[j] < 0 ? -1 : 0;
      }
      if (j < mask_info.end_pos.size() && mask_info.end_pos[j] == 1) {
        finish = strides_v[j] < 0 ? -(x_shape[i] + 1) : x_shape[i];
      }
      if (j < mask_info.new_axis_pos.size() && mask_info.new_axis_pos[j] == 1) {
        infer_shape.push_back(1);
        j += 1;
        continue;
      }
      if (j < mask_info.shrink_axis_pos.size() && mask_info.shrink_axis_pos[j] == 1) {
        if (!(-x_shape[i] <= start && start < x_shape[i]) || strides < 0) {
          MS_EXCEPTION(IndexError) << "For '" << primitive->name() << "', the 'strides[" << j << "]' cannot be "
                                   << "negative number and 'begin[" << j << "]' must be in [-" << x_shape[i] << ", "
                                   << x_shape[i] << ") when 'shrink_axis_mask' is greater than 0, but got 'strides["
                                   << j << "]': " << strides << ", 'begin[" << j << "]': " << start << ".";
        }
        j += 1;
        i += 1;
        continue;
      }
    } else {
      start = 0;
      finish = x_shape[i];
      strides = 1;
    }
    slicing_length = ComputeSlicingLength(start, finish, strides, x_dim_size);
    infer_shape.push_back(slicing_length);
    i += 1;
    j += 1;
  }
  EllipsisInferShape(primitive, x_shape, begin_v, end_v, strides_v, &infer_shape, i, j, has_ellipsis, mask_info);
  return infer_shape;
}

MaskInfo GetAllMask(const std::vector<AbstractBasePtr> &input_args, const size_t &length) {
  MaskInfo mask_info;
  auto begin_mask = GetMaskValue(input_args[kIndex4], "begin_mask");
  auto end_mask = GetMaskValue(input_args[kIndex5], "end_mask");
  auto ellipsis_mask = GetMaskValue(input_args[kIndex6], "ellipsis_mask");
  auto new_axis_mask = GetMaskValue(input_args[kIndex7], "new_axis_mask");
  auto shrink_axis_mask = GetMaskValue(input_args[kIndex8], "shrink_axis_mask");
  mask_info.begin_pos = TenToTwo(begin_mask, length);
  mask_info.end_pos = TenToTwo(end_mask, length);
  mask_info.ellipsis_pos = TenToTwo(ellipsis_mask, length);
  mask_info.new_axis_pos = TenToTwo(new_axis_mask, length);
  mask_info.shrink_axis_pos = TenToTwo(shrink_axis_mask, length);
  auto ellipsis_size = std::count(mask_info.ellipsis_pos.begin(), mask_info.ellipsis_pos.end(), 1);
  if (ellipsis_size > 1) {
    MS_EXCEPTION(ValueError) << "For 'StridedSlice' only support one ellipsis in the index, but got " << ellipsis_mask;
  }
  return mask_info;
}
}  // namespace

BaseShapePtr StridedSliceFuncImpl::InferShape(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  ShapeVector ret_in_shape;
  const int64_t cor_input_num = 9;
  auto input_len = SizeToLong(input_args.size());
  MS_CHECK_VALUE(input_len == cor_input_num, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                               "input_args number", input_len, kEqual, cor_input_num, primitive));
  auto prim_name = primitive->name();
  auto x_shape = input_args[kInputIndex0]->GetShape()->GetShapeVector();
  if (x_shape.size() == 0) {
    MS_EXCEPTION(TypeError) << "For 'StridedSlice', input can not be a scalar.";
  }
  auto begin_info = GetSliceInfo(input_args[kIndex1], "begin");
  auto end_info = GetSliceInfo(input_args[kIndex2], "end");
  auto strides_info = GetSliceInfo(input_args[kIndex3], "strides");
  std::vector<bool> check_vec = {IsDynamicRank(x_shape), begin_info.is_rank_unknown, end_info.is_rank_unknown,
                                 strides_info.is_rank_unknown};
  if (std::any_of(check_vec.begin(), check_vec.end(), [](const bool &flag) { return flag; })) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
  }

  if (begin_info.length != strides_info.length || end_info.length != strides_info.length) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', 'begin', 'end' and 'strides' must have the same length, "
                             << "but got length of 'begin': " << begin_info.length << ", 'end': " << end_info.length
                             << ", 'strides': " << strides_info.length << ".";
  }

  auto mask_info = GetAllMask(input_args, begin_info.length);
  bool slice_dynamic = false;
  if (begin_info.is_value_unknown || end_info.is_value_unknown || strides_info.is_value_unknown || IsDynamic(x_shape)) {
    slice_dynamic = true;
  }
  if (!slice_dynamic) {
    ret_in_shape = ComputeInferShape(primitive, begin_info.slice_value, end_info.slice_value, strides_info.slice_value,
                                     x_shape, mask_info);
    return std::make_shared<abstract::Shape>(ret_in_shape);
  }
  ret_in_shape = DynamicComputeInferShape(primitive, begin_info, end_info, strides_info, x_shape, mask_info);

  return std::make_shared<abstract::Shape>(ret_in_shape);
}

TypePtr StridedSliceFuncImpl::InferType(const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  const size_t x_index = 0;
  auto x_type = input_args[x_index]->GetType();
  return x_type->Clone();
}
}  // namespace ops
}  // namespace mindspore
