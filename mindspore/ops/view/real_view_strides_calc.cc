/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include <memory>
#include <set>
#include <unordered_map>
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "view/real_view_strides_calc.h"

namespace mindspore::ops {

// Complex numbers are stored as pairs of real numbers, so strides need to be doubled
constexpr int kComplexStrideMultiplier = 2;

TypeId GetRealTypeFromComplex(TypeId complex_type) {
  static const std::unordered_map<TypeId, TypeId> complex_to_real_map = {
    {kNumberTypeComplex, kNumberTypeFloat16},     // complex -> float16
    {kNumberTypeComplex64, kNumberTypeFloat32},   // complex64 -> float32
    {kNumberTypeComplex128, kNumberTypeFloat64},  // complex128 -> float64
  };

  auto it = complex_to_real_map.find(complex_type);
  if (it != complex_to_real_map.end()) {
    return it->second;
  }
  // if not found, return unknown
  return kTypeUnknown;
}

BasicCalcResult RealImagViewBasicTypeCalc(const tensor::TensorPtr &input_tensor, bool is_real) {
  MS_LOG(DEBUG) << "RealViewBasicTypeCalc Call start";
  auto input_data_type = input_tensor->data_type();
  TypeId data_type = GetRealTypeFromComplex(input_data_type);
  auto is_complex_data_type = data_type != kTypeUnknown;
  if (!is_real && !is_complex_data_type) {  // RealView
    MS_EXCEPTION(TypeError) << "For primitive [ImagView], "
                            << "the input tensor data type must be complex64 or complex128, "
                            << "but got " << TypeIdToString(input_data_type) << ".";
  }
  data_type = is_complex_data_type ? data_type : input_data_type;
  auto old_tensor_info = GetOldTensorInfo(input_tensor);
  MS_EXCEPTION_IF_NULL(old_tensor_info);

  auto new_shape = old_tensor_info->old_shape;
  auto new_strides = old_tensor_info->old_strides;
  auto ori_shape = old_tensor_info->ori_shape;
  auto ori_strides = old_tensor_info->ori_strides;
  size_t old_storage_offset = old_tensor_info->old_offset;
  int dim_size = SizeToLong(new_shape.size());
  int ori_dim_size = SizeToLong(ori_shape.size());

  auto new_storage_offset = old_storage_offset;
  if (is_complex_data_type && dim_size > 0 && ori_dim_size > 0) {
    // if old tensor has shape, and because of the complex storage, the new stride is doubled
    for (int i = 0; i < dim_size; i++) {
      new_strides[i] *= kComplexStrideMultiplier;
    }
    // we reexplained the complex storage into two parts, so the original shape is doubled, eg. complex64 to 2 float32
    ori_shape[ori_dim_size - 1] *= kComplexStrideMultiplier;
    ori_strides[ori_dim_size - 1] = 1;
    // recalculate the original strides with the new shape
    for (int i = ori_dim_size - 2; i >= 0; i--) {
      ori_strides[i] = ori_strides[i + 1] * ori_shape[i + 1];
    }
    new_storage_offset *= kComplexStrideMultiplier;
  }

  // for imag part, the storage offset need to add 1
  if (is_complex_data_type && (!is_real)) {
    new_storage_offset += 1;
  }
  MS_LOG(DEBUG) << "RealViewBasicTypeCalc Call end";
  bool is_contiguous = IsContiguous(new_shape, new_strides);
  return {std::make_shared<TensorStorageInfo>(new_shape, new_strides, new_storage_offset, ori_shape, ori_strides,
                                              is_contiguous),
          data_type};
}

BasicCalcResult RealViewBasicTypeCalc(const tensor::TensorPtr &input_tensor) {
  MS_EXCEPTION_IF_NULL(input_tensor);
  return RealImagViewBasicTypeCalc(input_tensor, true);
}

}  // namespace mindspore::ops
