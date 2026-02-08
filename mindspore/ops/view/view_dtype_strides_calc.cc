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

#include "view/view_dtype_strides_calc.h"
#include <algorithm>
#include <vector>
#include <memory>
#include "ir/dtype/type_id.h"
#include "view/view_strides_calculator.h"
#include "utils/check_convert_utils.h"

namespace mindspore::ops {
inline void CheckStridesValidation(const std::vector<int64_t> &strides, int64_t size_ratio, TypeId old_dtype,
                                   TypeId new_dtype, bool is_upsizing = false) {
  if (strides.back() != 1) {
    MS_EXCEPTION(ValueError) << "For ViewDtype, strides[-1] must be 1 to view " << TypeIdToString(old_dtype) << " as "
                             << TypeIdToString(new_dtype) << " (different element sizes), but got " << strides.back();
  }
  if (is_upsizing) {
    if (std::any_of(strides.begin(), strides.end() - 1,
                    [size_ratio](const auto &stride) { return stride % size_ratio != 0; })) {
      MS_EXCEPTION(ValueError) << "For ViewDtype, strides without last dim must be divisible by " << size_ratio
                               << " to view " << TypeIdToString(old_dtype) << " as " << TypeIdToString(new_dtype)
                               << " (different element sizes), but got strides " << strides;
    }
  }
}

inline void ComputeStridesForDownsizingElementSize(std::vector<int64_t> *strides, int64_t size_ratio, TypeId old_dtype,
                                                   TypeId new_dtype) {
  std::for_each(strides->begin(), strides->end() - 1, [size_ratio](auto &stride) { stride *= size_ratio; });
  strides->back() = 1;
}

inline void ComputeStridesForUpsizingElementSize(std::vector<int64_t> *strides, int64_t size_ratio, TypeId old_dtype,
                                                 TypeId new_dtype) {
  std::for_each(strides->begin(), strides->end() - 1, [size_ratio](auto &stride) { stride /= size_ratio; });
  strides->back() = 1;
}

BasicCalcResult ViewDtypeBasicTypeCalc(const tensor::TensorPtr &input_tensor, const int64_t &dtype) {
  MS_EXCEPTION_IF_NULL(input_tensor);

  auto view_shape = input_tensor->shape();
  auto view_strides = input_tensor->stride();
  auto [storage_shape, storage_strides, storage_offset] =
    GetOriShapeStridesAndOffset(view_shape, view_strides, input_tensor->storage_info());

  TypeId old_dtype = input_tensor->data_type();
  TypeId new_dtype = static_cast<TypeId>(dtype);
  if (old_dtype == new_dtype) {
    if (input_tensor->storage_info() != nullptr) {
      return {input_tensor->storage_info(), old_dtype};
    }
    return {std::make_shared<TensorStorageInfo>(std::move(view_shape), std::move(view_strides), storage_offset,
                                                std::move(storage_shape), std::move(storage_strides),
                                                IsContiguous(view_shape, view_strides)),
            old_dtype};
  }

  int64_t old_element_size = SizeToLong(abstract::TypeIdSize(old_dtype));
  int64_t new_element_size = SizeToLong(abstract::TypeIdSize(new_dtype));
  if (old_element_size == new_element_size) {
    return {std::make_shared<TensorStorageInfo>(std::move(view_shape), std::move(view_strides), storage_offset,
                                                std::move(storage_shape), std::move(storage_strides),
                                                IsContiguous(view_shape, view_strides)),
            new_dtype};
  } else if (view_shape.empty()) {
    MS_EXCEPTION(ValueError) << "For ViewDtype, scalar tensor (shape: ()) cannot be viewed "
                             << TypeIdToString(old_dtype) << " as " << TypeIdToString(new_dtype)
                             << " (different element sizes)";
  } else if (old_element_size > new_element_size) {
    int64_t size_ratio = old_element_size / new_element_size;
    view_shape.back() *= size_ratio;
    storage_shape.back() *= size_ratio;
    CheckStridesValidation(view_strides, size_ratio, old_dtype, new_dtype);
    ComputeStridesForDownsizingElementSize(&view_strides, size_ratio, old_dtype, new_dtype);
    ComputeStridesForDownsizingElementSize(&storage_strides, size_ratio, old_dtype, new_dtype);
    storage_offset = storage_offset * LongToSize(size_ratio);
  } else {
    int64_t size_ratio = new_element_size / old_element_size;
    if (view_shape.back() % size_ratio != 0) {
      MS_EXCEPTION(ValueError) << "For ViewDtype, last dimension must be divisible by " << size_ratio << " to view "
                               << TypeIdToString(old_dtype) << " as " << TypeIdToString(new_dtype)
                               << " (different element sizes), but got " << view_shape.back();
    }

    if (storage_offset % LongToSize(size_ratio) != 0) {
      MS_EXCEPTION(ValueError) << "For ViewDtype, storage offset must be divisible by " << size_ratio << " to view "
                               << TypeIdToString(old_dtype) << " as " << TypeIdToString(new_dtype)
                               << " (different element sizes), but got " << storage_offset;
    }

    view_shape.back() /= size_ratio;
    storage_shape.back() /= size_ratio;
    CheckStridesValidation(view_strides, size_ratio, old_dtype, new_dtype, true);
    ComputeStridesForUpsizingElementSize(&view_strides, size_ratio, old_dtype, new_dtype);
    ComputeStridesForUpsizingElementSize(&storage_strides, size_ratio, old_dtype, new_dtype);
    storage_offset = storage_offset / LongToSize(size_ratio);
  }
  return {std::make_shared<TensorStorageInfo>(std::move(view_shape), std::move(view_strides), storage_offset,
                                              std::move(storage_shape), std::move(storage_strides),
                                              IsContiguous(view_shape, view_strides)),
          new_dtype};
}
}  // namespace mindspore::ops
