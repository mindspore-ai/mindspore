/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_IR_TENSOR_STORAGE_INFO_H_
#define MINDSPORE_CORE_IR_TENSOR_STORAGE_INFO_H_

#include <memory>
#include <vector>
#include <string>
#include <utility>
#include <sstream>
#include "mindapi/base/type_id.h"

namespace mindspore {
struct TensorStorageInfo {
 public:
  TensorStorageInfo(std::vector<int64_t> new_shape, std::vector<int64_t> new_strides, std::vector<int64_t> ori_shape,
                    std::vector<int64_t> ori_strides, bool is_contiguous)
      : shape(std::move(new_shape)),
        strides(std::move(new_strides)),
        ori_shape(std::move(ori_shape)),
        ori_strides(std::move(ori_strides)),
        is_contiguous(is_contiguous) {}
  TensorStorageInfo(std::vector<int64_t> new_shape, std::vector<int64_t> new_strides, size_t new_storage_offset,
                    std::vector<int64_t> ori_shape, std::vector<int64_t> ori_strides, bool is_contiguous)
      : shape(std::move(new_shape)),
        strides(std::move(new_strides)),
        storage_offset(std::move(new_storage_offset)),
        ori_shape(std::move(ori_shape)),
        ori_strides(std::move(ori_strides)),
        is_contiguous(is_contiguous) {}

  template <typename T>
  std::string VectorToString(const std::vector<T> &values) {
    std::stringstream ss;
    ss << "[";
    auto size = values.size();
    for (size_t i = 0; i < size; ++i) {
      ss << values[i];
      if (i != size - 1) {
        ss << ", ";
      }
    }
    ss << "]";
    return ss.str();
  }

  std::string ToString() {
    std::stringstream buf;
    buf << "TensorStorageInfo(shape=" << VectorToString(shape);
    buf << "  strides=" << VectorToString(strides);
    buf << "  storage_offset=" << std::to_string(storage_offset);
    buf << "  ori_shape=" << VectorToString(ori_shape);
    buf << "  ori_strides=" << VectorToString(ori_strides);
    buf << "  is_contiguous=" << std::to_string(is_contiguous);
    buf << "  data_type=" << std::to_string(data_type);
    buf << ")";
    return buf.str();
  }

  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  size_t storage_offset{0};
  std::vector<int64_t> ori_shape;
  std::vector<int64_t> ori_strides;
  bool is_contiguous{false};
  TypeId data_type{kTypeUnknown};
};
using TensorStorageInfoPtr = std::shared_ptr<TensorStorageInfo>;
using TensorStorageInfoPtrList = std::vector<TensorStorageInfoPtr>;

}  // namespace mindspore
#endif  // MINDSPORE_CORE_IR_TENSOR_STORAGE_INFO_H_
