/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_TENSOR_LAYOUT_TENSOR_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_TENSOR_LAYOUT_TENSOR_INFO_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/tensor_layout/tensor_layout.h"

namespace mindspore {
namespace parallel {
using Shapes = std::vector<Shape>;

class TensorInfo {
 public:
  TensorInfo(const TensorLayout &tensor_layout, Shape shape, Shape slice_shape)
      : tensor_layout_(tensor_layout), shape_(std::move(shape)), slice_shape_(std::move(slice_shape)) {}
  explicit TensorInfo(const TensorLayout &tensor_layout) : tensor_layout_(tensor_layout) {
    shape_ = tensor_layout.base_tensor_shape().array();
    slice_shape_ = tensor_layout.base_slice_shape().array();
  }
  // trivial default constructor will not initialize c language types.
  TensorInfo() = default;
  ~TensorInfo() = default;
  TensorLayout tensor_layout() const { return tensor_layout_; }
  Shape slice_shape() const { return slice_shape_; }
  Shape shape() const { return shape_; }
  void set_reduce_dim(const std::vector<int64_t> &dim) { reduce_dim_ = dim; }
  std::vector<int64_t> reduce_dim() const { return reduce_dim_; }
  Dimensions InferStrategy() const {
    Dimensions stra;
    for (size_t i = 0; i < shape_.size(); ++i) {
      if ((slice_shape_[i] == 0) || (shape_[i] % slice_shape_[i] != 0)) {
        return stra;
      }
      int64_t dim = shape_[i] / slice_shape_[i];
      stra.push_back(dim);
    }
    return stra;
  }
  bool operator==(const TensorInfo &other) {
    if (this->slice_shape_ != other.slice_shape_) {
      return false;
    }
    if (this->tensor_layout_ != other.tensor_layout_) {
      return false;
    }
    return true;
  }

 private:
  TensorLayout tensor_layout_;
  Shape shape_;
  Shape slice_shape_;
  // reduce method's reduce dim
  std::vector<int64_t> reduce_dim_;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_TENSOR_LAYOUT_TENSOR_INFO_H_
