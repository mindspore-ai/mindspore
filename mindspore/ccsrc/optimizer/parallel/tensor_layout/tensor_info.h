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

#ifndef MINDSPORE_CCSRC_OPTIMIZER_PARALLEL_TENSOR_LAYOUT_TENSOR_INFO_H_
#define MINDSPORE_CCSRC_OPTIMIZER_PARALLEL_TENSOR_LAYOUT_TENSOR_INFO_H_

#include <cstdint>
#include <string>
#include <vector>
#include <utility>

#include "optimizer/parallel/status.h"
#include "optimizer/parallel/tensor_layout/tensor_layout.h"
#include "optimizer/parallel/device_matrix.h"

namespace mindspore {
namespace parallel {

using Shapes = std::vector<Shape>;

class TensorInfo {
 public:
  TensorInfo(const TensorLayout& tensor_layout, Shape shape, Shape slice_shape)
      : tensor_layout_(tensor_layout), shape_(std::move(shape)), slice_shape_(std::move(slice_shape)) {}
  explicit TensorInfo(const TensorLayout& tensor_layout) : tensor_layout_(tensor_layout) {
    shape_ = tensor_layout.tensor_shape().array();
    slice_shape_ = tensor_layout.slice_shape().array();
  }
  // trivial default constructor will not initialize c language types.
  TensorInfo() = default;
  ~TensorInfo() = default;
  TensorLayout tensor_layout() const { return tensor_layout_; }
  Shape slice_shape() const { return slice_shape_; }
  Shape shape() const { return shape_; }
  void set_reduce_dim(const std::vector<int32_t>& dim) { reduce_dim_ = dim; }
  std::vector<int32_t> reduce_dim() const { return reduce_dim_; }

 private:
  TensorLayout tensor_layout_;
  Shape shape_;
  Shape slice_shape_;
  // reduce method's reduce dim
  std::vector<int32_t> reduce_dim_;
};

}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_OPTIMIZER_PARALLEL_TENSOR_LAYOUT_TENSOR_INFO_H_
