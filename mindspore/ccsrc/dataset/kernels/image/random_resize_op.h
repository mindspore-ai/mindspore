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
#ifndef DATASET_KERNELS_IMAGE_RANDOM_RESIZE_OP_H_
#define DATASET_KERNELS_IMAGE_RANDOM_RESIZE_OP_H_

#include <memory>
#include <random>

#include "dataset/core/tensor.h"
#include "dataset/kernels/image/resize_op.h"
#include "dataset/kernels/tensor_op.h"
#include "dataset/util/random.h"
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {
class RandomResizeOp : public ResizeOp {
 public:
  // Default values, also used by python_bindings.cc
  static const int32_t kDefTargetWidth;

  explicit RandomResizeOp(int32_t size_1, int32_t size_2 = kDefTargetWidth) : ResizeOp(size_1, size_2) {
    random_generator_.seed(GetSeed());
  }

  ~RandomResizeOp() = default;

  // Description: A function that prints info about the node
  void Print(std::ostream &out) const override {
    out << "RandomResizeOp: " << ResizeOp::size1_ << " " << ResizeOp::size2_;
  }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

 private:
  std::mt19937 random_generator_;
  std::uniform_int_distribution<int> distribution_{0, 3};
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_KERNELS_IMAGE_RANDOM_RESIZE_OP_H_
