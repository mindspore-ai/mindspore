/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RANDOM_RESIZE_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RANDOM_RESIZE_OP_H_

#include <memory>
#include <random>
#include <string>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/image/resize_op.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class RandomResizeOp : public ResizeOp {
 public:
  // Default values, also used by python_bindings.cc
  static const int32_t kDefTargetWidth;

  explicit RandomResizeOp(int32_t size_1, int32_t size_2 = kDefTargetWidth) : ResizeOp(size_1, size_2) {
    random_generator_.seed(GetSeed());
    is_deterministic_ = false;
  }

  ~RandomResizeOp() = default;

  // Description: A function that prints info about the node
  void Print(std::ostream &out) const override { out << Name() << ": " << ResizeOp::size1_ << " " << ResizeOp::size2_; }

  Status Compute(const TensorRow &input, TensorRow *output) override;

  std::string Name() const override { return kRandomResizeOp; }

  uint32_t NumInput() override { return -1; }

  uint32_t NumOutput() override { return -1; }

 private:
  std::mt19937 random_generator_;
  std::uniform_int_distribution<int> distribution_{0, 3};
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RANDOM_RESIZE_OP_H_
