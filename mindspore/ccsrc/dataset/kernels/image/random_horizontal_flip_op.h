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
#ifndef DATASET_KERNELS_IMAGE_RANDOM_HORIZONTAL_FLIP_OP_H_
#define DATASET_KERNELS_IMAGE_RANDOM_HORIZONTAL_FLIP_OP_H_

#include <memory>
#include <random>

#include "dataset/core/tensor.h"
#include "dataset/kernels/tensor_op.h"
#include "dataset/util/random.h"
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {
class RandomHorizontalFlipOp : public TensorOp {
 public:
  // Default values, also used by python_bindings.cc
  static const float kDefProbability;

  explicit RandomHorizontalFlipOp(float probability = kDefProbability) : distribution_(probability) {
    rnd_.seed(GetSeed());
  }

  ~RandomHorizontalFlipOp() override = default;

  // Provide stream operator for displaying it
  friend std::ostream &operator<<(std::ostream &out, const RandomHorizontalFlipOp &so) {
    so.Print(out);
    return out;
  }

  void Print(std::ostream &out) const override { out << "RandomHorizontalFlipOp"; }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

 private:
  std::mt19937 rnd_;
  std::bernoulli_distribution distribution_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_KERNELS_IMAGE_RANDOM_HORIZONTAL_FLIP_OP_H_
