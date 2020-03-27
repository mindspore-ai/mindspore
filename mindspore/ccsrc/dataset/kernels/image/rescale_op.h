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
#ifndef DATASET_KERNELS_IMAGE_RESCALE_OP_H_
#define DATASET_KERNELS_IMAGE_RESCALE_OP_H_

#include <memory>
#include <vector>

#include "dataset/core/tensor.h"
#include "dataset/kernels/tensor_op.h"
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {
class RescaleOp : public TensorOp {
 public:
  RescaleOp(float rescale_ratio, float shift_ratio) : rescale_(rescale_ratio), shift_(shift_ratio) {}

  ~RescaleOp() override = default;

  void Print(std::ostream &out) const override {
    out << "RescaleOp: shift: " << shift_ << ", Rescale: " << rescale_ << std::endl;
  }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;
  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

 private:
  float rescale_;
  float shift_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // DATASET_KERNELS_IMAGE_RESCALE_OP_H_
