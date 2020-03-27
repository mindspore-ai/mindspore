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
#ifndef DATASET_KERNELS_IMAGE_NORMALIZE_OP_H_
#define DATASET_KERNELS_IMAGE_NORMALIZE_OP_H_

#include <memory>

#include "dataset/core/cv_tensor.h"
#include "dataset/core/tensor.h"
#include "dataset/kernels/tensor_op.h"
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {
class NormalizeOp : public TensorOp {
 public:
  NormalizeOp(float mean_r, float mean_g, float mean_b, float std_r, float std_g, float std_b);

  ~NormalizeOp() override = default;

  void Print(std::ostream &out) const override;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

 private:
  std::shared_ptr<CVTensor> mean_;
  std::shared_ptr<CVTensor> std_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_KERNELS_IMAGE_NORMALIZE_OP_H_
