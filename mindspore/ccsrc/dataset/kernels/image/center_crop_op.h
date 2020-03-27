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
#ifndef DATASET_KERNELS_IMAGE_CENTER_CROP_OP_H_
#define DATASET_KERNELS_IMAGE_CENTER_CROP_OP_H_

#include <memory>
#include <vector>

#include "dataset/core/tensor.h"
#include "dataset/kernels/tensor_op.h"
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {
class CenterCropOp : public TensorOp {
 public:
  // Default values, also used by python_bindings.cc
  static const int32_t kDefWidth;

  explicit CenterCropOp(int32_t het, int32_t wid = kDefWidth) : crop_het_(het), crop_wid_(wid == 0 ? het : wid) {}

  ~CenterCropOp() override = default;

  void Print(std::ostream &out) const override;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;
  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

 private:
  int32_t crop_het_;
  int32_t crop_wid_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_KERNELS_IMAGE_CENTER_CROP_OP_H_
