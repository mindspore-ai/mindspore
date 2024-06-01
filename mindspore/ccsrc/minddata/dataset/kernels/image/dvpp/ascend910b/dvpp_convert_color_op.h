/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_CONVERT_COLOR_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_CONVERT_COLOR_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/core/device_tensor_ascend910b.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"

namespace mindspore {
namespace dataset {
class DvppConvertColorOp : public TensorOp {
 public:
  explicit DvppConvertColorOp(ConvertMode convertMode) : convert_mode_(convertMode) {}

  ~DvppConvertColorOp() override = default;

  Status Compute(const std::shared_ptr<DeviceTensorAscend910B> &input,
                 std::shared_ptr<DeviceTensorAscend910B> *output) override;

  std::string Name() const override { return kDvppConvertColorOp; }

  bool IsDvppOp() override { return true; }

 private:
  ConvertMode convert_mode_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_CONVERT_COLOR_OP_H_
