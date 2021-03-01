/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_DVPP_NORMALIZE_JPEG_OP_H
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_DVPP_NORMALIZE_JPEG_OP_H

#include <memory>
#include <string>
#include <vector>
#include "minddata/dataset/core/device_tensor.h"
#include "minddata/dataset/core/device_resource.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"
#include "mindspore/core/utils/log_adapter.h"

namespace mindspore {
namespace dataset {
class DvppNormalizeOp : public TensorOp {
 public:
  explicit DvppNormalizeOp(std::vector<float> mean, std::vector<float> std) : mean_(mean), std_(std) {}

  ~DvppNormalizeOp() = default;

  Status Compute(const std::shared_ptr<DeviceTensor> &input, std::shared_ptr<DeviceTensor> *output) override;

  std::string Name() const override { return kDvppNormalizeOp; }

  Status SetAscendResource(const std::shared_ptr<DeviceResource> &resource) override;

 private:
  std::vector<float> mean_;
  std::vector<float> std_;
};

}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_DVPP_NORMALIZE_JPEG_OP_H
