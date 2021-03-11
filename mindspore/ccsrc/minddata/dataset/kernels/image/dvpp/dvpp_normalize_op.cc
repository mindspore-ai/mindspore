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

#include <algorithm>
#include "minddata/dataset/kernels/image/dvpp/dvpp_normalize_op.h"

namespace mindspore {
namespace dataset {
Status DvppNormalizeOp::Compute(const std::shared_ptr<DeviceTensor> &input, std::shared_ptr<DeviceTensor> *output) {
  const TensorShape dvpp_shape({1, 1, 1});
  const DataType dvpp_data_type(DataType::DE_UINT8);
  mindspore::dataset::DeviceTensor::CreateEmpty(dvpp_shape, dvpp_data_type, output);
  std::vector<uint32_t> yuv_shape = input->GetYuvStrideShape();
  (*output)->SetAttributes(input->GetDeviceMutableBuffer(), input->DeviceDataSize(), yuv_shape[0], yuv_shape[1],
                           yuv_shape[2], yuv_shape[3]);
  if (!((*output)->HasDeviceData())) {
    std::string error = "[ERROR] Fail to get the output result from device memory!";
    RETURN_STATUS_UNEXPECTED(error);
  }
  return Status::OK();
}

Status DvppNormalizeOp::SetAscendResource(const std::shared_ptr<DeviceResource> &resource) { return Status::OK(); }

}  // namespace dataset
}  // namespace mindspore
