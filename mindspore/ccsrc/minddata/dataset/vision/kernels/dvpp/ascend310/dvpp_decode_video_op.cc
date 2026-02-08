/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/vision/kernels/dvpp/ascend310/dvpp_decode_video_op.h"

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/core/cv_tensor.h"
#include "minddata/dataset/core/data_type.h"
#include "minddata/dataset/core/device_tensor.h"
#include "minddata/dataset/vision/kernels/dvpp/acl_adapter.h"

namespace mindspore::dataset {
const VdecOutputFormat DvppDecodeVideoOp::kDefVdecOutputFormat = VdecOutputFormat::kYuvSemiplanar420;
const char DvppDecodeVideoOp::kDefOutput[] = "./output";

Status DvppDecodeVideoOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);

  try {
    CHECK_FAIL_RETURN_UNEXPECTED(input->GetBuffer() != nullptr, "The input video buffer is empty.");
    auto *buffer = const_cast<uint8_t *>(input->GetBuffer());
    auto data_size = input->SizeInBytes();
    // assuem that output equals to input
    RETURN_IF_NOT_OK(mindspore::dataset::Tensor::CreateFromTensor(input, output));

    ResourceInfo resource;
    resource.deviceIds.insert(0);
    APP_ERROR ret = AclAdapter::GetInstance().InitResource(&resource);
    if (ret != APP_ERR_OK) {
      AclAdapter::GetInstance().Release();
      std::string error = "DvppDecodeVideo: Error in Init D-chip: " + std::to_string(ret);
      RETURN_STATUS_UNEXPECTED(error);
    }
    int deviceId = *(resource.deviceIds.begin());
    void *context = AclAdapter::GetInstance().GetContext(deviceId);
    // initialize the resource of D-chip and set up all configures

    auto dvpp_video = AclAdapter::GetInstance().CreateDvppVideo(context, buffer, data_size, width_, height_,
                                                                static_cast<uint32_t>(en_type_),
                                                                static_cast<uint32_t>(format_), output_);
    AclLiteError res = AclAdapter::GetInstance().InitDvppVideo(dvpp_video);
    if (res != ACLLITE_OK) {
      (void)AclAdapter::GetInstance().CloseDvppVideo(dvpp_video);
      AclAdapter::GetInstance().Release();
      std::string error = "DvppDecodeVideo: Failed to initialize DvppVideo:" + std::to_string(res);
      RETURN_STATUS_UNEXPECTED(error);
    }

    res = AclAdapter::GetInstance().DvppVideoDumpFrame(dvpp_video);
    if (res != ACLLITE_OK) {
      (void)AclAdapter::GetInstance().CloseDvppVideo(dvpp_video);
      AclAdapter::GetInstance().Release();
      std::string error = "DvppDecodeVideo: Error in DumpFrame:" + std::to_string(res);
      RETURN_STATUS_UNEXPECTED(error);
    }
    (void)AclAdapter::GetInstance().CloseDvppVideo(dvpp_video);
  } catch (const std::exception &e) {
    std::string error = "[ERROR] Error in DvppDecodeVideoOp:" + std::string(e.what());
    RETURN_STATUS_UNEXPECTED(error);
  }
  return Status::OK();
}

Status DvppDecodeVideoOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  CHECK_FAIL_RETURN_UNEXPECTED(!inputs.empty(), "DvppDecodeVideo: inputs cannot be empty.");
  if (inputs[0].Rank() == 1) {
    outputs = inputs;
  }
  CHECK_FAIL_RETURN_UNEXPECTED(!outputs.empty(), "DvppDecodeVideo: Invalid input shape.");
  return Status::OK();
}
}  // namespace mindspore::dataset
