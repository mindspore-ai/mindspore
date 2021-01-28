/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include <string>
#include <vector>
#include <iostream>
#include "minddata/dataset/kernels/image/dvpp/utils/AclProcess.h"
#include "minddata/dataset/core/cv_tensor.h"
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/kernels/image/dvpp/utils/CommonDataType.h"
#include "minddata/dataset/core/data_type.h"
#include "minddata/dataset/kernels/image/dvpp/dvpp_decode_resize_crop_jpeg_op.h"
#include "include/api/context.h"

namespace mindspore {
namespace dataset {
Status DvppDecodeResizeCropJpegOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  if (!IsNonEmptyJPEG(input)) {
    RETURN_STATUS_UNEXPECTED("SoftDvppDecodeReiszeJpegOp only support process jpeg image.");
  }
  try {
    CHECK_FAIL_RETURN_UNEXPECTED(input->GetBuffer() != nullptr, "The input image buffer is empty.");
    unsigned char *buffer = const_cast<unsigned char *>(input->GetBuffer());
    RawData imageInfo;
    uint32_t filesize = input->SizeInBytes();
    imageInfo.lenOfByte = filesize;
    imageInfo.data = std::make_shared<uint8_t>();
    imageInfo.data.reset(new uint8_t[filesize], std::default_delete<uint8_t[]>());
    memcpy_s(imageInfo.data.get(), filesize, buffer, filesize);
    // First part end, whose function is to transform data from a Tensor to imageinfo data structure which can be
    // applied on device
    ResourceInfo resource;
    resource.aclConfigPath = "";
    resource.deviceIds.insert(mindspore::GlobalContext::GetGlobalDeviceID());
    std::shared_ptr<ResourceManager> instance = ResourceManager::GetInstance();
    APP_ERROR ret = instance->InitResource(resource);
    if (ret != APP_ERR_OK) {
      instance->Release();
      std::string error = "Error in Init D-chip:" + std::to_string(ret);
      RETURN_STATUS_UNEXPECTED(error);
    }
    int deviceId = *(resource.deviceIds.begin());
    aclrtContext context = instance->GetContext(deviceId);
    // Second part end where we initialize the resource of D chip and set up all configures
    AclProcess process(resized_width_, resized_height_, crop_width_, crop_height_, context);
    process.set_mode(true);
    ret = process.InitResource();
    if (ret != APP_ERR_OK) {
      instance->Release();
      std::string error = "Error in Init resource:" + std::to_string(ret);
      RETURN_STATUS_UNEXPECTED(error);
    }
    ret = process.Process(imageInfo);
    if (ret != APP_ERR_OK) {
      instance->Release();
      std::string error = "Error in dvpp processing:" + std::to_string(ret);
      RETURN_STATUS_UNEXPECTED(error);
    }
    // Third part end where we execute the core function of dvpp
    auto data = std::static_pointer_cast<unsigned char>(process.Get_Memory_Data());
    unsigned char *ret_ptr = data.get();
    std::shared_ptr<DvppDataInfo> CropOut = process.Get_Device_Memory_Data();
    dsize_t dvpp_length = CropOut->dataSize;
    const TensorShape dvpp_shape({dvpp_length, 1, 1});
    const DataType dvpp_data_type(DataType::DE_UINT8);
    mindspore::dataset::Tensor::CreateFromMemory(dvpp_shape, dvpp_data_type, ret_ptr, output);
    if (!((*output)->HasData())) {
      std::string error = "[ERROR] Fail to get the Output result from memory!";
      RETURN_STATUS_UNEXPECTED(error);
    }
    process.device_memory_release();
    process.Release();
    // Last part end where we transform the processed data into a tensor which can be applied in later units.
  } catch (const cv::Exception &e) {
    std::string error = "[ERROR] Fail in DvppDecodeResizeCropJpegOp:" + std::string(e.what());
    RETURN_STATUS_UNEXPECTED(error);
  }
  return Status::OK();
}

Status DvppDecodeResizeCropJpegOp::OutputShape(const std::vector<TensorShape> &inputs,
                                               std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  TensorShape out({-1, 1, 1});  // we don't know what is output image size, but we know it should be 3 channels
  if (inputs[0].Rank() == 1) outputs.emplace_back(out);
  if (!outputs.empty()) return Status::OK();
  return Status(StatusCode::kMDUnexpectedError, "Input has a wrong shape");
}

}  // namespace dataset
}  // namespace mindspore
