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
#include "minddata/dataset/kernels/image/dvpp/dvpp_crop_jpeg_op.h"
#include <string>
#include <vector>
#include <iostream>
#include "include/api/context.h"
#include "minddata/dataset/core/data_type.h"
#include "minddata/dataset/core/device_tensor.h"
#include "minddata/dataset/kernels/image/dvpp/utils/CommonDataType.h"
#include "minddata/dataset/kernels/image/image_utils.h"

namespace mindspore {
namespace dataset {
Status DvppCropJpegOp::Compute(const std::shared_ptr<DeviceTensor> &input, std::shared_ptr<DeviceTensor> *output) {
  IO_CHECK(input, output);
  try {
    CHECK_FAIL_RETURN_UNEXPECTED(input->GetDeviceBuffer() != nullptr, "The input image buffer is empty.");
    std::string last_step = "Resize";
    DvppDataInfo *imageinfo = AclAdapter::GetInstance().GetResizedDeviceData(processor_.get());
    if (!imageinfo->data) {
      last_step = "Decode";
    }
    APP_ERROR ret = AclAdapter::GetInstance().JPEG_C(processor_.get(), last_step);
    if (ret != APP_ERR_OK) {
      ret = AclAdapter::GetInstance().ReleaseAclProcess(processor_.get());
      CHECK_FAIL_RETURN_UNEXPECTED(ret == APP_ERR_OK, "Release memory failed.");
      std::string error = "Error in dvpp crop processing:" + std::to_string(ret);
      RETURN_STATUS_UNEXPECTED(error);
    }
    DvppDataInfo *CropOut = AclAdapter::GetInstance().GetCropedDeviceData(processor_.get());
    const TensorShape dvpp_shape({1, 1, 1});
    const DataType dvpp_data_type(DataType::DE_UINT8);
    RETURN_IF_NOT_OK(mindspore::dataset::DeviceTensor::CreateEmpty(dvpp_shape, dvpp_data_type, output));
    RETURN_IF_NOT_OK((*output)->SetAttributes(CropOut->data, CropOut->dataSize, CropOut->width, CropOut->widthStride,
                                              CropOut->height, CropOut->heightStride));
    if (!((*output)->HasDeviceData())) {
      std::string error = "[ERROR] Fail to get the Output result from device memory!";
      RETURN_STATUS_UNEXPECTED(error);
    }
  } catch (const std::exception &e) {
    std::string error = "[ERROR] Fail in DvppCropJpegOp:" + std::string(e.what());
    RETURN_STATUS_UNEXPECTED(error);
  }
  return Status::OK();
}

Status DvppCropJpegOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  if (!IsNonEmptyJPEG(input)) {
    RETURN_STATUS_UNEXPECTED("DvppCropJpegOp only support process jpeg image.");
  }
  try {
    CHECK_FAIL_RETURN_UNEXPECTED(input->GetBuffer() != nullptr, "The input image buffer is empty.");
    unsigned char *buffer = const_cast<unsigned char *>(input->GetBuffer());
    DvppDataInfo imageinfo;
    imageinfo.dataSize = input->SizeInBytes();
    imageinfo.data = static_cast<uint8_t *>(buffer);
    std::vector<uint32_t> yuv_shape_ = input->GetYuvShape();
    const size_t yuv_shape_size = 4;
    CHECK_FAIL_RETURN_UNEXPECTED(yuv_shape_.size() == yuv_shape_size, "yuv_shape requires 4 elements.");
    imageinfo.width = yuv_shape_[0];
    imageinfo.widthStride = yuv_shape_[1];
    imageinfo.height = yuv_shape_[2];
    imageinfo.heightStride = yuv_shape_[3];
    imageinfo.format = 1;  // 1 means PIXEL_FORMAT_YUV_SEMIPLANAR_420
    ResourceInfo resource;
    resource.deviceIds.insert(0);
    APP_ERROR ret = AclAdapter::GetInstance().InitResource(&resource);
    if (ret != APP_ERR_OK) {
      AclAdapter::GetInstance().Release();
      std::string error = "Error in Init D-chip:" + std::to_string(ret);
      RETURN_STATUS_UNEXPECTED(error);
    }
    int deviceId = *(resource.deviceIds.begin());
    void *context = AclAdapter::GetInstance().GetContext(deviceId);
    // Second part end where we initialize the resource of D-chip and set up all configures
    std::shared_ptr<void> process(
      AclAdapter::GetInstance().CreateAclProcessWithPara(crop_width_, crop_height_, context, true, nullptr, nullptr),
      [](void *ptr) { AclAdapter::GetInstance().DestroyAclProcess(ptr); });
    ret = AclAdapter::GetInstance().InitAclProcess(process.get());
    if (ret != APP_ERR_OK) {
      AclAdapter::GetInstance().Release();
      std::string error = "Error in Init resource:" + std::to_string(ret);
      RETURN_STATUS_UNEXPECTED(error);
    }

    ret = AclAdapter::GetInstance().JPEG_C_WITH_DATA(process.get(), imageinfo);
    if (ret != APP_ERR_OK) {
      AclAdapter::GetInstance().Release();
      std::string error = "Error in dvpp crop processing:" + std::to_string(ret);
      RETURN_STATUS_UNEXPECTED(error);
    }

    // Third part end where we execute the core function of dvpp
    unsigned char *ret_ptr = static_cast<unsigned char *>(AclAdapter::GetInstance().GetMemoryData(process.get()));
    DvppDataInfo *CropOut(AclAdapter::GetInstance().GetCropedDeviceData(process.get()));
    dsize_t dvpp_length = CropOut->dataSize;
    const TensorShape dvpp_shape({dvpp_length, 1, 1});
    uint32_t crop_height = CropOut->height;
    uint32_t crop_heightStride = CropOut->heightStride;
    uint32_t crop_width = CropOut->width;
    uint32_t crop_widthStride = CropOut->widthStride;
    const DataType dvpp_data_type(DataType::DE_UINT8);
    RETURN_IF_NOT_OK(mindspore::dataset::Tensor::CreateFromMemory(dvpp_shape, dvpp_data_type, ret_ptr, output));
    RETURN_IF_NOT_OK((*output)->SetYuvShape(crop_width, crop_widthStride, crop_height, crop_heightStride));
    if (!((*output)->HasData())) {
      std::string error = "[ERROR] Fail to get the Output result from memory!";
      RETURN_STATUS_UNEXPECTED(error);
    }
    ret = AclAdapter::GetInstance().DeviceMemoryRelease(process.get());
    CHECK_FAIL_RETURN_UNEXPECTED(ret == APP_ERR_OK, "Release device memory failed.");
    ret = AclAdapter::GetInstance().ReleaseAclProcess(process.get());
    CHECK_FAIL_RETURN_UNEXPECTED(ret == APP_ERR_OK, "Release host memory failed.");
    // Last part end where we transform the processed data into a tensor which can be applied in later units.
  } catch (const std::exception &e) {
    std::string error = "[ERROR] Fail in DvppCropJpegOp:" + std::string(e.what());
    RETURN_STATUS_UNEXPECTED(error);
  }
  return Status::OK();
}

Status DvppCropJpegOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  TensorShape out({-1, 1, 1});  // we don't know what is output image size, but we know it should be 1 channels
  if (inputs.size() < 1) {
    RETURN_STATUS_UNEXPECTED("DvppCropJpegOp::OutputShape inputs is null");
  }
  if (inputs[0].Rank() == 1) {
    (void)outputs.emplace_back(out);
  }
  if (!outputs.empty()) {
    return Status::OK();
  }
  return Status(StatusCode::kMDUnexpectedError, "Input has a wrong shape");
}

Status DvppCropJpegOp::SetAscendResource(const std::shared_ptr<DeviceResource> &resource) {
  processor_ = resource->GetInstance();
  if (!processor_) {
    RETURN_STATUS_UNEXPECTED("Resource initialize fail, please check your env");
  }
  APP_ERROR ret = AclAdapter::GetInstance().SetCropParas(processor_.get(), crop_width_, crop_height_);
  CHECK_FAIL_RETURN_UNEXPECTED(ret == APP_ERR_OK, "SetCropParas failed.");
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
