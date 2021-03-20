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

#include <string>
#include <vector>
#include <iostream>
#include "include/api/context.h"
#include "minddata/dataset/core/cv_tensor.h"
#include "minddata/dataset/core/data_type.h"
#include "minddata/dataset/core/device_tensor.h"
#include "minddata/dataset/kernels/image/dvpp/dvpp_resize_jpeg_op.h"
#include "minddata/dataset/kernels/image/dvpp/utils/CommonDataType.h"
#include "minddata/dataset/kernels/image/dvpp/utils/MDAclProcess.h"
#include "minddata/dataset/kernels/image/image_utils.h"

namespace mindspore {
namespace dataset {
Status DvppResizeJpegOp::Compute(const std::shared_ptr<DeviceTensor> &input, std::shared_ptr<DeviceTensor> *output) {
  IO_CHECK(input, output);
  try {
    CHECK_FAIL_RETURN_UNEXPECTED(input->GetDeviceBuffer() != nullptr, "The input image buffer is empty.");
    std::string last_step = "Decode";
    std::shared_ptr<DvppDataInfo> imageinfo(processor_->Get_Decode_DeviceData());
    if (!imageinfo->data) {
      last_step = "Crop";
    }
    APP_ERROR ret = processor_->JPEG_R(last_step);
    if (ret != APP_ERR_OK) {
      processor_->Release();
      std::string error = "Error in dvpp processing:" + std::to_string(ret);
      RETURN_STATUS_UNEXPECTED(error);
    }
    std::shared_ptr<DvppDataInfo> ResizeOut(processor_->Get_Resized_DeviceData());
    const TensorShape dvpp_shape({1, 1, 1});
    const DataType dvpp_data_type(DataType::DE_UINT8);
    mindspore::dataset::DeviceTensor::CreateEmpty(dvpp_shape, dvpp_data_type, output);
    (*output)->SetAttributes(ResizeOut->data, ResizeOut->dataSize, ResizeOut->width, ResizeOut->widthStride,
                             ResizeOut->height, ResizeOut->heightStride);
    if (!((*output)->HasDeviceData())) {
      std::string error = "[ERROR] Fail to get the Output result from device memory!";
      RETURN_STATUS_UNEXPECTED(error);
    }
  } catch (const cv::Exception &e) {
    std::string error = "[ERROR] Fail in DvppResizeJpegOp:" + std::string(e.what());
    RETURN_STATUS_UNEXPECTED(error);
  }
  return Status::OK();
}

Status DvppResizeJpegOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  if (!IsNonEmptyJPEG(input)) {
    RETURN_STATUS_UNEXPECTED("DvppReiszeJpegOp only support process jpeg image.");
  }
  try {
    CHECK_FAIL_RETURN_UNEXPECTED(input->GetBuffer() != nullptr, "The input image buffer is empty.");
    unsigned char *buffer = const_cast<unsigned char *>(input->GetBuffer());
    DvppDataInfo imageinfo;
    imageinfo.dataSize = input->SizeInBytes();
    imageinfo.data = static_cast<uint8_t *>(buffer);
    std::vector<uint32_t> yuv_shape_ = input->GetYuvShape();
    imageinfo.width = yuv_shape_[0];
    imageinfo.widthStride = yuv_shape_[1];
    imageinfo.height = yuv_shape_[2];
    imageinfo.heightStride = yuv_shape_[3];
    imageinfo.format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    ResourceInfo resource;
    resource.aclConfigPath = "";
    resource.deviceIds.insert(0);
    std::shared_ptr<ResourceManager> instance = ResourceManager::GetInstance();
    APP_ERROR ret = instance->InitResource(resource);
    if (ret != APP_ERR_OK) {
      instance->Release();
      std::string error = "Error in Init D-chip:" + std::to_string(ret);
      RETURN_STATUS_UNEXPECTED(error);
    }
    int deviceId = *(resource.deviceIds.begin());
    aclrtContext context = instance->GetContext(deviceId);
    // Second part end where we initialize the resource of D-chip and set up all configures
    MDAclProcess process(resized_width_, resized_height_, context, false);
    ret = process.InitResource();
    if (ret != APP_ERR_OK) {
      instance->Release();
      std::string error = "Error in Init resource:" + std::to_string(ret);
      RETURN_STATUS_UNEXPECTED(error);
    }

    ret = process.JPEG_R(imageinfo);
    if (ret != APP_ERR_OK) {
      instance->Release();
      std::string error = "Error in dvpp processing:" + std::to_string(ret);
      RETURN_STATUS_UNEXPECTED(error);
    }

    // Third part end where we execute the core function of dvpp
    auto data = std::static_pointer_cast<unsigned char>(process.Get_Memory_Data());
    unsigned char *ret_ptr = data.get();
    std::shared_ptr<DvppDataInfo> ResizeOut(process.Get_Resized_DeviceData());
    dsize_t dvpp_length = ResizeOut->dataSize;
    const TensorShape dvpp_shape({dvpp_length, 1, 1});
    uint32_t resized_height = ResizeOut->height;
    uint32_t resized_heightStride = ResizeOut->heightStride;
    uint32_t resized_width = ResizeOut->width;
    uint32_t resized_widthStride = ResizeOut->widthStride;
    const DataType dvpp_data_type(DataType::DE_UINT8);
    mindspore::dataset::Tensor::CreateFromMemory(dvpp_shape, dvpp_data_type, ret_ptr, output);
    (*output)->SetYuvShape(resized_width, resized_widthStride, resized_height, resized_heightStride);
    if (!((*output)->HasData())) {
      std::string error = "[ERROR] Fail to get the Output result from memory!";
      RETURN_STATUS_UNEXPECTED(error);
    }
    process.device_memory_release();
    process.Release();
    // Last part end where we transform the processed data into a tensor which can be applied in later units.
  } catch (const cv::Exception &e) {
    std::string error = "[ERROR] Fail in DvppResizeJpegOp:" + std::string(e.what());
    RETURN_STATUS_UNEXPECTED(error);
  }
  return Status::OK();
}

Status DvppResizeJpegOp::SetAscendResource(const std::shared_ptr<DeviceResource> &resource) {
  processor_ = std::static_pointer_cast<MDAclProcess>(resource->GetInstance());
  if (!processor_) {
    RETURN_STATUS_UNEXPECTED("Resource initialize fail, please check your env");
  }
  processor_->SetResizeParas(resized_width_, resized_height_);
  return Status::OK();
}

Status DvppResizeJpegOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  TensorShape out({-1, 1, 1});  // we don't know what is output image size, but we know it should be 1 channels
  if (inputs[0].Rank() == 1) outputs.emplace_back(out);
  if (!outputs.empty()) return Status::OK();
  return Status(StatusCode::kMDUnexpectedError, "Input has a wrong shape");
}

}  // namespace dataset
}  // namespace mindspore
