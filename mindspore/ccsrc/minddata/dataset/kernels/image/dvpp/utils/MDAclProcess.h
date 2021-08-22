/*
 * Copyright (c) 2020-2021.Huawei Technologies Co., Ltd. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MDACLMANAGER_H
#define MDACLMANAGER_H

#include <climits>
#include <string>
#include <string.h>
#include <map>
#include <iostream>
#include <memory>
#include "acl/acl.h"
#include "CommonDataType.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/core/data_type.h"
#include "mindspore/ccsrc/minddata/dataset/core/device_tensor.h"
#include "mindspore/ccsrc/minddata/dataset/core/tensor.h"
#include "mindspore/core/utils/log_adapter.h"
#include "mindspore/ccsrc/minddata/dataset/util/status.h"
#include "ErrorCode.h"
#include "DvppCommon.h"
#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

mode_t SetFileDefaultUmask();

class MDAclProcess {
 public:
  MDAclProcess(uint32_t resizeWidth, uint32_t resizeHeight, uint32_t cropWidth, uint32_t cropHeight,
               aclrtContext context, bool is_crop = true, aclrtStream stream = nullptr,
               std::shared_ptr<DvppCommon> dvppCommon = nullptr);

  MDAclProcess(uint32_t ParaWidth, uint32_t ParaHeight, aclrtContext context, bool is_crop = false,
               aclrtStream stream = nullptr, std::shared_ptr<DvppCommon> dvppCommon = nullptr);

  MDAclProcess(aclrtContext context, bool is_crop = false, aclrtStream stream = nullptr,
               std::shared_ptr<DvppCommon> dvppCommon = nullptr);

  ~MDAclProcess(){};

  // Release all the resource
  APP_ERROR Release();
  // Create resource for this sample
  APP_ERROR InitResource();
  // Get Ascend Resource core: context and stream which are created when InitResource()
  aclrtContext GetContext();
  aclrtContext GetStream();

  // Process the result
  APP_ERROR JPEG_DRC(const RawData &ImageInfo);
  APP_ERROR JPEG_DRC();
  // Procss the image without crop
  APP_ERROR JPEG_DR(const RawData &ImageInfo);
  APP_ERROR JPEG_DR();
  // Process the JPEG image only with decode
  APP_ERROR JPEG_D(const RawData &ImageInfo);
  APP_ERROR JPEG_D();
  // Process the JPEG image only with resize
  APP_ERROR JPEG_R(const DvppDataInfo &ImageInfo);
  APP_ERROR JPEG_R(std::string &last_step);
  // Process the JPEG image only with crop
  APP_ERROR JPEG_C(const DvppDataInfo &ImageInfo);
  APP_ERROR JPEG_C(std::string &last_step);
  // Process the PNG image only with decode
  APP_ERROR PNG_D(const RawData &ImageInfo);
  APP_ERROR PNG_D();
  // API for access memory
  std::shared_ptr<void> Get_Memory_Data();
  // API for access device memory of croped data
  std::shared_ptr<DvppDataInfo> Get_Croped_DeviceData();
  // API for access device memory of resized data
  std::shared_ptr<DvppDataInfo> Get_Resized_DeviceData();
  // API for access device memory of decode data
  std::shared_ptr<DvppDataInfo> Get_Decode_DeviceData();

  APP_ERROR H2D_Sink(const std::shared_ptr<mindspore::dataset::Tensor> &input,
                     std::shared_ptr<mindspore::dataset::DeviceTensor> &device_input);

  APP_ERROR D2H_Pop(const std::shared_ptr<mindspore::dataset::DeviceTensor> &device_output,
                    std::shared_ptr<mindspore::dataset::Tensor> &output);

  // D-chip memory release
  APP_ERROR device_memory_release();

  std::shared_ptr<DvppCommon> GetDeviceModule();

  std::vector<uint32_t> Get_Primary_Shape();

  // Set Dvpp parameters
  APP_ERROR SetResizeParas(uint32_t width, uint32_t height);
  APP_ERROR SetCropParas(uint32_t width, uint32_t height);

 private:
  // Crop definition
  void CropConfigFilter(CropRoiConfig &cfg, DvppCropInputInfo &cropinfo, DvppDataInfo &resizeinfo);
  // Resize definition
  APP_ERROR ResizeConfigFilter(DvppDataInfo &resizeinfo, const uint32_t pri_w_, const uint32_t pri_h_);
  // Initialize DVPP modules used by this sample
  APP_ERROR InitModule();
  // Dvpp process with crop
  APP_ERROR JPEG_DRC_(const RawData &ImageInfo);
  // Dvpp process without crop
  APP_ERROR JPEG_DR_(const RawData &ImageInfo);
  // Impl of JPEG_D
  APP_ERROR JPEG_D_(const RawData &ImageInfo);
  APP_ERROR JPEG_D_();
  // Impl of JPEG_R
  APP_ERROR JPEG_R_(const DvppDataInfo &ImageInfo);
  APP_ERROR JPEG_R_(std::string &last_step);
  // Impl of JPEG_C
  APP_ERROR JPEG_C_(const DvppDataInfo &ImageInfo);
  APP_ERROR JPEG_C_(std::string &last_step);
  // Impl of PNG_D
  APP_ERROR PNG_D_(const RawData &ImageInfo);
  APP_ERROR PNG_D_();

  aclrtContext context_;
  aclrtStream stream_;
  std::shared_ptr<DvppCommon> dvppCommon_;  // dvpp object
  std::shared_ptr<void> processedInfo_;     // processed data (On host)
  uint32_t resizeWidth_;                    // dvpp resize width
  uint32_t resizeHeight_;                   // dvpp resize height
  uint32_t cropWidth_;                      // dvpp crop width
  uint32_t cropHeight_;                     // dvpp crop height
  bool contain_crop_;                       // Initialize with crop or not
};

#endif
