/*
 * Copyright (c) 2020.Huawei Technologies Co., Ltd. All rights reserved.
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

#include "AclProcess.h"
#include <sys/stat.h>
#include <sys/time.h>
#include <thread>

namespace {
const int BUFFER_SIZE = 2048;
const mode_t DEFAULT_FILE_PERMISSION = 0077;
}  // namespace

mode_t SetFileDefaultUmask() { return umask(DEFAULT_FILE_PERMISSION); }

/*
 * @description: Constructor
 * @param: resizeWidth specifies the resized width
 * @param: resizeHeight specifies the resized hegiht
 * @param: stream is used to maintain the execution order of operations
 * @param: context is used to manage the life cycle of objects
 * @param: dvppCommon is a class for decoding and resizing
 */
AclProcess::AclProcess(uint32_t resizeWidth, uint32_t resizeHeight, uint32_t cropWidth, uint32_t cropHeight,
                       aclrtContext context, aclrtStream stream, std::shared_ptr<DvppCommon> dvppCommon)
    : resizeWidth_(resizeWidth),
      resizeHeight_(resizeHeight),
      cropWidth_(cropWidth),
      cropHeight_(cropHeight),
      context_(context),
      stream_(stream),
      dvppCommon_(dvppCommon) {
  repeat_ = true;
}

/*
 * @description: Release AclProcess resources
 * @return: aclError which is error code of ACL API
 */
APP_ERROR AclProcess::Release() {
  // Release objects resource
  APP_ERROR ret = dvppCommon_->DeInit();
  dvppCommon_->ReleaseDvppBuffer();

  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to deinitialize dvppCommon_, ret = " << ret;
    return ret;
  }
  MS_LOG(INFO) << "dvppCommon_ object deinitialized successfully";
  dvppCommon_.reset();

  // Release stream
  if (stream_ != nullptr) {
    ret = aclrtDestroyStream(stream_);
    if (ret != APP_ERR_OK) {
      MS_LOG(ERROR) << "Failed to destroy stream, ret = " << ret;
      stream_ = nullptr;
      return ret;
    }
    stream_ = nullptr;
  }
  MS_LOG(INFO) << "The stream is destroyed successfully";
  return APP_ERR_OK;
}

/*
 * @description: Initialize DvppCommon object
 * @return: aclError which is error code of ACL API
 */
APP_ERROR AclProcess::InitModule() {
  // Create Dvpp JpegD object
  dvppCommon_ = std::make_shared<DvppCommon>(stream_);
  if (dvppCommon_ == nullptr) {
    MS_LOG(ERROR) << "Failed to create dvppCommon_ object";
    return APP_ERR_COMM_INIT_FAIL;
  }
  MS_LOG(INFO) << "DvppCommon object created successfully";
  APP_ERROR ret = dvppCommon_->Init();
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to initialize dvppCommon_ object, ret = " << ret;
    return ret;
  }
  MS_LOG(INFO) << "DvppCommon object initialized successfully";
  return APP_ERR_OK;
}

/*
 * @description: Initialize AclProcess resources
 * @return: aclError which is error code of ACL API
 */
APP_ERROR AclProcess::InitResource() {
  APP_ERROR ret = aclrtSetCurrentContext(context_);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to get ACL context, ret = " << ret;
    return ret;
  }
  MS_LOG(INFO) << "The context is created successfully";
  ret = aclrtCreateStream(&stream_);  // Create stream for application
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to create ACL stream, ret = " << ret;
    return ret;
  }
  MS_LOG(INFO) << "The stream is created successfully";
  // Initialize dvpp module
  if (InitModule() != APP_ERR_OK) {
    return APP_ERR_COMM_INIT_FAIL;
  }
  return APP_ERR_OK;
}

/*
 * @description: Read image files, and perform decoding and scaling
 * @param: imageFile specifies the image path to be processed
 * @return: aclError which is error code of ACL API
 */
APP_ERROR AclProcess::Preprocess(const RawData &ImageInfo) {
  // Decode process
  APP_ERROR ret = dvppCommon_->CombineJpegdProcess(ImageInfo, PIXEL_FORMAT_YUV_SEMIPLANAR_420, true);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to process decode, ret = " << ret << ".";
    return ret;
  }
  // Get output of decoded jpeg image
  std::shared_ptr<DvppDataInfo> decodeOutData = dvppCommon_->GetDecodedImage();
  if (decodeOutData == nullptr) {
    MS_LOG(ERROR) << "Decode output buffer is null.";
    return APP_ERR_COMM_INVALID_POINTER;
  }
  // Define output of resize jpeg image
  DvppDataInfo resizeOut;
  resizeOut.width = resizeWidth_;
  resizeOut.height = resizeHeight_;
  resizeOut.format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
  // Run resize application function
  ret = dvppCommon_->CombineResizeProcess(*(decodeOutData.get()), resizeOut, true);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to process resize, ret = " << ret << ".";
    return ret;
  }
  // Get output of resize jpeg image
  std::shared_ptr<DvppDataInfo> resizeOutData = dvppCommon_->GetResizedImage();
  if (resizeOutData == nullptr) {
    MS_LOG(ERROR) << "resize output buffer is null.";
    return APP_ERR_COMM_INVALID_POINTER;
  }
  // Define output of crop jpeg image
  DvppDataInfo cropOut;
  cropOut.width = cropWidth_;
  cropOut.height = cropHeight_;
  cropOut.format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
  // Define input of crop jpeg image
  DvppCropInputInfo cropInfo;
  cropInfo.dataInfo = *(resizeOutData.get());
  // Define crop parameters
  CropRoiConfig cropCfg;
  CropConfigFilter(cropCfg, cropInfo);
  ret = dvppCommon_->CombineCropProcess(cropInfo, cropOut, true);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to process center crop, ret = " << ret << ".";
    return ret;
  }
  return APP_ERR_OK;
}

/*
 * @description: Decode and scale the picture, and write the result to a file
 * @param: imageFile specifies the image path to be processed
 * @return: aclError which is error code of ACL API
 */
APP_ERROR AclProcess::Process(const RawData &ImageInfo) {
  struct timeval begin = {0};
  struct timeval end = {0};
  gettimeofday(&begin, nullptr);
  // deal with image
  APP_ERROR ret = Preprocess(ImageInfo);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to preprocess, ret = " << ret;
    return ret;
  }
  gettimeofday(&end, nullptr);
  // Calculate the time cost of preprocess
  const double costMs = SEC2MS * (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) / SEC2MS;
  const double fps = 1 * SEC2MS / costMs;
  MS_LOG(INFO) << "[dvpp Delay] cost: " << costMs << "ms\tfps: " << fps;
  // Get output of resize module
  std::shared_ptr<DvppDataInfo> CropOutData = dvppCommon_->GetCropedImage();
  if (CropOutData->dataSize == 0) {
    MS_LOG(ERROR) << "CropOutData return NULL";
    return APP_ERR_COMM_INVALID_POINTER;
  }
  // Alloc host memory for the inference output according to the size of output
  void *resHostBuf = nullptr;
  ret = aclrtMallocHost(&resHostBuf, CropOutData->dataSize);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to allocate memory from host ret = " << ret;
    return ret;
  }
  std::shared_ptr<void> outBuf(resHostBuf, aclrtFreeHost);
  processedInfo_ = outBuf;
  // Memcpy the output data from device to host
  ret = aclrtMemcpy(outBuf.get(), CropOutData->dataSize, CropOutData->data, CropOutData->dataSize,
                    ACL_MEMCPY_DEVICE_TO_HOST);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to copy memory from device to host, ret = " << ret;
    return ret;
  }
  return APP_ERR_OK;
}

void AclProcess::CropConfigFilter(CropRoiConfig &cfg, DvppCropInputInfo &cropinfo) {
  cfg.up = (resizeHeight_ - cropHeight_) / 2;
  if (cfg.up % 2 != 0) {
    cfg.up++;
  }
  cfg.down = resizeHeight_ - (resizeHeight_ - cropHeight_) / 2;
  if (cfg.down % 2 == 0) {
    cfg.down--;
  }
  cfg.left = (resizeWidth_ - cropWidth_) / 2;
  if (cfg.left % 2 != 0) {
    cfg.left++;
  }
  cfg.right = resizeWidth_ - (resizeWidth_ - cropWidth_) / 2;
  if (cfg.right % 2 == 0) {
    cfg.right--;
  }
  cropinfo.roi = cfg;
}

/*
 * @description: Obtain result data of memory
 * @param: processed_data is result data info pointer
 * @return: Address of data in the memory
 */
std::shared_ptr<void> AclProcess::Get_Memory_Data() { return processedInfo_; }

std::shared_ptr<DvppDataInfo> AclProcess::Get_Device_Memory_Data() { return dvppCommon_->GetCropedImage(); }

void AclProcess::set_mode(bool flag) { repeat_ = flag; }

bool AclProcess::get_mode() { return repeat_; }

void AclProcess::device_memory_release() { dvppCommon_->ReleaseDvppBuffer(); }
