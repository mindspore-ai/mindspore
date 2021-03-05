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

#include "minddata/dataset/include/constants.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/kernels/image/image_utils.h"
#include "MDAclProcess.h"
#include <sys/time.h>
#include <thread>
#include <sys/stat.h>

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
MDAclProcess::MDAclProcess(uint32_t resizeWidth, uint32_t resizeHeight, uint32_t cropWidth, uint32_t cropHeight,
                           aclrtContext context, bool is_crop, aclrtStream stream,
                           std::shared_ptr<DvppCommon> dvppCommon)
    : resizeWidth_(resizeWidth),
      resizeHeight_(resizeHeight),
      cropWidth_(cropWidth),
      cropHeight_(cropHeight),
      context_(context),
      stream_(stream),
      contain_crop_(is_crop),
      dvppCommon_(dvppCommon),
      processedInfo_(nullptr) {}

MDAclProcess::MDAclProcess(uint32_t ParaWidth, uint32_t ParaHeight, aclrtContext context, bool is_crop,
                           aclrtStream stream, std::shared_ptr<DvppCommon> dvppCommon)
    : contain_crop_(is_crop), context_(context), stream_(stream), dvppCommon_(dvppCommon), processedInfo_(nullptr) {
  if (is_crop) {
    resizeWidth_ = 0;
    resizeHeight_ = 0;
    cropWidth_ = ParaWidth;
    cropHeight_ = ParaHeight;
  } else {
    resizeWidth_ = ParaWidth;
    resizeHeight_ = ParaHeight;
    cropWidth_ = 0;
    cropHeight_ = 0;
  }
}

MDAclProcess::MDAclProcess(aclrtContext context, bool is_crop, aclrtStream stream,
                           std::shared_ptr<DvppCommon> dvppCommon)
    : resizeWidth_(0),
      resizeHeight_(0),
      cropWidth_(0),
      cropHeight_(0),
      contain_crop_(is_crop),
      context_(context),
      stream_(stream),
      dvppCommon_(dvppCommon),
      processedInfo_(nullptr) {}
/*
 * @description: Release MDAclProcess resources
 * @return: aclError which is error code of ACL API
 */
APP_ERROR MDAclProcess::Release() {
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
APP_ERROR MDAclProcess::InitModule() {
  // Create Dvpp JpegD object
  dvppCommon_ = std::make_shared<DvppCommon>(context_, stream_);
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
 * @description: Initialize MDAclProcess resources
 * @return: aclError which is error code of ACL API
 */
APP_ERROR MDAclProcess::InitResource() {
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

std::shared_ptr<DvppCommon> MDAclProcess::GetDeviceModule() { return dvppCommon_; }

aclrtContext MDAclProcess::GetContext() { return context_; }

aclrtStream MDAclProcess::GetStream() { return stream_; }

/*
 * Sink data from Tensor(On host) to DeviceTensor(On device)
 * Two cases are different, jpeg and png
 */
APP_ERROR MDAclProcess::H2D_Sink(const std::shared_ptr<mindspore::dataset::Tensor> &input,
                                 std::shared_ptr<mindspore::dataset::DeviceTensor> &device_input) {
  // Recall the context created in InitResource()
  APP_ERROR ret = aclrtSetCurrentContext(context_);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to get ACL context, ret = " << ret;
    return ret;
  }

  RawData imageinfo;
  uint32_t filesize = input->SizeInBytes();
  // MS_LOG(INFO) << "Filesize on host is: " << filesize;
  imageinfo.lenOfByte = filesize;
  unsigned char *buffer = const_cast<unsigned char *>(input->GetBuffer());
  imageinfo.data = static_cast<void *>(buffer);

  // Transfer RawData(Raw image) from host to device, which we call sink
  if (IsNonEmptyJPEG(input)) {  // case JPEG
    ret = dvppCommon_->SinkImageH2D(imageinfo, PIXEL_FORMAT_YUV_SEMIPLANAR_420);
  } else {  // case PNG
    ret = dvppCommon_->SinkImageH2D(imageinfo);
  }
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to transport Tensor to device, ret = " << ret;
    return ret;
  }
  auto deviceInputData = dvppCommon_->GetInputImage();
  // std::cout << "[DEBUG]Sink data sunccessfully, Filesize on device is: " << deviceInputData->dataSize << std::endl;
  const mindspore::dataset::DataType dvpp_data_type(mindspore::dataset::DataType::DE_UINT8);
  const mindspore::dataset::TensorShape dvpp_shape({1, 1, 1});
  mindspore::dataset::DeviceTensor::CreateEmpty(dvpp_shape, dvpp_data_type, &device_input);
  device_input->SetAttributes(deviceInputData->data, deviceInputData->dataSize, deviceInputData->width,
                              deviceInputData->widthStride, deviceInputData->height, deviceInputData->heightStride);
  return APP_ERR_OK;
}

APP_ERROR MDAclProcess::D2H_Pop(const std::shared_ptr<mindspore::dataset::DeviceTensor> &device_output,
                                std::shared_ptr<mindspore::dataset::Tensor> &output) {
  void *resHostBuf = nullptr;
  APP_ERROR ret = aclrtMallocHost(&resHostBuf, device_output->DeviceDataSize());
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to allocate memory from host ret = " << ret;
    return ret;
  }
  std::shared_ptr<void> outBuf(resHostBuf, aclrtFreeHost);
  processedInfo_ = outBuf;
  // Memcpy the output data from device to host
  ret = aclrtMemcpy(outBuf.get(), device_output->DeviceDataSize(), device_output->GetDeviceBuffer(),
                    device_output->DeviceDataSize(), ACL_MEMCPY_DEVICE_TO_HOST);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to copy memory from device to host, ret = " << ret;
    return ret;
  }
  auto data = std::static_pointer_cast<unsigned char>(processedInfo_);
  unsigned char *ret_ptr = data.get();

  mindspore::dataset::dsize_t dvppDataSize = device_output->DeviceDataSize();
  const mindspore::dataset::TensorShape dvpp_shape({dvppDataSize, 1, 1});
  uint32_t _output_width_ = device_output->GetYuvStrideShape()[0];
  uint32_t _output_widthStride_ = device_output->GetYuvStrideShape()[1];
  uint32_t _output_height_ = device_output->GetYuvStrideShape()[2];
  uint32_t _output_heightStride_ = device_output->GetYuvStrideShape()[3];
  const mindspore::dataset::DataType dvpp_data_type(mindspore::dataset::DataType::DE_UINT8);
  mindspore::dataset::Tensor::CreateFromMemory(dvpp_shape, dvpp_data_type, ret_ptr, &output);
  output->SetYuvShape(_output_width_, _output_widthStride_, _output_height_, _output_heightStride_);
  if (!output->HasData()) {
    return APP_ERR_COMM_ALLOC_MEM;
  }
  return APP_ERR_OK;
}

APP_ERROR MDAclProcess::JPEG_D(const RawData &ImageInfo) {
  MS_LOG(WARNING) << "It's deprecated to use kCpu as input device for Dvpp operators to compute, because it's slow and "
                     "unsafe, we recommend you to set input device as MapTargetDevice::kAscend for Dvpp operators. "
                     "This API will be removed later";
  struct timeval begin = {0};
  struct timeval end = {0};
  gettimeofday(&begin, nullptr);
  // deal with image
  APP_ERROR ret = JPEG_D_(ImageInfo);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to decode, ret = " << ret;
    return ret;
  }
  gettimeofday(&end, nullptr);
  // Calculate the time cost of preprocess
  const double costMs = SEC2MS * (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) / SEC2MS;
  const double fps = 1 * SEC2MS / costMs;
  MS_LOG(INFO) << "[dvpp decode Delay] cost: " << costMs << "ms\tfps: " << fps;
  // Get output of resize module
  std::shared_ptr<DvppDataInfo> DecodeOutData = dvppCommon_->GetDecodedImage();
  if (!DecodeOutData) {
    MS_LOG(ERROR) << "Decode Data returns NULL";
    return APP_ERR_COMM_INVALID_POINTER;
  }

  // Alloc host memory for the inference output according to the size of output
  void *resHostBuf = nullptr;
  ret = aclrtMallocHost(&resHostBuf, DecodeOutData->dataSize);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to allocate memory from host ret = " << ret;
    return ret;
  }
  std::shared_ptr<void> outBuf(resHostBuf, aclrtFreeHost);
  processedInfo_ = outBuf;
  // Memcpy the output data from device to host
  ret = aclrtMemcpy(outBuf.get(), DecodeOutData->dataSize, DecodeOutData->data, DecodeOutData->dataSize,
                    ACL_MEMCPY_DEVICE_TO_HOST);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to copy memory from device to host, ret = " << ret;
    return ret;
  }
  return APP_ERR_OK;
}

APP_ERROR MDAclProcess::JPEG_D() {
  struct timeval begin = {0};
  struct timeval end = {0};
  gettimeofday(&begin, nullptr);
  // deal with image
  APP_ERROR ret = JPEG_D_();
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to decode, ret = " << ret;
    return ret;
  }
  gettimeofday(&end, nullptr);
  // Calculate the time cost of preprocess
  const double costMs = SEC2MS * (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) / SEC2MS;
  const double fps = 1 * SEC2MS / costMs;
  MS_LOG(INFO) << "[dvpp decode Delay] cost: " << costMs << "ms\tfps: " << fps;
  // Get output of resize module
  std::shared_ptr<DvppDataInfo> DecodeOutData = dvppCommon_->GetDecodedImage();
  if (!DecodeOutData) {
    MS_LOG(ERROR) << "Decode Data returns NULL";
    return APP_ERR_COMM_INVALID_POINTER;
  }
  return APP_ERR_OK;
}

APP_ERROR MDAclProcess::JPEG_D_(const RawData &ImageInfo) {
  APP_ERROR ret = dvppCommon_->CombineJpegdProcess(ImageInfo, PIXEL_FORMAT_YUV_SEMIPLANAR_420, true);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to process decode, ret = " << ret << ".";
    return ret;
  }
  return APP_ERR_OK;
}

APP_ERROR MDAclProcess::JPEG_D_() {
  auto input_ = dvppCommon_->GetInputImage();
  auto decode_output_ = dvppCommon_->GetDecodedImage();
  APP_ERROR ret = dvppCommon_->SinkCombineJpegdProcess(input_, decode_output_, true);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to process decode, ret = " << ret << ".";
    return ret;
  }
  return APP_ERR_OK;
}

APP_ERROR MDAclProcess::JPEG_R(const DvppDataInfo &ImageInfo) {
  MS_LOG(WARNING) << "It's deprecated to use kCpu as input device for Dvpp operators to compute, because it's slow and "
                     "unsafe, we recommend you to set input device as MapTargetDevice::kAscend for Dvpp operators. "
                     "This API will be removed later";
  struct timeval begin = {0};
  struct timeval end = {0};
  gettimeofday(&begin, nullptr);
  // deal with image
  APP_ERROR ret = JPEG_R_(ImageInfo);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to resize, ret = " << ret;
    return ret;
  }
  gettimeofday(&end, nullptr);
  // Calculate the time cost of preprocess
  const double costMs = SEC2MS * (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) / SEC2MS;
  const double fps = 1 * SEC2MS / costMs;
  MS_LOG(INFO) << "[dvpp resize Delay] cost: " << costMs << "ms\tfps: " << fps;
  // Get output of resize module
  std::shared_ptr<DvppDataInfo> ResizeOutData = dvppCommon_->GetResizedImage();
  if (!ResizeOutData) {
    MS_LOG(ERROR) << "Resize Data returns NULL";
    return APP_ERR_COMM_INVALID_POINTER;
  }
  // Alloc host memory for the inference output according to the size of output
  void *resHostBuf = nullptr;
  ret = aclrtMallocHost(&resHostBuf, ResizeOutData->dataSize);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to allocate memory from host ret = " << ret;
    return ret;
  }
  std::shared_ptr<void> outBuf(resHostBuf, aclrtFreeHost);
  processedInfo_ = outBuf;
  // Memcpy the output data from device to host
  ret = aclrtMemcpy(outBuf.get(), ResizeOutData->dataSize, ResizeOutData->data, ResizeOutData->dataSize,
                    ACL_MEMCPY_DEVICE_TO_HOST);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to copy memory from device to host, ret = " << ret;
    return ret;
  }
  return APP_ERR_OK;
}

APP_ERROR MDAclProcess::JPEG_R(std::string &last_step) {
  struct timeval begin = {0};
  struct timeval end = {0};
  gettimeofday(&begin, nullptr);
  // deal with image
  APP_ERROR ret = JPEG_R_(last_step);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to resize, ret = " << ret;
    return ret;
  }
  gettimeofday(&end, nullptr);
  // Calculate the time cost of preprocess
  const double costMs = SEC2MS * (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) / SEC2MS;
  const double fps = 1 * SEC2MS / costMs;
  MS_LOG(INFO) << "[dvpp resize Delay] cost: " << costMs << "ms\tfps: " << fps;
  // Get output of resize module
  std::shared_ptr<DvppDataInfo> ResizeOutData = dvppCommon_->GetResizedImage();
  if (!ResizeOutData) {
    MS_LOG(ERROR) << "Resize Data returns NULL";
    return APP_ERR_COMM_INVALID_POINTER;
  }
  return APP_ERR_OK;
}

APP_ERROR MDAclProcess::JPEG_R_(const DvppDataInfo &ImageInfo) {
  APP_ERROR ret = dvppCommon_->TransferYuvDataH2D(ImageInfo);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to copy data from host to device, ret = " << ret << ".";
    return ret;
  }
  std::shared_ptr<DvppDataInfo> decoded_image = dvppCommon_->GetDecodedImage();
  uint32_t pri_h = decoded_image->heightStride;
  uint32_t pri_w = decoded_image->widthStride;
  // Define the resize shape
  DvppDataInfo resizeOut;
  ResizeConfigFilter(resizeOut, pri_w, pri_h);
  ret = dvppCommon_->CombineResizeProcess(*decoded_image, resizeOut, true);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to process resize, ret = " << ret << ".";
    return ret;
  }
  return APP_ERR_OK;
}

APP_ERROR MDAclProcess::JPEG_R_(std::string &last_step) {
  std::shared_ptr<DvppDataInfo> input_image = std::make_shared<DvppDataInfo>();
  if (last_step == "Decode") {
    input_image = dvppCommon_->GetDecodedImage();
  } else {
    input_image = dvppCommon_->GetCropedImage();
  }
  if (!input_image->data) {
    MS_LOG(ERROR) << "Failed to get data for resize, please verify last step operation";
    return APP_ERR_DVPP_RESIZE_FAIL;
  }
  uint32_t pri_h = input_image->heightStride;
  uint32_t pri_w = input_image->widthStride;
  DvppDataInfo resizeOut;
  ResizeConfigFilter(resizeOut, pri_w, pri_h);
  APP_ERROR ret = dvppCommon_->CombineResizeProcess(*input_image, resizeOut, true);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to process resize, ret = " << ret << ".";
    return ret;
  }
  return APP_ERR_OK;
}

APP_ERROR MDAclProcess::JPEG_C(const DvppDataInfo &ImageInfo) {
  MS_LOG(WARNING) << "It's deprecated to use kCpu as input device for Dvpp operators to compute, because it's slow and "
                     "unsafe, we recommend you to set input device as MapTargetDevice::kAscend for Dvpp operators. "
                     "This API will be removed later";
  struct timeval begin = {0};
  struct timeval end = {0};
  gettimeofday(&begin, nullptr);
  // deal with image
  APP_ERROR ret = JPEG_C_(ImageInfo);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to crop image, ret = " << ret;
    return ret;
  }
  gettimeofday(&end, nullptr);
  // Calculate the time cost of preprocess
  const double costMs = SEC2MS * (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) / SEC2MS;
  const double fps = 1 * SEC2MS / costMs;
  MS_LOG(INFO) << "[dvpp crop Delay] cost: " << costMs << "ms\tfps: " << fps;
  // Get output of resize module
  std::shared_ptr<DvppDataInfo> CropOutData = dvppCommon_->GetCropedImage();
  if (!CropOutData) {
    MS_LOG(ERROR) << "Crop Data returns NULL";
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

APP_ERROR MDAclProcess::JPEG_C(std::string &last_step) {
  struct timeval begin = {0};
  struct timeval end = {0};
  gettimeofday(&begin, nullptr);
  // deal with image
  APP_ERROR ret = JPEG_C_(last_step);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to crop image, ret = " << ret;
    return ret;
  }
  gettimeofday(&end, nullptr);
  // Calculate the time cost of preprocess
  const double costMs = SEC2MS * (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) / SEC2MS;
  const double fps = 1 * SEC2MS / costMs;
  MS_LOG(INFO) << "[dvpp crop Delay] cost: " << costMs << "ms\tfps: " << fps;
  // Get output of resize module
  std::shared_ptr<DvppDataInfo> CropOutData = dvppCommon_->GetCropedImage();
  if (!CropOutData) {
    MS_LOG(ERROR) << "Crop Data returns NULL";
    return APP_ERR_COMM_INVALID_POINTER;
  }
  return APP_ERR_OK;
}

APP_ERROR MDAclProcess::JPEG_C_(const DvppDataInfo &ImageInfo) {
  APP_ERROR ret = dvppCommon_->TransferYuvDataH2D(ImageInfo);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to copy data from host to device, ret = " << ret << ".";
    return ret;
  }
  // Unneccessary to be image after resize, maybe after decode, we store both of them in DecodedImage()
  std::shared_ptr<DvppDataInfo> resized_image = dvppCommon_->GetDecodedImage();
  uint32_t pri_h = resized_image->heightStride;
  uint32_t pri_w = resized_image->widthStride;
  // Validate the crop shape
  DvppDataInfo cropOut;
  cropOut.width = cropWidth_;
  cropOut.height = cropHeight_;
  if (cropOut.width > pri_w || cropOut.height > pri_h) {
    MS_LOG(ERROR) << "Crop size can not excceed resize, please verify your input [CROP SIZE] parameters";
    return APP_ERR_COMM_INVALID_PARAM;
  }
  cropOut.format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
  DvppCropInputInfo cropInfo;
  cropInfo.dataInfo = *resized_image;
  // Define crop area
  CropRoiConfig cropCfg;
  CropConfigFilter(cropCfg, cropInfo, *resized_image);
  ret = dvppCommon_->CombineCropProcess(cropInfo, cropOut, true);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to process center crop, ret = " << ret << ".";
    return ret;
  }
  return APP_ERR_OK;
}

APP_ERROR MDAclProcess::JPEG_C_(std::string &last_step) {
  std::shared_ptr<DvppDataInfo> input_image = std::make_shared<DvppDataInfo>();
  if (last_step == "Resize") {
    input_image = dvppCommon_->GetResizedImage();
  } else {
    input_image = dvppCommon_->GetDecodedImage();
  }
  if (!input_image->data) {
    MS_LOG(ERROR) << "Failed to get input data for crop, please verify last step operation";
    return APP_ERR_DVPP_CROP_FAIL;
  }
  uint32_t pri_h = input_image->heightStride;
  uint32_t pri_w = input_image->widthStride;
  DvppDataInfo cropOut;
  cropOut.width = cropWidth_;
  cropOut.height = cropHeight_;
  if (cropOut.width > pri_w || cropOut.height > pri_h) {
    MS_LOG(ERROR) << "Crop size can not excceed resize, please verify your input [CROP SIZE] parameters";
    return APP_ERR_COMM_INVALID_PARAM;
  }
  cropOut.format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
  DvppCropInputInfo cropInfo;
  cropInfo.dataInfo = *input_image;
  // Define crop area
  CropRoiConfig cropCfg;
  CropConfigFilter(cropCfg, cropInfo, *input_image);
  APP_ERROR ret = dvppCommon_->CombineCropProcess(cropInfo, cropOut, true);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to process center crop, ret = " << ret << ".";
    return ret;
  }
  return APP_ERR_OK;
}

APP_ERROR MDAclProcess::PNG_D(const RawData &ImageInfo) {
  MS_LOG(WARNING) << "It's deprecated to use kCpu as input device for Dvpp operators to compute, because it's slow and "
                     "unsafe, we recommend you to set input device as MapTargetDevice::kAscend for Dvpp operators. "
                     "This API will be removed later";
  struct timeval begin = {0};
  struct timeval end = {0};
  gettimeofday(&begin, nullptr);
  // deal with image
  APP_ERROR ret = PNG_D_(ImageInfo);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to decode, ret = " << ret;
    return ret;
  }
  gettimeofday(&end, nullptr);
  // Calculate the time cost of preprocess
  const double costMs = SEC2MS * (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) / SEC2MS;
  const double fps = 1 * SEC2MS / costMs;
  MS_LOG(INFO) << "[dvpp Delay] cost: " << costMs << "ms\tfps: " << fps;
  // Get output of resize module
  std::shared_ptr<DvppDataInfo> DecodeOutData = dvppCommon_->GetDecodedImage();
  if (!DecodeOutData) {
    MS_LOG(ERROR) << "ResizedOutData returns NULL";
    return APP_ERR_COMM_INVALID_POINTER;
  }
  // Alloc host memory for the inference output according to the size of output
  void *resHostBuf = nullptr;
  ret = aclrtMallocHost(&resHostBuf, DecodeOutData->dataSize);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to allocate memory from host ret = " << ret;
    return ret;
  }
  std::shared_ptr<void> outBuf(resHostBuf, aclrtFreeHost);
  processedInfo_ = outBuf;
  // Memcpy the output data from device to host
  ret = aclrtMemcpy(outBuf.get(), DecodeOutData->dataSize, DecodeOutData->data, DecodeOutData->dataSize,
                    ACL_MEMCPY_DEVICE_TO_HOST);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to copy memory from device to host, ret = " << ret;
    return ret;
  }
  return APP_ERR_OK;
}

APP_ERROR MDAclProcess::PNG_D() {
  struct timeval begin = {0};
  struct timeval end = {0};
  gettimeofday(&begin, nullptr);
  // deal with image
  APP_ERROR ret = PNG_D_();
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to decode, ret = " << ret;
    return ret;
  }
  gettimeofday(&end, nullptr);
  // Calculate the time cost of preprocess
  const double costMs = SEC2MS * (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) / SEC2MS;
  const double fps = 1 * SEC2MS / costMs;
  MS_LOG(INFO) << "[dvpp decode Delay] cost: " << costMs << "ms\tfps: " << fps;
  // Get output of resize module
  std::shared_ptr<DvppDataInfo> DecodeOutData = dvppCommon_->GetDecodedImage();
  if (!DecodeOutData) {
    MS_LOG(ERROR) << "Decode Data returns NULL";
    return APP_ERR_COMM_INVALID_POINTER;
  }
  return APP_ERR_OK;
}

APP_ERROR MDAclProcess::PNG_D_(const RawData &ImageInfo) {
  APP_ERROR ret = dvppCommon_->CombinePngdProcess(ImageInfo, PIXEL_FORMAT_RGB_888, true);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to process decode, ret = " << ret << ".";
    return ret;
  }
  return APP_ERR_OK;
}

APP_ERROR MDAclProcess::PNG_D_() {
  auto input_ = dvppCommon_->GetInputImage();
  auto decode_output_ = dvppCommon_->GetDecodedImage();
  APP_ERROR ret = dvppCommon_->SinkCombinePngdProcess(input_, decode_output_, true);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to process decode, ret = " << ret << ".";
    return ret;
  }
  return APP_ERR_OK;
}

/*
 * @description: Decode and scale the picture, and write the result to a file
 * @param: imageFile specifies the image path to be processed
 * @return: aclError which is error code of ACL API
 */
APP_ERROR MDAclProcess::JPEG_DRC(const RawData &ImageInfo) {
  MS_LOG(WARNING) << "It's deprecated to use kCpu as input device for Dvpp operators to compute, because it's slow and "
                     "unsafe, we recommend you to set input device as MapTargetDevice::kAscend for Dvpp operators. "
                     "This API will be removed later";
  struct timeval begin = {0};
  struct timeval end = {0};
  gettimeofday(&begin, nullptr);
  // deal with image
  APP_ERROR ret = JPEG_DRC_(ImageInfo);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to decode or resize or crop, ret = " << ret;
    return ret;
  }
  gettimeofday(&end, nullptr);
  // Calculate the time cost of preprocess
  const double costMs = SEC2MS * (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) / SEC2MS;
  const double fps = 1 * SEC2MS / costMs;
  MS_LOG(INFO) << "[dvpp Delay] cost: " << costMs << "ms\tfps: " << fps;
  // Get output of resize module
  /*  测试Device内存
   */
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

APP_ERROR MDAclProcess::JPEG_DRC() {
  struct timeval begin = {0};
  struct timeval end = {0};
  gettimeofday(&begin, nullptr);
  // deal with image
  APP_ERROR ret = JPEG_D_();
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to decode, ret = " << ret;
    return ret;
  }
  std::string last_step = "Decode";
  ret = JPEG_R_(last_step);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to resize, ret = " << ret;
    return ret;
  }
  last_step = "Resize";
  ret = JPEG_C_(last_step);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to crop, ret = " << ret;
    return ret;
  }
  // Get output of crop module
  std::shared_ptr<DvppDataInfo> CropOutData = dvppCommon_->GetCropedImage();
  if (!CropOutData) {
    MS_LOG(ERROR) << "Decode Data returns NULL";
    return APP_ERR_COMM_INVALID_POINTER;
  }
  gettimeofday(&end, nullptr);
  // Calculate the time cost of preprocess
  const double costMs = SEC2MS * (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) / SEC2MS;
  const double fps = 1 * SEC2MS / costMs;
  MS_LOG(INFO) << "[dvpp (Decode + Resize + Crop) Delay] cost: " << costMs << "ms\tfps: " << fps;
  return APP_ERR_OK;
}

/*
 * @description: Read image files, and perform decoding and scaling
 * @param: imageFile specifies the image path to be processed
 * @return: aclError which is error code of ACL API
 */
APP_ERROR MDAclProcess::JPEG_DRC_(const RawData &ImageInfo) {
  // Decode process
  APP_ERROR ret = dvppCommon_->CombineJpegdProcess(ImageInfo, PIXEL_FORMAT_YUV_SEMIPLANAR_420, true);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to process decode, ret = " << ret << ".";
    return ret;
  }
  // Get output of decoded jpeg image, decodeOutData locates on device
  std::shared_ptr<DvppDataInfo> decodeOutData = dvppCommon_->GetDecodedImage();

  if (decodeOutData == nullptr) {
    MS_LOG(ERROR) << "Decode output buffer is null.";
    return APP_ERR_COMM_INVALID_POINTER;
  }
  uint32_t pri_h = decodeOutData->heightStride;
  uint32_t pri_w = decodeOutData->widthStride;
  // Define output of resize jpeg image
  DvppDataInfo resizeOut;
  ResizeConfigFilter(resizeOut, pri_w, pri_h);
  // Run resize application function
  ret = dvppCommon_->CombineResizeProcess(*decodeOutData, resizeOut, true);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to process resize, ret = " << ret << ".";
    return ret;
  }
  // Get output of resize jpeg image, resizeOutData locates on device
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
  cropInfo.dataInfo = *resizeOutData;
  // Define crop area
  CropRoiConfig cropCfg;
  CropConfigFilter(cropCfg, cropInfo, resizeOut);
  ret = dvppCommon_->CombineCropProcess(cropInfo, cropOut, true);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to process center crop, ret = " << ret << ".";
    return ret;
  }
  return APP_ERR_OK;
}

APP_ERROR MDAclProcess::JPEG_DR(const RawData &ImageInfo) {
  MS_LOG(WARNING) << "It's deprecated to use kCpu as input device for Dvpp operators to compute, because it's slow and "
                     "unsafe, we recommend you to set input device as MapTargetDevice::kAscend for Dvpp operators. "
                     "This API will be removed later";
  struct timeval begin = {0};
  struct timeval end = {0};
  gettimeofday(&begin, nullptr);
  // deal with image
  APP_ERROR ret = JPEG_DR_(ImageInfo);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to decode or resize, ret = " << ret;
    return ret;
  }
  gettimeofday(&end, nullptr);
  // Calculate the time cost of preprocess
  const double costMs = SEC2MS * (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) / SEC2MS;
  const double fps = 1 * SEC2MS / costMs;
  MS_LOG(INFO) << "[dvpp Delay] cost: " << costMs << "ms\tfps: " << fps;
  // Get output of resize module
  std::shared_ptr<DvppDataInfo> ResizeOutData = dvppCommon_->GetResizedImage();
  if (!ResizeOutData) {
    MS_LOG(ERROR) << "ResizedOutData returns NULL";
    return APP_ERR_COMM_INVALID_POINTER;
  }
  // Alloc host memory for the inference output according to the size of output
  void *resHostBuf = nullptr;
  ret = aclrtMallocHost(&resHostBuf, ResizeOutData->dataSize);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to allocate memory from host ret = " << ret;
    return ret;
  }
  std::shared_ptr<void> outBuf(resHostBuf, aclrtFreeHost);
  processedInfo_ = outBuf;
  // Memcpy the output data from device to host
  ret = aclrtMemcpy(outBuf.get(), ResizeOutData->dataSize, ResizeOutData->data, ResizeOutData->dataSize,
                    ACL_MEMCPY_DEVICE_TO_HOST);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to copy memory from device to host, ret = " << ret;
    return ret;
  }
  return APP_ERR_OK;
}

APP_ERROR MDAclProcess::JPEG_DR() {
  struct timeval begin = {0};
  struct timeval end = {0};
  gettimeofday(&begin, nullptr);
  // deal with image
  APP_ERROR ret = JPEG_D_();
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to decode, ret = " << ret;
    return ret;
  }
  std::string last_step = "Decode";
  ret = JPEG_R_(last_step);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to resize, ret = " << ret;
    return ret;
  }
  // Get output of resize module
  std::shared_ptr<DvppDataInfo> ResizeOutData = dvppCommon_->GetResizedImage();
  if (!ResizeOutData) {
    MS_LOG(ERROR) << "Decode Data returns NULL";
    return APP_ERR_COMM_INVALID_POINTER;
  }
  gettimeofday(&end, nullptr);
  // Calculate the time cost of preprocess
  const double costMs = SEC2MS * (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) / SEC2MS;
  const double fps = 1 * SEC2MS / costMs;
  MS_LOG(INFO) << "[dvpp (Decode + Resize) Delay] cost: " << costMs << "ms\tfps: " << fps;
  return APP_ERR_OK;
}

APP_ERROR MDAclProcess::JPEG_DR_(const RawData &ImageInfo) {
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
  uint32_t pri_h = decodeOutData->heightStride;
  uint32_t pri_w = decodeOutData->widthStride;
  DvppDataInfo resizeOut;
  ResizeConfigFilter(resizeOut, pri_w, pri_h);
  // Run resize application function
  ret = dvppCommon_->CombineResizeProcess(*decodeOutData, resizeOut, true);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to process resize, ret = " << ret << ".";
    return ret;
  }
  return APP_ERR_OK;
}

void MDAclProcess::CropConfigFilter(CropRoiConfig &cfg, DvppCropInputInfo &cropinfo, DvppDataInfo &resizeinfo) {
  if (resizeHeight_ != 0) {
    cfg.up = (resizeHeight_ - cropHeight_) / 2;
    if (cfg.up % 2 != 0) {
      cfg.up++;
    }
    cfg.down = resizeHeight_ - (resizeHeight_ - cropHeight_) / 2;
    if (cfg.down % 2 == 0) {
      cfg.down--;
    }
  } else {
    cfg.up = (resizeinfo.height - cropHeight_) / 2;
    if (cfg.up % 2 != 0) {
      cfg.up++;
    }
    cfg.down = resizeinfo.height - (resizeinfo.height - cropHeight_) / 2;
    if (cfg.down % 2 == 0) {
      cfg.down--;
    }
  }
  if (resizeWidth_ != 0) {
    cfg.left = (resizeWidth_ - cropWidth_) / 2;
    if (cfg.left % 2 != 0) {
      cfg.left++;
    }
    cfg.right = resizeWidth_ - (resizeWidth_ - cropWidth_) / 2;
    if (cfg.right % 2 == 0) {
      cfg.right--;
    }
  } else {
    cfg.left = (resizeinfo.width - cropWidth_) / 2;
    if (cfg.left % 2 != 0) {
      cfg.left++;
    }
    cfg.right = resizeinfo.width - (resizeinfo.width - cropWidth_) / 2;
    if (cfg.right % 2 == 0) {
      cfg.right--;
    }
  }
  cropinfo.roi = cfg;
}

APP_ERROR MDAclProcess::ResizeConfigFilter(DvppDataInfo &resizeinfo, const uint32_t pri_w_, const uint32_t pri_h_) {
  if (resizeWidth_ != 0) {  // 如果输入参数个数为2，按指定参数缩放
    resizeinfo.width = resizeWidth_;
    resizeinfo.widthStride = DVPP_ALIGN_UP(resizeWidth_, VPC_STRIDE_WIDTH);
    resizeinfo.height = resizeHeight_;
    resizeinfo.heightStride = DVPP_ALIGN_UP(resizeHeight_, VPC_STRIDE_HEIGHT);
  } else {  // 如果输入参数个数为1，保持原图片比例缩放
    if (pri_h_ >= pri_w_) {
      resizeinfo.width = resizeHeight_;  // 若输入参数个数为1，则只有resizeHeight_有值
      resizeinfo.widthStride = DVPP_ALIGN_UP(resizeinfo.width, VPC_STRIDE_WIDTH);
      resizeinfo.height = uint32_t(resizeHeight_ * pri_h_ / pri_w_);
      resizeinfo.heightStride = DVPP_ALIGN_UP(resizeinfo.height, VPC_STRIDE_HEIGHT);
    } else {
      resizeinfo.width = uint32_t(resizeHeight_ * pri_w_ / pri_h_);
      resizeinfo.widthStride = DVPP_ALIGN_UP(resizeinfo.width, VPC_STRIDE_WIDTH);
      resizeinfo.height = resizeHeight_;
      resizeinfo.heightStride = DVPP_ALIGN_UP(resizeinfo.height, VPC_STRIDE_HEIGHT);
    }
  }
  resizeinfo.format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
  return APP_ERR_OK;
}
/*
 * @description: Obtain result data of memory
 * @param: processed_data is result data info pointer
 * @return: Address of data in the memory
 */
std::shared_ptr<void> MDAclProcess::Get_Memory_Data() { return processedInfo_; }

std::shared_ptr<DvppDataInfo> MDAclProcess::Get_Croped_DeviceData() { return dvppCommon_->GetCropedImage(); }

std::shared_ptr<DvppDataInfo> MDAclProcess::Get_Resized_DeviceData() { return dvppCommon_->GetResizedImage(); }

std::shared_ptr<DvppDataInfo> MDAclProcess::Get_Decode_DeviceData() { return dvppCommon_->GetDecodedImage(); }

APP_ERROR MDAclProcess::SetResizeParas(uint32_t width, uint32_t height) {
  resizeWidth_ = width;
  resizeHeight_ = height;
  return APP_ERR_OK;
}

APP_ERROR MDAclProcess::SetCropParas(uint32_t width, uint32_t height) {
  cropWidth_ = width;
  cropHeight_ = height;
  return APP_ERR_OK;
}

APP_ERROR MDAclProcess::device_memory_release() {
  dvppCommon_->ReleaseDvppBuffer();
  MS_LOG(INFO) << "Device memory release successfully";
  return APP_ERR_OK;
}

std::vector<uint32_t> MDAclProcess::Get_Primary_Shape() {
  std::vector<uint32_t> pri_shape;
  if (!dvppCommon_) {
    pri_shape.emplace_back(dvppCommon_->GetDecodedImage()->heightStride);
    pri_shape.emplace_back(dvppCommon_->GetDecodedImage()->widthStride);
  }
  return pri_shape;
}
