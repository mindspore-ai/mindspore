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

#include <iostream>
#include <memory>

#include "mindspore/core/utils/log_adapter.h"
#include "DvppCommon.h"
#include "CommonDataType.h"

static auto g_resizeConfigDeleter = [](acldvppResizeConfig *p) { acldvppDestroyResizeConfig(p); };
static auto g_picDescDeleter = [](acldvppPicDesc *picDesc) { acldvppDestroyPicDesc(picDesc); };
static auto g_roiConfigDeleter = [](acldvppRoiConfig *p) { acldvppDestroyRoiConfig(p); };
static auto g_jpegeConfigDeleter = [](acldvppJpegeConfig *p) { acldvppDestroyJpegeConfig(p); };

DvppCommon::DvppCommon(aclrtContext dvppContext, aclrtStream dvppStream)
    : dvppContext_(dvppContext), dvppStream_(dvppStream) {}

DvppCommon::DvppCommon(const VdecConfig &vdecConfig) : vdecConfig_(vdecConfig) {}

/*
 * @description: Create a channel for processing image data,
 *               the channel description is created by acldvppCreateChannelDesc
 * @return: APP_ERR_OK if success, other values if failure
 */
APP_ERROR DvppCommon::Init(void) {
  dvppChannelDesc_ = acldvppCreateChannelDesc();
  if (dvppChannelDesc_ == nullptr) {
    return APP_ERR_COMM_INVALID_POINTER;
  }

  APP_ERROR ret = acldvppCreateChannel(dvppChannelDesc_);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to create dvpp channel: " << GetAppErrCodeInfo(ret) << ".";
    acldvppDestroyChannelDesc(dvppChannelDesc_);
    dvppChannelDesc_ = nullptr;
    return ret;
  }

  return APP_ERR_OK;
}

/*
 * @description: Create a channel for processing video data,
 *               the channel description is created by aclvdecCreateChannelDesc
 * @return: APP_ERR_OK if success, other values if failure
 */
APP_ERROR DvppCommon::InitVdec() {
  isVdec_ = true;
  // create vdec channelDesc
  vdecChannelDesc_ = aclvdecCreateChannelDesc();
  if (vdecChannelDesc_ == nullptr) {
    MS_LOG(ERROR) << "Failed to create vdec channel description.";
    return APP_ERR_ACL_FAILURE;
  }

  // channelId: 0-15
  aclError ret = aclvdecSetChannelDescChannelId(vdecChannelDesc_, vdecConfig_.channelId);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Failed to set vdec channel id, ret = " << ret << ".";
    return APP_ERR_ACL_FAILURE;
  }

  ret = aclvdecSetChannelDescThreadId(vdecChannelDesc_, vdecConfig_.threadId);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Failed to set thread id, ret = " << ret << ".";
    return APP_ERR_ACL_FAILURE;
  }

  // callback func
  ret = aclvdecSetChannelDescCallback(vdecChannelDesc_, vdecConfig_.callback);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Failed to set vdec callback function, ret = " << ret << ".";
    return APP_ERR_ACL_FAILURE;
  }

  ret = aclvdecSetChannelDescEnType(vdecChannelDesc_, vdecConfig_.inFormat);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Failed to set encoded type of input video, ret = " << ret << ".";
    return APP_ERR_ACL_FAILURE;
  }

  ret = aclvdecSetChannelDescOutPicFormat(vdecChannelDesc_, vdecConfig_.outFormat);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Failed to set vdec output format, ret = " << ret << ".";
    return APP_ERR_ACL_FAILURE;
  }

  // create vdec channel
  ret = aclvdecCreateChannel(vdecChannelDesc_);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Failed to create vdec channel, ret = " << ret << ".";
    return APP_ERR_ACL_FAILURE;
  }

  MS_LOG(INFO) << "Vdec init resource successfully.";
  return APP_ERR_OK;
}

/*
 * @description: If isVdec_ is true, destroy the channel and the channel description used by video.
 *               Otherwise destroy the channel and the channel description used by image.
 * @return: APP_ERR_OK if success, other values if failure
 */
APP_ERROR DvppCommon::DeInit(void) {
  if (isVdec_) {
    return DestroyResource();
  }

  // Obtain the dvppContext_ allocated by AscendResource which contains the dvppStream_, they mush bind each other
  APP_ERROR ret = aclrtSetCurrentContext(dvppContext_);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to get ACL context, ret = " << ret;
    return ret;
  }

  ret = aclrtSynchronizeStream(dvppStream_);  // APP_ERROR ret
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to synchronize stream, ret = " << ret << ".";
    return ret;
  }

  ret = acldvppDestroyChannel(dvppChannelDesc_);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to destory dvpp channel, ret = " << ret << ".";
    return ret;
  }

  ret = acldvppDestroyChannelDesc(dvppChannelDesc_);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to destroy dvpp channel description, ret = " << ret << ".";
    return ret;
  }
  return APP_ERR_OK;
}

/*
 * @description: Destroy the channel and the channel description used by video.
 * @return: APP_ERR_OK if success, other values if failure
 */
APP_ERROR DvppCommon::DestroyResource() {
  APP_ERROR ret = APP_ERR_OK;
  isVdec_ = true;
  if (vdecChannelDesc_ != nullptr) {
    ret = aclvdecDestroyChannel(vdecChannelDesc_);
    if (ret != APP_ERR_OK) {
      MS_LOG(ERROR) << "Failed to destory dvpp channel, ret = " << ret;
    }
    aclvdecDestroyChannelDesc(vdecChannelDesc_);
    vdecChannelDesc_ = nullptr;
  }
  return ret;
}

/*
 * @description: Release the memory that is allocated in the interfaces which are started with "Combine"
 */
void DvppCommon::ReleaseDvppBuffer() {
  if (cropImage_ != nullptr) {
    RELEASE_DVPP_DATA(cropImage_->data);
  }
  if (resizedImage_ != nullptr) {
    RELEASE_DVPP_DATA(resizedImage_->data);
  }
  if (decodedImage_ != nullptr) {
    RELEASE_DVPP_DATA(decodedImage_->data);
  }
  if (inputImage_ != nullptr) {
    RELEASE_DVPP_DATA(inputImage_->data);
  }
  if (encodedImage_ != nullptr) {
    RELEASE_DVPP_DATA(encodedImage_->data);
  }
}

/*
 * @description: Get the size of buffer used to save image for VPC according to width, height and format
 * @param  width specifies the width of the output image
 * @param  height specifies the height of the output image
 * @param  format specifies the format of the output image
 * @param: vpcSize is used to save the result size
 * @return: APP_ERR_OK if success, other values if failure
 */
APP_ERROR DvppCommon::GetVpcDataSize(uint32_t width, uint32_t height, acldvppPixelFormat format, uint32_t &vpcSize) {
  // Check the invalid format of VPC function and calculate the output buffer size
  if (format != PIXEL_FORMAT_YUV_SEMIPLANAR_420 && format != PIXEL_FORMAT_YVU_SEMIPLANAR_420 &&
      format != PIXEL_FORMAT_RGB_888) {
    MS_LOG(ERROR) << "Format[" << format << "] for VPC is not supported, just support NV12 or NV21 or RGB888.";
    return APP_ERR_COMM_INVALID_PARAM;
  }
  uint32_t widthStride = DVPP_ALIGN_UP(width, VPC_WIDTH_ALIGN);
  if (format == PIXEL_FORMAT_RGB_888) {
    widthStride *= 3;
  }

  uint32_t heightStride = DVPP_ALIGN_UP(height, VPC_HEIGHT_ALIGN);
  vpcSize = widthStride * heightStride * YUV_BGR_SIZE_CONVERT_3 / YUV_BGR_SIZE_CONVERT_2;
  return APP_ERR_OK;
}

/*
 * @description: Get the aligned width and height of the input image according to the image format
 * @param: width specifies the width before alignment
 * @param: height specifies the height before alignment
 * @param: format specifies the image format
 * @param: widthStride is used to save the width after alignment
 * @param: heightStride is used to save the height after alignment
 * @return: APP_ERR_OK if success, other values if failure
 */
APP_ERROR DvppCommon::GetVpcInputStrideSize(uint32_t width, uint32_t height, acldvppPixelFormat format,
                                            uint32_t &widthStride, uint32_t &heightStride) {
  uint32_t inputWidthStride;
  // Check the invalidty of input format and calculate the input width stride
  if (format >= PIXEL_FORMAT_YUV_400 && format <= PIXEL_FORMAT_YVU_SEMIPLANAR_444) {
    // If format is YUV SP, keep widthStride not change.
    inputWidthStride = DVPP_ALIGN_UP(width, VPC_STRIDE_WIDTH);
  } else if (format >= PIXEL_FORMAT_YUYV_PACKED_422 && format <= PIXEL_FORMAT_VYUY_PACKED_422) {
    // If format is YUV422 packed, image size = H x W * 2;
    inputWidthStride = DVPP_ALIGN_UP(width, VPC_STRIDE_WIDTH) * YUV422_WIDTH_NU;
  } else if (format >= PIXEL_FORMAT_YUV_PACKED_444 && format <= PIXEL_FORMAT_BGR_888) {
    // If format is YUV444 packed or RGB, image size = H x W * 3;
    inputWidthStride = DVPP_ALIGN_UP(width, VPC_STRIDE_WIDTH) * YUV444_RGB_WIDTH_NU;
  } else if (format >= PIXEL_FORMAT_ARGB_8888 && format <= PIXEL_FORMAT_BGRA_8888) {
    // If format is XRGB8888, image size = H x W * 4
    inputWidthStride = DVPP_ALIGN_UP(width, VPC_STRIDE_WIDTH) * XRGB_WIDTH_NU;
  } else {
    MS_LOG(ERROR) << "Input format[" << format << "] for VPC is invalid, please check it.";
    return APP_ERR_COMM_INVALID_PARAM;
  }
  uint32_t inputHeightStride = DVPP_ALIGN_UP(height, VPC_STRIDE_HEIGHT);
  // Check the input validity width stride.
  if (inputWidthStride > MAX_RESIZE_WIDTH || inputWidthStride < MIN_RESIZE_WIDTH) {
    MS_LOG(ERROR) << "Input width stride " << inputWidthStride << " is invalid, not in [" << MIN_RESIZE_WIDTH << ", "
                  << MAX_RESIZE_WIDTH << "].";
    return APP_ERR_COMM_INVALID_PARAM;
  }
  // Check the input validity height stride.
  if (inputHeightStride > MAX_RESIZE_HEIGHT || inputHeightStride < MIN_RESIZE_HEIGHT) {
    MS_LOG(ERROR) << "Input height stride " << inputHeightStride << " is invalid, not in [" << MIN_RESIZE_HEIGHT << ", "
                  << MAX_RESIZE_HEIGHT << "].";
    return APP_ERR_COMM_INVALID_PARAM;
  }
  widthStride = inputWidthStride;
  heightStride = inputHeightStride;
  return APP_ERR_OK;
}

/*
 * @description: Get the aligned width and height of the output image according to the image format
 * @param: width specifies the width before alignment
 * @param: height specifies the height before alignment
 * @param: format specifies the image format
 * @param: widthStride is used to save the width after alignment
 * @param: heightStride is used to save the height after alignment
 * @return: APP_ERR_OK if success, other values if failure
 */
APP_ERROR DvppCommon::GetVpcOutputStrideSize(uint32_t width, uint32_t height, acldvppPixelFormat format,
                                             uint32_t &widthStride, uint32_t &heightStride) {
  // Check the invalidty of output format and calculate the output width and height
  if (format != PIXEL_FORMAT_YUV_SEMIPLANAR_420 && format != PIXEL_FORMAT_YVU_SEMIPLANAR_420 &&
      format != PIXEL_FORMAT_RGB_888) {
    MS_LOG(ERROR) << "Output format[" << format << "] for VPC is not supported, just support NV12 or NV21 or RGB888.";
    return APP_ERR_COMM_INVALID_PARAM;
  }

  widthStride = DVPP_ALIGN_UP(width, VPC_STRIDE_WIDTH);
  if (format == PIXEL_FORMAT_RGB_888) {
    widthStride *= 3;
  }

  heightStride = DVPP_ALIGN_UP(height, VPC_STRIDE_HEIGHT);
  return APP_ERR_OK;
}

/*
 * @description: Set picture description information and execute resize function
 * @param: input specifies the input image information
 * @param: output specifies the output image information
 * @param: withSynchronize specifies whether to execute synchronously
 * @param: processType specifies whether to perform proportional scaling, default is non-proportional resize
 * @return: APP_ERR_OK if success, other values if failure
 * @attention: This function can be called only when the DvppCommon object is initialized with Init
 */
APP_ERROR DvppCommon::VpcResize(DvppDataInfo &input, DvppDataInfo &output, bool withSynchronize,
                                VpcProcessType processType) {
  // Return special error code when the DvppCommon object is initialized with InitVdec
  if (isVdec_) {
    MS_LOG(ERROR) << "VpcResize cannot be called by the DvppCommon object which is initialized with InitVdec.";
    return APP_ERR_DVPP_OBJ_FUNC_MISMATCH;
  }

  acldvppPicDesc *inputDesc = acldvppCreatePicDesc();
  acldvppPicDesc *outputDesc = acldvppCreatePicDesc();
  resizeInputDesc_.reset(inputDesc, g_picDescDeleter);
  resizeOutputDesc_.reset(outputDesc, g_picDescDeleter);

  // Set dvpp picture descriptin info of input image
  APP_ERROR ret = SetDvppPicDescData(input, *resizeInputDesc_);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to set dvpp input picture description, ret = " << ret << ".";
    return ret;
  }

  // Set dvpp picture descriptin info of output image
  ret = SetDvppPicDescData(output, *resizeOutputDesc_);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to set dvpp output picture description, ret = " << ret << ".";
    return ret;
  }
  if (processType == VPC_PT_DEFAULT) {
    return ResizeProcess(*resizeInputDesc_, *resizeOutputDesc_, withSynchronize);
  }

  // Get crop area according to the processType
  // When the processType is VPC_PT_FILL, the image will be cropped if the image size is different from the target
  // resolution
  CropRoiConfig cropRoi = {0};
  GetCropRoi(input, output, processType, cropRoi);

  // The width and height of the original image will be resized by the same ratio
  // The cropped image will be pasted on the upper left corner or the middle location or the whole location according to
  // the processType
  CropRoiConfig pasteRoi = {0};
  GetPasteRoi(input, output, processType, pasteRoi);

  return ResizeWithPadding(*resizeInputDesc_, *resizeOutputDesc_, cropRoi, pasteRoi, withSynchronize);
}

/*
 * @description: Set image description information
 * @param: dataInfo specifies the image information
 * @param: picsDesc specifies the picture description information to be set
 * @return: APP_ERR_OK if success, other values if failure
 */
APP_ERROR DvppCommon::SetDvppPicDescData(const DvppDataInfo &dataInfo, acldvppPicDesc &picDesc) {
  APP_ERROR ret = acldvppSetPicDescData(&picDesc, dataInfo.data);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to set data for dvpp picture description, ret = " << ret << ".";
    return ret;
  }
  ret = acldvppSetPicDescSize(&picDesc, dataInfo.dataSize);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to set size for dvpp picture description, ret = " << ret << ".";
    return ret;
  }
  ret = acldvppSetPicDescFormat(&picDesc, dataInfo.format);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to set format for dvpp picture description, ret = " << ret << ".";
    return ret;
  }
  ret = acldvppSetPicDescWidth(&picDesc, dataInfo.width);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to set width for dvpp picture description, ret = " << ret << ".";
    return ret;
  }
  ret = acldvppSetPicDescHeight(&picDesc, dataInfo.height);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to set height for dvpp picture description, ret = " << ret << ".";
    return ret;
  }
  if (!isVdec_) {
    ret = acldvppSetPicDescWidthStride(&picDesc, dataInfo.widthStride);
    if (ret != APP_ERR_OK) {
      MS_LOG(ERROR) << "Failed to set aligned width for dvpp picture description, ret = " << ret << ".";
      return ret;
    }
    ret = acldvppSetPicDescHeightStride(&picDesc, dataInfo.heightStride);
    if (ret != APP_ERR_OK) {
      MS_LOG(ERROR) << "Failed to set aligned height for dvpp picture description, ret = " << ret << ".";
      return ret;
    }
  }

  return APP_ERR_OK;
}

/*
 * @description: Check whether the image format and zoom ratio meet the requirements
 * @param: input specifies the input image information
 * @param: output specifies the output image information
 * @return: APP_ERR_OK if success, other values if failure
 */
APP_ERROR DvppCommon::CheckResizeParams(const DvppDataInfo &input, const DvppDataInfo &output) {
  if (output.format != PIXEL_FORMAT_YUV_SEMIPLANAR_420 && output.format != PIXEL_FORMAT_YVU_SEMIPLANAR_420 &&
      output.format != PIXEL_FORMAT_RGB_888) {
    MS_LOG(ERROR) << "Output format[" << output.format << "] for VPC is not supported, only NV12 or NV21 or RGB888.";
    return APP_ERR_COMM_INVALID_PARAM;
  }
  if (((float)output.height / input.height) < MIN_RESIZE_SCALE ||
      ((float)output.height / input.height) > MAX_RESIZE_SCALE) {
    MS_LOG(ERROR) << "Resize scale should be in range [1/16, 16], which is " << (output.height / input.height) << ".";
    return APP_ERR_COMM_INVALID_PARAM;
  }
  if (((float)output.width / input.width) < MIN_RESIZE_SCALE ||
      ((float)output.width / input.width) > MAX_RESIZE_SCALE) {
    MS_LOG(ERROR) << "Resize scale should be in range [1/16, 16], which is " << (output.width / input.width) << ".";
    return APP_ERR_COMM_INVALID_PARAM;
  }
  return APP_ERR_OK;
}

/*
 * @description: Scale the input image to the size specified by the output image and
 *               saves the result to the output image (non-proportionate scaling)
 * @param: inputDesc specifies the description information of the input image
 * @param: outputDesc specifies the description information of the output image
 * @param: withSynchronize specifies whether to execute synchronously
 * @return: APP_ERR_OK if success, other values if failure
 */
APP_ERROR DvppCommon::ResizeProcess(acldvppPicDesc &inputDesc, acldvppPicDesc &outputDesc, bool withSynchronize) {
  acldvppResizeConfig *resizeConfig = acldvppCreateResizeConfig();
  if (resizeConfig == nullptr) {
    MS_LOG(ERROR) << "Failed to create dvpp resize config.";
    return APP_ERR_COMM_INVALID_POINTER;
  }

  resizeConfig_.reset(resizeConfig, g_resizeConfigDeleter);
  APP_ERROR ret = acldvppVpcResizeAsync(dvppChannelDesc_, &inputDesc, &outputDesc, resizeConfig_.get(), dvppStream_);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to resize asynchronously, ret = " << ret << ".";
    return ret;
  }

  if (withSynchronize) {
    ret = aclrtSynchronizeStream(dvppStream_);
    if (ret != APP_ERR_OK) {
      MS_LOG(ERROR) << "Failed to synchronize stream, ret = " << ret << ".";
      return ret;
    }
  }

  return APP_ERR_OK;
}

/*
 * @description: Crop the image from the input image based on the specified area and
 *               paste the cropped image to the specified position of the target image
 *               as the output image
 * @param: inputDesc specifies the description information of the input image
 * @param: outputDesc specifies the description information of the output image
 * @param: cropRoi specifies the cropped area
 * @param: pasteRoi specifies the pasting area
 * @param: withSynchronize specifies whether to execute synchronously
 * @return: APP_ERR_OK if success, other values if failure
 * @attention: If the width and height of the crop area are different from those of the
 *             paste area, the image is scaled again
 */
APP_ERROR DvppCommon::ResizeWithPadding(acldvppPicDesc &inputDesc, acldvppPicDesc &outputDesc, CropRoiConfig &cropRoi,
                                        CropRoiConfig &pasteRoi, bool withSynchronize) {
  acldvppRoiConfig *cropRoiCfg = acldvppCreateRoiConfig(cropRoi.left, cropRoi.right, cropRoi.up, cropRoi.down);
  if (cropRoiCfg == nullptr) {
    MS_LOG(ERROR) << "Failed to create dvpp roi config for corp area.";
    return APP_ERR_COMM_FAILURE;
  }
  cropAreaConfig_.reset(cropRoiCfg, g_roiConfigDeleter);

  acldvppRoiConfig *pastRoiCfg = acldvppCreateRoiConfig(pasteRoi.left, pasteRoi.right, pasteRoi.up, pasteRoi.down);
  if (pastRoiCfg == nullptr) {
    MS_LOG(ERROR) << "Failed to create dvpp roi config for paster area.";
    return APP_ERR_COMM_FAILURE;
  }
  pasteAreaConfig_.reset(pastRoiCfg, g_roiConfigDeleter);

  APP_ERROR ret = acldvppVpcCropAndPasteAsync(dvppChannelDesc_, &inputDesc, &outputDesc, cropAreaConfig_.get(),
                                              pasteAreaConfig_.get(), dvppStream_);
  if (ret != APP_ERR_OK) {
    // release resource.
    MS_LOG(ERROR) << "Failed to crop and paste asynchronously, ret = " << ret << ".";
    return ret;
  }
  if (withSynchronize) {
    ret = aclrtSynchronizeStream(dvppStream_);
    if (ret != APP_ERR_OK) {
      MS_LOG(ERROR) << "Failed tp synchronize stream, ret = " << ret << ".";
      return ret;
    }
  }
  return APP_ERR_OK;
}

/*
 * @description: Get crop area
 * @param: input specifies the input image information
 * @param: output specifies the output image information
 * @param: processType specifies whether to perform proportional scaling
 * @param: cropRoi is used to save the info of the crop roi area
 * @return: APP_ERR_OK if success, other values if failure
 */
void DvppCommon::GetCropRoi(const DvppDataInfo &input, const DvppDataInfo &output, VpcProcessType processType,
                            CropRoiConfig &cropRoi) {
  // When processType is not VPC_PT_FILL, crop area is the whole input image
  if (processType != VPC_PT_FILL) {
    cropRoi.right = CONVERT_TO_ODD(input.width - ODD_NUM_1);
    cropRoi.down = CONVERT_TO_ODD(input.height - ODD_NUM_1);
    return;
  }

  bool widthRatioSmaller = true;
  // The scaling ratio is based on the smaller ratio to ensure the smallest edge to fill the targe edge
  float resizeRatio = static_cast<float>(input.width) / output.width;
  if (resizeRatio > (static_cast<float>(input.height) / output.height)) {
    resizeRatio = static_cast<float>(input.height) / output.height;
    widthRatioSmaller = false;
  }

  const int halfValue = 2;
  // The left and up must be even, right and down must be odd which is required by acl
  if (widthRatioSmaller) {
    cropRoi.left = 0;
    cropRoi.right = CONVERT_TO_ODD(input.width - ODD_NUM_1);
    cropRoi.up = CONVERT_TO_EVEN(static_cast<uint32_t>((input.height - output.height * resizeRatio) / halfValue));
    cropRoi.down = CONVERT_TO_ODD(input.height - cropRoi.up - ODD_NUM_1);
    return;
  }

  cropRoi.up = 0;
  cropRoi.down = CONVERT_TO_ODD(input.height - ODD_NUM_1);
  cropRoi.left = CONVERT_TO_EVEN(static_cast<uint32_t>((input.width - output.width * resizeRatio) / halfValue));
  cropRoi.right = CONVERT_TO_ODD(input.width - cropRoi.left - ODD_NUM_1);
  return;
}

/*
 * @description: Get paste area
 * @param: input specifies the input image information
 * @param: output specifies the output image information
 * @param: processType specifies whether to perform proportional scaling
 * @param: pasteRio is used to save the info of the paste area
 * @return: APP_ERR_OK if success, other values if failure
 */
void DvppCommon::GetPasteRoi(const DvppDataInfo &input, const DvppDataInfo &output, VpcProcessType processType,
                             CropRoiConfig &pasteRoi) {
  if (processType == VPC_PT_FILL) {
    pasteRoi.right = CONVERT_TO_ODD(output.width - ODD_NUM_1);
    pasteRoi.down = CONVERT_TO_ODD(output.height - ODD_NUM_1);
    return;
  }

  bool widthRatioLarger = true;
  // The scaling ratio is based on the larger ratio to ensure the largest edge to fill the targe edge
  float resizeRatio = static_cast<float>(input.width) / output.width;
  if (resizeRatio < (static_cast<float>(input.height) / output.height)) {
    resizeRatio = static_cast<float>(input.height) / output.height;
    widthRatioLarger = false;
  }

  // Left and up is 0 when the roi paste on the upper left corner
  if (processType == VPC_PT_PADDING) {
    pasteRoi.right = (input.width / resizeRatio) - ODD_NUM_1;
    pasteRoi.down = (input.height / resizeRatio) - ODD_NUM_1;
    pasteRoi.right = CONVERT_TO_ODD(pasteRoi.right);
    pasteRoi.down = CONVERT_TO_ODD(pasteRoi.down);
    return;
  }

  const int halfValue = 2;
  // Left and up is 0 when the roi paste on the middler location
  if (widthRatioLarger) {
    pasteRoi.left = 0;
    pasteRoi.right = output.width - ODD_NUM_1;
    pasteRoi.up = (output.height - (input.height / resizeRatio)) / halfValue;
    pasteRoi.down = output.height - pasteRoi.up - ODD_NUM_1;
  } else {
    pasteRoi.up = 0;
    pasteRoi.down = output.height - ODD_NUM_1;
    pasteRoi.left = (output.width - (input.width / resizeRatio)) / halfValue;
    pasteRoi.right = output.width - pasteRoi.left - ODD_NUM_1;
  }

  // The left must be even and align to 16, up must be even, right and down must be odd which is required by acl
  pasteRoi.left = DVPP_ALIGN_UP(CONVERT_TO_EVEN(pasteRoi.left), VPC_WIDTH_ALIGN);
  pasteRoi.right = CONVERT_TO_ODD(pasteRoi.right);
  pasteRoi.up = CONVERT_TO_EVEN(pasteRoi.up);
  pasteRoi.down = CONVERT_TO_ODD(pasteRoi.down);
  return;
}

/*
 * @description: Resize the image specified by input and save the result to member variable resizedImage_
 * @param: input specifies the input image information
 * @param: output specifies the output image information
 * @param: withSynchronize specifies whether to execute synchronously
 * @param: processType specifies whether to perform proportional scaling, default is non-proportional resize
 * @return: APP_ERR_OK if success, other values if failure
 * @attention: This function can be called only when the DvppCommon object is initialized with Init
 */
APP_ERROR DvppCommon::CombineResizeProcess(DvppDataInfo &input, DvppDataInfo &output, bool withSynchronize,
                                           VpcProcessType processType) {
  // Return special error code when the DvppCommon object is initialized with InitVdec
  if (isVdec_) {
    MS_LOG(ERROR)
      << "CombineResizeProcess cannot be called by the DvppCommon object which is initialized with InitVdec.";
    return APP_ERR_DVPP_OBJ_FUNC_MISMATCH;
  }

  APP_ERROR ret = CheckResizeParams(input, output);
  if (ret != APP_ERR_OK) {
    return ret;
  }
  // Get widthStride and heightStride for input and output image according to the format
  ret =
    GetVpcInputStrideSize(input.widthStride, input.heightStride, input.format, input.widthStride, input.heightStride);
  if (ret != APP_ERR_OK) {
    return ret;
  }

  resizedImage_ = std::make_shared<DvppDataInfo>();
  resizedImage_->width = output.width;
  resizedImage_->height = output.height;
  resizedImage_->format = output.format;
  ret = GetVpcOutputStrideSize(output.width, output.height, output.format, resizedImage_->widthStride,
                               resizedImage_->heightStride);
  if (ret != APP_ERR_OK) {
    return ret;
  }
  // Get output buffer size for resize output
  ret = GetVpcDataSize(output.width, output.height, output.format, resizedImage_->dataSize);
  if (ret != APP_ERR_OK) {
    return ret;
  }
  // Malloc buffer for output of resize module
  // Need to pay attention to release of the buffer
  ret = acldvppMalloc((void **)(&(resizedImage_->data)), resizedImage_->dataSize);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to malloc " << resizedImage_->dataSize << " bytes on dvpp for resize, ret = " << ret
                  << ".";
    return ret;
  }

  aclrtMemset(resizedImage_->data, resizedImage_->dataSize, YUV_GREYER_VALUE, resizedImage_->dataSize);
  resizedImage_->frameId = input.frameId;
  ret = VpcResize(input, *resizedImage_, withSynchronize, processType);
  if (ret != APP_ERR_OK) {
    // Release the output buffer when resize failed, otherwise release it after use
    RELEASE_DVPP_DATA(resizedImage_->data);
  }
  return ret;
}

/*
 * @description: Set picture description information and execute crop function
 * @param: cropInput specifies the input image information and cropping area
 * @param: output specifies the output image information
 * @param: withSynchronize specifies whether to execute synchronously
 * @return: APP_ERR_OK if success, other values if failure
 * @attention: This function can be called only when the DvppCommon object is initialized with Init
 */
APP_ERROR DvppCommon::VpcCrop(const DvppCropInputInfo &cropInput, const DvppDataInfo &output, bool withSynchronize) {
  // Return special error code when the DvppCommon object is initialized with InitVdec
  if (isVdec_) {
    MS_LOG(ERROR) << "VpcCrop cannot be called by the DvppCommon object which is initialized with InitVdec.";
    return APP_ERR_DVPP_OBJ_FUNC_MISMATCH;
  }

  acldvppPicDesc *inputDesc = acldvppCreatePicDesc();
  acldvppPicDesc *outputDesc = acldvppCreatePicDesc();
  cropInputDesc_.reset(inputDesc, g_picDescDeleter);
  cropOutputDesc_.reset(outputDesc, g_picDescDeleter);

  // Set dvpp picture descriptin info of input image
  APP_ERROR ret = SetDvppPicDescData(cropInput.dataInfo, *cropInputDesc_);
  if (ret != APP_ERR_OK) {
    return ret;
  }
  // Set dvpp picture descriptin info of output image
  ret = SetDvppPicDescData(output, *cropOutputDesc_);
  if (ret != APP_ERR_OK) {
    return ret;
  }
  return CropProcess(*cropInputDesc_, *cropOutputDesc_, cropInput.roi, withSynchronize);
}

/*
 * @description: Check whether the size of the cropped data and the cropped area meet the requirements
 * @param: input specifies the image information and the information about the area to be cropped
 * @return: APP_ERR_OK if success, other values if failure
 */
APP_ERROR DvppCommon::CheckCropParams(const DvppCropInputInfo &input) {
  APP_ERROR ret;
  uint32_t payloadSize;
  ret = GetVpcDataSize(input.dataInfo.widthStride, input.dataInfo.heightStride, PIXEL_FORMAT_YUV_SEMIPLANAR_420,
                       payloadSize);
  if (ret != APP_ERR_OK) {
    return ret;
  }
  if (payloadSize != input.dataInfo.dataSize) {
    MS_LOG(ERROR) << "Input data size: " << payloadSize
                  << " to crop does not match input yuv image size: " << input.dataInfo.dataSize << ".";
    return APP_ERR_COMM_INVALID_PARAM;
  }

  if ((!CHECK_EVEN(input.roi.left)) || (!CHECK_EVEN(input.roi.up)) || (!CHECK_ODD(input.roi.right)) ||
      (!CHECK_ODD(input.roi.down))) {
    MS_LOG(ERROR) << "Crop area left and top(" << input.roi.left << ", " << input.roi.up
                  << ") must be even, right bottom(" << input.roi.right << "," << input.roi.down << ") must be odd.";
    return APP_ERR_COMM_INVALID_PARAM;
  }

  // Calculate crop width and height according to the input location
  uint32_t cropWidth = input.roi.right - input.roi.left + ODD_NUM_1;
  uint32_t cropHeight = input.roi.down - input.roi.up + ODD_NUM_1;
  if ((cropWidth < MIN_CROP_WIDTH) || (cropHeight < MIN_CROP_HEIGHT)) {
    MS_LOG(ERROR) << "Crop area width:" << cropWidth << " need to be larger than 10 and height:" << cropHeight
                  << " need to be larger than 6.";
    return APP_ERR_COMM_INVALID_PARAM;
  }

  if ((input.roi.left + cropWidth > input.dataInfo.width) || (input.roi.up + cropHeight > input.dataInfo.height)) {
    MS_LOG(ERROR) << "Target rectangle start location(" << input.roi.left << "," << input.roi.up << ") with size("
                  << cropWidth << "," << cropHeight << ") is out of the input image(" << input.dataInfo.width << ","
                  << input.dataInfo.height << ") to be cropped.";
    return APP_ERR_COMM_INVALID_PARAM;
  }

  return APP_ERR_OK;
}

/*
 * @description: It is used to crop an input image based on a specified region and
 *               store the cropped image to the output memory as an output image
 * @param: inputDesc specifies the description information of the input image
 * @param: outputDesc specifies the description information of the output image
 * @param: CropRoiConfig specifies the cropped area
 * @param: withSynchronize specifies whether to execute synchronously
 * @return: APP_ERR_OK if success, other values if failure
 * @attention: if the region of the output image is inconsistent with the crop area, the image is scaled again
 */
APP_ERROR DvppCommon::CropProcess(acldvppPicDesc &inputDesc, acldvppPicDesc &outputDesc, const CropRoiConfig &cropArea,
                                  bool withSynchronize) {
  uint32_t leftOffset = CONVERT_TO_EVEN(cropArea.left);
  uint32_t rightOffset = CONVERT_TO_ODD(cropArea.right);
  uint32_t upOffset = CONVERT_TO_EVEN(cropArea.up);
  uint32_t downOffset = CONVERT_TO_ODD(cropArea.down);

  auto cropRioCfg = acldvppCreateRoiConfig(leftOffset, rightOffset, upOffset, downOffset);
  if (cropRioCfg == nullptr) {
    MS_LOG(ERROR) << "DvppCommon: create dvpp vpc resize failed.";
    return APP_ERR_DVPP_RESIZE_FAIL;
  }
  cropRoiConfig_.reset(cropRioCfg, g_roiConfigDeleter);

  APP_ERROR ret = acldvppVpcCropAsync(dvppChannelDesc_, &inputDesc, &outputDesc, cropRoiConfig_.get(), dvppStream_);
  if (ret != APP_ERR_OK) {
    // release resource.
    MS_LOG(ERROR) << "Failed to crop, ret = " << ret << ".";
    return ret;
  }
  if (withSynchronize) {
    ret = aclrtSynchronizeStream(dvppStream_);
    if (ret != APP_ERR_OK) {
      MS_LOG(ERROR) << "Failed to synchronize stream, ret = " << ret << ".";
      return ret;
    }
  }
  return APP_ERR_OK;
}

/*
 * @description: Crop the image specified by the input parameter and saves the result to member variable cropImage_
 * @param: input specifies the input image information and cropping area
 * @param: output specifies the output image information
 * @param: withSynchronize specifies whether to execute synchronously
 * @return: APP_ERR_OK if success, other values if failure
 * @attention: This function can be called only when the DvppCommon object is initialized with Init
 */
APP_ERROR DvppCommon::CombineCropProcess(DvppCropInputInfo &input, DvppDataInfo &output, bool withSynchronize) {
  // Return special error code when the DvppCommon object is initialized with InitVdec
  if (isVdec_) {
    MS_LOG(ERROR) << "CombineCropProcess cannot be called by the DvppCommon object which is initialized with InitVdec.";
    return APP_ERR_DVPP_OBJ_FUNC_MISMATCH;
  }

  // Get widthStride and heightStride for input and output image according to the format
  APP_ERROR ret = GetVpcInputStrideSize(input.dataInfo.width, input.dataInfo.height, input.dataInfo.format,
                                        input.dataInfo.widthStride, input.dataInfo.heightStride);
  if (ret != APP_ERR_OK) {
    return ret;
  }
  ret = CheckCropParams(input);
  if (ret != APP_ERR_OK) {
    return ret;
  }
  // cropImage_所持有的成员变量 uint8_t *data通过acldvppMalloc()接口申请，位于Device上
  cropImage_ = std::make_shared<DvppDataInfo>();
  cropImage_->width = output.width;
  cropImage_->height = output.height;
  cropImage_->format = output.format;
  ret = GetVpcOutputStrideSize(output.width, output.height, output.format, cropImage_->widthStride,
                               cropImage_->heightStride);
  if (ret != APP_ERR_OK) {
    return ret;
  }
  // Get output buffer size for resize output
  ret = GetVpcDataSize(output.width, output.height, output.format, cropImage_->dataSize);
  if (ret != APP_ERR_OK) {
    return ret;
  }

  // Malloc buffer for output of resize module
  // Need to pay attention to release of the buffer
  ret = acldvppMalloc((void **)(&(cropImage_->data)), cropImage_->dataSize);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to malloc " << cropImage_->dataSize << " bytes on dvpp for resize, ret = " << ret << ".";
    return ret;
  }
  cropImage_->frameId = input.dataInfo.frameId;
  ret = VpcCrop(input, *cropImage_, withSynchronize);
  if (ret != APP_ERR_OK) {
    // Release the output buffer when resize failed, otherwise release it after use
    RELEASE_DVPP_DATA(cropImage_->data);
  }
  return ret;
}

/*
 * @description: Set the description of the output image and decode
 * @param: input specifies the input image information
 * @param: output specifies the output image information
 * @param: withSynchronize specifies whether to execute synchronously
 * @return: APP_ERR_OK if success, other values if failure
 * @attention: This function can be called only when the DvppCommon object is initialized with Init
 */
APP_ERROR DvppCommon::JpegDecode(DvppDataInfo &input, DvppDataInfo &output, bool withSynchronize) {
  // Return special error code when the DvppCommon object is initialized with InitVdec
  if (isVdec_) {
    MS_LOG(ERROR) << "JpegDecode cannot be called by the DvppCommon object which is initialized with InitVdec.";
    return APP_ERR_DVPP_OBJ_FUNC_MISMATCH;
  }

  acldvppPicDesc *outputDesc = acldvppCreatePicDesc();
  decodeOutputDesc_.reset(outputDesc, g_picDescDeleter);

  APP_ERROR ret = SetDvppPicDescData(output, *decodeOutputDesc_);
  if (ret != APP_ERR_OK) {
    return ret;
  }

  ret = acldvppJpegDecodeAsync(dvppChannelDesc_, input.data, input.dataSize, decodeOutputDesc_.get(), dvppStream_);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to decode jpeg, ret = " << ret << ".";
    return ret;
  }
  if (withSynchronize) {
    ret = aclrtSynchronizeStream(dvppStream_);
    if (ret != APP_ERR_OK) {
      MS_LOG(ERROR) << "Failed to synchronize stream, ret = " << ret << ".";
      return APP_ERR_DVPP_JPEG_DECODE_FAIL;
    }
  }
  return APP_ERR_OK;
}

/*
 * @description: Set the description of the output image and decode
 * @param: input specifies the input image information
 * @param: output specifies the output image information
 * @param: withSynchronize specifies whether to execute synchronously
 * @return: APP_ERR_OK if success, other values if failure
 * @attention: This function can be called only when the DvppCommon object is initialized with Init
 */
APP_ERROR DvppCommon::PngDecode(DvppDataInfo &input, DvppDataInfo &output, bool withSynchronize) {
  // Return special error code when the DvppCommon object is initialized with InitVdec
  if (isVdec_) {
    MS_LOG(ERROR) << "PngDecode cannot be called by the DvppCommon object which is initialized with InitVdec.";
    return APP_ERR_DVPP_OBJ_FUNC_MISMATCH;
  }

  acldvppPicDesc *outputDesc = acldvppCreatePicDesc();
  decodeOutputDesc_.reset(outputDesc, g_picDescDeleter);

  APP_ERROR ret = SetDvppPicDescData(output, *decodeOutputDesc_);
  if (ret != APP_ERR_OK) {
    return ret;
  }

  ret = acldvppPngDecodeAsync(dvppChannelDesc_, input.data, input.dataSize, decodeOutputDesc_.get(), dvppStream_);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to decode png, ret = " << ret << ".";
    return ret;
  }
  if (withSynchronize) {
    ret = aclrtSynchronizeStream(dvppStream_);
    if (ret != APP_ERR_OK) {
      MS_LOG(ERROR) << "Failed to synchronize stream, ret = " << ret << ".";
      return APP_ERR_DVPP_JPEG_DECODE_FAIL;
    }
  }
  return APP_ERR_OK;
}

/*
 * @description: Get the aligned width and height of the image after decoding
 * @param: width specifies the width before alignment
 * @param: height specifies the height before alignment
 * @param: widthStride is used to save the width after alignment
 * @param: heightStride is used to save the height after alignment
 * @return: APP_ERR_OK if success, other values if failure
 */
void DvppCommon::GetJpegDecodeStrideSize(uint32_t width, uint32_t height, uint32_t &widthStride,
                                         uint32_t &heightStride) {
  widthStride = DVPP_ALIGN_UP(width, JPEGD_STRIDE_WIDTH);
  heightStride = DVPP_ALIGN_UP(height, JPEGD_STRIDE_HEIGHT);
}

void DvppCommon::GetPngDecodeStrideSize(uint32_t width, uint32_t height, uint32_t &widthStride, uint32_t &heightStride,
                                        acldvppPixelFormat format) {
  if (format == PIXEL_FORMAT_RGB_888) {
    widthStride = DVPP_ALIGN_UP(width * 3, JPEGD_STRIDE_WIDTH);
    heightStride = DVPP_ALIGN_UP(height, JPEGD_STRIDE_HEIGHT);
  } else {
    widthStride = DVPP_ALIGN_UP(width * 4, JPEGD_STRIDE_WIDTH);
    heightStride = DVPP_ALIGN_UP(height, JPEGD_STRIDE_HEIGHT);
  }
}

/*
 * @description: Get picture width and height and number of channels from image data
 * @param: data specifies the memory to store the image data
 * @param: dataSize specifies the size of the image data
 * @param: width is used to save the image width
 * @param: height is used to save the image height
 * @param: components is used to save the number of channels
 * @return: APP_ERR_OK if success, other values if failure
 */
APP_ERROR DvppCommon::GetJpegImageInfo(const void *data, uint32_t dataSize, uint32_t &width, uint32_t &height,
                                       int32_t &components) {
  uint32_t widthTmp;
  uint32_t heightTmp;
  int32_t componentsTmp;
  APP_ERROR ret = acldvppJpegGetImageInfo(data, dataSize, &widthTmp, &heightTmp, &componentsTmp);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to get image info of jpeg, ret = " << ret << ".";
    return ret;
  }
  if (widthTmp > MAX_JPEGD_WIDTH || widthTmp < MIN_JPEGD_WIDTH) {
    MS_LOG(ERROR) << "Input width is invalid, not in [" << MIN_JPEGD_WIDTH << ", " << MAX_JPEGD_WIDTH << "].";
    return APP_ERR_COMM_INVALID_PARAM;
  }
  if (heightTmp > MAX_JPEGD_HEIGHT || heightTmp < MIN_JPEGD_HEIGHT) {
    MS_LOG(ERROR) << "Input height is invalid, not in [" << MIN_JPEGD_HEIGHT << ", " << MAX_JPEGD_HEIGHT << "].";
    return APP_ERR_COMM_INVALID_PARAM;
  }
  width = widthTmp;
  height = heightTmp;
  components = componentsTmp;
  return APP_ERR_OK;
}

/*
 * @description: Get picture width and height and number of channels from PNG image data
 * @param: data specifies the memory to store the image data
 * @param: dataSize specifies the size of the image data
 * @param: width is used to save the image width
 * @param: height is used to save the image height
 * @param: components is used to save the number of channels
 * @return: APP_ERR_OK if success, other values if failure
 */
APP_ERROR DvppCommon::GetPngImageInfo(const void *data, uint32_t dataSize, uint32_t &width, uint32_t &height,
                                      int32_t &components) {
  uint32_t widthTmp;
  uint32_t heightTmp;
  int32_t componentsTmp;
  APP_ERROR ret = acldvppPngGetImageInfo(data, dataSize, &widthTmp, &heightTmp, &componentsTmp);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to get image info of PNG, ret = " << ret << ".";
    return ret;
  }
  if (widthTmp > MAX_PNGD_WIDTH || widthTmp < MIN_PNGD_WIDTH) {
    MS_LOG(ERROR) << "Input width is invalid, not in [" << MIN_PNGD_WIDTH << ", " << MAX_PNGD_WIDTH << "].";
    return APP_ERR_COMM_INVALID_PARAM;
  }
  if (heightTmp > MAX_PNGD_HEIGHT || heightTmp < MIN_PNGD_HEIGHT) {
    MS_LOG(ERROR) << "Input height is invalid, not in [" << MIN_PNGD_HEIGHT << ", " << MAX_PNGD_HEIGHT << "].";
    return APP_ERR_COMM_INVALID_PARAM;
  }
  width = widthTmp;
  height = heightTmp;
  components = componentsTmp;
  return APP_ERR_OK;
}

/*
 * @description: Get the size of the buffer for storing decoded images based on the image data, size, and format
 * @param: data specifies the memory to store the image data
 * @param: dataSize specifies the size of the image data
 * @param: format specifies the image format
 * @param: decSize is used to store the result size
 * @return: APP_ERR_OK if success, other values if failure
 */
APP_ERROR DvppCommon::GetJpegDecodeDataSize(const void *data, uint32_t dataSize, acldvppPixelFormat format,
                                            uint32_t &decSize) {
  uint32_t outputSize;
  APP_ERROR ret = acldvppJpegPredictDecSize(data, dataSize, format, &outputSize);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to predict decode size of jpeg image, ret = " << ret << ".";
    return ret;
  }
  decSize = outputSize;
  return APP_ERR_OK;
}

/*
 * @description: Get the size of the buffer for storing decoded images based on the PNG image data, size, and format
 * @param: data specifies the memory to store the image data
 * @param: dataSize specifies the size of the image data
 * @param: format specifies the image format
 * @param: decSize is used to store the result size
 * @return: APP_ERR_OK if success, other values if failure
 */
APP_ERROR DvppCommon::GetPngDecodeDataSize(const void *data, uint32_t dataSize, acldvppPixelFormat format,
                                           uint32_t &decSize) {
  uint32_t outputSize;
  APP_ERROR ret = acldvppPngPredictDecSize(data, dataSize, format, &outputSize);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to predict decode size of png image, ret = " << ret << ".";
    return ret;
  }
  decSize = outputSize;
  return APP_ERR_OK;
}

/*
 * @description: Decode the image specified by imageInfo and save the result to member variable decodedImage_
 * @param: imageInfo specifies image information
 * @param: format specifies the image format
 * @param: withSynchronize specifies whether to execute synchronously
 * @return: APP_ERR_OK if success, other values if failure
 * @attention: This function can be called only when the DvppCommon object is initialized with Init
 */
APP_ERROR DvppCommon::CombineJpegdProcess(const RawData &imageInfo, acldvppPixelFormat format, bool withSynchronize) {
  // Return special error code when the DvppCommon object is initialized with InitVdec
  if (isVdec_) {
    MS_LOG(ERROR)
      << "CombineJpegdProcess cannot be called by the DvppCommon object which is initialized with InitVdec.";
    return APP_ERR_DVPP_OBJ_FUNC_MISMATCH;
  }

  int32_t components;
  // Member variable of inputImage_, uint8_t *data will be on device
  inputImage_ = std::make_shared<DvppDataInfo>();
  inputImage_->format = format;
  APP_ERROR ret =
    // GetJpegImageInfo(imageInfo.data.get(), imageInfo.lenOfByte, inputImage_->width, inputImage_->height, components);
    GetJpegImageInfo(imageInfo.data, imageInfo.lenOfByte, inputImage_->width, inputImage_->height, components);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to get input image info, ret = " << ret << ".";
    return ret;
  }

  // Get the buffer size(On device) of decode output according to the input data and output format
  uint32_t outBuffSize;
  // ret = GetJpegDecodeDataSize(imageInfo.data.get(), imageInfo.lenOfByte, format, outBuffSize);
  ret = GetJpegDecodeDataSize(imageInfo.data, imageInfo.lenOfByte, format, outBuffSize);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to get size of decode output buffer, ret = " << ret << ".";
    return ret;
  }

  // In TransferImageH2D function, device buffer will be alloced to store the input image before decode
  // Need to pay attention to release of the buffer
  ret = TransferImageH2D(imageInfo, inputImage_);
  if (ret != APP_ERR_OK) {
    return ret;
  }

  decodedImage_ = std::make_shared<DvppDataInfo>();
  decodedImage_->format = format;
  decodedImage_->width = inputImage_->width;
  decodedImage_->height = inputImage_->height;
  GetJpegDecodeStrideSize(inputImage_->width, inputImage_->height, decodedImage_->widthStride,
                          decodedImage_->heightStride);
  decodedImage_->dataSize = outBuffSize;
  // Malloc dvpp buffer to store the output data after decoding
  // Need to pay attention to release of the buffer
  ret = acldvppMalloc((void **)&decodedImage_->data, decodedImage_->dataSize);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to malloc memory on dvpp, ret = " << ret << ".";
    RELEASE_DVPP_DATA(inputImage_->data);
    return ret;
  }

  ret = JpegDecode(*inputImage_, *decodedImage_, withSynchronize);
  if (ret != APP_ERR_OK) {
    // Release the output buffer when decode failed, otherwise release it after use
    RELEASE_DVPP_DATA(inputImage_->data);
    inputImage_->data = nullptr;
    RELEASE_DVPP_DATA(decodedImage_->data);
    decodedImage_->data = nullptr;
    return ret;
  }

  return APP_ERR_OK;
}

APP_ERROR DvppCommon::SinkCombineJpegdProcess(std::shared_ptr<DvppDataInfo> &input,
                                              std::shared_ptr<DvppDataInfo> &output, bool withSynchronize) {
  // Both input and output are locate on device, so we must release them if fail in decode
  APP_ERROR ret = JpegDecode(*input, *output, withSynchronize);
  if (ret != APP_ERR_OK) {
    // Release the output buffer when decode failed, otherwise release it after use
    RELEASE_DVPP_DATA(inputImage_->data);
    inputImage_->data = nullptr;
    RELEASE_DVPP_DATA(decodedImage_->data);
    decodedImage_->data = nullptr;
    return ret;
  }
  return APP_ERR_OK;
}

APP_ERROR DvppCommon::SinkCombinePngdProcess(std::shared_ptr<DvppDataInfo> &input,
                                             std::shared_ptr<DvppDataInfo> &output, bool withSynchronize) {
  // Both input and output are locate on device, so we must release them if fail in decode
  APP_ERROR ret = PngDecode(*input, *output, withSynchronize);
  if (ret != APP_ERR_OK) {
    // Release the output buffer when decode failed, otherwise release it after use
    RELEASE_DVPP_DATA(inputImage_->data);
    inputImage_->data = nullptr;
    RELEASE_DVPP_DATA(decodedImage_->data);
    decodedImage_->data = nullptr;
    return ret;
  }
  return APP_ERR_OK;
}

/*
 * @description: Decode the image specified by imageInfo and save the result to member variable decodedImage_
 * This function is for PNG format image
 * @param: imageInfo specifies image information
 * @param: format specifies the image format
 * @param: withSynchronize specifies whether to execute synchronously
 * @return: APP_ERR_OK if success, other values if failure
 * @attention: This function can be called only when the DvppCommon object is initialized with Init
 */
APP_ERROR DvppCommon::CombinePngdProcess(const RawData &imageInfo, acldvppPixelFormat format, bool withSynchronize) {
  // Return special error code when the DvppCommon object is initialized with InitVdec
  if (isVdec_) {
    MS_LOG(ERROR) << "CombinePngdProcess cannot be called by the DvppCommon object which is initialized with InitVdec.";
    return APP_ERR_DVPP_OBJ_FUNC_MISMATCH;
  }

  int32_t components;
  inputImage_ = std::make_shared<DvppDataInfo>();
  inputImage_->format = format;
  APP_ERROR ret =
    // GetJpegImageInfo(imageInfo.data.get(), imageInfo.lenOfByte, inputImage_->width, inputImage_->height, components);
    GetPngImageInfo(imageInfo.data, imageInfo.lenOfByte, inputImage_->width, inputImage_->height, components);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to get input image info, ret = " << ret << ".";
    return ret;
  }

  // Get the buffer size of decode output according to the input data and output format
  uint32_t outBuffSize;
  // ret = GetJpegDecodeDataSize(imageInfo.data.get(), imageInfo.lenOfByte, format, outBuffSize);
  ret = GetPngDecodeDataSize(imageInfo.data, imageInfo.lenOfByte, format, outBuffSize);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to get size of decode output buffer, ret = " << ret << ".";
    return ret;
  }

  // In TransferImageH2D function, device buffer will be alloced to store the input image
  // Need to pay attention to release of the buffer
  ret = TransferImageH2D(imageInfo, inputImage_);
  if (ret != APP_ERR_OK) {
    return ret;
  }

  decodedImage_ = std::make_shared<DvppDataInfo>();
  decodedImage_->format = format;
  decodedImage_->width = inputImage_->width;
  decodedImage_->height = inputImage_->height;
  GetPngDecodeStrideSize(inputImage_->width, inputImage_->height, decodedImage_->widthStride,
                         decodedImage_->heightStride, format);
  decodedImage_->dataSize = outBuffSize;
  // Malloc dvpp buffer to store the output data after decoding
  // Need to pay attention to release of the buffer
  ret = acldvppMalloc((void **)&decodedImage_->data, decodedImage_->dataSize);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to malloc memory on dvpp, ret = " << ret << ".";
    RELEASE_DVPP_DATA(inputImage_->data);
    return ret;
  }
  ret = PngDecode(*inputImage_, *decodedImage_, withSynchronize);
  if (ret != APP_ERR_OK) {
    // Release the output buffer when decode failed, otherwise release it after use
    RELEASE_DVPP_DATA(inputImage_->data);
    inputImage_->data = nullptr;
    RELEASE_DVPP_DATA(decodedImage_->data);
    decodedImage_->data = nullptr;
    return ret;
  }

  return APP_ERR_OK;
}

/*
 * @description: Transfer data from host to device
 * @param: imageInfo specifies the image data on the host
 * @return: APP_ERR_OK if success, other values if failure
 */
APP_ERROR DvppCommon::TransferYuvDataH2D(const DvppDataInfo &imageinfo) {
  if (imageinfo.dataSize <= 0) {
    MS_LOG(ERROR) << "The input buffer size on host should not be empty.";
    return APP_ERR_COMM_INVALID_PARAM;
  }
  uint8_t *device_ptr = nullptr;
  APP_ERROR ret = acldvppMalloc((void **)&device_ptr, imageinfo.dataSize);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to malloc " << imageinfo.dataSize << " bytes on dvpp, ret = " << ret << ".";
    return ret;
  }
  ret = aclrtMemcpyAsync(device_ptr, imageinfo.dataSize, imageinfo.data, imageinfo.dataSize, ACL_MEMCPY_HOST_TO_DEVICE,
                         dvppStream_);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to copy " << imageinfo.dataSize << " bytes from host to device, ret = " << ret << ".";
    RELEASE_DVPP_DATA(device_ptr);
    return ret;
  }
  ret = aclrtSynchronizeStream(dvppStream_);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to synchronize stream, ret = " << ret << ".";
    RELEASE_DVPP_DATA(device_ptr);
    return ret;
  }
  /* Important!!! decodedImage_ speifies the image in deocded format(RGB OR YUV)
   * Not essentailly to be the image after decode.(Specifies the data not in RAW encode format)
   * It can also be the image after resize(Very important)
   */
  decodedImage_ = std::make_shared<DvppDataInfo>();
  decodedImage_->data = device_ptr;
  decodedImage_->dataSize = imageinfo.dataSize;
  decodedImage_->height = imageinfo.height;
  decodedImage_->heightStride = imageinfo.heightStride;
  decodedImage_->width = imageinfo.width;
  decodedImage_->widthStride = imageinfo.widthStride;
  return APP_ERR_OK;
}

/*
 * @description: Transfer data from host to device
 * @param: imageInfo specifies the image data on the host
 * @param: jpegInput is used to save the buffer and its size which is allocate on the device
 * @return: APP_ERR_OK if success, other values if failure
 */
APP_ERROR DvppCommon::TransferImageH2D(const RawData &imageInfo, const std::shared_ptr<DvppDataInfo> &jpegInput) {
  // Check image buffer size validity
  if (imageInfo.lenOfByte <= 0) {
    MS_LOG(ERROR) << "The input buffer size on host should not be empty.";
    return APP_ERR_COMM_INVALID_PARAM;
  }

  uint8_t *inDevBuff = nullptr;  // This pointer will be on device
  APP_ERROR ret = acldvppMalloc((void **)&inDevBuff, imageInfo.lenOfByte);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to malloc " << imageInfo.lenOfByte << " bytes on dvpp, ret = " << ret << ".";
    return ret;
  }

  // Copy the image data from host to device
  // ret = aclrtMemcpyAsync(inDevBuff, imageInfo.lenOfByte, imageInfo.data.get(), imageInfo.lenOfByte,
  //                       ACL_MEMCPY_HOST_TO_DEVICE, dvppStream_);
  ret = aclrtMemcpyAsync(inDevBuff, imageInfo.lenOfByte, imageInfo.data, imageInfo.lenOfByte, ACL_MEMCPY_HOST_TO_DEVICE,
                         dvppStream_);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to copy " << imageInfo.lenOfByte << " bytes from host to device, ret = " << ret << ".";
    RELEASE_DVPP_DATA(inDevBuff);
    return ret;
  }
  // Attention: We must call the aclrtSynchronizeStream to ensure the task of memory replication has been completed
  // after calling aclrtMemcpyAsync
  ret = aclrtSynchronizeStream(dvppStream_);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to synchronize stream, ret = " << ret << ".";
    RELEASE_DVPP_DATA(inDevBuff);
    return ret;
  }
  jpegInput->data = inDevBuff;
  jpegInput->dataSize = imageInfo.lenOfByte;
  return APP_ERR_OK;
}

/*
 * Sink RawData(On host) into DvppDataInfo(On device)
 */
APP_ERROR DvppCommon::SinkImageH2D(const RawData &imageInfo, acldvppPixelFormat format) {
  if (isVdec_) {
    MS_LOG(ERROR)
      << "CombineJpegdProcess cannot be called by the DvppCommon object which is initialized with InitVdec.";
    return APP_ERR_DVPP_OBJ_FUNC_MISMATCH;
  }

  APP_ERROR ret = aclrtSetCurrentContext(dvppContext_);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to get ACL context, ret = " << ret;
    return ret;
  }

  int32_t components;
  // Member variable of inputImage_, uint8_t *data will be on device
  inputImage_ = std::make_shared<DvppDataInfo>();
  inputImage_->format = format;
  ret = GetJpegImageInfo(imageInfo.data, imageInfo.lenOfByte, inputImage_->width, inputImage_->height, components);

  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to get input image info, ret = " << ret << ".";
    return ret;
  }

  // Get the buffer size(On device) of decode output according to the input data and output format
  uint32_t outBufferSize;
  // ret = GetJpegDecodeDataSize(imageInfo.data.get(), imageInfo.lenOfByte, format, outBuffSize);
  ret = GetJpegDecodeDataSize(imageInfo.data, imageInfo.lenOfByte, format, outBufferSize);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to get size of decode output buffer, ret = " << ret << ".";
    return ret;
  }
  // In TransferImageH2D function, device buffer will be alloced to store the input image before decode
  // Need to pay attention to release of the buffer
  ret = TransferImageH2D(imageInfo, inputImage_);
  if (ret != APP_ERR_OK) {
    return ret;
  }
  // This part is to define the data after decode (MALLOC ON DEVICE!!)
  decodedImage_ = std::make_shared<DvppDataInfo>();
  decodedImage_->format = format;
  decodedImage_->width = inputImage_->width;
  decodedImage_->height = inputImage_->height;
  GetJpegDecodeStrideSize(inputImage_->width, inputImage_->height, decodedImage_->widthStride,
                          decodedImage_->heightStride);
  // Obtain all the attributes of inputImage_
  GetJpegDecodeStrideSize(inputImage_->width, inputImage_->height, inputImage_->widthStride, inputImage_->heightStride);
  decodedImage_->dataSize = outBufferSize;
  // Malloc dvpp buffer to store the output data after decoding
  // Need to pay attention to release of the buffer
  ret = acldvppMalloc((void **)&decodedImage_->data, decodedImage_->dataSize);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to malloc memory on dvpp, ret = " << ret << ".";
    RELEASE_DVPP_DATA(inputImage_->data);
    return ret;
  }
  return APP_ERR_OK;
}

APP_ERROR DvppCommon::SinkImageH2D(const RawData &imageInfo) {
  if (isVdec_) {
    MS_LOG(ERROR) << "CombinePngdProcess cannot be called by the DvppCommon object which is initialized with InitVdec.";
    return APP_ERR_DVPP_OBJ_FUNC_MISMATCH;
  }

  APP_ERROR ret = aclrtSetCurrentContext(dvppContext_);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to get ACL context, ret = " << ret;
    return ret;
  }

  int32_t components;
  inputImage_ = std::make_shared<DvppDataInfo>();
  acldvppPixelFormat format = PIXEL_FORMAT_RGB_888;
  inputImage_->format = format;

  ret = GetPngImageInfo(imageInfo.data, imageInfo.lenOfByte, inputImage_->width, inputImage_->height, components);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to get input image info, ret = " << ret << ".";
    return ret;
  }

  // Get the buffer size of decode output according to the input data and output format
  uint32_t outBuffSize;
  // ret = GetJpegDecodeDataSize(imageInfo.data.get(), imageInfo.lenOfByte, format, outBuffSize);
  ret = GetPngDecodeDataSize(imageInfo.data, imageInfo.lenOfByte, format, outBuffSize);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to get size of decode output buffer, ret = " << ret << ".";
    return ret;
  }

  // In TransferImageH2D function, device buffer will be alloced to store the input image
  // Need to pay attention to release of the buffer
  ret = TransferImageH2D(imageInfo, inputImage_);
  if (ret != APP_ERR_OK) {
    return ret;
  }

  decodedImage_ = std::make_shared<DvppDataInfo>();
  decodedImage_->format = format;
  decodedImage_->width = inputImage_->width;
  decodedImage_->height = inputImage_->height;
  GetPngDecodeStrideSize(inputImage_->width, inputImage_->height, decodedImage_->widthStride,
                         decodedImage_->heightStride, format);
  decodedImage_->dataSize = outBuffSize;
  // Malloc dvpp buffer to store the output data after decoding
  // Need to pay attention to release of the buffer
  ret = acldvppMalloc((void **)&decodedImage_->data, decodedImage_->dataSize);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to malloc memory on dvpp, ret = " << ret << ".";
    RELEASE_DVPP_DATA(inputImage_->data);
    return ret;
  }
  return APP_ERR_OK;
}

/*
 * @description: Create and set the description of a video stream
 * @param: data specifies the information about the video stream
 * @return: APP_ERR_OK if success, other values if failure
 */
APP_ERROR DvppCommon::CreateStreamDesc(std::shared_ptr<DvppDataInfo> data) {
  // Malloc input device memory which need to be released in vdec callback function
  void *modelInBuff = nullptr;
  APP_ERROR ret = acldvppMalloc(&modelInBuff, data->dataSize);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to malloc dvpp data with " << data->dataSize << " bytes, ret = " << ret << ".";
    return APP_ERR_ACL_BAD_ALLOC;
  }
  // copy input to device memory
  ret = aclrtMemcpy(modelInBuff, data->dataSize, static_cast<uint8_t *>(data->data), data->dataSize,
                    ACL_MEMCPY_HOST_TO_DEVICE);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to copy memory with " << data->dataSize << " bytes from host to device, ret = " << ret
                  << ".";
    acldvppFree(modelInBuff);
    modelInBuff = nullptr;
    return APP_ERR_ACL_FAILURE;
  }
  // Create input stream desc which need to be destoryed in vdec callback function
  streamInputDesc_ = acldvppCreateStreamDesc();
  if (streamInputDesc_ == nullptr) {
    MS_LOG(ERROR) << "Failed to create input stream description.";
    return APP_ERR_ACL_FAILURE;
  }
  ret = acldvppSetStreamDescData(streamInputDesc_, modelInBuff);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to set data for stream desdescription, ret = " << ret << ".";
    return ret;
  }
  // set size for dvpp stream desc
  ret = acldvppSetStreamDescSize(streamInputDesc_, data->dataSize);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to set data size for stream desdescription,  ret = " << ret << ".";
    return ret;
  }
  return APP_ERR_OK;
}

/*
 * @description: Decode the video based on the video stream specified by data and user-defined data,
 *               and outputs the image of each frame
 * @param: data specifies the information about the video stream
 * @param: userdata is specified for user-defined data
 * @return: APP_ERR_OK if success, other values if failure
 * @attention: This function can be called only when the DvppCommon object is initialized with InitVdec
 */
APP_ERROR DvppCommon::CombineVdecProcess(std::shared_ptr<DvppDataInfo> data, void *userData) {
  // Return special error code when the DvppCommon object is not initialized with InitVdec
  if (!isVdec_) {
    MS_LOG(ERROR)
      << "CombineVdecProcess cannot be called by the DvppCommon object which is not initialized with InitVdec.";
    return APP_ERR_DVPP_OBJ_FUNC_MISMATCH;
  }
  // create stream desc
  APP_ERROR ret = CreateStreamDesc(data);
  if (ret != APP_ERR_OK) {
    return ret;
  }

  uint32_t dataSize;
  ret = GetVideoDecodeDataSize(vdecConfig_.inputWidth, vdecConfig_.inputHeight, vdecConfig_.outFormat, dataSize);
  if (ret != APP_ERR_OK) {
    return ret;
  }

  void *picOutBufferDev = nullptr;
  // picOutBufferDev need to be destoryed in vdec callback function
  ret = acldvppMalloc(&picOutBufferDev, dataSize);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to malloc memory with " << dataSize << " bytes, ret = " << ret << ".";
    return APP_ERR_ACL_BAD_ALLOC;
  }

  // picOutputDesc_ will be destoryed in vdec callback function
  picOutputDesc_ = acldvppCreatePicDesc();
  if (picOutputDesc_ == NULL) {
    return APP_ERR_ACL_BAD_ALLOC;
  }

  DvppDataInfo dataInfo;
  dataInfo.width = vdecConfig_.inputWidth;
  dataInfo.height = vdecConfig_.inputHeight;
  dataInfo.format = vdecConfig_.outFormat;
  dataInfo.dataSize = dataSize;
  dataInfo.data = static_cast<uint8_t *>(picOutBufferDev);
  ret = SetDvppPicDescData(dataInfo, *picOutputDesc_);
  if (ret != APP_ERR_OK) {
    return ret;
  }

  // send frame
  ret = aclvdecSendFrame(vdecChannelDesc_, streamInputDesc_, picOutputDesc_, nullptr, userData);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to send frame, ret = " << ret << ".";
    return APP_ERR_ACL_FAILURE;
  }

  return APP_ERR_OK;
}

/*
 * @description: Send eos frame when video stream ends
 * @return: APP_ERR_OK if success, other values if failure
 */
APP_ERROR DvppCommon::VdecSendEosFrame() const {
  // create input stream desc
  acldvppStreamDesc *eosStreamDesc = acldvppCreateStreamDesc();
  if (eosStreamDesc == nullptr) {
    MS_LOG(ERROR) << "Fail to create dvpp stream desc for eos.";
    return ACL_ERROR_FAILURE;
  }

  // set eos for eos stream desc
  APP_ERROR ret = acldvppSetStreamDescEos(eosStreamDesc, true);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Fail to set eos for stream desc, ret = " << ret << ".";
    acldvppDestroyStreamDesc(eosStreamDesc);
    return ret;
  }

  // send eos and synchronize
  ret = aclvdecSendFrame(vdecChannelDesc_, eosStreamDesc, nullptr, nullptr, nullptr);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Fail to send eos, ret = " << ret << ".";
    acldvppDestroyStreamDesc(eosStreamDesc);
    return ret;
  }

  // destory input stream desc
  ret = acldvppDestroyStreamDesc(eosStreamDesc);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Fail to destory dvpp stream desc for eos, ret = " << ret << ".";
    return ret;
  }
  return ret;
}

/*
 * @description: Get the aligned width and height of the output image after video decoding
 * @param: width specifies the width before alignment
 * @param: height specifies the height before alignment
 * @param: format specifies the format of the output image
 * @param: widthStride is used to save the width after alignment
 * @param: heightStride is used to save the height after alignment
 * @return: APP_ERR_OK if success, other values if failure
 */
APP_ERROR DvppCommon::GetVideoDecodeStrideSize(uint32_t width, uint32_t height, acldvppPixelFormat format,
                                               uint32_t &widthStride, uint32_t &heightStride) {
  // Check the invalidty of output format and calculate the output width and height
  if (format != PIXEL_FORMAT_YUV_SEMIPLANAR_420 && format != PIXEL_FORMAT_YVU_SEMIPLANAR_420) {
    MS_LOG(ERROR) << "Input format[" << format << "] for VPC is not supported, just support NV12 or NV21.";
    return APP_ERR_COMM_INVALID_PARAM;
  }
  widthStride = DVPP_ALIGN_UP(width, VDEC_STRIDE_WIDTH);
  heightStride = DVPP_ALIGN_UP(height, VDEC_STRIDE_HEIGHT);
  return APP_ERR_OK;
}

/*
 * @description: Get the buffer size for storing results after video decoding
 * @param  width specifies the width of the output image after video decoding
 * @param  height specifies the height of the output image after video decoding
 * @param  format specifies the format of the output image
 * @param: vpcSize is used to save the result size
 * @return: APP_ERR_OK if success, other values if failure
 */
APP_ERROR DvppCommon::GetVideoDecodeDataSize(uint32_t width, uint32_t height, acldvppPixelFormat format,
                                             uint32_t &vdecSize) {
  // Check the invalid format of vdec output and calculate the output buffer size
  if (format != PIXEL_FORMAT_YUV_SEMIPLANAR_420 && format != PIXEL_FORMAT_YVU_SEMIPLANAR_420) {
    MS_LOG(ERROR) << "Format[" << format << "] for VPC is not supported, just support NV12 or NV21.";
    return APP_ERR_COMM_INVALID_PARAM;
  }
  uint32_t widthStride = DVPP_ALIGN_UP(width, VDEC_STRIDE_WIDTH);
  uint32_t heightStride = DVPP_ALIGN_UP(height, VDEC_STRIDE_HEIGHT);
  vdecSize = widthStride * heightStride * YUV_BGR_SIZE_CONVERT_3 / YUV_BGR_SIZE_CONVERT_2;
  return APP_ERR_OK;
}

/*
 * @description: Encode a YUV image into a JPG image
 * @param: input specifies the input image information
 * @param: output specifies the output image information
 * @param: jpegeConfig specifies the encoding configuration data
 * @param: withSynchronize specifies whether to execute synchronously
 * @return: APP_ERR_OK if success, other values if failure
 * @attention: This function can be called only when the DvppCommon object is initialized with Init
 */
APP_ERROR DvppCommon::JpegEncode(DvppDataInfo &input, DvppDataInfo &output, acldvppJpegeConfig *jpegeConfig,
                                 bool withSynchronize) {
  // Return special error code when the DvppCommon object is initialized with InitVdec
  if (isVdec_) {
    MS_LOG(ERROR) << "JpegEncode cannot be called by the DvppCommon object which is initialized with InitVdec.";
    return APP_ERR_DVPP_OBJ_FUNC_MISMATCH;
  }

  APP_ERROR ret = SetDvppPicDescData(input, *encodeInputDesc_);
  if (ret != APP_ERR_OK) {
    return ret;
  }

  ret = acldvppJpegEncodeAsync(dvppChannelDesc_, encodeInputDesc_.get(), output.data, &output.dataSize, jpegeConfig,
                               dvppStream_);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to encode image, ret = " << ret << ".";
    return ret;
  }
  if (withSynchronize) {
    ret = aclrtSynchronizeStream(dvppStream_);
    if (ret != APP_ERR_OK) {
      MS_LOG(ERROR) << "Failed to aclrtSynchronizeStream, ret = " << ret << ".";
      return APP_ERR_DVPP_JPEG_ENCODE_FAIL;
    }
  }
  MS_LOG(INFO) << "Encode successfully.";
  return APP_ERR_OK;
}

/*
 * @description: Get the aligned width, height, and data size of the input image
 * @param: inputImage specifies the input image information
 * @return: APP_ERR_OK if success, other values if failure
 */
APP_ERROR DvppCommon::GetJpegEncodeStrideSize(std::shared_ptr<DvppDataInfo> &inputImage) {
  uint32_t inputWidth = inputImage->width;
  uint32_t inputHeight = inputImage->height;
  acldvppPixelFormat format = inputImage->format;
  uint32_t widthStride;
  uint32_t heightStride;
  uint32_t encodedBufferSize;
  // Align up the input width and height and calculate buffer size of encoded input file
  if (format == PIXEL_FORMAT_YUV_SEMIPLANAR_420 || format == PIXEL_FORMAT_YVU_SEMIPLANAR_420) {
    widthStride = DVPP_ALIGN_UP(inputWidth, JPEGE_STRIDE_WIDTH);
    heightStride = DVPP_ALIGN_UP(inputHeight, JPEGE_STRIDE_HEIGHT);
    encodedBufferSize = widthStride * heightStride * YUV_BYTES_NU / YUV_BYTES_DE;
  } else if (format == PIXEL_FORMAT_YUYV_PACKED_422 || format == PIXEL_FORMAT_UYVY_PACKED_422 ||
             format == PIXEL_FORMAT_YVYU_PACKED_422 || format == PIXEL_FORMAT_VYUY_PACKED_422) {
    widthStride = DVPP_ALIGN_UP(inputWidth * YUV422_WIDTH_NU, JPEGE_STRIDE_WIDTH);
    heightStride = DVPP_ALIGN_UP(inputHeight, JPEGE_STRIDE_HEIGHT);
    encodedBufferSize = widthStride * heightStride;
  } else {
    return APP_ERR_COMM_INVALID_PARAM;
  }
  if (encodedBufferSize == 0) {
    MS_LOG(ERROR) << "Input host buffer size is empty.";
    return APP_ERR_COMM_INVALID_PARAM;
  }
  inputImage->widthStride = widthStride;
  inputImage->heightStride = heightStride;
  inputImage->dataSize = encodedBufferSize;
  return APP_ERR_OK;
}

/*
 * @description: Estimate the size of the output memory required by image encoding according to
 *               the input image description and image encoding configuration data
 * @param: input specifies specifies the input image information
 * @param: jpegeConfig specifies the encoding configuration data
 * @param: encSize is used to save the result size
 * @return: APP_ERR_OK if success, other values if failure
 * @attention: This function can be called only when the DvppCommon object is initialized with Init
 */
APP_ERROR DvppCommon::GetJpegEncodeDataSize(DvppDataInfo &input, acldvppJpegeConfig *jpegeConfig, uint32_t &encSize) {
  // Return special error code when the DvppCommon object is initialized with InitVdec
  if (isVdec_) {
    MS_LOG(ERROR)
      << "GetJpegEncodeDataSize cannot be called by the DvppCommon object which is initialized with InitVdec.";
    return APP_ERR_DVPP_OBJ_FUNC_MISMATCH;
  }

  acldvppPicDesc *inputDesc = acldvppCreatePicDesc();
  encodeInputDesc_.reset(inputDesc, g_picDescDeleter);

  APP_ERROR ret = SetDvppPicDescData(input, *encodeInputDesc_);
  if (ret != APP_ERR_OK) {
    return ret;
  }

  uint32_t outputSize;
  ret = acldvppJpegPredictEncSize(encodeInputDesc_.get(), jpegeConfig, &outputSize);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to predict encode size of jpeg image, ret = " << ret << ".";
    return ret;
  }
  encSize = outputSize;
  return APP_ERR_OK;
}

/*
 * @description: Set the encoding configuration data
 * @param: level specifies the encode quality range
 * @param: jpegeConfig specifies the encoding configuration data
 * @return: APP_ERR_OK if success, other values if failure
 */
APP_ERROR DvppCommon::SetEncodeLevel(uint32_t level, acldvppJpegeConfig &jpegeConfig) {
  // Set the encoding quality
  // The coding quality range [0, 100]
  // The level 0 coding quality is similar to the level 100
  // The smaller the value in [1, 100], the worse the quality of the output picture
  auto ret = (APP_ERROR)acldvppSetJpegeConfigLevel(&jpegeConfig, level);
  if (ret != APP_ERR_OK) {
    return ret;
  }
  return APP_ERR_OK;
}

/*
 * @description: Encode the image specified by imageInfo and save the result to member variable encodedImage_
 * @param: imageInfo specifies image information
 * @param: width specifies the width of the input image
 * @param: height specifies the height of the input image
 * @param: format specifies the format of the input image
 * @param: withSynchronize specifies whether to execute synchronously
 * @return: APP_ERR_OK if success, other values if failure
 * @attention: This function can be called only when the DvppCommon object is initialized with Init
 */
APP_ERROR DvppCommon::CombineJpegeProcess(const RawData &imageInfo, uint32_t width, uint32_t height,
                                          acldvppPixelFormat format, bool withSynchronize) {
  // Return special error code when the DvppCommon object is initialized with InitVdec
  if (isVdec_) {
    MS_LOG(ERROR)
      << "CombineJpegeProcess cannot be called by the DvppCommon object which is initialized with InitVdec.";
    return APP_ERR_DVPP_OBJ_FUNC_MISMATCH;
  }
  inputImage_ = std::make_shared<DvppDataInfo>();
  inputImage_->format = format;
  inputImage_->width = width;
  inputImage_->height = height;
  // In TransferImageH2D function, device buffer will be alloced to store the input image
  // Need to pay attention to release of the buffer
  APP_ERROR ret = TransferImageH2D(imageInfo, inputImage_);
  if (ret != APP_ERR_OK) {
    return ret;
  }
  // Get stride size of encoded image
  ret = GetJpegEncodeStrideSize(inputImage_);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to get encode stride size of input image file, ret = " << ret << ".";
    return ret;
  }

  auto jpegeConfig = acldvppCreateJpegeConfig();
  jpegeConfig_.reset(jpegeConfig, g_jpegeConfigDeleter);

  uint32_t encodeLevel = 100;
  ret = SetEncodeLevel(encodeLevel, *jpegeConfig_);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to set encode level, ret = " << ret << ".";
    return ret;
  }

  // Get the buffer size of encode output according to the input data and jpeg encode config
  uint32_t encodeOutBufferSize;
  ret = GetJpegEncodeDataSize(*inputImage_, jpegeConfig_.get(), encodeOutBufferSize);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to get size of encode output buffer, ret = " << ret << ".";
    return ret;
  }

  encodedImage_ = std::make_shared<DvppDataInfo>();
  encodedImage_->dataSize = encodeOutBufferSize;
  // Malloc dvpp buffer to store the output data after decoding
  // Need to pay attention to release of the buffer
  ret = acldvppMalloc((void **)&encodedImage_->data, encodedImage_->dataSize);
  if (ret != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed to malloc memory on dvpp, ret = " << ret << ".";
    acldvppFree(inputImage_->data);
    return ret;
  }

  // Encode input image
  ret = JpegEncode(*inputImage_, *encodedImage_, jpegeConfig_.get(), withSynchronize);
  if (ret != APP_ERR_OK) {
    // Release the output buffer when decode failed, otherwise release it after use
    acldvppFree(inputImage_->data);
    acldvppFree(encodedImage_->data);
    return ret;
  }
  return APP_ERR_OK;
}

std::shared_ptr<DvppDataInfo> DvppCommon::GetInputImage() { return inputImage_; }

std::shared_ptr<DvppDataInfo> DvppCommon::GetDecodedImage() { return decodedImage_; }

std::shared_ptr<DvppDataInfo> DvppCommon::GetResizedImage() { return resizedImage_; }

std::shared_ptr<DvppDataInfo> DvppCommon::GetEncodedImage() { return encodedImage_; }

std::shared_ptr<DvppDataInfo> DvppCommon::GetCropedImage() { return cropImage_; }

DvppCommon::~DvppCommon() {}
