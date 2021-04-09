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

#include <iostream>
#include <memory>

#include "../inc/DvppCommon.h"
#include "../inc/CommonDataType.h"

static auto g_resizeConfigDeleter = [](acldvppResizeConfig *p) { acldvppDestroyResizeConfig(p); };
static auto g_picDescDeleter = [](acldvppPicDesc *picDesc) { acldvppDestroyPicDesc(picDesc); };
static auto g_roiConfigDeleter = [](acldvppRoiConfig *p) { acldvppDestroyRoiConfig(p); };

DvppCommon::DvppCommon(aclrtStream dvppStream):dvppStream_(dvppStream) {}

/*
 * @description: Create a channel for processing image data,
 *               the channel description is created by acldvppCreateChannelDesc
 * @return: OK if success, other values if failure
 */
int DvppCommon::Init(void) {
    dvppChannelDesc_ = acldvppCreateChannelDesc();
    if (dvppChannelDesc_ == nullptr) {
        return -1;
    }
    int ret = acldvppCreateChannel(dvppChannelDesc_);
    if (ret != 0) {
        std::cout << "Failed to create dvpp channel, ret = " << ret << "." << std::endl;
        acldvppDestroyChannelDesc(dvppChannelDesc_);
        dvppChannelDesc_ = nullptr;
        return ret;
    }

    return OK;
}

/*
 * @description:  destroy the channel and the channel description used by image.
 * @return: OK if success, other values if failure
 */
int DvppCommon::DeInit(void) {
    int ret = aclrtSynchronizeStream(dvppStream_);  // int ret
    if (ret != OK) {
        std::cout << "Failed to synchronize stream, ret = " << ret << "." << std::endl;
        return ret;
    }

    ret = acldvppDestroyChannel(dvppChannelDesc_);
    if (ret != OK) {
        std::cout << "Failed to destroy dvpp channel, ret = " << ret << "." << std::endl;
        return ret;
    }

    ret = acldvppDestroyChannelDesc(dvppChannelDesc_);
    if (ret != OK) {
        std::cout << "Failed to destroy dvpp channel description, ret = " << ret << "." << std::endl;
        return ret;
    }
    return OK;
}

/*
 * @description: Release the memory that is allocated in the interfaces which are started with "Combine"
 */
void DvppCommon::ReleaseDvppBuffer() {
    if (resizedImage_ != nullptr) {
        RELEASE_DVPP_DATA(resizedImage_->data);
    }
    if (decodedImage_ != nullptr) {
        RELEASE_DVPP_DATA(decodedImage_->data);
    }
    if (inputImage_ != nullptr) {
        RELEASE_DVPP_DATA(inputImage_->data);
    }
}

/*
 * @description: Get the size of buffer used to save image for VPC according to width, height and format
 * @param  width specifies the width of the output image
 * @param  height specifies the height of the output image
 * @param  format specifies the format of the output image
 * @param: vpcSize is used to save the result size
 * @return: OK if success, other values if failure
 */
int DvppCommon::GetVpcDataSize(uint32_t width, uint32_t height, acldvppPixelFormat format, uint32_t *vpcSize) {
    if (format != PIXEL_FORMAT_YUV_SEMIPLANAR_420 && format != PIXEL_FORMAT_YVU_SEMIPLANAR_420) {
        std::cout << "Format[" << format << "] for VPC is not supported, just support NV12 or NV21." << std::endl;
        return INVALID_PARAM;
    }
    uint32_t widthStride = DVPP_ALIGN_UP(width, VPC_WIDTH_ALIGN);
    uint32_t heightStride = DVPP_ALIGN_UP(height, VPC_HEIGHT_ALIGN);
    *vpcSize = widthStride * heightStride * YUV_BGR_SIZE_CONVERT_3 / YUV_BGR_SIZE_CONVERT_2;
    return OK;
}

/*
 * @description: Get the aligned width and height of the input image according to the image format
 * @param: width specifies the width before alignment
 * @param: height specifies the height before alignment
 * @param: format specifies the image format
 * @param: widthStride is used to save the width after alignment
 * @param: heightStride is used to save the height after alignment
 * @return: OK if success, other values if failure
 */
int DvppCommon::GetVpcInputStrideSize(uint32_t width, uint32_t height, acldvppPixelFormat format,
                                      uint32_t *widthStride, uint32_t *heightStride) {
    uint32_t inputWidthStride;
    if (format >= PIXEL_FORMAT_YUV_400 && format <= PIXEL_FORMAT_YVU_SEMIPLANAR_444) {
        inputWidthStride = DVPP_ALIGN_UP(width, VPC_STRIDE_WIDTH);
    } else if (format >= PIXEL_FORMAT_YUYV_PACKED_422 && format <= PIXEL_FORMAT_VYUY_PACKED_422) {
        inputWidthStride = DVPP_ALIGN_UP(width, VPC_STRIDE_WIDTH) * YUV422_WIDTH_NU;
    } else if (format >= PIXEL_FORMAT_YUV_PACKED_444 && format <= PIXEL_FORMAT_BGR_888) {
        inputWidthStride = DVPP_ALIGN_UP(width, VPC_STRIDE_WIDTH) * YUV444_RGB_WIDTH_NU;
    } else if (format >= PIXEL_FORMAT_ARGB_8888 && format <= PIXEL_FORMAT_BGRA_8888) {
        inputWidthStride = DVPP_ALIGN_UP(width, VPC_STRIDE_WIDTH) * XRGB_WIDTH_NU;
    } else {
        std::cout << "Input format[" << format << "] for VPC is invalid, please check it." << std::endl;
        return INVALID_PARAM;
    }
    uint32_t inputHeightStride = DVPP_ALIGN_UP(height, VPC_STRIDE_HEIGHT);
    if (inputWidthStride > MAX_RESIZE_WIDTH || inputWidthStride < MIN_RESIZE_WIDTH) {
        std::cout << "Input width stride " << inputWidthStride << " is invalid, not in [" << MIN_RESIZE_WIDTH \
                 << ", " << MAX_RESIZE_WIDTH << "]." << std::endl;
        return INVALID_PARAM;
    }

    if (inputHeightStride > MAX_RESIZE_HEIGHT || inputHeightStride < MIN_RESIZE_HEIGHT) {
        std::cout << "Input height stride " << inputHeightStride << " is invalid, not in [" << MIN_RESIZE_HEIGHT \
                 << ", " << MAX_RESIZE_HEIGHT << "]." << std::endl;
        return INVALID_PARAM;
    }
    *widthStride = inputWidthStride;
    *heightStride = inputHeightStride;
    return OK;
}

/*
 * @description: Get the aligned width and height of the output image according to the image format
 * @param: width specifies the width before alignment
 * @param: height specifies the height before alignment
 * @param: format specifies the image format
 * @param: widthStride is used to save the width after alignment
 * @param: heightStride is used to save the height after alignment
 * @return: OK if success, other values if failure
 */
int DvppCommon::GetVpcOutputStrideSize(uint32_t width, uint32_t height, acldvppPixelFormat format,
                                       uint32_t *widthStride, uint32_t *heightStride) {
    if (format != PIXEL_FORMAT_YUV_SEMIPLANAR_420 && format != PIXEL_FORMAT_YVU_SEMIPLANAR_420) {
        std::cout << "Output format[" << format << "] is not supported, just support NV12 or NV21." << std::endl;
        return INVALID_PARAM;
    }

    *widthStride = DVPP_ALIGN_UP(width, VPC_STRIDE_WIDTH);
    *heightStride = DVPP_ALIGN_UP(height, VPC_STRIDE_HEIGHT);
    return OK;
}

/*
 * @description: Set picture description information and execute resize function
 * @param: input specifies the input image information
 * @param: output specifies the output image information
 * @param: withSynchronize specifies whether to execute synchronously
 * @param: processType specifies whether to perform proportional scaling, default is non-proportional resize
 * @return: OK if success, other values if failure
 * @attention: This function can be called only when the DvppCommon object is initialized with Init
 */
int DvppCommon::VpcResize(std::shared_ptr<DvppDataInfo> input, std::shared_ptr<DvppDataInfo> output,
                          bool withSynchronize, VpcProcessType processType) {
    acldvppPicDesc *inputDesc = acldvppCreatePicDesc();
    acldvppPicDesc *outputDesc = acldvppCreatePicDesc();
    resizeInputDesc_.reset(inputDesc, g_picDescDeleter);
    resizeOutputDesc_.reset(outputDesc, g_picDescDeleter);

    // Set dvpp picture descriptin info of input image
    int ret = SetDvppPicDescData(input, resizeInputDesc_);
    if (ret != OK) {
        std::cout << "Failed to set dvpp input picture description, ret = " << ret << "." << std::endl;
        return ret;
    }

    // Set dvpp picture descriptin info of output image
    ret = SetDvppPicDescData(output, resizeOutputDesc_);
    if (ret != OK) {
        std::cout << "Failed to set dvpp output picture description, ret = " << ret << "." << std::endl;
        return ret;
    }
    if (processType == VPC_PT_DEFAULT) {
        return ResizeProcess(resizeInputDesc_, resizeOutputDesc_, withSynchronize);
    }

    // Get crop area according to the processType
    CropRoiConfig cropRoi = {0};
    GetCropRoi(input, output, processType, &cropRoi);

    // The width and height of the original image will be resized by the same ratio
    CropRoiConfig pasteRoi = {0};
    GetPasteRoi(input, output, processType, &pasteRoi);

    return ResizeWithPadding(resizeInputDesc_, resizeOutputDesc_, cropRoi, pasteRoi, withSynchronize);
}

/*
 * @description: Set image description information
 * @param: dataInfo specifies the image information
 * @param: picsDesc specifies the picture description information to be set
 * @return: OK if success, other values if failure
 */
int DvppCommon::SetDvppPicDescData(std::shared_ptr<DvppDataInfo> dataInfo, std::shared_ptr<acldvppPicDesc>picDesc) {
    int ret = acldvppSetPicDescData(picDesc.get(), dataInfo->data);
    if (ret != OK) {
        std::cout << "Failed to set data for dvpp picture description, ret = " << ret << "." << std::endl;
        return ret;
    }
    ret = acldvppSetPicDescSize(picDesc.get(), dataInfo->dataSize);
    if (ret != OK) {
        std::cout << "Failed to set size for dvpp picture description, ret = " << ret << "." << std::endl;
        return ret;
    }
    ret = acldvppSetPicDescFormat(picDesc.get(), dataInfo->format);
    if (ret != OK) {
        std::cout << "Failed to set format for dvpp picture description, ret = " << ret << "." << std::endl;
        return ret;
    }
    ret = acldvppSetPicDescWidth(picDesc.get(), dataInfo->width);
    if (ret != OK) {
        std::cout << "Failed to set width for dvpp picture description, ret = " << ret << "." << std::endl;
        return ret;
    }
    ret = acldvppSetPicDescHeight(picDesc.get(), dataInfo->height);
    if (ret != OK) {
        std::cout << "Failed to set height for dvpp picture description, ret = " << ret << "." << std::endl;
        return ret;
    }

    ret = acldvppSetPicDescWidthStride(picDesc.get(), dataInfo->widthStride);
    if (ret != OK) {
        std::cout << "Failed to set aligned width for dvpp picture description, ret = " << ret << "." << std::endl;
        return ret;
    }
    ret = acldvppSetPicDescHeightStride(picDesc.get(), dataInfo->heightStride);
    if (ret != OK) {
        std::cout << "Failed to set aligned height for dvpp picture description, ret = " << ret << "." << std::endl;
        return ret;
    }

    return OK;
}

/*
 * @description: Check whether the image format and zoom ratio meet the requirements
 * @param: input specifies the input image information
 * @param: output specifies the output image information
 * @return: OK if success, other values if failure
 */
int DvppCommon::CheckResizeParams(const DvppDataInfo &input, const DvppDataInfo &output) {
    if (output.format != PIXEL_FORMAT_YUV_SEMIPLANAR_420 && output.format != PIXEL_FORMAT_YVU_SEMIPLANAR_420) {
        std::cout << "Output format[" << output.format << "]is not supported, just support NV12 or NV21." << std::endl;
        return INVALID_PARAM;
    }
    float heightScale = static_cast<float>(output.height) / input.height;
    if (heightScale < MIN_RESIZE_SCALE || heightScale > MAX_RESIZE_SCALE) {
        std::cout << "Resize scale should be in range [1/16, 32], which is " << heightScale << "." << std::endl;
        return INVALID_PARAM;
    }
    float widthScale = static_cast<float>(output.width) / input.width;
    if (widthScale < MIN_RESIZE_SCALE || widthScale > MAX_RESIZE_SCALE) {
        std::cout << "Resize scale should be in range [1/16, 32], which is " << widthScale << "." << std::endl;
        return INVALID_PARAM;
    }
    return OK;
}

/*
 * @description: Scale the input image to the size specified by the output image and
 *               saves the result to the output image (non-proportionate scaling)
 * @param: inputDesc specifies the description information of the input image
 * @param: outputDesc specifies the description information of the output image
 * @param: withSynchronize specifies whether to execute synchronously
 * @return: OK if success, other values if failure
 */
int DvppCommon::ResizeProcess(std::shared_ptr<acldvppPicDesc>inputDesc,
                              std::shared_ptr<acldvppPicDesc>outputDesc,
                              bool withSynchronize) {
    acldvppResizeConfig *resizeConfig = acldvppCreateResizeConfig();
    if (resizeConfig == nullptr) {
        std::cout << "Failed to create dvpp resize config." << std::endl;
        return INVALID_POINTER;
    }

    resizeConfig_.reset(resizeConfig, g_resizeConfigDeleter);
    int ret = acldvppVpcResizeAsync(dvppChannelDesc_, inputDesc.get(), outputDesc.get(),
                                    resizeConfig_.get(), dvppStream_);
    if (ret != OK) {
        std::cout << "Failed to resize asynchronously, ret = " << ret << "." << std::endl;
        return ret;
    }

    if (withSynchronize) {
        ret = aclrtSynchronizeStream(dvppStream_);
        if (ret != OK) {
            std::cout << "Failed to synchronize stream, ret = " << ret << "." << std::endl;
            return ret;
        }
    }
    return OK;
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
 * @return: OK if success, other values if failure
 * @attention: If the width and height of the crop area are different from those of the
 *             paste area, the image is scaled again
 */
int DvppCommon::ResizeWithPadding(std::shared_ptr<acldvppPicDesc> inputDesc,
                                  std::shared_ptr<acldvppPicDesc> outputDesc,
                                  const CropRoiConfig &cropRoi, const CropRoiConfig &pasteRoi, bool withSynchronize) {
    acldvppRoiConfig *cropRoiCfg = acldvppCreateRoiConfig(cropRoi.left, cropRoi.right, cropRoi.up, cropRoi.down);
    if (cropRoiCfg == nullptr) {
        std::cout << "Failed to create dvpp roi config for corp area." << std::endl;
        return INVALID_POINTER;
    }
    cropAreaConfig_.reset(cropRoiCfg, g_roiConfigDeleter);

    acldvppRoiConfig *pastRoiCfg = acldvppCreateRoiConfig(pasteRoi.left, pasteRoi.right, pasteRoi.up, pasteRoi.down);
    if (pastRoiCfg == nullptr) {
        std::cout << "Failed to create dvpp roi config for paster area." << std::endl;
        return INVALID_POINTER;
    }
    pasteAreaConfig_.reset(pastRoiCfg, g_roiConfigDeleter);

    int ret = acldvppVpcCropAndPasteAsync(dvppChannelDesc_, inputDesc.get(), outputDesc.get(), cropAreaConfig_.get(),
                                          pasteAreaConfig_.get(), dvppStream_);
    if (ret != OK) {
        // release resource.
        std::cout << "Failed to crop and paste asynchronously, ret = " << ret << "." << std::endl;
        return ret;
    }
    if (withSynchronize) {
        ret = aclrtSynchronizeStream(dvppStream_);
        if (ret != OK) {
            std::cout << "Failed tp synchronize stream, ret = " << ret << "." << std::endl;
            return ret;
        }
    }
    return OK;
}

/*
 * @description: Get crop area
 * @param: input specifies the input image information
 * @param: output specifies the output image information
 * @param: processType specifies whether to perform proportional scaling
 * @param: cropRoi is used to save the info of the crop roi area
 * @return: OK if success, other values if failure
 */
void DvppCommon::GetCropRoi(std::shared_ptr<DvppDataInfo> input, std::shared_ptr<DvppDataInfo> output,
                            VpcProcessType processType, CropRoiConfig *cropRoi) {
    // When processType is not VPC_PT_FILL, crop area is the whole input image
    if (processType != VPC_PT_FILL) {
        cropRoi->right = CONVERT_TO_ODD(input->width - ODD_NUM_1);
        cropRoi->down = CONVERT_TO_ODD(input->height - ODD_NUM_1);
        return;
    }

    bool widthRatioSmaller = true;
    // The scaling ratio is based on the smaller ratio to ensure the smallest edge to fill the targe edge
    float resizeRatio = static_cast<float>(input->width) / output->width;
    if (resizeRatio > (static_cast<float>(input->height) / output->height)) {
        resizeRatio = static_cast<float>(input->height) / output->height;
        widthRatioSmaller = false;
    }

    const int halfValue = 2;
    // The left and up must be even, right and down must be odd which is required by acl
    if (widthRatioSmaller) {
        cropRoi->left = 0;
        cropRoi->right = CONVERT_TO_ODD(input->width - ODD_NUM_1);
        cropRoi->up = CONVERT_TO_EVEN(static_cast<uint32_t>((input->height - output->height * resizeRatio) /
                                                            halfValue));
        cropRoi->down = CONVERT_TO_ODD(input->height - cropRoi->up - ODD_NUM_1);
        return;
    }

    cropRoi->up = 0;
    cropRoi->down = CONVERT_TO_ODD(input->height - ODD_NUM_1);
    cropRoi->left = CONVERT_TO_EVEN(static_cast<uint32_t>((input->width - output->width * resizeRatio) / halfValue));
    cropRoi->right = CONVERT_TO_ODD(input->width - cropRoi->left - ODD_NUM_1);
    return;
}

/*
 * @description: Get paste area
 * @param: input specifies the input image information
 * @param: output specifies the output image information
 * @param: processType specifies whether to perform proportional scaling
 * @param: pasteRio is used to save the info of the paste area
 * @return: OK if success, other values if failure
 */
void DvppCommon::GetPasteRoi(std::shared_ptr<DvppDataInfo> input, std::shared_ptr<DvppDataInfo> output,
                             VpcProcessType processType, CropRoiConfig *pasteRoi) {
    if (processType == VPC_PT_FILL) {
        pasteRoi->right = CONVERT_TO_ODD(output->width - ODD_NUM_1);
        pasteRoi->down = CONVERT_TO_ODD(output->height - ODD_NUM_1);
        return;
    }

    bool widthRatioLarger = true;
    // The scaling ratio is based on the larger ratio to ensure the largest edge to fill the targe edge
    float resizeRatio = static_cast<float>(input->width) / output->width;
    if (resizeRatio < (static_cast<float>(input->height) / output->height)) {
        resizeRatio = static_cast<float>(input->height) / output->height;
        widthRatioLarger = false;
    }

    // Left and up is 0 when the roi paste on the upper left corner
    if (processType == VPC_PT_PADDING) {
        pasteRoi->right = (input->width / resizeRatio) - ODD_NUM_1;
        pasteRoi->down = (input->height / resizeRatio) - ODD_NUM_1;
        pasteRoi->right = CONVERT_TO_ODD(pasteRoi->right);
        pasteRoi->down = CONVERT_TO_ODD(pasteRoi->down);
        return;
    }

    const int halfValue = 2;
    // Left and up is 0 when the roi paste on the middler location
    if (widthRatioLarger) {
        pasteRoi->left = 0;
        pasteRoi->right = output->width - ODD_NUM_1;
        pasteRoi->up = (output->height - (input->height / resizeRatio)) / halfValue;
        pasteRoi->down = output->height - pasteRoi->up - ODD_NUM_1;
    } else {
        pasteRoi->up = 0;
        pasteRoi->down = output->height - ODD_NUM_1;
        pasteRoi->left = (output->width - (input->width / resizeRatio)) / halfValue;
        pasteRoi->right = output->width - pasteRoi->left - ODD_NUM_1;
    }

    // The left must be even and align to 16, up must be even, right and down must be odd which is required by acl
    pasteRoi->left = DVPP_ALIGN_UP(CONVERT_TO_EVEN(pasteRoi->left), VPC_WIDTH_ALIGN);
    pasteRoi->right = CONVERT_TO_ODD(pasteRoi->right);
    pasteRoi->up = CONVERT_TO_EVEN(pasteRoi->up);
    pasteRoi->down = CONVERT_TO_ODD(pasteRoi->down);
    return;
}

/*
 * @description: Resize the image specified by input and save the result to member variable resizedImage_
 * @param: input specifies the input image information
 * @param: output specifies the output image information
 * @param: withSynchronize specifies whether to execute synchronously
 * @param: processType specifies whether to perform proportional scaling, default is non-proportional resize
 * @return: OK if success, other values if failure
 * @attention: This function can be called only when the DvppCommon object is initialized with Init
 */
int DvppCommon::CombineResizeProcess(std::shared_ptr<DvppDataInfo> input, const DvppDataInfo &output,
                                     bool withSynchronize, VpcProcessType processType) {
    int ret = CheckResizeParams(*input, output);
    if (ret != OK) {
        return ret;
    }
    // Get widthStride and heightStride for input and output image according to the format
    ret = GetVpcInputStrideSize(input->widthStride, input->heightStride, input->format,
                                &(input->widthStride), &(input->heightStride));
    if (ret != OK) {
        return ret;
    }

    resizedImage_ = std::make_shared<DvppDataInfo>();
    resizedImage_->width = output.width;
    resizedImage_->height = output.height;
    resizedImage_->format = output.format;
    ret = GetVpcOutputStrideSize(output.width, output.height, output.format, &(resizedImage_->widthStride),
                                 &(resizedImage_->heightStride));
    if (ret != OK) {
        return ret;
    }
    // Get output buffer size for resize output
    ret = GetVpcDataSize(output.width, output.height, output.format, &(resizedImage_->dataSize));
    if (ret != OK) {
        return ret;
    }
    // Malloc buffer for output of resize module
    // Need to pay attention to release of the buffer
    ret = acldvppMalloc(reinterpret_cast<void **>(&(resizedImage_->data)), resizedImage_->dataSize);
    if (ret != OK) {
        std::cout << "Failed to malloc " << resizedImage_->dataSize << " bytes on dvpp for resize" << std::endl;
        return ret;
    }

    aclrtMemset(resizedImage_->data, resizedImage_->dataSize, YUV_GREYER_VALUE, resizedImage_->dataSize);
    resizedImage_->frameId = input->frameId;
    ret = VpcResize(input, resizedImage_, withSynchronize, processType);
    if (ret != OK) {
        // Release the output buffer when resize failed, otherwise release it after use
        RELEASE_DVPP_DATA(resizedImage_->data);
    }
    return ret;
}

/*
 * @description: Set the description of the output image and decode
 * @param: input specifies the input image information
 * @param: output specifies the output image information
 * @param: withSynchronize specifies whether to execute synchronously
 * @return: OK if success, other values if failure
 * @attention: This function can be called only when the DvppCommon object is initialized with Init
 */
int DvppCommon::JpegDecode(std::shared_ptr<DvppDataInfo> input,
                           std::shared_ptr<DvppDataInfo> output,
                           bool withSynchronize) {
    acldvppPicDesc *outputDesc = acldvppCreatePicDesc();
    decodeOutputDesc_.reset(outputDesc, g_picDescDeleter);

    int ret = SetDvppPicDescData(output, decodeOutputDesc_);
    if (ret != OK) {
        return ret;
    }

    ret = acldvppJpegDecodeAsync(dvppChannelDesc_, input->data, input->dataSize, decodeOutputDesc_.get(), dvppStream_);
    if (ret != OK) {
        std::cout << "Failed to decode jpeg, ret = " << ret << "." << std::endl;
        return ret;
    }
    if (withSynchronize) {
        ret = aclrtSynchronizeStream(dvppStream_);
        if (ret != OK) {
            std::cout << "Failed to synchronize stream, ret = " << ret << "." << std::endl;
            return DECODE_FAIL;
        }
    }
    return OK;
}

/*
 * @description: Get the aligned width and height of the image after decoding
 * @param: width specifies the width before alignment
 * @param: height specifies the height before alignment
 * @param: widthStride is used to save the width after alignment
 * @param: heightStride is used to save the height after alignment
 * @return: OK if success, other values if failure
 */
void DvppCommon::GetJpegDecodeStrideSize(uint32_t width, uint32_t height,
                                         uint32_t *widthStride, uint32_t *heightStride) {
    *widthStride = DVPP_ALIGN_UP(width, JPEGD_STRIDE_WIDTH);
    *heightStride = DVPP_ALIGN_UP(height, JPEGD_STRIDE_HEIGHT);
}

/*
 * @description: Get picture width and height and number of channels from image data
 * @param: data specifies the memory to store the image data
 * @param: dataSize specifies the size of the image data
 * @param: width is used to save the image width
 * @param: height is used to save the image height
 * @param: components is used to save the number of channels
 * @return: OK if success, other values if failure
 */
int DvppCommon::GetJpegImageInfo(const void *data, uint32_t dataSize, uint32_t *width, uint32_t *height,
                                 int32_t *components) {
    uint32_t widthTmp;
    uint32_t heightTmp;
    int32_t componentsTmp;
    int ret = acldvppJpegGetImageInfo(data, dataSize, &widthTmp, &heightTmp, &componentsTmp);
    if (ret != OK) {
        std::cout << "Failed to get image info of jpeg, ret = " << ret << "." << std::endl;
        return ret;
    }
    if (widthTmp > MAX_JPEGD_WIDTH || widthTmp < MIN_JPEGD_WIDTH) {
        std::cout << "Input width is invalid, not in [" << MIN_JPEGD_WIDTH << ", "
            << MAX_JPEGD_WIDTH << "]." << std::endl;
        return INVALID_PARAM;
    }
    if (heightTmp > MAX_JPEGD_HEIGHT || heightTmp < MIN_JPEGD_HEIGHT) {
        std::cout << "Input height is invalid, not in [" << MIN_JPEGD_HEIGHT << ", "
            << MAX_JPEGD_HEIGHT << "]." << std::endl;
        return INVALID_PARAM;
    }
    *width = widthTmp;
    *height = heightTmp;
    *components = componentsTmp;
    return OK;
}

/*
 * @description: Get the size of the buffer for storing decoded images based on the image data, size, and format
 * @param: data specifies the memory to store the image data
 * @param: dataSize specifies the size of the image data
 * @param: format specifies the image format
 * @param: decSize is used to store the result size
 * @return: OK if success, other values if failure
 */
int DvppCommon::GetJpegDecodeDataSize(const void *data, uint32_t dataSize, acldvppPixelFormat format,
                                      uint32_t *decSize) {
    uint32_t outputSize;
    int ret = acldvppJpegPredictDecSize(data, dataSize, format, &outputSize);
    if (ret != OK) {
        std::cout << "Failed to predict decode size of jpeg image, ret = " << ret << "." << std::endl;
        return ret;
    }
    *decSize = outputSize;
    return OK;
}

/*
 * @description: Decode the image specified by imageInfo and save the result to member variable decodedImage_
 * @param: imageInfo specifies image information
 * @param: format specifies the image format
 * @param: withSynchronize specifies whether to execute synchronously
 * @return: OK if success, other values if failure
 * @attention: This function can be called only when the DvppCommon object is initialized with Init
 */
int DvppCommon::CombineJpegdProcess(const RawData& imageInfo, acldvppPixelFormat format, bool withSynchronize) {
    int32_t components;
    inputImage_ = std::make_shared<DvppDataInfo>();
    inputImage_->format = format;
    int ret = GetJpegImageInfo(imageInfo.data.get(), imageInfo.lenOfByte, &(inputImage_->width), &(inputImage_->height),
                               &components);
    if (ret != OK) {
        std::cout << "Failed to get input image info, ret = " << ret << "." << std::endl;
        return ret;
    }

    // Get the buffer size of decode output according to the input data and output format
    uint32_t outBuffSize;
    ret = GetJpegDecodeDataSize(imageInfo.data.get(), imageInfo.lenOfByte, format, &outBuffSize);
    if (ret != OK) {
        std::cout << "Failed to get size of decode output buffer, ret = " << ret << "." << std::endl;
        return ret;
    }

    // In TransferImageH2D function, device buffer will be allocated to store the input image
    // Need to pay attention to release of the buffer
    ret = TransferImageH2D(imageInfo, inputImage_);
    if (ret != OK) {
        return ret;
    }

    decodedImage_ = std::make_shared<DvppDataInfo>();
    decodedImage_->format = format;
    decodedImage_->width = inputImage_->width;
    decodedImage_->height = inputImage_->height;
    GetJpegDecodeStrideSize(inputImage_->width, inputImage_->height, &(decodedImage_->widthStride),
                            &(decodedImage_->heightStride));
    decodedImage_->dataSize = outBuffSize;
    // Need to pay attention to release of the buffer
    ret = acldvppMalloc(reinterpret_cast<void **>(&decodedImage_->data), decodedImage_->dataSize);
    if (ret != OK) {
        std::cout << "Failed to malloc memory on dvpp, ret = " << ret << "." << std::endl;
        RELEASE_DVPP_DATA(inputImage_->data);
        return ret;
    }

    ret = JpegDecode(inputImage_, decodedImage_, withSynchronize);
    if (ret != OK) {
        RELEASE_DVPP_DATA(inputImage_->data);
        inputImage_->data = nullptr;
        RELEASE_DVPP_DATA(decodedImage_->data);
        decodedImage_->data = nullptr;
        return ret;
    }

    return OK;
}

/*
 * @description: Transfer data from host to device
 * @param: imageInfo specifies the image data on the host
 * @param: jpegInput is used to save the buffer and its size which is allocate on the device
 * @return: OK if success, other values if failure
 */
int DvppCommon::TransferImageH2D(const RawData& imageInfo, const std::shared_ptr<DvppDataInfo>& jpegInput) {
    if (imageInfo.lenOfByte == 0) {
        std::cout << "The input buffer size on host should not be empty." << std::endl;
        return INVALID_PARAM;
    }

    uint8_t* inDevBuff = nullptr;
    int ret = acldvppMalloc(reinterpret_cast<void **>(&inDevBuff), imageInfo.lenOfByte);
    if (ret != OK) {
        std::cout << "Failed to malloc " << imageInfo.lenOfByte << " bytes on dvpp, ret = " << ret << "." << std::endl;
        return ret;
    }

    // Copy the image data from host to device
    ret = aclrtMemcpyAsync(inDevBuff, imageInfo.lenOfByte, imageInfo.data.get(), imageInfo.lenOfByte,
                           ACL_MEMCPY_HOST_TO_DEVICE, dvppStream_);
    if (ret != OK) {
        std::cout << "Failed to copy " << imageInfo.lenOfByte << " bytes from host to device" << std::endl;
        RELEASE_DVPP_DATA(inDevBuff);
        return ret;
    }
    // Attention: We must call the aclrtSynchronizeStream to ensure the task of memory replication has been completed
    // after calling aclrtMemcpyAsync
    ret = aclrtSynchronizeStream(dvppStream_);
    if (ret != OK) {
        std::cout << "Failed to synchronize stream, ret = " << ret << "." << std::endl;
        RELEASE_DVPP_DATA(inDevBuff);
        return ret;
    }
    jpegInput->data = inDevBuff;
    jpegInput->dataSize = imageInfo.lenOfByte;
    return OK;
}

std::shared_ptr<DvppDataInfo> DvppCommon::GetInputImage() {
    return inputImage_;
}

std::shared_ptr<DvppDataInfo> DvppCommon::GetDecodedImage() {
    return decodedImage_;
}

std::shared_ptr<DvppDataInfo> DvppCommon::GetResizedImage() {
    return resizedImage_;
}

DvppCommon::~DvppCommon() {}
