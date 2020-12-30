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

#ifndef DVPP_COMMON_H
#define DVPP_COMMON_H
#include <memory>

#include "CommonDataType.h"
#include "acl/ops/acl_dvpp.h"

const int MODULUS_NUM_2 = 2;
const uint32_t ODD_NUM_1 = 1;

const uint32_t JPEGD_STRIDE_WIDTH = 128;  // Jpegd module output width need to align up to 128
const uint32_t JPEGD_STRIDE_HEIGHT = 16;  // Jpegd module output height need to align up to 16
const uint32_t VPC_STRIDE_WIDTH = 16;  // Vpc module output width need to align up to 16
const uint32_t VPC_STRIDE_HEIGHT = 2;  // Vpc module output height need to align up to 2
const uint32_t YUV422_WIDTH_NU = 2;  // Width of YUV422, WidthStride = Width * 2
const uint32_t YUV444_RGB_WIDTH_NU = 3;  // Width of YUV444 and RGB888, WidthStride = Width * 3
const uint32_t XRGB_WIDTH_NU = 4;  // Width of XRGB8888, WidthStride = Width * 4
const uint32_t JPEG_OFFSET = 8;  // Offset of input file for jpegd module
const uint32_t MAX_JPEGD_WIDTH = 8192;  // Max width of jpegd module
const uint32_t MAX_JPEGD_HEIGHT = 8192;  // Max height of jpegd module
const uint32_t MIN_JPEGD_WIDTH = 32;  // Min width of jpegd module
const uint32_t MIN_JPEGD_HEIGHT = 32;  // Min height of jpegd module
const uint32_t MAX_RESIZE_WIDTH = 4096;  // Max width stride of resize module
const uint32_t MAX_RESIZE_HEIGHT = 4096;  // Max height stride of resize module
const uint32_t MIN_RESIZE_WIDTH = 32;  // Min width stride of resize module
const uint32_t MIN_RESIZE_HEIGHT = 6;  // Min height stride of resize module
const float MIN_RESIZE_SCALE = 0.03125;   // Min resize scale of resize module
const float MAX_RESIZE_SCALE = 16.0;  // Min resize scale of resize module
const uint32_t MAX_VPC_WIDTH = 4096;  // Max width of picture to VPC(resize/crop)
const uint32_t MAX_VPC_HEIGHT = 4096;  // Max height of picture to VPC(resize/crop)
const uint32_t MIN_VPC_WIDTH  = 32;    // Min width of picture to VPC(resize/crop)
const uint32_t MIN_VPC_HEIGHT = 6;    // Min height of picture to VPC(resize/crop)
const uint32_t MIN_CROP_WIDTH = 10;   // Min width of crop area
const uint32_t MIN_CROP_HEIGHT = 6;    // Min height of crop area
const uint8_t YUV_GREYER_VALUE = 128;   // Filling value of the resized YUV image

#define CONVERT_TO_ODD(NUM) (((NUM) % MODULUS_NUM_2 != 0) ? (NUM) : ((NUM) - 1))
#define CONVERT_TO_EVEN(NUM) (((NUM) % MODULUS_NUM_2 == 0) ? (NUM) : ((NUM) - 1))
#define CHECK_ODD(num) ((num) % MODULUS_NUM_2 != 0)
#define CHECK_EVEN(num) ((num) % MODULUS_NUM_2 == 0)
#define RELEASE_DVPP_DATA(dvppDataPtr) do { \
    int retMacro; \
    if (dvppDataPtr != nullptr) { \
        retMacro = acldvppFree(dvppDataPtr); \
        if (retMacro != OK) { \
            std::cout << "Failed to free memory on dvpp, ret = " << retMacro << "." << std::endl; \
        } \
        dvppDataPtr = nullptr; \
    } \
} while (0);

class DvppCommon {
 public:
    explicit DvppCommon(aclrtStream dvppStream);
    ~DvppCommon();
    int Init(void);
    int DeInit(void);

    static int GetVpcDataSize(uint32_t widthVpc, uint32_t heightVpc, acldvppPixelFormat format,
                                    uint32_t *vpcSize);

    static int GetVpcInputStrideSize(uint32_t width, uint32_t height, acldvppPixelFormat format,
                                           uint32_t *widthStride, uint32_t *heightStride);

    static int GetVpcOutputStrideSize(uint32_t width, uint32_t height, acldvppPixelFormat format,
                                            uint32_t *widthStride, uint32_t *heightStride);

    static void GetJpegDecodeStrideSize(uint32_t width, uint32_t height, uint32_t *widthStride, uint32_t *heightStride);
    static int GetJpegImageInfo(const void *data, uint32_t dataSize, uint32_t *width, uint32_t *height,
                                      int32_t *components);

    static int GetJpegDecodeDataSize(const void *data, uint32_t dataSize, acldvppPixelFormat format,
                                           uint32_t *decSize);

    int VpcResize(std::shared_ptr<DvppDataInfo> input, std::shared_ptr<DvppDataInfo> output, bool withSynchronize,
                        VpcProcessType processType = VPC_PT_DEFAULT);

    int JpegDecode(std::shared_ptr<DvppDataInfo> input, std::shared_ptr<DvppDataInfo> output, bool withSynchronize);

    int CombineResizeProcess(std::shared_ptr<DvppDataInfo> input, const DvppDataInfo &output, bool withSynchronize,
                                   VpcProcessType processType = VPC_PT_DEFAULT);

    int CombineJpegdProcess(const RawData& imageInfo, acldvppPixelFormat format, bool withSynchronize);

    std::shared_ptr<DvppDataInfo> GetInputImage();
    std::shared_ptr<DvppDataInfo> GetDecodedImage();
    std::shared_ptr<DvppDataInfo> GetResizedImage();

    void ReleaseDvppBuffer();

 private:
    int SetDvppPicDescData(std::shared_ptr<DvppDataInfo> dataInfo, std::shared_ptr<acldvppPicDesc>picDesc);
    int ResizeProcess(std::shared_ptr<acldvppPicDesc> inputDesc,
                      std::shared_ptr<acldvppPicDesc> outputDesc, bool withSynchronize);

    int ResizeWithPadding(std::shared_ptr<acldvppPicDesc> inputDesc, std::shared_ptr<acldvppPicDesc> outputDesc,
                          const CropRoiConfig &cropRoi, const CropRoiConfig &pasteRoi, bool withSynchronize);

    void GetCropRoi(std::shared_ptr<DvppDataInfo> input, std::shared_ptr<DvppDataInfo> output,
                    VpcProcessType processType, CropRoiConfig *cropRoi);

    void GetPasteRoi(std::shared_ptr<DvppDataInfo> input, std::shared_ptr<DvppDataInfo> output,
                     VpcProcessType processType, CropRoiConfig *pasteRoi);

    int CheckResizeParams(const DvppDataInfo &input, const DvppDataInfo &output);
    int TransferImageH2D(const RawData& imageInfo, const std::shared_ptr<DvppDataInfo>& jpegInput);
    int CreateStreamDesc(std::shared_ptr<DvppDataInfo> data);
    int DestroyResource();

    std::shared_ptr<acldvppRoiConfig> cropAreaConfig_ = nullptr;
    std::shared_ptr<acldvppRoiConfig> pasteAreaConfig_ = nullptr;

    std::shared_ptr<acldvppPicDesc> resizeInputDesc_ = nullptr;
    std::shared_ptr<acldvppPicDesc> resizeOutputDesc_ = nullptr;
    std::shared_ptr<acldvppPicDesc> decodeOutputDesc_ = nullptr;
    std::shared_ptr<acldvppResizeConfig> resizeConfig_ = nullptr;

    acldvppChannelDesc *dvppChannelDesc_ = nullptr;
    aclrtStream dvppStream_ = nullptr;
    std::shared_ptr<DvppDataInfo> inputImage_ = nullptr;
    std::shared_ptr<DvppDataInfo> decodedImage_ = nullptr;
    std::shared_ptr<DvppDataInfo> resizedImage_ = nullptr;
};
#endif
