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

#ifndef DVPP_COMMON_H
#define DVPP_COMMON_H

#include "CommonDataType.h"
#include "ErrorCode.h"
#include "acl/ops/acl_dvpp.h"

const int MODULUS_NUM_2 = 2;
const uint32_t ODD_NUM_1 = 1;

struct Rect {
  /* left location of the rectangle */
  uint32_t x;
  /* top location of the rectangle */
  uint32_t y;
  /* with of the rectangle */
  uint32_t width;
  /* height of the rectangle */
  uint32_t height;
};

struct DvppBaseData {
  uint32_t dataSize;  // Size of data in byte
  uint8_t *data;
};

struct VdecConfig {
  int inputWidth = 0;
  int inputHeight = 0;
  acldvppStreamFormat inFormat = H264_MAIN_LEVEL;                  // stream format renference acldvppStreamFormat
  acldvppPixelFormat outFormat = PIXEL_FORMAT_YUV_SEMIPLANAR_420;  // output format renference acldvppPixelFormat
  uint32_t channelId = 0;                                          // user define channelId: 0-15
  uint32_t deviceId = 0;
  pthread_t threadId = 0;          // thread for callback
  aclvdecCallback callback = {0};  // user define how to process vdec out data
  bool runflag = true;
};

struct DeviceStreamData {
  std::vector<ObjectDetectInfo> detectResult;
  uint32_t framId;
  uint32_t channelId;
};

const uint32_t JPEGD_STRIDE_WIDTH = 128;  // Jpegd module output width need to align up to 128
const uint32_t JPEGD_STRIDE_HEIGHT = 16;  // Jpegd module output height need to align up to 16
const uint32_t PNGD_STRIDE_WIDTH = 128;   // Pngd module output width need to align up to 128
const uint32_t PNGD_STRIDE_HEIGHT = 16;   // Pngd module output height need to align up to 16
const uint32_t JPEGE_STRIDE_WIDTH = 16;   // Jpege module input width need to align up to 16
const uint32_t JPEGE_STRIDE_HEIGHT = 1;   // Jpege module input height remains unchanged
const uint32_t VPC_STRIDE_WIDTH = 16;     // Vpc module output width need to align up to 16
const uint32_t VPC_STRIDE_HEIGHT = 2;     // Vpc module output height need to align up to 2
const uint32_t VDEC_STRIDE_WIDTH = 16;    // Vdec module output width need to align up to 16
const uint32_t VDEC_STRIDE_HEIGHT = 2;    // Vdec module output width need to align up to 2
const uint32_t YUV_BYTES_NU = 3;          // Numerator of yuv image, H x W x 3 / 2
const uint32_t YUV_BYTES_DE = 2;          // Denominator of yuv image, H x W x 3 / 2
const uint32_t YUV422_WIDTH_NU = 2;       // Width of YUV422, WidthStride = Width * 2
const uint32_t YUV444_RGB_WIDTH_NU = 3;   // Width of YUV444 and RGB888, WidthStride = Width * 3
const uint32_t XRGB_WIDTH_NU = 4;         // Width of XRGB8888, WidthStride = Width * 4
const uint32_t JPEG_OFFSET = 8;           // Offset of input file for jpegd module
const uint32_t MAX_JPEGD_WIDTH = 8192;    // Max width of jpegd module
const uint32_t MAX_JPEGD_HEIGHT = 8192;   // Max height of jpegd module
const uint32_t MIN_JPEGD_WIDTH = 32;      // Min width of jpegd module
const uint32_t MIN_JPEGD_HEIGHT = 32;     // Min height of jpegd module
const uint32_t MAX_PNGD_WIDTH = 4096;     // Max width of pngd module
const uint32_t MAX_PNGD_HEIGHT = 4096;    // Max height of pngd module
const uint32_t MIN_PNGD_WIDTH = 32;       // Min width of pngd module
const uint32_t MIN_PNGD_HEIGHT = 32;      // Min height of pngd module
const uint32_t MAX_JPEGE_WIDTH = 8192;    // Max width of jpege module
const uint32_t MAX_JPEGE_HEIGHT = 8192;   // Max height of jpege module
const uint32_t MIN_JPEGE_WIDTH = 32;      // Min width of jpege module
const uint32_t MIN_JPEGE_HEIGHT = 32;     // Min height of jpege module
const uint32_t MAX_RESIZE_WIDTH = 4096;   // Max width stride of resize module
const uint32_t MAX_RESIZE_HEIGHT = 4096;  // Max height stride of resize module
const uint32_t MIN_RESIZE_WIDTH = 32;     // Min width stride of resize module
const uint32_t MIN_RESIZE_HEIGHT = 6;     // Min height stride of resize module
const float MIN_RESIZE_SCALE = 0.03125;   // Min resize scale of resize module
const float MAX_RESIZE_SCALE = 16.0;      // Min resize scale of resize module
const uint32_t MAX_VPC_WIDTH = 4096;      // Max width of picture to VPC(resize/crop)
const uint32_t MAX_VPC_HEIGHT = 4096;     // Max height of picture to VPC(resize/crop)
const uint32_t MIN_VPC_WIDTH = 32;        // Min width of picture to VPC(resize/crop)
const uint32_t MIN_VPC_HEIGHT = 6;        // Min height of picture to VPC(resize/crop)
const uint32_t MIN_CROP_WIDTH = 10;       // Min width of crop area
const uint32_t MIN_CROP_HEIGHT = 6;       // Min height of crop area
const uint8_t YUV_GREYER_VALUE = 128;     // Filling value of the resized YUV image

#define CONVERT_TO_ODD(NUM) (((NUM) % MODULUS_NUM_2 != 0) ? (NUM) : ((NUM)-1))   // Convert the input to odd num
#define CONVERT_TO_EVEN(NUM) (((NUM) % MODULUS_NUM_2 == 0) ? (NUM) : ((NUM)-1))  // Convert the input to even num
#define CHECK_ODD(num) ((num) % MODULUS_NUM_2 != 0)
#define CHECK_EVEN(num) ((num) % MODULUS_NUM_2 == 0)
#define RELEASE_DVPP_DATA(dvppDataPtr)                                               \
  do {                                                                               \
    APP_ERROR retMacro;                                                              \
    if (dvppDataPtr != nullptr) {                                                    \
      retMacro = acldvppFree(dvppDataPtr);                                           \
      if (retMacro != APP_ERR_OK) {                                                  \
        MS_LOG(ERROR) << "Failed to free memory on dvpp, ret = " << retMacro << "."; \
      }                                                                              \
      dvppDataPtr = nullptr;                                                         \
    }                                                                                \
  } while (0);

class DvppCommon {
 public:
  explicit DvppCommon(aclrtContext dvppContext, aclrtStream dvppStream);
  explicit DvppCommon(const VdecConfig &vdecConfig);  // Need by vdec
  ~DvppCommon();
  APP_ERROR Init(void);
  APP_ERROR InitVdec();  // Needed by vdec
  APP_ERROR DeInit(void);

  static APP_ERROR GetVpcDataSize(uint32_t widthVpc, uint32_t heightVpc, acldvppPixelFormat format, uint32_t &vpcSize);
  static APP_ERROR GetVpcInputStrideSize(uint32_t width, uint32_t height, acldvppPixelFormat format,
                                         uint32_t &widthStride, uint32_t &heightStride);
  static APP_ERROR GetVpcOutputStrideSize(uint32_t width, uint32_t height, acldvppPixelFormat format,
                                          uint32_t &widthStride, uint32_t &heightStride);
  static void GetJpegDecodeStrideSize(uint32_t width, uint32_t height, uint32_t &widthStride, uint32_t &heightStride);
  static void GetPngDecodeStrideSize(uint32_t width, uint32_t height, uint32_t &widthStride, uint32_t &heightStride,
                                     acldvppPixelFormat format);
  static APP_ERROR GetJpegImageInfo(const void *data, uint32_t dataSize, uint32_t &width, uint32_t &height,
                                    int32_t &components);
  static APP_ERROR GetPngImageInfo(const void *data, uint32_t dataSize, uint32_t &width, uint32_t &height,
                                   int32_t &components);
  static APP_ERROR GetJpegDecodeDataSize(const void *data, uint32_t dataSize, acldvppPixelFormat format,
                                         uint32_t &decSize);
  static APP_ERROR GetPngDecodeDataSize(const void *data, uint32_t dataSize, acldvppPixelFormat format,
                                        uint32_t &decSize);
  static APP_ERROR GetJpegEncodeStrideSize(std::shared_ptr<DvppDataInfo> &input);
  static APP_ERROR SetEncodeLevel(uint32_t level, acldvppJpegeConfig &jpegeConfig);
  static APP_ERROR GetVideoDecodeStrideSize(uint32_t width, uint32_t height, acldvppPixelFormat format,
                                            uint32_t &widthStride, uint32_t &heightStride);
  static APP_ERROR GetVideoDecodeDataSize(uint32_t width, uint32_t height, acldvppPixelFormat format,
                                          uint32_t &vdecSize);

  // The following interfaces can be called only when the DvppCommon object is initialized with Init
  APP_ERROR VpcResize(DvppDataInfo &input, DvppDataInfo &output, bool withSynchronize,
                      VpcProcessType processType = VPC_PT_DEFAULT);
  APP_ERROR VpcCrop(const DvppCropInputInfo &input, const DvppDataInfo &output, bool withSynchronize);
  APP_ERROR JpegDecode(DvppDataInfo &input, DvppDataInfo &output, bool withSynchronize);

  APP_ERROR PngDecode(DvppDataInfo &input, DvppDataInfo &output, bool withSynchronize);

  APP_ERROR JpegEncode(DvppDataInfo &input, DvppDataInfo &output, acldvppJpegeConfig *jpegeConfig,
                       bool withSynchronize);

  APP_ERROR GetJpegEncodeDataSize(DvppDataInfo &input, acldvppJpegeConfig *jpegeConfig, uint32_t &encSize);

  // These functions started with "Combine" encapsulate the DVPP process together, malloc DVPP memory,
  // transfer pictures from host to device, and then execute the DVPP operation.
  // The caller needs to pay attention to the release of the memory alloced in these functions.
  // You can call the ReleaseDvppBuffer function to release memory after use completely.
  APP_ERROR CombineResizeProcess(DvppDataInfo &input, DvppDataInfo &output, bool withSynchronize,
                                 VpcProcessType processType = VPC_PT_DEFAULT);
  APP_ERROR CombineCropProcess(DvppCropInputInfo &input, DvppDataInfo &output, bool withSynchronize);
  APP_ERROR CombineJpegdProcess(const RawData &imageInfo, acldvppPixelFormat format, bool withSynchronize);

  APP_ERROR SinkCombineJpegdProcess(std::shared_ptr<DvppDataInfo> &input, std::shared_ptr<DvppDataInfo> &output,
                                    bool withSynchronize);
  APP_ERROR SinkCombinePngdProcess(std::shared_ptr<DvppDataInfo> &input, std::shared_ptr<DvppDataInfo> &output,
                                   bool withSynchronize);

  APP_ERROR CombineJpegeProcess(const RawData &imageInfo, uint32_t width, uint32_t height, acldvppPixelFormat format,
                                bool withSynchronize);
  // New feature for PNG format image decode
  APP_ERROR CombinePngdProcess(const RawData &imageInfo, acldvppPixelFormat format, bool withSynchronize);

  // The following interface can be called only when the DvppCommon object is initialized with InitVdec
  APP_ERROR CombineVdecProcess(std::shared_ptr<DvppDataInfo> data, void *userData);

  // Get the private member variables which are assigned in the interfaces which are started with "Combine"
  std::shared_ptr<DvppDataInfo> GetInputImage();
  std::shared_ptr<DvppDataInfo> GetDecodedImage();
  std::shared_ptr<DvppDataInfo> GetResizedImage();
  std::shared_ptr<DvppDataInfo> GetEncodedImage();
  std::shared_ptr<DvppDataInfo> GetCropedImage();

  // Transfer DvppDataInfo from host to device, aim at resize and crop
  APP_ERROR TransferYuvDataH2D(const DvppDataInfo &imageinfo);
  // Transfer RawData(image) from host to device, aim at decode
  APP_ERROR TransferImageH2D(const RawData &imageInfo, const std::shared_ptr<DvppDataInfo> &jpegInput);
  // Transfer RawData(image on host) to inputImage_(data on device), this is for data sink mode
  APP_ERROR SinkImageH2D(const RawData &imageInfo, acldvppPixelFormat format);
  // This overload function is for PNG image sink, hence we ignore the pixel format
  APP_ERROR SinkImageH2D(const RawData &imageInfo);

  // Release the memory that is allocated in the interfaces which are started with "Combine"
  void ReleaseDvppBuffer();
  APP_ERROR VdecSendEosFrame() const;

 private:
  APP_ERROR SetDvppPicDescData(const DvppDataInfo &dataInfo, acldvppPicDesc &picDesc);
  APP_ERROR ResizeProcess(acldvppPicDesc &inputDesc, acldvppPicDesc &outputDesc, bool withSynchronize);
  APP_ERROR ResizeWithPadding(acldvppPicDesc &inputDesc, acldvppPicDesc &outputDesc, CropRoiConfig &cropRoi,
                              CropRoiConfig &pasteRoi, bool withSynchronize);
  void GetCropRoi(const DvppDataInfo &input, const DvppDataInfo &output, VpcProcessType processType,
                  CropRoiConfig &cropRoi);
  void GetPasteRoi(const DvppDataInfo &input, const DvppDataInfo &output, VpcProcessType processType,
                   CropRoiConfig &pasteRoi);
  APP_ERROR CropProcess(acldvppPicDesc &inputDesc, acldvppPicDesc &outputDesc, const CropRoiConfig &cropArea,
                        bool withSynchronize);
  APP_ERROR CheckResizeParams(const DvppDataInfo &input, const DvppDataInfo &output);
  APP_ERROR CheckCropParams(const DvppCropInputInfo &input);
  APP_ERROR CreateStreamDesc(std::shared_ptr<DvppDataInfo> data);
  APP_ERROR DestroyResource();

  std::shared_ptr<acldvppRoiConfig> cropAreaConfig_ = nullptr;
  std::shared_ptr<acldvppRoiConfig> pasteAreaConfig_ = nullptr;

  std::shared_ptr<acldvppPicDesc> cropInputDesc_ = nullptr;
  std::shared_ptr<acldvppPicDesc> cropOutputDesc_ = nullptr;
  std::shared_ptr<acldvppRoiConfig> cropRoiConfig_ = nullptr;

  std::shared_ptr<acldvppPicDesc> encodeInputDesc_ = nullptr;
  std::shared_ptr<acldvppJpegeConfig> jpegeConfig_ = nullptr;

  std::shared_ptr<acldvppPicDesc> resizeInputDesc_ = nullptr;
  std::shared_ptr<acldvppPicDesc> resizeOutputDesc_ = nullptr;
  std::shared_ptr<acldvppResizeConfig> resizeConfig_ = nullptr;

  std::shared_ptr<acldvppPicDesc> decodeOutputDesc_ = nullptr;

  acldvppChannelDesc *dvppChannelDesc_ = nullptr;

  // Ascend resource core: (context, stream) is a pair so must bind with each other to make all function perform well
  aclrtContext dvppContext_ = nullptr;
  aclrtStream dvppStream_ = nullptr;

  std::shared_ptr<DvppDataInfo> inputImage_ = nullptr;
  std::shared_ptr<DvppDataInfo> decodedImage_ = nullptr;
  std::shared_ptr<DvppDataInfo> encodedImage_ = nullptr;
  std::shared_ptr<DvppDataInfo> resizedImage_ = nullptr;
  std::shared_ptr<DvppDataInfo> cropImage_ = nullptr;
  bool isVdec_ = false;
  aclvdecChannelDesc *vdecChannelDesc_ = nullptr;
  acldvppStreamDesc *streamInputDesc_ = nullptr;
  acldvppPicDesc *picOutputDesc_ = nullptr;
  VdecConfig vdecConfig_;
};
#endif
