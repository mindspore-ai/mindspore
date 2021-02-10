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

#ifndef COMMONDATATYPE_H
#define COMMONDATATYPE_H
#ifndef ENABLE_DVPP_INTERFACE
#define ENABLE_DVPP_INTERFACE
#endif
#include <stdio.h>
#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"

#define DVPP_ALIGN_UP(x, align) ((((x) + ((align)-1)) / (align)) * (align))

const uint32_t VIDEO_H264 = 0;
const uint32_t VIDEO_H265 = 1;

const float SEC2MS = 1000.0;
const uint32_t VIDEO_PROCESS_THREAD = 16;
const int YUV_BGR_SIZE_CONVERT_3 = 3;
const int YUV_BGR_SIZE_CONVERT_2 = 2;
const int DVPP_JPEG_OFFSET = 8;
const int VPC_WIDTH_ALIGN = 16;
const int VPC_HEIGHT_ALIGN = 2;
const int JPEG_WIDTH_ALIGN = 128;
const int JPEG_HEIGHT_ALIGN = 16;
const int VPC_OFFSET_ALIGN = 2;

// Data type of tensor
enum OpAttrType {
  BOOL = 0,
  INT = 1,
  FLOAT = 2,
  STRING = 3,
  LIST_BOOL = 4,
  LIST_INT = 6,
  LIST_FLOAT = 7,
  LIST_STRING = 8,
  LIST_LIST_INT = 9,
};

// operator attribution describe
// type decide whether the other attribute needed to set a value
struct OpAttr {
  std::string name;
  OpAttrType type;
  int num;                                           // LIST_BOOL/INT/FLOAT/STRING/LIST_LIST_INT need
  uint8_t numBool;                                   // BOOL need
  int64_t numInt;                                    // INT need
  float numFloat;                                    // FLOAT need
  std::string numString;                             // STRING need
  std::vector<uint8_t> valuesBool;                   // LIST_BOOL need
  std::vector<int64_t> valuesInt;                    // LIST_INT need
  std::vector<float> valuesFloat;                    // LIST_FLOAT need
  std::vector<std::string> valuesString;             // LIST_STRING need
  std::vector<int> numLists;                         // LIST_LIST_INT need
  std::vector<std::vector<int64_t>> valuesListList;  // LIST_LIST_INT need
};

// Description of image data
struct ImageInfo {
  uint32_t width;                 // Image width
  uint32_t height;                // Image height
  uint32_t lenOfByte;             // Size of image data, bytes
  std::shared_ptr<uint8_t> data;  // Smart pointer of image data
};

// Description of data in device
struct RawData {
  size_t lenOfByte;  // Size of memory, bytes
  void *data;        // Pointer of data
};

// Description of data in device
struct StreamData {
  size_t size;                 // Size of memory, bytes
  std::shared_ptr<void> data;  // Smart pointer of data
};

// Description of stream data
struct StreamInfo {
  std::string format;
  uint32_t height;
  uint32_t width;
  uint32_t channelId;
  std::string streamPath;
};

// define the structure of an rectangle
struct Rectangle {
  uint32_t leftTopX;
  uint32_t leftTopY;
  uint32_t rightBottomX;
  uint32_t rightBottomY;
};

struct ObjectDetectInfo {
  int32_t classId;
  float confidence;
  Rectangle location;
};

enum VpcProcessType {
  VPC_PT_DEFAULT = 0,
  VPC_PT_PADDING,  // Resize with locked ratio and paste on upper left corner
  VPC_PT_FIT,      // Resize with locked ratio and paste on middle location
  VPC_PT_FILL,     // Resize with locked ratio and paste on whole locatin, the input image may be cropped
};

struct DvppDataInfo {
  uint32_t width = 0;                                           // Width of image
  uint32_t height = 0;                                          // Height of image
  uint32_t widthStride = 0;                                     // Width after align up
  uint32_t heightStride = 0;                                    // Height after align up
  acldvppPixelFormat format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;  // Format of image
  uint32_t frameId = 0;                                         // Needed by video
  uint32_t dataSize = 0;                                        // Size of data in byte
  uint8_t *data = nullptr;                                      // Image data
};

struct CropRoiConfig {
  uint32_t left;
  uint32_t right;
  uint32_t down;
  uint32_t up;
};

struct DvppCropInputInfo {
  DvppDataInfo dataInfo;
  CropRoiConfig roi;
};

// Description of matrix info
struct MatrixInfo {
  uint32_t row = 0;                      // row of matrix
  uint32_t col = 0;                      // col of matrix
  uint32_t dataSize = 0;                 // size of memory, bytes
  std::shared_ptr<void> data = nullptr;  // data of matrix
  aclDataType dataType = ACL_FLOAT16;    // data Type of matrix
};

// Description of coefficient info
struct CoefficientInfo {
  std::shared_ptr<void> data = nullptr;  // data of coefficient
  aclDataType dataType = ACL_FLOAT16;    // dataType
};

// define the input of BLAS operator such as producing:
// C = alpha * A * B + beta * C
struct BlasInput {
  MatrixInfo A;
  MatrixInfo B;
  MatrixInfo C;
  CoefficientInfo alpha;
  CoefficientInfo beta;
};

extern bool g_vdecNotified[VIDEO_PROCESS_THREAD];
extern bool g_vpcNotified[VIDEO_PROCESS_THREAD];
extern bool g_inferNotified[VIDEO_PROCESS_THREAD];
extern bool g_postNotified[VIDEO_PROCESS_THREAD];

#endif
