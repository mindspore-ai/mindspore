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

#include <stdio.h>
#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"

#define DVPP_ALIGN_UP(x, align) ((((x) + ((align)-1)) / (align)) * (align))

#define OK 0
#define ERROR -1
#define INVALID_POINTER -2
#define READ_FILE_FAIL -3
#define OPEN_FILE_FAIL -4
#define INIT_FAIL -5
#define INVALID_PARAM -6
#define DECODE_FAIL -7

const float SEC2MS = 1000.0;
const int YUV_BGR_SIZE_CONVERT_3 = 3;
const int YUV_BGR_SIZE_CONVERT_2 = 2;
const int VPC_WIDTH_ALIGN = 16;
const int VPC_HEIGHT_ALIGN = 2;

// Description of image data
struct ImageInfo {
    uint32_t width;  // Image width
    uint32_t height;  // Image height
    uint32_t lenOfByte;  // Size of image data, bytes
    std::shared_ptr<uint8_t> data;  // Smart pointer of image data
};

// Description of data in device
struct RawData {
    size_t lenOfByte;  // Size of memory, bytes
    std::shared_ptr<void> data;  // Smart pointer of data
};

// define the structure of an rectangle
struct Rectangle {
    uint32_t leftTopX;
    uint32_t leftTopY;
    uint32_t rightBottomX;
    uint32_t rightBottomY;
};

enum VpcProcessType {
    VPC_PT_DEFAULT = 0,
    VPC_PT_PADDING,     // Resize with locked ratio and paste on upper left corner
    VPC_PT_FIT,         // Resize with locked ratio and paste on middle location
    VPC_PT_FILL,        // Resize with locked ratio and paste on whole locatin, the input image may be cropped
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

#endif
