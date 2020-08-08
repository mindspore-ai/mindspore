/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef ACL_STUB_INC_ACL_DVPP_H
#define ACL_STUB_INC_ACL_DVPP_H
#include <stddef.h>
#include "acl/acl.h"
#include "acl/acl_base.h"

typedef struct acldvppPicDesc acldvppPicDesc;
typedef struct acldvppRoiConfig acldvppRoiConfig;
typedef struct acldvppResizeConfig acldvppResizeConfig;
typedef struct acldvppChannelDesc acldvppChannelDesc;
typedef struct acldvppStreamDesc acldvppStreamDesc;
typedef struct acldvppBatchPicDesc acldvppBatchPicDesc;

enum acldvppPixelFormat {
  PIXEL_FORMAT_YUV_400 = 0,
  PIXEL_FORMAT_YUV_SEMIPLANAR_420 = 1,  // YUV
  PIXEL_FORMAT_YVU_SEMIPLANAR_420 = 2,  // YVU
  PIXEL_FORMAT_YUV_SEMIPLANAR_422 = 3,  // YUV
  PIXEL_FORMAT_YVU_SEMIPLANAR_422 = 4,  // YVU
  PIXEL_FORMAT_YUV_SEMIPLANAR_444 = 5,  // YUV
  PIXEL_FORMAT_YVU_SEMIPLANAR_444 = 6,  // YVU

};

enum acldvppStreamFormat {
  H265_MAIN_LEVEL = 0,
  H254_BASELINE_LEVEL = 1,
  H254_MAIN_LEVEL,
  H254_HIGH_LEVEL,
};

enum acldvppChannelMode { DVPP_CHNMODE_VPC = 1, DVPP_CHNMODE_JPEGD = 2, DVPP_CHNMODE_JPEGE = 4 };

aclError acldvppMalloc(void **devPtr, size_t size);
aclError acldvppFree(void *devPtr);
acldvppChannelDesc *acldvppCreateChannelDesc();
aclError acldvppDestroyChannelDesc(acldvppChannelDesc *channelDesc);
acldvppPicDesc *acldvppCreatePicDesc();
aclError acldvppDestroyPicDesc(acldvppPicDesc *picDesc);
aclError acldvppSetPicDescSize(acldvppPicDesc *picDesc, uint32_t size);
aclError acldvppSetPicDescFormat(acldvppPicDesc *picDesc, acldvppPixelFormat format);
aclError acldvppSetPicDescWidth(acldvppPicDesc *picDesc, uint32_t width);
aclError acldvppSetPicDescHeight(acldvppPicDesc *picDesc, uint32_t height);
aclError acldvppSetPicDescData(acldvppPicDesc *picDesc, void *dataDev);
aclError acldvppSetPicDescWidthStride(acldvppPicDesc *picDesc, uint32_t widthStride);
aclError acldvppSetPicDescHeightStride(acldvppPicDesc *picDesc, uint32_t heightStride);
aclError acldvppSetPicDescRetCode(acldvppPicDesc *picDesc, uint32_t retCode);

uint32_t acldvppGetPicDescSize(acldvppPicDesc *picDesc);
acldvppPixelFormat acldvppGetPicDescFormat(acldvppPicDesc *picDesc);
uint32_t acldvppGetPicDescWidth(acldvppPicDesc *picDesc);
uint32_t acldvppGetPicDescHeight(acldvppPicDesc *picDesc);
void *acldvppGetPicDescData(acldvppPicDesc *picDesc);
uint32_t acldvppGetPicDescWidthStride(acldvppPicDesc *picDesc);
uint32_t acldvppGetPicDescHeightStride(acldvppPicDesc *picDesc);
uint32_t acldvppGetPicDescRetCode(acldvppPicDesc *picDesc);

acldvppRoiConfig *acldvppCreateRoiConfig(uint32_t left, uint32_t right, uint32_t top, uint32_t bottom);
aclError acldvppDestroyRoiConfig(acldvppRoiConfig *roiConfig);
aclError acldvppSetRoiConfigLeft(acldvppRoiConfig *roiConfig, uint32_t left);
aclError acldvppSetRoiConfigRight(acldvppRoiConfig *roiConfig, uint32_t right);
aclError acldvppSetRoiConfigTop(acldvppRoiConfig *roiConfig, uint32_t top);
aclError acldvppSetRoiConfigBottom(acldvppRoiConfig *roiConfig, uint32_t bottom);
aclError acldvppSetRoiConfig(acldvppRoiConfig *roiConfig, uint32_t left, uint32_t right, uint32_t top, uint32_t bottom);

acldvppResizeConfig *acldvppCreateResizeConfig();
aclError acldvppDestroyResizeConfig(acldvppResizeConfig *resizeConfig);

aclError acldvppJpegPredictDecSize(const void *data, uint32_t dataSize, acldvppPixelFormat ouputPixelFormat,
                                   uint32_t *decSize);

aclError acldvppCreateChannel(acldvppChannelDesc *channelDesc);
aclError acldvppDestroyChannel(acldvppChannelDesc *channelDesc);

aclError acldvppVpcResizeAsync(acldvppChannelDesc *channelDesc, acldvppPicDesc *inputDesc, acldvppPicDesc *outputDesc,
                               acldvppResizeConfig *resizeConfig, aclrtStream stream);

aclError acldvppVpcCropAsync(acldvppChannelDesc *channelDesc, acldvppPicDesc *inputDesc, acldvppPicDesc *outputDesc,
                             acldvppRoiConfig *cropArea, aclrtStream stream);

aclError acldvppVpcCropAndPasteAsync(acldvppChannelDesc *channelDesc, acldvppPicDesc *inputDesc,
                                     acldvppPicDesc *outputDesc, acldvppRoiConfig *cropArea,
                                     acldvppRoiConfig *pasteArea, aclrtStream stream);

aclError acldvppVpcBatchCropAsync(acldvppChannelDesc *channelDesc, acldvppBatchPicDesc *srcBatchDesc, uint32_t *roiNums,
                                  uint32_t size, acldvppBatchPicDesc *dstBatchDesc, acldvppRoiConfig *cropAreas[],
                                  aclrtStream stream);

aclError acldvppJpegDecodeAsync(acldvppChannelDesc *channelDesc, const void *data, uint32_t size,
                                acldvppPicDesc *outputDesc, aclrtStream stream);

acldvppBatchPicDesc *acldvppCreateBatchPicDesc(uint32_t batchSize);
acldvppPicDesc *acldvppGetPicDesc(acldvppBatchPicDesc *batchPicDesc, uint32_t index);
aclError acldvppDestroyBatchPicDesc(acldvppBatchPicDesc *batchPicDesc);

#endif  // ACL_STUB_INC_ACL_DVPP_H