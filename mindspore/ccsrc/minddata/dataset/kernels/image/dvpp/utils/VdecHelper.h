/**
* Copyright 2022 Huawei Technologies Co., Ltd
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at

* http://www.apache.org/licenses/LICENSE-2.0

* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_UTILS_VDECHELPER_H
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_UTILS_VDECHELPER_H
#include <cstdint>
#include <iostream>

#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"
#include "AclLiteError.h"
#include "AclLiteType.h"

class VdecHelper {
 public:
  VdecHelper(int channel, uint32_t width, uint32_t height, int type, aclvdecCallback callback,
             uint32_t outFormat = PIXEL_FORMAT_YUV_SEMIPLANAR_420);
  ~VdecHelper();

  static void *SubscribeReportThreadFunc(void *arg);

  AclLiteError Init();
  void DestroyResource();
  void DestroyChannel();

  AclLiteError Process(std::shared_ptr<FrameData> frameData, void *userData);
  AclLiteError SetFormat(uint32_t format);
  AclLiteError VideoParamCheck();
  bool IsExit() { return isExit_; }
  aclrtContext GetContext() { return context_; }

 private:
  AclLiteError CreateVdecChannelDesc();
  AclLiteError CreateInputStreamDesc(std::shared_ptr<FrameData> frameData);
  AclLiteError CreateOutputPicDesc(size_t size);
  void UnsubscribReportThread();

 private:
  int channelId_;

  /* 1：YUV420 semi-planner（nv12）
     2：YVU420 semi-planner（nv21）
  */
  uint32_t format_;

  /* 0：H265 main level
   * 1：H264 baseline level
   * 2：H264 main level
   * 3：H264 high level
   */
  uint32_t enType_;

  uint32_t frameWidth_;
  uint32_t frameHeight_;
  uint32_t alignWidth_;
  uint32_t alignHeight_;
  uint32_t outputPicSize_;
  void *outputPicBuf_;
  aclvdecCallback callback_;
  aclrtContext context_;
  aclrtStream stream_;

  aclvdecChannelDesc *vdecChannelDesc_;
  acldvppStreamDesc *inputStreamDesc_;
  acldvppPicDesc *outputPicDesc_;

  pthread_t subscribeThreadId_;
  bool isExit_;
  bool isReleased_;
  bool isChannelExit_;
};

#endif
