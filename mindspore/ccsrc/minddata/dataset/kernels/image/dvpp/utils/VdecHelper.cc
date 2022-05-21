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
#include "AclLiteUtils.h"
#include "VdecHelper.h"

using namespace std;

namespace {
const uint32_t kFrameWidthMax = 4096;
const uint32_t kFrameHeightMax = 4096;
}  // namespace

VdecHelper::VdecHelper(int channelId, uint32_t width, uint32_t height, int type, aclvdecCallback callback,
                       uint32_t outFormat)
    : channelId_(channelId),
      format_(outFormat),
      enType_(type),
      frameWidth_(width),
      frameHeight_(height),
      callback_(callback),
      isExit_(false),
      isReleased_(false),
      isChannelExit_(false) {
  alignWidth_ = ALIGN_UP16(frameWidth_);
  alignHeight_ = ALIGN_UP2(frameHeight_);
  outputPicSize_ = YUV420SP_SIZE(alignWidth_, alignHeight_);

  vdecChannelDesc_ = nullptr;
  inputStreamDesc_ = nullptr;
  outputPicDesc_ = nullptr;
  outputPicBuf_ = nullptr;

  aclError aclRet;
  ACLLITE_LOG_INFO("get current context");
  aclRet = aclrtGetCurrentContext(&context_);
  if ((aclRet != ACL_SUCCESS) || (context_ == nullptr)) {
    ACLLITE_LOG_ERROR("VdecHelper : Get current acl context error:%d", aclRet);
  }

  ACLLITE_LOG_INFO("VDEC width %d, height %d", frameWidth_, frameHeight_);
}

VdecHelper::~VdecHelper() { DestroyResource(); }

void VdecHelper::DestroyChannel() {
  if (isReleased_) {
    return;
  }
  aclError ret;
  if (vdecChannelDesc_ != nullptr) {
    ret = aclvdecDestroyChannel(vdecChannelDesc_);
    if (ret != ACL_SUCCESS) {
      ACLLITE_LOG_ERROR("Vdec destroy channel failed, errorno: %d", ret);
    }
    ACLLITE_LOG_INFO("Vdec destory Channel ok");
    aclvdecDestroyChannelDesc(vdecChannelDesc_);
    ACLLITE_LOG_INFO("Vdec destory ChannelDesc ok");
    vdecChannelDesc_ = nullptr;
    isChannelExit_ = true;
  }
}

void VdecHelper::DestroyResource() {
  if (isReleased_) {
    return;
  }
  while (!isChannelExit_) {
    usleep(1000);
  }
  UnsubscribReportThread();

  // destory stream
  aclError ret;
  if (stream_ != nullptr) {
    ret = aclrtDestroyStream(stream_);
    if (ret != ACL_SUCCESS) {
      ACLLITE_LOG_ERROR("Vdec destroy stream failed");
    }
    stream_ = nullptr;
  }
  isReleased_ = true;
}

void *VdecHelper::SubscribeReportThreadFunc(void *arg) {
  ACLLITE_LOG_INFO("Start vdec subscribe thread...");

  // Notice: create context for this thread
  VdecHelper *vdec = (VdecHelper *)arg;
  aclrtContext context = vdec->GetContext();
  aclError ret = aclrtSetCurrentContext(context);
  if (ret != ACL_SUCCESS) {
    ACLLITE_LOG_ERROR("Video decoder set context failed, error: %d", ret);
  }

  while (!vdec->IsExit()) {
    // Notice: timeout 1000ms
    aclrtProcessReport(1000);
  }

  ACLLITE_LOG_INFO("Vdec subscribe thread exit!");

  return reinterpret_cast<void *>(ACLLITE_OK);
}

void VdecHelper::UnsubscribReportThread() {
  if ((subscribeThreadId_ == 0) || (stream_ == nullptr)) {
    return;
  }

  (void)aclrtUnSubscribeReport(static_cast<uint64_t>(subscribeThreadId_), stream_);
  // destory thread
  isExit_ = true;

  void *res = nullptr;
  int joinThreadErr = pthread_join(subscribeThreadId_, &res);
  if (joinThreadErr) {
    ACLLITE_LOG_ERROR("Join thread failed, threadId = %lu, err = %d", subscribeThreadId_, joinThreadErr);
  } else {
    if (reinterpret_cast<uint64_t>(res) != 0) {
      ACLLITE_LOG_ERROR("thread run failed. ret is %lu.", reinterpret_cast<uint64_t>(res));
    }
  }
  ACLLITE_LOG_INFO("Destory report thread success.");
}

AclLiteError VdecHelper::Init() {
  ACLLITE_LOG_INFO("Vdec process init start...");
  aclError aclRet = aclrtCreateStream(&stream_);
  if (aclRet != ACL_SUCCESS) {
    ACLLITE_LOG_ERROR("Vdec create stream failed, errorno:%d", aclRet);
    return ACLLITE_ERROR_CREATE_STREAM;
  }
  ACLLITE_LOG_INFO("Vdec create stream ok");

  int ret = pthread_create(&subscribeThreadId_, nullptr, SubscribeReportThreadFunc, reinterpret_cast<void *>(this));
  if (ret) {
    ACLLITE_LOG_ERROR("Start vdec subscribe thread failed, return:%d", ret);
    return ACLLITE_ERROR_CREATE_THREAD;
  }
  (void)aclrtSubscribeReport(static_cast<uint64_t>(subscribeThreadId_), stream_);

  ret = CreateVdecChannelDesc();
  if (ret != ACLLITE_OK) {
    ACLLITE_LOG_ERROR("Create vdec channel failed");
    return ret;
  }

  return ACLLITE_OK;
}

AclLiteError VdecHelper::CreateVdecChannelDesc() {
  vdecChannelDesc_ = aclvdecCreateChannelDesc();
  if (vdecChannelDesc_ == nullptr) {
    ACLLITE_LOG_ERROR("Create vdec channel desc failed");
    return ACLLITE_ERROR_CREATE_DVPP_CHANNEL_DESC;
  }

  // channelId: 0-15
  aclError ret = aclvdecSetChannelDescChannelId(vdecChannelDesc_, channelId_);
  if (ret != ACL_SUCCESS) {
    ACLLITE_LOG_ERROR("Set vdec channel id to %d failed, errorno:%d", channelId_, ret);
    return ACLLITE_ERROR_SET_VDEC_CHANNEL_ID;
  }

  ret = aclvdecSetChannelDescThreadId(vdecChannelDesc_, subscribeThreadId_);
  if (ret != ACL_SUCCESS) {
    ACLLITE_LOG_ERROR("Set vdec channel thread id failed, errorno:%d", ret);
    return ACLLITE_ERROR_SET_VDEC_CHANNEL_THREAD_ID;
  }

  // callback func
  ret = aclvdecSetChannelDescCallback(vdecChannelDesc_, callback_);
  if (ret != ACL_SUCCESS) {
    ACLLITE_LOG_ERROR("Set vdec channel callback failed, errorno:%d", ret);
    return ACLLITE_ERROR_SET_VDEC_CALLBACK;
  }

  ret = aclvdecSetChannelDescEnType(vdecChannelDesc_, static_cast<acldvppStreamFormat>(enType_));
  if (ret != ACL_SUCCESS) {
    ACLLITE_LOG_ERROR("Set vdec channel entype failed, errorno:%d", ret);
    return ACLLITE_ERROR_SET_VDEC_ENTYPE;
  }

  ret = aclvdecSetChannelDescOutPicFormat(vdecChannelDesc_, static_cast<acldvppPixelFormat>(format_));
  if (ret != ACL_SUCCESS) {
    ACLLITE_LOG_ERROR("Set vdec channel pic format failed, errorno:%d", ret);
    return ACLLITE_ERROR_SET_VDEC_PIC_FORMAT;
  }

  // create vdec channel
  ACLLITE_LOG_INFO("Start create vdec channel by desc...");
  ret = aclvdecCreateChannel(vdecChannelDesc_);
  if (ret != ACL_SUCCESS) {
    ACLLITE_LOG_ERROR("fail to create vdec channel");
    return ACLLITE_ERROR_CREATE_VDEC_CHANNEL;
  }
  ACLLITE_LOG_INFO("Create vdec channel ok");

  return ACLLITE_OK;
}

AclLiteError VdecHelper::CreateInputStreamDesc(std::shared_ptr<FrameData> frameData) {
  inputStreamDesc_ = acldvppCreateStreamDesc();
  if (inputStreamDesc_ == nullptr) {
    ACLLITE_LOG_ERROR("Create input stream desc failed");
    return ACLLITE_ERROR_CREATE_STREAM_DESC;
  }

  aclError ret;
  // to the last data,send an endding signal to dvpp vdec
  if (frameData->isFinished) {
    ret = acldvppSetStreamDescEos(inputStreamDesc_, 1);
    if (ret != ACL_SUCCESS) {
      ACLLITE_LOG_ERROR("Set EOS to input stream desc failed, errorno:%d", ret);
      return ACLLITE_ERROR_SET_STREAM_DESC_EOS;
    }
    return ACLLITE_OK;
  }

  ret = acldvppSetStreamDescData(inputStreamDesc_, frameData->data);
  if (ret != ACL_SUCCESS) {
    ACLLITE_LOG_ERROR("Set input stream data failed, errorno:%d", ret);
    return ACLLITE_ERROR_SET_STREAM_DESC_DATA;
  }

  // set size for dvpp stream desc
  ret = acldvppSetStreamDescSize(inputStreamDesc_, frameData->size);
  if (ret != ACL_SUCCESS) {
    ACLLITE_LOG_ERROR("Set input stream size failed, errorno:%d", ret);
    return ACLLITE_ERROR_SET_STREAM_DESC_SIZE;
  }

  acldvppSetStreamDescTimestamp(inputStreamDesc_, frameData->frameId);

  return ACLLITE_OK;
}

AclLiteError VdecHelper::CreateOutputPicDesc(size_t size) {
  // Malloc output device memory
  aclError ret = acldvppMalloc(&outputPicBuf_, size);
  if (ret != ACL_SUCCESS) {
    ACLLITE_LOG_ERROR(
      "Malloc vdec output buffer failed when create "
      "vdec output desc, errorno:%d",
      ret);
    return ACLLITE_ERROR_MALLOC_DVPP;
  }

  outputPicDesc_ = acldvppCreatePicDesc();
  if (outputPicDesc_ == nullptr) {
    ACLLITE_LOG_ERROR("Create vdec output pic desc failed");
    return ACLLITE_ERROR_CREATE_PIC_DESC;
  }

  ret = acldvppSetPicDescData(outputPicDesc_, outputPicBuf_);
  if (ret != ACL_SUCCESS) {
    ACLLITE_LOG_ERROR("Set vdec output pic desc data failed, errorno:%d", ret);
    return ACLLITE_ERROR_SET_PIC_DESC_DATA;
  }

  ret = acldvppSetPicDescSize(outputPicDesc_, size);
  if (ret != ACL_SUCCESS) {
    ACLLITE_LOG_ERROR("Set vdec output pic size failed, errorno:%d", ret);
    return ACLLITE_ERROR_SET_PIC_DESC_SIZE;
  }

  ret = acldvppSetPicDescFormat(outputPicDesc_, static_cast<acldvppPixelFormat>(format_));
  if (ret != ACL_SUCCESS) {
    ACLLITE_LOG_ERROR("Set vdec output pic format failed, errorno:%d", ret);
    return ACLLITE_ERROR_SET_PIC_DESC_FORMAT;
  }

  return ACLLITE_OK;
}

AclLiteError VdecHelper::Process(std::shared_ptr<FrameData> frameData, void *userData) {
  // create input desc
  AclLiteError atlRet = CreateInputStreamDesc(frameData);
  if (atlRet != ACLLITE_OK) {
    ACLLITE_LOG_ERROR("Create stream desc failed");
    return atlRet;
  }
  // create out desc
  atlRet = CreateOutputPicDesc(outputPicSize_);
  if (atlRet != ACLLITE_OK) {
    ACLLITE_LOG_ERROR("Create pic desc failed");
    return atlRet;
  }
  // send data to dvpp vdec to decode
  aclError ret = aclvdecSendFrame(vdecChannelDesc_, inputStreamDesc_, outputPicDesc_, nullptr, userData);
  if (ret != ACL_SUCCESS) {
    ACLLITE_LOG_ERROR("Send frame to vdec failed, errorno:%d", ret);
    return ACLLITE_ERROR_VDEC_SEND_FRAME;
  }

  return ACLLITE_OK;
}

AclLiteError VdecHelper::SetFormat(uint32_t format) {
  if ((format != PIXEL_FORMAT_YUV_SEMIPLANAR_420) && (format != PIXEL_FORMAT_YVU_SEMIPLANAR_420)) {
    ACLLITE_LOG_ERROR(
      "Set video decode output image format to %d failed, "
      "only support %d(YUV420SP NV12) and %d(YUV420SP NV21)",
      format, (int)PIXEL_FORMAT_YUV_SEMIPLANAR_420, (int)PIXEL_FORMAT_YVU_SEMIPLANAR_420);
    return ACLLITE_ERROR_VDEC_FORMAT_INVALID;
  }

  format_ = format;
  ACLLITE_LOG_INFO("Set video decode output image format to %d ok", format);

  return ACLLITE_OK;
}

AclLiteError VdecHelper::VideoParamCheck() {
  if ((frameWidth_ == 0) || (frameWidth_ > kFrameWidthMax)) {
    ACLLITE_LOG_ERROR("video frame width %d is invalid, the legal range is [0, %d]", frameWidth_, kFrameWidthMax);
    return ACLLITE_ERROR_VDEC_INVALID_PARAM;
  }
  if ((frameHeight_ == 0) || (frameHeight_ > kFrameHeightMax)) {
    ACLLITE_LOG_ERROR("video frame height %d is invalid, the legal range is [0, %d]", frameHeight_, kFrameHeightMax);
    return ACLLITE_ERROR_VDEC_INVALID_PARAM;
  }
  if ((format_ != PIXEL_FORMAT_YUV_SEMIPLANAR_420) && (format_ != PIXEL_FORMAT_YVU_SEMIPLANAR_420)) {
    ACLLITE_LOG_ERROR(
      "video decode image format %d invalid, "
      "only support %d(YUV420SP NV12) and %d(YUV420SP NV21)",
      format_, (int)PIXEL_FORMAT_YUV_SEMIPLANAR_420, (int)PIXEL_FORMAT_YVU_SEMIPLANAR_420);
    return ACLLITE_ERROR_VDEC_INVALID_PARAM;
  }
  if (enType_ > (uint32_t)H264_HIGH_LEVEL) {
    ACLLITE_LOG_ERROR("Input video stream format %d invalid", enType_);
    return ACLLITE_ERROR_VDEC_INVALID_PARAM;
  }

  return ACLLITE_OK;
}
