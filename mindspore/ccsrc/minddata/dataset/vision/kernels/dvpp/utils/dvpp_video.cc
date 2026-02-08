/**
 # Adapted from
 # https://gitee.com/ascend/samples/blob/master/cplusplus/common/acllite/src/VideoCapture.cpp
 * ============================================================================
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   1 Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *   2 Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 *   3 Neither the names of the copyright holders nor the names of the
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * ============================================================================
 */
#include "minddata/dataset/vision/kernels/dvpp/utils/dvpp_video.h"

#include <malloc.h>
#include <unistd.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <thread>

#include "minddata/dataset/vision/kernels/dvpp/utils/AclLiteUtils.h"
#include "minddata/dataset/util/thread_bind_core_utils.h"
#include "utils/log_adapter.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace {
const int64_t kUsec = 1000000;
const uint32_t kDecodeFrameQueueSize = 256;
const int kDecodeQueueOpWait = 10000;      //  decode wait 10ms/frame
const int kFrameEnQueueRetryTimes = 1000;  //  max wait time for the frame to enter in queue
const int kQueueOpRetryTimes = 1000;
const int kOutputJamWait = 10000;
const int kInvalidTpye = -1;
const int kWaitDecodeFinishInterval = 1000;

const uint32_t DVPP_VIDEO_H264 = 0;
const uint32_t DVPP_VIDEO_H265 = 1;

ChannelIdGenerator channelIdGenerator;
}  // namespace

FrameExtarct::FrameExtarct(uint8_t *data, uint32_t size, uint32_t width, uint32_t height, uint32_t type)
    : data_(data), size_(size), frameWidth_(width), frameHeight_(height) {
  isFinished_ = false;
  isStop_ = false;
  videoType_ = (type == 0) ? DVPP_VIDEO_H265 : DVPP_VIDEO_H264;
}

void FrameExtarct::ExtractFrameH264(const uint8_t *buf_ptr, int *size_ptr) {
  if (buf_ptr == nullptr || size_ptr == nullptr || *size_ptr <= 0) {
    return;
  }
  bool isFindStart = false;
  bool isFindEnd = false;
  int size = *size_ptr;
  int i = 0;
  for (; i < size - 8; i++) {
    if (FindStartH264(buf_ptr, i)) {
      isFindStart = true;
      i += 8;
      break;
    }
  }

  for (; i < size - 8; i++) {
    if (FindEndH264(buf_ptr, i)) {
      isFindEnd = true;
      break;
    }
  }

  if (i > 0) {
    *size_ptr = i;
  }

  if (!isFindStart) {
    MS_LOG(ERROR) << "Channel can not find H265 start code, please check input video coding protocol is H264.";
    return;
  }
  if (!isFindEnd) {
    *size_ptr = i + 8;
  }
}

void FrameExtarct::ExtractFrameH265(const uint8_t *buf_ptr, int *size_ptr) {
  if (buf_ptr == nullptr || size_ptr == nullptr || *size_ptr <= 0) {
    return;
  }
  bool isFindStart = false;
  bool isFindEnd = false;
  int i = 0;
  for (; i < *size_ptr - 6; i++) {
    if (FindStartH265(buf_ptr, i)) {
      isFindStart = true;
      i += 6;
      break;
    }
  }

  for (; i < *size_ptr - 6; i++) {
    if (FindEndH265(buf_ptr, i)) {
      isFindEnd = true;
      break;
    }
  }
  if (i > 0) {
    *size_ptr = i;
  }

  if (!isFindStart) {
    MS_LOG(ERROR) << "Channel can not find H265 start code, please check input video coding protocol is H265.";
    return;
  }
  if (!isFindEnd) {
    *size_ptr = i + 6;
  }
}

void FrameExtarct::Decode(FrameProcessCallBack callback, void *callbackParam) {
  MS_LOG(INFO) << "Start extarct frame from video...";

  int32_t usedBytes = 0;
  uint32_t count = 0;
  bool processOk = true;

  while (!isStop_ && processOk) {
    uint8_t *bufPointer;
    bufPointer = data_ + usedBytes;
    int32_t readlen = size_ - usedBytes;
    if (readlen <= 0) {
      break;
    }

    if (videoType_ == DVPP_VIDEO_H264) {  // H264
      ExtractFrameH264(bufPointer, &readlen);
    } else if (videoType_ == DVPP_VIDEO_H265) {  // H265
      ExtractFrameH265(bufPointer, &readlen);
    }
    int ret = callback(callbackParam, bufPointer, readlen);
    if (ret != 0) {
      processOk = false;
    }
    count++;
    usedBytes = usedBytes + readlen;
  }
  // Frame count

  isFinished_ = true;
  MS_LOG(INFO) << "FrameExtarct decoder finished, frame count: " << count << ".";
}

DvppVideo::DvppVideo(aclrtContext context, uint8_t *data, uint32_t size, uint32_t width, uint32_t height, uint32_t type,
                     uint32_t out_format, const std::string &output)
    : data_(data),
      size_(size),
      frameWidth_(width),
      frameHeight_(height),
      format_(out_format),
      output_(output),
      isStop_(false),
      isReleased_(false),
      isJam_(false),
      status_(DecodeStatus::DECODE_UNINIT),
      context_(context),
      channelId_(INVALID_CHANNEL_ID),
      streamFormat_(type),
      frameId_(0),
      finFrameCnt_(0),
      lastDecodeTime_(0),
      frameExtarct_(nullptr),
      dvppVdec_(nullptr),
      frameImageQueue_(kDecodeFrameQueueSize) {}

DvppVideo::~DvppVideo() { DestroyResource(); }

void DvppVideo::DestroyResource() {
  if (isReleased_) {
    return;
  }
  // 1. stop ffmpeg
  isStop_ = true;
  frameExtarct_->StopDecode();
  while ((status_ >= DecodeStatus::DECODE_START) && (status_ < DecodeStatus::DECODE_FRAME_EXTRACT_FINISHED)) {
    (void)usleep(kWaitDecodeFinishInterval);
  }
  // 2. delete ffmpeg decoder
  delete frameExtarct_;
  frameExtarct_ = nullptr;

  // 3. release dvpp vdec
  dvppVdec_->DestroyResource();

  // 4. release image memory in decode output queue
  do {
    std::shared_ptr<ImageData> frame = FrameImageOutQueue(true);
    if (frame == nullptr) {
      break;
    }

    if (frame->data != nullptr) {
      auto ret = acldvppFree(frame->data.get());
      if (ret != ACL_SUCCESS) {
        MS_LOG(ERROR) << "acldvppFree failed, errorno: " << ret;
      }
      frame->data = nullptr;
    }
  } while (true);
  // 5. release channel id
  channelIdGenerator.ReleaseChannelId(channelId_);

  isReleased_ = true;
}

AclLiteError DvppVideo::InitResource() {
  aclError aclRet;
  // use current thread context default
  if (context_ == nullptr) {
    aclRet = CALL_ASCEND_API(aclrtGetCurrentContext, &context_);
    if ((aclRet != ACL_SUCCESS) || (context_ == nullptr)) {
      MS_LOG(ERROR) << "Get current acl context error: " << aclRet;
      return ACLLITE_ERROR_GET_ACL_CONTEXT;
    }
  }
  // Get current run mode
  aclRet = CALL_ASCEND_API(aclrtGetRunMode, &runMode_);
  if (aclRet != ACL_SUCCESS) {
    MS_LOG(ERROR) << "acl get run mode failed";
    return ACLLITE_ERROR_GET_RUM_MODE;
  }

  return ACLLITE_OK;
}

AclLiteError DvppVideo::InitVdecDecoder() {
  // Generate a unique channel id for video decoder
  channelId_ = channelIdGenerator.GenerateChannelId();
  if (channelId_ == INVALID_CHANNEL_ID) {
    MS_LOG(ERROR) << "Decoder number excessive " << VIDEO_CHANNEL_MAX;
    return ACLLITE_ERROR_TOO_MANY_VIDEO_DECODERS;
  }

  // Create dvpp vdec to decode h26x data
  dvppVdec_ = new VdecHelper(channelId_, frameWidth_, frameHeight_, streamFormat_, DvppVideo::DvppVdecCallback);

  AclLiteError ret = dvppVdec_->SetFormat(format_);
  if (ret != ACLLITE_OK) {
    MS_LOG(ERROR) << "Dvpp vdec set out format failed";
  }
  ret = dvppVdec_->Init();
  if (ret != ACLLITE_OK) {
    MS_LOG(ERROR) << "Dvpp vdec init failed";
  }

  ret = this->dvppVdec_->VideoParamCheck();
  if (ret != ACLLITE_OK) {
    this->SetStatus(DecodeStatus::DECODE_ERROR);
    MS_LOG(ERROR) << "Dvpp vdec check param failed " << ret;
    return ret;
  }

  return ret;
}

AclLiteError DvppVideo::InitFrameExtractor() {
  // Create ffmpeg decoder to parse video stream to h26x frame data
  frameExtarct_ = new FrameExtarct(data_, size_, frameWidth_, frameHeight_, streamFormat_);
  return ACLLITE_OK;
}

AclLiteError DvppVideo::Init() {
  // Open video stream, if open failed before, return error directly
  if (status_ == DecodeStatus::DECODE_ERROR) {
    return ACLLITE_ERROR_OPEN_VIDEO_UNREADY;
  }
  // If open ok already
  if (status_ != DecodeStatus::DECODE_UNINIT) {
    return ACLLITE_OK;
  }
  // Init acl resource
  AclLiteError ret = InitResource();
  if (ret != ACLLITE_OK) {
    this->SetStatus(DecodeStatus::DECODE_ERROR);
    MS_LOG(ERROR) << "Dvpp video init resource failed " << ret;
    return ret;
  }
  // Init ffmpeg decoder
  ret = InitFrameExtractor();
  if (ret != ACLLITE_OK) {
    this->SetStatus(DecodeStatus::DECODE_ERROR);
    MS_LOG(ERROR) << "Dvpp video init FrameExtractor failed " << ret;
    return ret;
  }
  // Init dvpp vdec decoder
  ret = InitVdecDecoder();
  if (ret != ACLLITE_OK) {
    this->SetStatus(DecodeStatus::DECODE_ERROR);
    MS_LOG(ERROR) << "Dvpp video init Vdec failed " << ret;
    return ret;
  }
  // Set init ok
  this->SetStatus(DecodeStatus::DECODE_READY);
  MS_LOG(INFO) << "Dvpp video init ok";

  return ACLLITE_OK;
}

// dvpp vdec callback
void DvppVideo::DvppVdecCallback(acldvppStreamDesc *input, acldvppPicDesc *output, void *userData) {
  auto *decoder = reinterpret_cast<DvppVideo *>(userData);
  // Get decoded image parameters
  std::shared_ptr<ImageData> image = std::make_shared<ImageData>();
  image->format = acldvppGetPicDescFormat(output);
  image->width = acldvppGetPicDescWidth(output);
  image->height = acldvppGetPicDescHeight(output);
  image->alignWidth = acldvppGetPicDescWidthStride(output);
  image->alignHeight = acldvppGetPicDescHeightStride(output);
  image->size = acldvppGetPicDescSize(output);

  void *vdecOutBufferDev = acldvppGetPicDescData(output);
  image->data = SHARED_PTR_DVPP_BUF(vdecOutBufferDev);

  // Put the decoded image to queue for read
  decoder->ProcessDecodedImage(image);
  // Release resource
  aclError ret = acldvppDestroyPicDesc(output);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Dvpp vdec destroy pic desc failed " << ret;
  }

  if (input != nullptr) {
    void *inputBuf = acldvppGetStreamDescData(input);
    if (inputBuf != nullptr) {
      ret = acldvppFree(inputBuf);
      if (ret != ACL_SUCCESS) {
        MS_LOG(ERROR) << "acldvppFree failed, errorno: " << ret;
      }
    }

    ret = acldvppDestroyStreamDesc(input);
    if (ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Dvpp vdec destroy input stream failed " << ret;
    }
  }
}

void DvppVideo::ProcessDecodedImage(std::shared_ptr<ImageData> frameData) {
  finFrameCnt_++;
  if (YUV420SP_SIZE(frameData->width, frameData->height) != frameData->size) {
    MS_LOG(ERROR) << "Invalid decoded frame parameter, width " << frameData->width << ", height " << frameData->height
                  << ", size " << frameData->size << ", buffer " << static_cast<void *>(frameData->data.get());
    return;
  }

  auto ret = FrameImageEnQueue(frameData);
  if (ret != ACLLITE_OK) {
    MS_LOG(ERROR) << "FrameImageEnQueue failed, errorno: " << ret;
  }

  if ((status_ == DecodeStatus::DECODE_FRAME_EXTRACT_FINISHED) && (finFrameCnt_ >= frameId_)) {
    MS_LOG(INFO) << "Last frame decoded by dvpp, change status to " << DecodeStatus::DECODE_DVPP_FINISHED;
    this->SetStatus(DecodeStatus::DECODE_DVPP_FINISHED);
  }
}

AclLiteError DvppVideo::FrameImageEnQueue(const std::shared_ptr<ImageData> &frameData) {
  for (int count = 0; count < kFrameEnQueueRetryTimes; count++) {
    if (frameImageQueue_.Push(frameData)) {
      return ACLLITE_OK;
    }
    (void)usleep(kDecodeQueueOpWait);
  }
  MS_LOG(ERROR) << "Video lost decoded image for queue full";

  return ACLLITE_ERROR_VDEC_QUEUE_FULL;
}

// start decoder
void DvppVideo::StartFrameDecoder() {
  if (status_ == DecodeStatus::DECODE_READY) {
    decodeThread_ = std::thread(FrameDecodeThreadFunction, reinterpret_cast<void *>(this));
    decodeThread_.detach();

    status_ = DecodeStatus::DECODE_START;
  }
}

// ffmpeg decoder entry
void DvppVideo::FrameDecodeThreadFunction(void *decoderSelf) {
  mindspore::dataset::BindThreadCoreForMindDataOp("dataset::DvppVideo::FrameDecodeThreadFunction");
  if (decoderSelf == nullptr) {
    return;
  }
  auto *thisPtr = reinterpret_cast<DvppVideo *>(decoderSelf);

  aclError aclRet = thisPtr->SetAclContext();
  if (aclRet != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Set frame decoder context failed, errorno: " << aclRet;
    return;
  }
  // start decode until complete
  thisPtr->DecodeH26xFrame();
  if (thisPtr->IsStop()) {
    thisPtr->SetStatus(DecodeStatus::DECODE_FINISHED);
    return;
  }
  thisPtr->SetStatus(DecodeStatus::DECODE_FRAME_EXTRACT_FINISHED);
  // when ffmpeg decode finish, send eos to vdec
  std::shared_ptr<FrameData> videoFrame = std::make_shared<FrameData>();
  videoFrame->isFinished = true;
  videoFrame->data = nullptr;
  videoFrame->size = 0;
  auto ret = thisPtr->dvppVdec_->Process(videoFrame, decoderSelf);
  if (ret != ACLLITE_OK) {
    MS_LOG(ERROR) << "DvppVdec processing failed, errorno: " << ret;
  }

  thisPtr->dvppVdec_->DestroyChannel();
  while ((thisPtr->GetStatus() != DecodeStatus::DECODE_DVPP_FINISHED) && !thisPtr->IsStop()) {
    (void)usleep(kWaitDecodeFinishInterval);
  }
}

// callback of ffmpeg decode frame
AclLiteError DvppVideo::FrameDecodeCallback(void *decoder, void *frameData, int frameSize) {
  if ((frameData == nullptr) || (frameSize == 0)) {
    MS_LOG(ERROR) << "Frame data is null";
    return ACLLITE_ERROR_H26X_FRAME;
  }

  // copy data to dvpp memory
  if (decoder == nullptr) {
    MS_LOG(ERROR) << "Decoder is nullptr";
    return ACLLITE_ERROR_H26X_FRAME;
  }
  auto *videoDecoder = reinterpret_cast<DvppVideo *>(decoder);

  void *buffer = CopyDataToDevice(frameData, frameSize, videoDecoder->runMode_, MemoryType::MEMORY_DVPP);
  if (buffer == nullptr) {
    MS_LOG(ERROR) << "Copy frame h26x data to dvpp failed";
    return ACLLITE_ERROR_COPY_DATA;
  }

  std::shared_ptr<FrameData> videoFrame = std::make_shared<FrameData>();
  videoDecoder->frameId_++;
  videoFrame->frameId = videoDecoder->frameId_;
  videoFrame->data = buffer;
  videoFrame->size = frameSize;
  // decode data by dvpp vdec
  AclLiteError ret = videoDecoder->dvppVdec_->Process(videoFrame, decoder);
  if (ret != ACLLITE_OK) {
    MS_LOG(ERROR) << "Dvpp vdec process " << videoDecoder->frameId_ << "th frame failed, error: " << ret;
    return ret;
  }
  return ACLLITE_OK;
}

// read decoded frame
AclLiteError DvppVideo::Read(std::shared_ptr<ImageData> *image_ptr) {
  if (image_ptr == nullptr) {
    MS_LOG(ERROR) << "image_ptr is nullptr";
    return ACLLITE_ERROR;
  }
  // return nullptr,if decode fail/finish
  if (status_ == DecodeStatus::DECODE_ERROR) {
    MS_LOG(ERROR) << "Read failed for decode failed";
    return ACLLITE_ERROR_VIDEO_DECODER_STATUS;
  }

  if (status_ == DecodeStatus::DECODE_FINISHED) {
    MS_LOG(INFO) << "No frame to read for decode finished";
    return ACLLITE_ERROR_DECODE_FINISH;
  }
  // start decode if status is ok
  if (status_ == DecodeStatus::DECODE_READY) {
    StartFrameDecoder();
    (void)usleep(kDecodeQueueOpWait);
  }
  // read frame from decode queue
  bool noWait = (status_ == DecodeStatus::DECODE_DVPP_FINISHED);
  std::shared_ptr<ImageData> frame = FrameImageOutQueue(noWait);
  if (noWait && (frame == nullptr)) {
    SetStatus(DecodeStatus::DECODE_FINISHED);
    MS_LOG(INFO) << "No frame to read anymore";
    return ACLLITE_ERROR_DECODE_FINISH;
  }

  if (frame == nullptr) {
    MS_LOG(ERROR) << "Empty frame image to read";
    return ACLLITE_ERROR_READ_EMPTY;
  }

  (*image_ptr)->format = frame->format;
  (*image_ptr)->width = frame->width;
  (*image_ptr)->height = frame->height;
  (*image_ptr)->alignWidth = frame->alignWidth;
  (*image_ptr)->alignHeight = frame->alignHeight;
  (*image_ptr)->size = frame->size;
  (*image_ptr)->data = frame->data;

  return ACLLITE_OK;
}

std::shared_ptr<ImageData> DvppVideo::FrameImageOutQueue(bool noWait) {
  std::shared_ptr<ImageData> image = frameImageQueue_.Pop();

  if (noWait || (image != nullptr)) {
    return image;
  }

  for (int count = 0; count < kQueueOpRetryTimes - 1; count++) {
    (void)usleep(kDecodeQueueOpWait);

    image = frameImageQueue_.Pop();
    if (image != nullptr) {
      return image;
    }
  }

  return nullptr;
}

// YUV data write to a file
void DvppVideo::SaveYuvFile(FILE *const fd, const ImageData &frame) {
  auto *addr = reinterpret_cast<uint8_t *>(frame.data.get());
  uint32_t imageSize = frame.width * frame.height * 3 / 2;  // Size = width * height * 3 / 2
  uint8_t *outImageBuf = nullptr;
  uint32_t outWidthStride = frame.alignWidth;
  uint32_t outHeightStride = frame.alignHeight;

  if (runMode_ == ACL_HOST) {
    // malloc host memory
    AclLiteError ret = CALL_ASCEND_API(aclrtMallocHost, reinterpret_cast<void **>(&outImageBuf), imageSize);
    if (ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Chn " << channelId_ << " malloc host memory " << imageSize << " failed, error code " << ret;
      return;
    }
  }

  if ((frame.width == outWidthStride) && (frame.height == outHeightStride)) {
    if (runMode_ == ACL_HOST) {
      // copy device data to host
      AclLiteError ret =
        CALL_ASCEND_API(aclrtMemcpy, outImageBuf, imageSize, addr, imageSize, ACL_MEMCPY_DEVICE_TO_HOST);
      if (ret != ACL_SUCCESS) {
        MS_LOG(ERROR) << "Chn " << channelId_ << " Copy aclrtMemcpy " << imageSize
                      << " from device to host failed, error code " << ret;
        ret = CALL_ASCEND_API(aclrtFreeHost, outImageBuf);
        if (ret != ACL_SUCCESS) {
          MS_LOG(ERROR) << "aclrtFreeHost failed, errorno: " << ret;
        }
        return;
      }

      (void)fwrite(outImageBuf, 1, imageSize, fd);
      ret = CALL_ASCEND_API(aclrtFreeHost, outImageBuf);
      if (ret != ACL_SUCCESS) {
        MS_LOG(ERROR) << "aclrtFreeHost failed, errorno: " << ret;
        return;
      }
    } else {
      (void)fwrite(addr, imageSize, 1, fd);
    }
  } else {
    if (runMode_ == ACL_HOST) {
      if (outImageBuf == nullptr) {
        return;
      }
      // Copy valid Y data
      for (uint32_t i = 0; i < frame.height; i++) {
        AclLiteError ret = CALL_ASCEND_API(aclrtMemcpy, outImageBuf + i * frame.width, frame.width,
                                           addr + i * outWidthStride, frame.width, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
          MS_LOG(ERROR) << "Chn " << channelId_ << " Copy aclrtMemcpy " << imageSize
                        << " from device to host failed, error code " << ret;
          ret = CALL_ASCEND_API(aclrtFreeHost, outImageBuf);
          if (ret != ACL_SUCCESS) {
            MS_LOG(ERROR) << "aclrtFreeHost failed, errorno: " << ret;
          }
          return;
        }
      }
      // Copy valid UV data
      for (uint32_t i = 0; i < frame.height / 2; i++) {
        AclLiteError ret = CALL_ASCEND_API(aclrtMemcpy, outImageBuf + i * frame.width + frame.width * frame.height,
                                           frame.width, addr + i * outWidthStride + outWidthStride * outHeightStride,
                                           frame.width, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
          MS_LOG(ERROR) << "Chn " << channelId_ << " Copy aclrtMemcpy " << imageSize
                        << " from device to host failed, error code " << ret;
          ret = CALL_ASCEND_API(aclrtFreeHost, outImageBuf);
          if (ret != ACL_SUCCESS) {
            MS_LOG(ERROR) << "aclrtFreeHost failed, errorno: " << ret;
          }
          return;
        }
      }

      (void)fwrite(outImageBuf, 1, imageSize, fd);
      aclError ret = CALL_ASCEND_API(aclrtFreeHost, outImageBuf);
      if (ret != ACL_SUCCESS) {
        MS_LOG(ERROR) << "aclrtFreeHost failed, errorno: " << ret;
      }
    } else {
      // Crop the invalid data, then write the valid data to a file
      outImageBuf = reinterpret_cast<uint8_t *>(malloc(imageSize));
      if (outImageBuf == nullptr) {
        MS_LOG(ERROR) << "Chn " << channelId_ << " Malloc failed";
        return;
      }
      // Copy valid Y data
      for (uint32_t i = 0; i < frame.height; i++) {
        int status = memcpy_s(outImageBuf + i * frame.width, frame.width, addr + i * outWidthStride, frame.width);
        if (status != EOK) {
          MS_LOG(ERROR) << "[Internal ERROR] memcpy failed.";
          free(outImageBuf);
          return;
        }
      }
      // Copy valid UV data
      for (uint32_t i = 0; i < frame.height / 2; i++) {
        int status = memcpy_s(outImageBuf + i * frame.width + frame.width * frame.height, frame.width,
                              addr + i * outWidthStride + outWidthStride * outHeightStride, frame.width);
        if (status != EOK) {
          MS_LOG(ERROR) << "[Internal ERROR] memcpy failed.";
          free(outImageBuf);
          return;
        }
      }

      (void)fwrite(outImageBuf, 1, imageSize, fd);
      free(outImageBuf);
    }
  }
}

AclLiteError DvppVideo::DumpFrame() {
  auto frame = std::make_shared<ImageData>();
  int frameCnt = 0;
  while (true) {
    AclLiteError ret = Read(&frame);
    if (ret != ACLLITE_OK) {
      if (ret == ACLLITE_ERROR_DECODE_FINISH) {
        MS_LOG(INFO) << "Dump all " << frameCnt << " frames to " << output_;
        return ACLLITE_OK;
      } else {
        MS_LOG(ERROR) << "Dump " << frameCnt << "td frame failed";
        return ret;
      }
    }
    frameCnt++;
    std::string full_path = output_ + "/" + "frame_" + std::to_string(frameCnt) + ".yuv";
    MS_LOG(INFO) << "Dump the " << frameCnt << "th frame to " << full_path;
    FILE *outFileFp = fopen(full_path.c_str(), "wb+");
    SaveYuvFile(outFileFp, *frame);
    (void)fflush(outFileFp);
    (void)fclose(outFileFp);
  }
}

AclLiteError DvppVideo::SetAclContext() {
  if (context_ == nullptr) {
    MS_LOG(ERROR) << "Video decoder context is null";
    return ACLLITE_ERROR_SET_ACL_CONTEXT;
  }

  aclError ret = CALL_ASCEND_API(aclrtSetCurrentContext, context_);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Video decoder set context failed, error: " << ret;
    return ACLLITE_ERROR_SET_ACL_CONTEXT;
  }

  return ACLLITE_OK;
}

AclLiteError DvppVideo::Close() {
  DestroyResource();
  return ACLLITE_OK;
}
