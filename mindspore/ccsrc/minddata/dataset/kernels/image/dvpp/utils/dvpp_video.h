/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_DVPP_VIDEO_H
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_DVPP_VIDEO_H

#include <dirent.h>
#include <unistd.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "ThreadSafeQueue.h"
#include "VdecHelper.h"

constexpr int INVALID_CHANNEL_ID = -1;
constexpr int INVALID_STREAM_FORMAT = -1;
constexpr int VIDEO_CHANNEL_MAX = 23;

using FrameProcessCallBack = int (*)(void *callback_param, void *frame_data, int frame_size);

enum class DecodeStatus {
  DECODE_ERROR = -1,
  DECODE_UNINIT = 0,
  DECODE_READY = 1,
  DECODE_START = 2,
  DECODE_FRAME_EXTRACT_FINISHED = 3,
  DECODE_DVPP_FINISHED = 4,
  DECODE_FINISHED = 5
};

class ChannelIdGenerator {
 public:
  ChannelIdGenerator() noexcept {
    for (int i = 0; i < VIDEO_CHANNEL_MAX; i++) {
      channelId_[i] = INVALID_CHANNEL_ID;
    }
  }
  ~ChannelIdGenerator() {}

  int GenerateChannelId(void) {
    std::lock_guard<std::mutex> lock(mutex_lock_);
    for (int i = 0; i < VIDEO_CHANNEL_MAX; i++) {
      if (channelId_[i] == INVALID_CHANNEL_ID) {
        channelId_[i] = i;
        return i;
      }
    }

    return INVALID_CHANNEL_ID;
  }

  void ReleaseChannelId(int channelId) {
    std::lock_guard<std::mutex> lock(mutex_lock_);
    if ((channelId >= 0) && (channelId < VIDEO_CHANNEL_MAX)) {
      channelId_[channelId] = INVALID_CHANNEL_ID;
    }
  }

 private:
  int channelId_[VIDEO_CHANNEL_MAX];
  mutable std::mutex mutex_lock_;
};

class FrameExtarct {
 public:
  FrameExtarct(uint8_t *data, uint32_t size, uint32_t width, uint32_t height, uint32_t type);
  ~FrameExtarct() {}
  void Decode(FrameProcessCallBack callback, void *callbackParam);
  void ExtractFrameH264(const uint8_t *buf_ptr, int *size_ptr);
  void ExtractFrameH265(const uint8_t *buf_ptr, int *size_ptr);
  int IsFinished() { return isFinished_; }
  void StopDecode() { isStop_ = true; }

 private:
  inline bool FindStartH264(const uint8_t *buf, int idx) {
    int32_t tmp = buf[idx + 3] & 0x1F;
    // Find 00 00 01
    return (buf[idx] == 0) && (buf[idx + 1] == 0) && (buf[idx + 2] == 1) &&
           (((tmp == 0x5 || tmp == 0x1) && ((buf[idx + 4] & 0x80) == 0x80)) ||
            (tmp == 20 && (buf[idx + 7] & 0x80) == 0x80));
  }

  inline bool FindEndH264(const uint8_t *buf, int idx) {
    // Find 00 00 01
    int32_t tmp = buf[idx + 3] & 0x1F;
    return (buf[idx] == 0) && (buf[idx + 1] == 0) && (buf[idx + 2] == 1) &&
           ((tmp == 15) || (tmp == 7) || (tmp == 8) || (tmp == 6) ||
            ((tmp == 5 || tmp == 1) && ((buf[idx + 4] & 0x80) == 0x80)) ||
            (tmp == 20 && (buf[idx + 7] & 0x80) == 0x80));
  }

  inline bool FindStartH265(const uint8_t *buf, int idx) {
    uint32_t tmp = (buf[idx + 3] & 0x7E) >> 1;
    // Find 00 00 01
    return (buf[idx + 0] == 0) && (buf[idx + 1] == 0) && (buf[idx + 2] == 1) && (tmp <= 21) &&
           ((buf[idx + 5] & 0x80) == 0x80);
  }

  inline bool FindEndH265(const uint8_t *buf, int idx) {
    uint32_t tmp = (buf[idx + 3] & 0x7E) >> 1;
    // Find 00 00 01
    return ((buf[idx + 0] == 0) && (buf[idx + 1] == 0) && (buf[idx + 2] == 1) &&
            ((tmp == 32) || (tmp == 33) || (tmp == 34) || (tmp == 39) || (tmp == 40) ||
             ((tmp <= 21) && (buf[idx + 5] & 0x80) == 0x80)));
  }

 private:
  uint8_t *data_;
  uint32_t size_;

  uint32_t frameWidth_;
  uint32_t frameHeight_;
  int videoType_;

  bool isFinished_;
  bool isStop_;
};

class DvppVideo {
 public:
  /**
   * @brief DvppVideo constructor
   */
  DvppVideo(aclrtContext context, uint8_t *data, uint32_t size, uint32_t width, uint32_t height, uint32_t type,
            uint32_t out_format, const std::string &output);

  /**
   * @brief DvppVideo destructor
   */
  ~DvppVideo();

  static void FrameDecodeThreadFunction(void *decoderSelf);
  static AclLiteError FrameDecodeCallback(void *context, void *frameData, int frameSize);
  static void DvppVdecCallback(acldvppStreamDesc *input, acldvppPicDesc *output, void *userdata);

  void ProcessDecodedImage(std::shared_ptr<ImageData> frameData);
  void DecodeH26xFrame() { frameExtarct_->Decode(&DvppVideo::FrameDecodeCallback, reinterpret_cast<void *>(this)); }

  AclLiteError Init();
  void SetStatus(DecodeStatus status) { status_ = status; }
  DecodeStatus GetStatus() { return status_; }

  AclLiteError Read(std::shared_ptr<ImageData> *image_ptr);

  AclLiteError DumpFrame();

  AclLiteError SetAclContext();
  AclLiteError Close();

  void DestroyResource();
  bool IsStop() { return isStop_; }
  bool IsJam() { return isJam_; }

 private:
  AclLiteError InitResource();
  AclLiteError InitVdecDecoder();
  AclLiteError InitFrameExtractor();
  void StartFrameDecoder();
  AclLiteError FrameImageEnQueue(std::shared_ptr<ImageData> frameData);
  std::shared_ptr<ImageData> FrameImageOutQueue(bool noWait = false);

  void SaveYuvFile(FILE *const fd, const ImageData &frame);

 private:
  uint8_t *data_;
  uint32_t size_;

  uint32_t frameWidth_;
  uint32_t frameHeight_;

  /* 1：YUV420 semi-planner（nv12）
     2：YVU420 semi-planner（nv21）
  */
  uint32_t format_;
  std::string output_;

  bool isStop_;
  bool isReleased_;
  bool isJam_;
  DecodeStatus status_;
  aclrtContext context_;
  aclrtRunMode runMode_;
  int channelId_;
  int streamFormat_;
  uint32_t frameId_;
  uint32_t finFrameCnt_;
  int64_t lastDecodeTime_;
  std::thread decodeThread_;
  FrameExtarct *frameExtarct_;
  VdecHelper *dvppVdec_;
  ThreadSafeQueue<std::shared_ptr<ImageData>> frameImageQueue_;
};

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_DVPP_VIDEO_H
