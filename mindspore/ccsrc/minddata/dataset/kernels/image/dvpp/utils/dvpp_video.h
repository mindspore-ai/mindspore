/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_UTILS_DVPP_VIDEO_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_UTILS_DVPP_VIDEO_H_

#include <dirent.h>
#include <unistd.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "minddata/dataset/kernels/image/dvpp/utils/ThreadSafeQueue.h"
#include "minddata/dataset/kernels/image/dvpp/utils/VdecHelper.h"

constexpr int INVALID_CHANNEL_ID = -1;
constexpr int INVALID_STREAM_FORMAT = -1;
constexpr int VIDEO_CHANNEL_MAX = 23;
constexpr int THIRD_ELEMENT_INDEX = 2;
constexpr int FOURTH_ELEMENT_INDEX = 3;
constexpr int FIFTH_ELEMENT_INDEX = 4;
constexpr int SIXTH_ELEMENT_INDEX = 5;
constexpr int EIGHTH_ELEMENT_INDEX = 7;

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
  ~ChannelIdGenerator() = default;

  int GenerateChannelId() {
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
  int channelId_[VIDEO_CHANNEL_MAX]{};
  mutable std::mutex mutex_lock_;
};

class FrameExtarct {
 public:
  FrameExtarct(uint8_t *data, uint32_t size, uint32_t width, uint32_t height, uint32_t type);
  ~FrameExtarct() = default;
  void Decode(FrameProcessCallBack callback, void *callbackParam);
  void ExtractFrameH264(const uint8_t *buf_ptr, int *size_ptr);
  void ExtractFrameH265(const uint8_t *buf_ptr, int *size_ptr);
  int IsFinished() const { return isFinished_; }
  void StopDecode() { isStop_ = true; }

 private:
  inline bool FindStartH264(const uint8_t *buf, int idx) {
    int32_t tmp = buf[idx + FOURTH_ELEMENT_INDEX] & 0x1F;
    // Find 00 00 01
    return (buf[idx] == 0) && (buf[idx + 1] == 0) && (buf[idx + THIRD_ELEMENT_INDEX] == 1) &&
           (((tmp == 0x5 || tmp == 0x1) && ((buf[idx + FIFTH_ELEMENT_INDEX] & 0x80) == 0x80)) ||
            (tmp == 0x14 && (buf[idx + EIGHTH_ELEMENT_INDEX] & 0x80) == 0x80));
  }

  inline bool FindEndH264(const uint8_t *buf, int idx) {
    // Find 00 00 01
    int32_t tmp = buf[idx + FOURTH_ELEMENT_INDEX] & 0x1F;
    return (buf[idx] == 0) && (buf[idx + 1] == 0) && (buf[idx + THIRD_ELEMENT_INDEX] == 1) &&
           ((tmp == 0xF) || (tmp == 0x7) || (tmp == 0x8) || (tmp == 0x6) ||
            ((tmp == 0x5 || tmp == 1) && ((buf[idx + FIFTH_ELEMENT_INDEX] & 0x80) == 0x80)) ||
            (tmp == 0x14 && (buf[idx + EIGHTH_ELEMENT_INDEX] & 0x80) == 0x80));
  }

  inline bool FindStartH265(const uint8_t *buf, int idx) {
    uint32_t tmp = (buf[idx + FOURTH_ELEMENT_INDEX] & 0x7EU) >> 1;
    // Find 00 00 01
    return (buf[idx + 0] == 0) && (buf[idx + 1] == 0) && (buf[idx + THIRD_ELEMENT_INDEX] == 1) && (tmp <= 0x15U) &&
           ((buf[idx + SIXTH_ELEMENT_INDEX] & 0x80) == 0x80);
  }

  inline bool FindEndH265(const uint8_t *buf, int idx) {
    uint32_t tmp = (buf[idx + FOURTH_ELEMENT_INDEX] & 0x7EU) >> 1;
    // Find 00 00 01
    return ((buf[idx + 0] == 0) && (buf[idx + 1] == 0) && (buf[idx + THIRD_ELEMENT_INDEX] == 1) &&
            ((tmp == 0x20U) || (tmp == 0x21U) || (tmp == 0x22U) || (tmp == 0x27U) || (tmp == 0x28U) ||
             ((tmp <= 0x15U) && (buf[idx + SIXTH_ELEMENT_INDEX] & 0x80) == 0x80)));
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
  bool IsStop() const { return isStop_; }
  bool IsJam() const { return isJam_; }

 private:
  AclLiteError InitResource();
  AclLiteError InitVdecDecoder();
  AclLiteError InitFrameExtractor();
  void StartFrameDecoder();
  AclLiteError FrameImageEnQueue(const std::shared_ptr<ImageData> &frameData);
  std::shared_ptr<ImageData> FrameImageOutQueue(bool noWait = false);

  void SaveYuvFile(FILE *fd, const ImageData &frame);

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
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_UTILS_DVPP_VIDEO_H_
