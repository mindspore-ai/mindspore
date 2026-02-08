/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_ASCEND910B_DVPP_VIDEO_UTILS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_ASCEND910B_DVPP_VIDEO_UTILS_H_

#include <cstdint>
#include <memory>
#include <mutex>

#include "acl/dvpp/hi_dvpp.h"

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/core/device_buffer.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"

namespace mindspore {
namespace dataset {
constexpr uint32_t VDEC_MAX_CHNL_NUM = 32;

enum class ChnStatus { CREATED, DESTROYED };

int64_t dvpp_sys_init();
int64_t dvpp_sys_exit();
int64_t dvpp_vdec_create_chnl(int64_t payload_type);
int64_t dvpp_vdec_start_get_frame(int64_t chn_id, int64_t total_frame);
int64_t dvpp_vdec_send_stream(int64_t chn_id, const std::shared_ptr<DeviceBuffer> &input, int64_t out_format,
                              bool display, std::shared_ptr<DeviceBuffer> *out);
std::shared_ptr<DeviceBuffer> dvpp_vdec_stop_get_frame(int64_t chn_id, int64_t total_frame);
int64_t dvpp_vdec_destroy_chnl(int64_t chn_id);
int64_t dvpp_memcpy(void *dst, size_t dest_max, const void *src, size_t count, int kind);

class VideoDecoder {
 public:
  static VideoDecoder &GetInstance();

  VideoDecoder();

  ~VideoDecoder();

  int32_t GetUnusedChn(uint32_t &chnl);

  void PutChn(uint32_t chnl);

  hi_s32 sys_init(hi_void);
  hi_s32 sys_exit(hi_void);
  hi_u32 get_tmv_buf_size(hi_payload_type type, hi_u32 width, hi_u32 height);
  hi_u32 get_pic_buf_size(hi_payload_type type, hi_pic_buf_attr *buf_attr);
  hi_s32 create_chn(hi_vdec_chn chn, const hi_vdec_chn_attr *attr);
  hi_s32 destroy_chn(hi_vdec_chn chn);
  hi_s32 sys_set_chn_csc_matrix(hi_vdec_chn chn);
  hi_s32 start_recv_stream(hi_vdec_chn chn);
  hi_s32 stop_recv_stream(hi_vdec_chn chn);
  hi_s32 query_status(hi_vdec_chn chn, hi_vdec_chn_status *status);
  hi_s32 reset_chn(hi_vdec_chn chn);
  hi_s32 send_stream(hi_vdec_chn chn, const hi_vdec_stream *stream, hi_vdec_pic_info *vdec_pic_info, hi_s32 milli_sec);
  hi_s32 get_frame(hi_vdec_chn chn, hi_video_frame_info *frame_info, hi_vdec_supplement_info *supplement,
                   hi_vdec_stream *stream, hi_s32 milli_sec);
  hi_s32 release_frame(hi_vdec_chn chn, const hi_video_frame_info *frame_info);

 private:
  std::mutex channel_mutex_[VDEC_MAX_CHNL_NUM];
  ChnStatus channel_status_[VDEC_MAX_CHNL_NUM];
  device::DeviceContext *device_context_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_ASCEND910B_DVPP_VIDEO_UTILS_H_
