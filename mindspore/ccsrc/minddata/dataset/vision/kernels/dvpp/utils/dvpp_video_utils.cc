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
#include "minddata/dataset/vision/kernels/dvpp/utils/dvpp_video_utils.h"

#include <vector>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <dlfcn.h>
#include <sys/time.h>
#include <sys/prctl.h>

#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "minddata/dataset/vision/kernels/dvpp/utils/dvpp_image_utils.h"

namespace mindspore {
namespace dataset {
constexpr uint32_t MAX_CHN_HEIGHT = 4096;
constexpr uint32_t MAX_CHN_WIDTH = 4096;
constexpr float BUF_SIZE_FACTOR = 1.5;
constexpr int32_t SEND_TIMEOUT = 1000;
constexpr uint32_t WAIT_TIMEOUT = 5000000;  // in us
constexpr uint32_t REF_FRAME_NUM = 16;
constexpr uint32_t DISPLAY_FRAME_NUM = 16;
constexpr uint32_t FRAME_BUF_CNT = REF_FRAME_NUM + DISPLAY_FRAME_NUM + 1;
constexpr uint8_t VDEC_DECODE_SUCCESS = 0;
constexpr uint8_t VDEC_DECODE_FAILED = 1;
constexpr uint8_t VDEC_DECODE_INTERLACED_FIELD_STREAM = 2;
constexpr uint8_t VDEC_DECODE_FRAME_NUMBER_ERROR = 3;
constexpr uint8_t VDEC_DECODE_FRAME_SIZE_ERROR = 4;

pthread_t vdec_thread_list[VDEC_MAX_CHNL_NUM] = {0};
uint32_t thread_exit_state_list[VDEC_MAX_CHNL_NUM] = {0};
std::vector<std::vector<std::shared_ptr<DeviceBuffer>>> out_buffer_queue(VDEC_MAX_CHNL_NUM);
std::mutex out_buffer_map_mutex[VDEC_MAX_CHNL_NUM];
std::map<hi_u64, std::shared_ptr<DeviceBuffer>> out_buffer_map[VDEC_MAX_CHNL_NUM];

struct GetThreadPara {
  uint32_t chn_id;
  uint32_t device_id;
  uint32_t total_frame;
  uint32_t success_cnt;
};

GetThreadPara get_thread_para[VDEC_MAX_CHNL_NUM];

static inline bool ValidChnNum(uint32_t chn) { return (chn < VDEC_MAX_CHNL_NUM); }

static inline void get_current_time_us(uint64_t &time_us) {
  struct timeval cur_time;
  gettimeofday(&cur_time, nullptr);
  time_us = static_cast<uint64_t>(cur_time.tv_sec) * 1000000 + cur_time.tv_usec;  // 1s = 1000000 us
}

VideoDecoder &VideoDecoder::GetInstance() {
  static VideoDecoder instance;
  return instance;
}

VideoDecoder::VideoDecoder() {
  for (uint32_t i = 0; i < VDEC_MAX_CHNL_NUM; ++i) {
    channel_status_[i] = ChnStatus::DESTROYED;
  }
}

VideoDecoder::~VideoDecoder() {}

int32_t VideoDecoder::GetUnusedChn(uint32_t &chn) {
  for (uint32_t i = 0; i < VDEC_MAX_CHNL_NUM; ++i) {
    const std::lock_guard<std::mutex> guard(channel_mutex_[i]);
    if (channel_status_[i] != ChnStatus::DESTROYED) {
      continue;
    } else {
      channel_status_[i] = ChnStatus::CREATED;
      chn = i;
      return 0;
    }
  }
  return -1;
}

void VideoDecoder::PutChn(uint32_t chn) {
  const std::lock_guard<std::mutex> guard(channel_mutex_[chn]);
  channel_status_[chn] = ChnStatus::DESTROYED;
}

static std::once_flag init_once_flag;

hi_s32 VideoDecoder::sys_init(void) {
  hi_s32 ret = HI_SUCCESS;
  std::call_once(init_once_flag, [this, &ret]() {
    auto ms_context = MsContext::GetInstance();
    if (ms_context == nullptr) {
      MS_EXCEPTION(RuntimeError) << "Failed to get mindspore context.";
    }
    device_context_ = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {device::GetDeviceTypeByName(ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET)),
       ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
    if (device_context_ == nullptr) {
      MS_EXCEPTION(RuntimeError) << "Failed to get device context.";
    }
    device_context_->Initialize();
    if (device_context_->device_res_manager_ == nullptr) {
      MS_EXCEPTION(RuntimeError) << "Device resource manager is nullptr.";
    }

    std::string soc_version;
    if (GetSocName(&soc_version) != APP_ERR_OK) {
      MS_EXCEPTION(RuntimeError) << "Failed to get Ascend version.";
    }
    if (soc_version.find("Ascend910B") == std::string::npos && soc_version.find("Ascend910_93") == std::string::npos) {
      device_context_ = nullptr;
      MS_EXCEPTION(RuntimeError) << "Only support running on Ascend910B or Ascend910_93, but got: " << soc_version;
    }
    ret = hi_mpi_sys_init();
  });
  return ret;
}

hi_s32 VideoDecoder::sys_exit(void) { return hi_mpi_sys_exit(); }

hi_u32 VideoDecoder::get_pic_buf_size(hi_payload_type type, hi_pic_buf_attr *buf_attr) {
  return hi_vdec_get_pic_buf_size(type, buf_attr);
}

hi_u32 VideoDecoder::get_tmv_buf_size(hi_payload_type type, hi_u32 width, hi_u32 height) {
  return hi_vdec_get_tmv_buf_size(type, width, height);
}

hi_s32 VideoDecoder::create_chn(hi_vdec_chn chn, const hi_vdec_chn_attr *attr) {
  auto ret = hi_mpi_vdec_create_chn(chn, attr);
  return ret;
}

hi_s32 VideoDecoder::destroy_chn(hi_vdec_chn chn) { return hi_mpi_vdec_destroy_chn(chn); }

hi_s32 VideoDecoder::sys_set_chn_csc_matrix(hi_vdec_chn chn) {
  return hi_mpi_sys_set_chn_csc_matrix(HI_ID_VDEC, chn, HI_CSC_MATRIX_BT601_NARROW, nullptr);
}

hi_s32 VideoDecoder::start_recv_stream(hi_vdec_chn chn) { return hi_mpi_vdec_start_recv_stream(chn); }

hi_s32 VideoDecoder::stop_recv_stream(hi_vdec_chn chn) { return hi_mpi_vdec_stop_recv_stream(chn); }

hi_s32 VideoDecoder::query_status(hi_vdec_chn chn, hi_vdec_chn_status *status) {
  return hi_mpi_vdec_query_status(chn, status);
}

hi_s32 VideoDecoder::reset_chn(hi_vdec_chn chn) { return hi_mpi_vdec_reset_chn(chn); }

hi_s32 VideoDecoder::send_stream(hi_vdec_chn chn, const hi_vdec_stream *stream, hi_vdec_pic_info *vdec_pic_info,
                                 hi_s32 milli_sec) {
  return hi_mpi_vdec_send_stream(chn, stream, vdec_pic_info, milli_sec);
}

hi_s32 VideoDecoder::get_frame(hi_vdec_chn chn, hi_video_frame_info *frame_info, hi_vdec_supplement_info *supplement,
                               hi_vdec_stream *stream, hi_s32 milli_sec) {
  return hi_mpi_vdec_get_frame(chn, frame_info, supplement, stream, milli_sec);
}

hi_s32 VideoDecoder::release_frame(hi_vdec_chn chn, const hi_video_frame_info *frame_info) {
  return hi_mpi_vdec_release_frame(chn, frame_info);
}

static void vdec_reset_chn(uint32_t chn) {
  int32_t ret = VideoDecoder::GetInstance().stop_recv_stream(chn);
  if (ret != 0) {
    MS_EXCEPTION(RuntimeError) << "Failed to call hi_mpi_vdec_stop_recv_stream on chn " << chn << ", ret: " << ret;
  }

  ret = VideoDecoder::GetInstance().reset_chn(chn);
  if (ret != 0) {
    MS_EXCEPTION(RuntimeError) << "Failed to call hi_mpi_vdec_reset_chn on chn " << chn << ", ret: " << ret;
  }

  ret = VideoDecoder::GetInstance().start_recv_stream(chn);
  if (ret != 0) {
    MS_EXCEPTION(RuntimeError) << "Failed to call hi_mpi_vdec_start_recv_stream on chn " << chn << ", ret: " << ret;
  }
}

static thread_local int32_t local_device = -1;
static std::unordered_map<int32_t, aclrtContext> used_devices;
std::recursive_mutex mtx;

aclError SetDevice(int32_t device) {
  if (device < 0) {
    MS_EXCEPTION(RuntimeError) << "Device id must be greater than 0.";
  }

  if (local_device == device) {
    return ACL_SUCCESS;
  }

  aclError err = aclrtSetDevice(device);
  if (err == ACL_SUCCESS) {
    local_device = device;
    std::lock_guard<std::recursive_mutex> lock(mtx);
    if (used_devices.find(local_device) == used_devices.end()) {
      aclError ret = aclrtGetCurrentContext(&used_devices[local_device]);
      if (ret != ACL_SUCCESS) {
        MS_EXCEPTION(RuntimeError) << "Failed to call aclrtGetCurrentContext, ret: " << ret;
      }
    }
  }
  return err;
}

void *get_pic(void *args) {
  prctl(PR_SET_NAME, "VdecGetPic", 0, 0, 0);
  GetThreadPara *para = reinterpret_cast<GetThreadPara *>(args);
  if (para == nullptr) {
    MS_EXCEPTION(RuntimeError) << "GetThreadPara must not be nullptr.";
  }
  uint32_t chn_id = para->chn_id;
  aclError device_ret = SetDevice(para->device_id);
  if (device_ret != ACL_SUCCESS) {
    MS_EXCEPTION(RuntimeError) << "Failed to set device, ret: " << device_ret;
  }

  int32_t ret = HI_SUCCESS;
  hi_video_frame_info frame{};
  hi_vdec_stream stream{};
  int32_t dec_result = VDEC_DECODE_SUCCESS;
  hi_u64 output_buffer = 0;
  uint32_t success_cnt = 0;
  uint32_t fail_cnt = 0;
  int32_t time_out = 0;

  auto out_queue = std::vector<std::shared_ptr<DeviceBuffer>>(para->total_frame);
  thread_exit_state_list[chn_id] = 0;

  while (thread_exit_state_list[chn_id] == 0) {
    ret = VideoDecoder::GetInstance().get_frame(chn_id, &frame, nullptr, &stream, time_out);
    if (ret == HI_SUCCESS) {
      output_buffer = static_cast<hi_u64>(reinterpret_cast<uintptr_t>(frame.v_frame.virt_addr[0]));
      dec_result = frame.v_frame.frame_flag;
      if (dec_result == VDEC_DECODE_SUCCESS) {  // 0: Decode success
        const std::lock_guard<std::mutex> guard(out_buffer_map_mutex[chn_id]);
        auto iter = out_buffer_map[chn_id].find(output_buffer);
        if (iter != out_buffer_map[chn_id].end()) {
          out_queue[success_cnt] = iter->second;
          out_buffer_map[chn_id].erase(iter);
          success_cnt++;
        }
      } else if (dec_result == VDEC_DECODE_FAILED) {  // 1: Decode fail
        fail_cnt++;
        MS_LOG(WARNING) << "Get a frame that failed to decode on chn " << chn_id
                        << ", total count of failures: " << fail_cnt;
      } else if (dec_result == VDEC_DECODE_INTERLACED_FIELD_STREAM) {
        // 2: This result is returned for the second field of the interlaced field stream, which is normal.
      } else if (dec_result == VDEC_DECODE_FRAME_NUMBER_ERROR) {  // 3: Reference frame number setting error
        fail_cnt++;
        MS_LOG(WARNING) << "Get a frame with reference number error on chn " << chn_id
                        << ", total count of failures: " << fail_cnt;
      } else if (dec_result == VDEC_DECODE_FRAME_SIZE_ERROR) {  // 4: Reference frame size setting error
        fail_cnt++;
        MS_LOG(WARNING) << "Get a frame with reference size error on chn " << chn_id
                        << ", total count of failures: " << fail_cnt;
      }
      // Release Frame
      ret = VideoDecoder::GetInstance().release_frame(chn_id, &frame);
      if (ret != 0) {
        MS_EXCEPTION(RuntimeError) << "Failed to call hi_mpi_vdec_release_frame on chn " << chn_id << ", ret: " << ret;
      }
    } else {
      usleep(500);  // 500us
    }
  }

  out_buffer_queue[chn_id] = out_queue;
  para->success_cnt = success_cnt;
  return (void *)HI_SUCCESS;
}

int64_t dvpp_sys_init() { return static_cast<int64_t>(VideoDecoder::GetInstance().sys_init()); }

int64_t dvpp_sys_exit() { return static_cast<int64_t>(VideoDecoder::GetInstance().sys_exit()); }

int64_t dvpp_vdec_create_chnl(int64_t payload_type) {
  if (payload_type != 96 && payload_type != 265) {  // H264:96 / H265:265
    MS_EXCEPTION(RuntimeError) << "Invalid payload type " << payload_type << ", only supports H264:96 or H265:265.";
  }
  uint32_t chn = 0;
  int32_t ret = VideoDecoder::GetInstance().GetUnusedChn(chn);
  if (ret != 0) {
    MS_EXCEPTION(RuntimeError) << "Failed to get unused chn.";
  }

  hi_vdec_chn_attr chn_attr{};
  chn_attr.type = static_cast<hi_payload_type>(payload_type);
  chn_attr.mode = HI_VDEC_SEND_MODE_FRAME;  // Only support frame mode
  chn_attr.pic_width = MAX_CHN_WIDTH;
  chn_attr.pic_height = MAX_CHN_HEIGHT;
  chn_attr.stream_buf_size = static_cast<hi_u32>(MAX_CHN_WIDTH * MAX_CHN_HEIGHT * BUF_SIZE_FACTOR);
  chn_attr.frame_buf_cnt = FRAME_BUF_CNT;
  hi_pic_buf_attr buf_attr{
    MAX_CHN_WIDTH, MAX_CHN_HEIGHT, 0, HI_DATA_BIT_WIDTH_10, HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420, HI_COMPRESS_MODE_NONE};
  chn_attr.frame_buf_size = VideoDecoder::GetInstance().get_pic_buf_size(chn_attr.type, &buf_attr);
  chn_attr.video_attr.ref_frame_num = REF_FRAME_NUM;
  chn_attr.video_attr.temporal_mvp_en = HI_TRUE;
  chn_attr.video_attr.tmv_buf_size =
    VideoDecoder::GetInstance().get_tmv_buf_size(chn_attr.type, MAX_CHN_WIDTH, MAX_CHN_HEIGHT);

  ret = VideoDecoder::GetInstance().create_chn(chn, &chn_attr);
  if (ret != HI_SUCCESS) {
    VideoDecoder::GetInstance().PutChn(chn);
    MS_EXCEPTION(RuntimeError) << "Failed to call hi_mpi_vdec_create_chn on chn " << chn << ", ret: " << ret;
    return -1;
  }

  ret = VideoDecoder::GetInstance().sys_set_chn_csc_matrix(chn);
  if (ret != HI_SUCCESS) {
    (void)VideoDecoder::GetInstance().destroy_chn(chn);
    VideoDecoder::GetInstance().PutChn(chn);
    MS_EXCEPTION(RuntimeError) << "Failed to call hi_mpi_sys_set_chn_csc_matrix on chn " << chn << ", ret: " << ret;
    return -1;
  }

  ret = VideoDecoder::GetInstance().start_recv_stream(chn);
  if (ret != HI_SUCCESS) {
    (void)VideoDecoder::GetInstance().destroy_chn(chn);
    VideoDecoder::GetInstance().PutChn(chn);
    MS_EXCEPTION(RuntimeError) << "Failed to call hi_mpi_vdec_start_recv_stream on chn " << chn << ", ret: " << ret;
    return -1;
  }

  return static_cast<int64_t>(chn);
}

aclError GetDevice(int32_t *device) {
  if (device == nullptr) {
    MS_EXCEPTION(RuntimeError) << "Input device must not be nullptr.";
  }
  if (local_device >= 0) {
    *device = local_device;
    return ACL_SUCCESS;
  }
  aclError err = aclrtGetDevice(device);
  if (err != ACL_SUCCESS) {
    MS_EXCEPTION(RuntimeError) << "Failed to call aclrtGetDevice, ret: " << err;
  }
  if (err == ACL_SUCCESS) {
    local_device = *device;
  } else if (err == ACL_ERROR_RT_CONTEXT_NULL && aclrtSetDevice(0) == ACL_SUCCESS) {
    *device = 0;
    local_device = 0;
    std::lock_guard<std::recursive_mutex> lock(mtx);
    if (used_devices.find(local_device) == used_devices.end()) {
      auto ret = aclrtGetCurrentContext(&used_devices[local_device]);
      if (ret != ACL_SUCCESS) {
        MS_EXCEPTION(RuntimeError) << "Failed to call aclrtGetCurrentContext, ret: " << ret;
      }
    }
    return ACL_SUCCESS;
  }
  return err;
}

int64_t dvpp_vdec_start_get_frame(int64_t chn_id, int64_t total_frame) {
  if (!ValidChnNum(chn_id)) {
    MS_EXCEPTION(RuntimeError) << "Chn " << chn_id << " is invalid.";
  }

  int32_t device_id = 0;
  aclError acl_ret = GetDevice(&device_id);
  if (acl_ret != 0) {
    MS_EXCEPTION(RuntimeError) << "Failed to get device id, ret: " << acl_ret;
  }

  get_thread_para[chn_id].chn_id = chn_id;
  get_thread_para[chn_id].device_id = device_id;
  get_thread_para[chn_id].total_frame = total_frame;
  get_thread_para[chn_id].success_cnt = 0;
  vdec_thread_list[chn_id] = 0;
  int32_t ret = pthread_create(&vdec_thread_list[chn_id], 0, get_pic, static_cast<void *>(&get_thread_para[chn_id]));
  if (ret != 0) {
    vdec_thread_list[chn_id] = 0;
    MS_EXCEPTION(RuntimeError) << "Failed to create VdecGetPic thread on chn " << chn_id << ", ret: " << ret;
    return -1;
  }

  return 0;
}

int64_t dvpp_vdec_send_stream(int64_t chn_id, const std::shared_ptr<DeviceBuffer> &input, int64_t out_format,
                              bool display, std::shared_ptr<DeviceBuffer> *out) {
  if (input == nullptr) {
    MS_EXCEPTION(RuntimeError) << "Input device buffer must not be nullptr.";
  }
  if (out == nullptr || (*out) == nullptr) {
    MS_EXCEPTION(RuntimeError) << "Output device buffer must not be nullptr.";
  }
  if (!ValidChnNum(chn_id)) {
    MS_EXCEPTION(RuntimeError) << "Chn " << chn_id << " is invalid.";
  }
  hi_pixel_format output_format = static_cast<hi_pixel_format>(out_format);
  if (output_format != HI_PIXEL_FORMAT_RGB_888 && output_format != HI_PIXEL_FORMAT_BGR_888 &&
      output_format != HI_PIXEL_FORMAT_RGB_888_PLANAR && output_format != HI_PIXEL_FORMAT_BGR_888_PLANAR) {
    MS_EXCEPTION(RuntimeError) << "Invalid output format: " << output_format << ", only supports "
                               << HI_PIXEL_FORMAT_RGB_888 << ", " << HI_PIXEL_FORMAT_BGR_888 << ", "
                               << HI_PIXEL_FORMAT_RGB_888_PLANAR << " or " << HI_PIXEL_FORMAT_BGR_888_PLANAR;
  }

  int64_t input_size_bytes = input->GetBufferSize();

  int64_t output_size_bytes = (*out)->GetBufferSize();

  hi_vdec_stream stream{};
  uint64_t current_send_time = 0;
  get_current_time_us(current_send_time);
  stream.pts = current_send_time;
  stream.addr = static_cast<hi_u8 *>(input->GetBuffer());
  stream.len = input_size_bytes;
  stream.end_of_frame = HI_TRUE;
  stream.end_of_stream = HI_FALSE;
  stream.need_display = display ? HI_TRUE : HI_FALSE;

  hi_vdec_pic_info out_pic_info{};
  out_pic_info.height = 0;
  out_pic_info.width = 0;
  out_pic_info.width_stride = 0;
  out_pic_info.height_stride = 0;
  out_pic_info.pixel_format = output_format;
  out_pic_info.vir_addr = 0;
  out_pic_info.buffer_size = 0;
  if (display) {
    out_pic_info.vir_addr = static_cast<hi_u64>(reinterpret_cast<uintptr_t>((*out)->GetBuffer()));
    out_pic_info.buffer_size = output_size_bytes;
  }

  uint32_t send_one_frame_cnt = 0;
  int32_t ret = 0;
  do {
    send_one_frame_cnt++;
    // Send one frame data
    ret = VideoDecoder::GetInstance().send_stream(chn_id, &stream, &out_pic_info, SEND_TIMEOUT);
    if (send_one_frame_cnt > 30) {  // if send stream timeout 30 times, end the decode process
      if (ret != 0) {
        vdec_reset_chn(chn_id);
      }
      break;
    }
  } while (ret == HI_ERR_VDEC_BUF_FULL);  // Try again
  if (ret != 0) {
    MS_EXCEPTION(RuntimeError) << "Failed to call hi_mpi_vdec_send_stream on chn " << chn_id << ", ret: " << ret;
  }

  if (display) {
    const std::lock_guard<std::mutex> guard(out_buffer_map_mutex[chn_id]);
    out_buffer_map[chn_id].insert(
      std::make_pair(static_cast<hi_u64>(reinterpret_cast<uintptr_t>((*out)->GetBuffer())), *out));
  }

  return 0;
}

std::shared_ptr<DeviceBuffer> dvpp_vdec_stop_get_frame(int64_t chn_id, int64_t total_frame) {
  hi_vdec_chn_status status{};
  hi_vdec_chn_status pre_status{};

  hi_vdec_stream stream{};
  hi_vdec_pic_info out_pic_info{};
  // Send stream end flag
  stream.addr = NULL;
  stream.len = 0;
  stream.end_of_frame = HI_FALSE;
  stream.end_of_stream = HI_TRUE;  // Stream end flag
  out_pic_info.vir_addr = 0;
  out_pic_info.buffer_size = 0;
  int32_t ret = VideoDecoder::GetInstance().send_stream(chn_id, &stream, &out_pic_info, -1);
  if (ret != 0) {
    MS_EXCEPTION(RuntimeError) << "Failed to call hi_mpi_vdec_send_stream on chn " << chn_id << ", ret: " << ret;
  }

  uint32_t wait_times = 0;
  uint32_t sleep_time = 10000;  // 10000us
  ret = VideoDecoder::GetInstance().stop_recv_stream(chn_id);
  if (ret != 0) {
    MS_EXCEPTION(RuntimeError) << "Failed to call hi_mpi_vdec_stop_recv_stream on chn " << chn_id << ", ret: " << ret;
  }

  while (wait_times < WAIT_TIMEOUT) {
    ret = VideoDecoder::GetInstance().query_status(chn_id, &status);
    if (ret != 0) {
      MS_EXCEPTION(RuntimeError) << "Failed to call hi_mpi_vdec_query_status on chn " << chn_id << ", ret: " << ret;
    }
    if (((status.left_stream_bytes == 0) && (status.left_decoded_frames == 0))) {
      break;
    }
    if (status.left_decoded_frames == pre_status.left_decoded_frames) {
      wait_times += sleep_time;
    } else {
      wait_times = 0;
    }
    pre_status = status;
    // 10000us
    usleep(sleep_time);

    if (wait_times >= WAIT_TIMEOUT) {
      vdec_reset_chn(chn_id);
      break;
    }
  }

  thread_exit_state_list[chn_id] = 1;  // notify get thread exit

  ret = pthread_join(vdec_thread_list[chn_id], nullptr);
  if (ret != 0) {
    MS_EXCEPTION(RuntimeError) << "Failed to join VdecGetPic thread on chn " << chn_id << ", ret: " << ret;
  }
  vdec_thread_list[chn_id] = 0;

  // all frame success or no frame success
  if (get_thread_para[chn_id].success_cnt == total_frame || get_thread_para[chn_id].success_cnt == 0) {
    out_buffer_queue[chn_id].clear();
    out_buffer_map[chn_id].clear();
    std::vector<size_t> shape = {0};
    return std::make_shared<DeviceBuffer>(shape);
  }

  // some frame failed
  std::shared_ptr<DeviceBuffer> buffer = out_buffer_queue[chn_id][0];
  if (buffer == nullptr) {
    MS_EXCEPTION(RuntimeError) << "Output device buffer must not be nullptr.";
  }
  auto new_shape = buffer->GetShape();
  (void)new_shape.insert(new_shape.begin(), get_thread_para[chn_id].success_cnt);
  auto result_tensor = std::make_shared<DeviceBuffer>(new_shape);
  for (int i = 0; i < get_thread_para[chn_id].success_cnt; i++) {
    if (out_buffer_queue[chn_id][i] == nullptr) {
      MS_EXCEPTION(RuntimeError) << "Output device buffer " << i << " must not be nullptr.";
    }
    std::shared_ptr<DeviceBuffer> dest = std::make_shared<DeviceBuffer>(result_tensor, i);
    auto acl_ret = aclrtMemcpy(dest->GetBuffer(), dest->GetBufferSize(), out_buffer_queue[chn_id][i]->GetBuffer(),
                               out_buffer_queue[chn_id][i]->GetBufferSize(), ACL_MEMCPY_DEVICE_TO_DEVICE);
    if (acl_ret != 0) {
      MS_EXCEPTION(RuntimeError) << "Failed to call aclrtMemcpy, ret: " << acl_ret;
    }
  }

  out_buffer_queue[chn_id].clear();
  out_buffer_map[chn_id].clear();
  return result_tensor;
}

int64_t dvpp_vdec_destroy_chnl(int64_t chn_id) {
  int32_t ret = VideoDecoder::GetInstance().destroy_chn(chn_id);
  VideoDecoder::GetInstance().PutChn(chn_id);
  if (ret != 0) {
    MS_EXCEPTION(RuntimeError) << "Failed to call hi_mpi_vdec_destroy_chn on chn " << chn_id << ", ret: " << ret;
  }
  return 0;
}

int64_t dvpp_memcpy(void *dst, size_t dest_max, const void *src, size_t count, int kind) {
  return aclrtMemcpy(dst, dest_max, src, count, static_cast<aclrtMemcpyKind>(kind));
}
}  // namespace dataset
}  // namespace mindspore
