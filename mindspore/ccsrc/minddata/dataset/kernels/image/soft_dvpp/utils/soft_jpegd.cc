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

#include "minddata/dataset/kernels/image/soft_dvpp/utils/soft_jpegd.h"
#include "minddata/dataset/kernels/image/soft_dvpp/utils/soft_dp_log.h"
#include "minddata/dataset/kernels/image/soft_dvpp/utils/soft_dp_tools.h"
#include "minddata/dataset/kernels/image/soft_dvpp/utils/soft_dp_check.h"
#include <turbojpeg.h>
#include <securec.h>

#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>

const uint32_t yuv400UvValue = 0x80;
const int32_t num2 = 2;
const uint32_t channel3 = 3;
const uint32_t zeroBufSize = 0;
const int32_t decodePadding = 1;
const int32_t minValue = 32;
const int32_t maxValue = 8192;
const int32_t decodeSucc = 0;
const int32_t decodeErr = 1;

SoftJpegd::SoftJpegd() : soft_decode_out_buf_(nullptr) {}

/*
 * @brief : Use libjpeg to determine the image format.
 * @param [in] jpeg_decompress_struct& libjpeg_handler : libjpeg
 * @param [in] VpcInfo& vpc_input_info : vpc input information
 */
void SetFormat(struct jpeg_decompress_struct *libjpeg_handler, struct VpcInfo *vpc_input_info) {
  // yuv400: component 1 1x1
  // yuv420: component 3 2x2 1x1 1x1
  // yuv422: component 3 2x1 1x1 1x1
  // yuv444: component 3 1x1 1x1 1x1
  if ((libjpeg_handler->num_components == 1) &&
      (libjpeg_handler->comp_info[0].h_samp_factor == libjpeg_handler->comp_info[0].v_samp_factor)) {
    vpc_input_info->format = INPUT_YUV420_PLANNER;
    vpc_input_info->is_fake420 = true;
  } else if ((libjpeg_handler->num_components == channel3) &&
             (libjpeg_handler->comp_info[1].h_samp_factor == libjpeg_handler->comp_info[2].h_samp_factor) &&
             (libjpeg_handler->comp_info[1].v_samp_factor == libjpeg_handler->comp_info[2].v_samp_factor)) {
    if (libjpeg_handler->comp_info[0].h_samp_factor == ((libjpeg_handler->comp_info[1].h_samp_factor) * num2)) {
      if (libjpeg_handler->comp_info[0].v_samp_factor == ((libjpeg_handler->comp_info[1].v_samp_factor) * num2)) {
        vpc_input_info->format = INPUT_YUV420_PLANNER;
      } else if (libjpeg_handler->comp_info[0].v_samp_factor == libjpeg_handler->comp_info[1].v_samp_factor) {
        vpc_input_info->format = INPUT_YUV422_PLANNER;
      }
    } else if (libjpeg_handler->comp_info[0].h_samp_factor == libjpeg_handler->comp_info[1].h_samp_factor) {
      if (libjpeg_handler->comp_info[0].v_samp_factor == libjpeg_handler->comp_info[1].v_samp_factor) {
        vpc_input_info->format = INPUT_YUV444_PLANNER;
      }
    }
  }
}

static void LibjpegErrorExit(j_common_ptr cinfo) {
  char jpegLastErrorMsg[JMSG_LENGTH_MAX];
  (*(cinfo->err->format_message))(cinfo, jpegLastErrorMsg);
  JPEGD_LOGE("run libjpeg get error : %s", jpegLastErrorMsg);
  throw std::runtime_error(jpegLastErrorMsg);
}

bool CallLibjpeg(struct jpeg_decompress_struct *libjpeg_handler, uint8_t *addr, uint32_t size) {
  struct jpeg_error_mgr libjpegErrorMsg;
  libjpeg_handler->err = jpeg_std_error(&libjpegErrorMsg);
  libjpegErrorMsg.error_exit = LibjpegErrorExit;

  try {
    jpeg_mem_src(libjpeg_handler, addr, size);
    jpeg_read_header(libjpeg_handler, TRUE);
    return true;
  } catch (...) {
    return false;
  }
}

/*
 * @brief : Obtains the JPEG header information through libjpeg to complete the decoding preparation process.
 * @param [in] jpeg_decompress_struct& libjpeg_handler : libjpeg.
 * @param [in] VpcInfo& vpc_input_info : vpc input information.
 * @param [in] SoftDpProcsessInfo& dp_soft_process_info : soft dp struct.
 * @return : decodeSucc：parse jpeg head succ， decodeErr:parse jpeg head fail.
 */
uint32_t PrepareDecode(jpeg_decompress_struct *libjpeg_handler, struct VpcInfo *vpc_input_info,
                       struct SoftDpProcsessInfo *dp_soft_process_info) {
  bool call_libjpeg_succ =
    CallLibjpeg(libjpeg_handler, dp_soft_process_info->input_buffer, dp_soft_process_info->input_buffer_size);
  if (!call_libjpeg_succ) {
    JPEGD_LOGE("CallLibjpeg failed!");
    return decodeErr;
  }

  SetFormat(libjpeg_handler, vpc_input_info);
  return decodeSucc;
}

/*
 * @brief : Check the parameters. The width and height range are as follows: [32,8192]
 * @param [in] int32_t height : image height
 * @param [in] int32_t width : image width
 * @return : decodeSucc：params are valid， decodeErr:params are invalid.
 */
uint32_t CheckInputParam(int32_t height, int32_t width) {
  JPEGD_CHECK_COND_FAIL_PRINT_RETURN((width >= minValue), decodeErr, "width(%d) should be >= 32.", width);
  JPEGD_CHECK_COND_FAIL_PRINT_RETURN((width <= maxValue), decodeErr, "width(%d) should be <= 8192.", width);
  JPEGD_CHECK_COND_FAIL_PRINT_RETURN((height >= minValue), decodeErr, "height(%d) should be >= 32.", height);
  JPEGD_CHECK_COND_FAIL_PRINT_RETURN((height <= maxValue), decodeErr, "height(%d) should be <= 8192.", height);
  return decodeSucc;
}

uint32_t SoftJpegd::AllocOutputBuffer(struct VpcInfo *vpc_input_info, int32_t *width, int32_t *height,
                                      int32_t *sub_sample) {
  CheckInputParam(*height, *width);
  uint32_t output_size = tjBufSizeYUV2(*width, decodePadding, *height, *sub_sample);
  JPEGD_LOGD("In this case the format= %d, output size=%d, real width=%d, the real height=%d, thread_id=%lu.",
             vpc_input_info->format, output_size, *width, *height, std::this_thread::get_id());
  if (output_size == zeroBufSize) {
    JPEGD_LOGE("get outbuffer size failed!");
    return decodeErr;
  }

  if (vpc_input_info->is_fake420) {
    *width = AlignUp(*width, num2);
    *height = AlignUp(*height, num2);
    output_size = (*width) * (*height) * channel3 / num2;
  }

  soft_decode_out_buf_ = new (std::nothrow) uint8_t[output_size];
  if (soft_decode_out_buf_ == nullptr) {
    JPEGD_LOGE("alloc outbuffer failed!");
    return decodeErr;
  }

  return decodeSucc;
}

uint32_t SoftJpegd::ConfigVpcInputData(struct VpcInfo *vpc_input_info, int32_t *width, int32_t *height) {
  vpc_input_info->real_height = *height;
  vpc_input_info->real_width = *width;

  if ((vpc_input_info->format == INPUT_YUV420_PLANNER || vpc_input_info->format == INPUT_YUV422_PLANNER) &&
      (*width % num2 == 1)) {
    *width = reinterpret_cast<int32_t>(AlignUp(*width, num2));
    JPEGD_LOGW("vpc width needs align up %d, height is %d.", *width, *height);
  }

  if ((vpc_input_info->format == INPUT_YUV420_PLANNER || vpc_input_info->format == INPUT_YUV422_PLANNER) &&
      (*height % num2 == 1)) {
    *height = reinterpret_cast<int32_t>(AlignUp(*height, num2));
    JPEGD_LOGW("vpc height needs align up %d, height is %d.", *width, *height);
  }

  vpc_input_info->addr = soft_decode_out_buf_;
  vpc_input_info->height = *height;
  vpc_input_info->width = *width;

  if (vpc_input_info->is_fake420) {
    uint8_t *u_start = vpc_input_info->addr + vpc_input_info->width * vpc_input_info->height;
    int32_t uv_size = vpc_input_info->width * vpc_input_info->height / num2;
    int32_t safe_ret = memset_s(reinterpret_cast<void *>((uintptr_t)u_start), uv_size, yuv400UvValue, uv_size);
    if (safe_ret != 0) {
      JPEGD_LOGE("config yuv400 uv memory failed.addr = 0x%llx, thread id = %lu", soft_decode_out_buf_,
                 std::this_thread::get_id());
      delete[] soft_decode_out_buf_;
      soft_decode_out_buf_ = nullptr;
      vpc_input_info->addr = nullptr;
      return decodeErr;
    }
  }

  return decodeSucc;
}

/*
 * @brief : destroy libjpeg source
 * @param [in] struct jpeg_decompress_struct &libjpeg_handler : libjpeg handle.
 * @param [in] tjhandle &handle : tjhandle.
 */
void DestroyLibjpegSource(struct jpeg_decompress_struct *libjpeg_handler, const tjhandle &handle) {
  (void)tjDestroy(handle);
  jpeg_destroy_decompress(libjpeg_handler);
}

uint32_t SoftJpegd::JpegdSoftwareDecodeProcess(struct VpcInfo *vpc_input_info,
                                               struct SoftDpProcsessInfo *soft_dp_process_info) {
  int32_t width = 0;
  int32_t height = 0;
  int32_t sub_sample = 0;
  int32_t color_spase = 0;
  struct jpeg_decompress_struct libjpeg_handler;
  jpeg_create_decompress(&libjpeg_handler);
  tjhandle handle = tjInitDecompress();
  int32_t prepare_decode_res = PrepareDecode(&libjpeg_handler, vpc_input_info, soft_dp_process_info);
  if (prepare_decode_res != decodeSucc) {
    JPEGD_LOGE("prepare decode failed!");
    DestroyLibjpegSource(&libjpeg_handler, handle);
    return decodeErr;
  }

  int32_t decode_header_res =
    tjDecompressHeader3(handle, soft_dp_process_info->input_buffer, soft_dp_process_info->input_buffer_size, &width,
                        &height, &sub_sample, &color_spase);
  if (decode_header_res != decodeSucc) {
    JPEGD_LOGE("Decompress header failed, width = %d, height = %d.", width, height);
    DestroyLibjpegSource(&libjpeg_handler, handle);
    return decodeErr;
  }

  int32_t alloc_out_buf_res = AllocOutputBuffer(vpc_input_info, &width, &height, &sub_sample);
  if (alloc_out_buf_res != decodeSucc) {
    JPEGD_LOGE("alloc output buffer failed!");
    DestroyLibjpegSource(&libjpeg_handler, handle);
    return decodeErr;
  }

  int32_t decode_res =
    tjDecompressToYUV2(handle, soft_dp_process_info->input_buffer, soft_dp_process_info->input_buffer_size,
                       soft_decode_out_buf_, width, decodePadding, height, JDCT_ISLOW);
  if (decode_res != decodeSucc) {
    JPEGD_LOGE("Decompress jpeg failed, addr is 0x%llx, thread id= %lu.", soft_decode_out_buf_,
               std::this_thread::get_id());
    delete[] soft_decode_out_buf_;
    soft_decode_out_buf_ = nullptr;
    DestroyLibjpegSource(&libjpeg_handler, handle);
    return decodeErr;
  }

  int32_t config_vpc_res = ConfigVpcInputData(vpc_input_info, &width, &height);
  if (config_vpc_res != decodeSucc) {
    DestroyLibjpegSource(&libjpeg_handler, handle);
    return decodeErr;
  }
  DestroyLibjpegSource(&libjpeg_handler, handle);
  return decodeSucc;
}
