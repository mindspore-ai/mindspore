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

#include "minddata/dataset/kernels/image/soft_dvpp/utils/soft_dp.h"
#include "minddata/dataset/kernels/image/soft_dvpp/utils/soft_dp_check.h"
#include "minddata/dataset/kernels/image/soft_dvpp/utils/soft_jpegd.h"
#include "minddata/dataset/kernels/image/soft_dvpp/utils/soft_vpc.h"
#include <thread>
const int32_t decodeSucc = 0;
const int32_t checkParamErr = 1;
const int32_t num2 = 2;

uint32_t DecodeAndResizeJpeg(SoftDpProcsessInfo *soft_dp_process_info) {
  if (soft_dp_process_info == nullptr || soft_dp_process_info->input_buffer == nullptr ||
      soft_dp_process_info->input_buffer_size <= 0 || soft_dp_process_info->output_buffer == nullptr ||
      soft_dp_process_info->output_buffer_size <= 0) {
    API_LOGE("The input buffer or out buffer is null or size is 0");
    return checkParamErr;
  }
  // height and width must be even
  if (soft_dp_process_info->output_width % 2 == 1 || soft_dp_process_info->output_height % 2 == 1) {
    API_LOGE("odd width and height dose not support in resize interface");
    return checkParamErr;
  }
  VpcInfo vpc_input_info;
  SoftJpegd soft_handler;
  int32_t ret = soft_handler.JpegdSoftwareDecodeProcess(&vpc_input_info, soft_dp_process_info);
  if (ret != decodeSucc) {
    API_LOGE("Jpegd decode fail in resize interface.");
    return ret;
  }

  // use vpc interface to resize and convert RGB, give user output buf and output size.
  auto crop = SoftDpCropInfo{.left = 0,
                             .right = static_cast<uint32_t>(vpc_input_info.real_width - 1),
                             .up = 0,
                             .down = static_cast<uint32_t>(vpc_input_info.real_height - 1)};

  VpcInfo output;
  output.addr = soft_dp_process_info->output_buffer;
  output.width = soft_dp_process_info->output_width;
  output.height = soft_dp_process_info->output_height;
  SoftVpc soft_vpc;
  ret = soft_vpc.Process(vpc_input_info, crop, output);
  return ret;
}

uint32_t DecodeAndCropAndResizeJpeg(SoftDpProcsessInfo *soft_dp_process_info, const SoftDpCropInfo &crop_info) {
  if (soft_dp_process_info == nullptr || soft_dp_process_info->input_buffer == nullptr ||
      soft_dp_process_info->input_buffer_size <= 0 || soft_dp_process_info->output_buffer == nullptr ||
      soft_dp_process_info->output_buffer_size <= 0) {
    API_LOGE("The input buffer or out buffer is null or size is 0");
    return checkParamErr;
  }
  // height and width must be even
  if (soft_dp_process_info->output_width % 2 == 1 || soft_dp_process_info->output_height % 2 == 1) {
    API_LOGE("odd width and height dose not support in crop and resize interface");
    return checkParamErr;
  }
  VpcInfo vpc_input_info;
  SoftJpegd soft_handler;

  int32_t ret = soft_handler.JpegdSoftwareDecodeProcess(&vpc_input_info, soft_dp_process_info);
  if (ret != decodeSucc) {
    API_LOGE("Jpegd decode fail in crop and resize interface.");
    return ret;
  }

  // use vpc interface to resize and crop and convert RGB, give user output buf and output size.
  VpcInfo output;
  output.addr = soft_dp_process_info->output_buffer;
  output.width = soft_dp_process_info->output_width;
  output.height = soft_dp_process_info->output_height;
  SoftDpCropInfo crop = crop_info;

  if ((vpc_input_info.real_width % num2 == 1) && ((uint32_t)vpc_input_info.real_width == crop.right)) {
    API_LOGD("crop width is equal the real width.");
    crop.right = vpc_input_info.real_width - 1;
  }

  if ((vpc_input_info.real_height % num2 == 1) && ((uint32_t)vpc_input_info.real_height == crop.down)) {
    API_LOGD("crop height is equal the real height.");
    crop.down = vpc_input_info.real_height - 1;
  }
  SoftVpc soft_vpc;
  ret = soft_vpc.Process(vpc_input_info, crop, output);
  return ret;
}
