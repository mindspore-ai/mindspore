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

const int32_t decodeSucc = 0;

uint32_t DecodeAndResizeJpeg(SoftDpProcsessInfo *soft_dp_process_info) {
  VpcInfo vpc_input_info;
  SoftJpegd soft_handler;
  int32_t ret = soft_handler.JpegdSoftwareDecodeProcess(&vpc_input_info, soft_dp_process_info);
  if (ret != decodeSucc) {
    API_LOGE("Jpegd decode fail in resize interface.");
    return ret;
  }

  // use vpc interface to resize and convert RGB, give user output buf and output size.
  SoftDpCropInfo crop;
  crop.left = 0;
  crop.right = vpc_input_info.real_width - 1;
  crop.up = 0;
  crop.down = vpc_input_info.real_height - 1;

  VpcInfo output;
  output.addr = soft_dp_process_info->output_buffer;
  output.width = soft_dp_process_info->output_width;
  output.height = soft_dp_process_info->output_height;
  SoftVpc soft_vpc;
  ret = soft_vpc.Process(vpc_input_info, crop, output);
  return ret;
}

uint32_t DecodeAndCropAndResizeJpeg(SoftDpProcsessInfo *soft_dp_process_info, const SoftDpCropInfo &crop_info) {
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

  SoftVpc soft_vpc;
  ret = soft_vpc.Process(vpc_input_info, crop, output);
  return ret;
}
