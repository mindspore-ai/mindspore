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

#ifndef SOFT_JPEGD_H
#define SOFT_JPEGD_H

#include <stdint.h>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include "./jpeglib.h"
#include "minddata/dataset/kernels/image/soft_dvpp/utils/soft_dp.h"
#include "minddata/dataset/kernels/image/soft_dvpp/utils/external_soft_dp.h"

class SoftJpegd {
 public:
  SoftJpegd();

  ~SoftJpegd() {}

  /*
   * @brief : decode interface
   * @param [in] VpcInfo& vpc_input_info : vpc input information
   * @param [in] SoftDpProcsessInfo& soft_dp_process_info : softDp process info
   * @return : decodeSucc：decode success, decodeErr:decode failed.
   */
  uint32_t JpegdSoftwareDecodeProcess(struct VpcInfo *vpc_input_info, struct SoftDpProcsessInfo *soft_dp_process_info);

 private:
  uint8_t *soft_decode_out_buf_;

  /*
   * @brief : alloc output buffer
   * @param [in] VpcInfo& vpc_input_info : vpc input information
   * @param [in] int32_t& width : output width
   * @param [in] int32_t& height : output height
   * @param [in] int32_t& sub_sample : level of chrominance subsampling in the image
   * @param [in] int32_t& color_spase : pointer to an integer variable that will receive one of the JPEG
   *                                   constants, indicating the colorspace of the JPEG image.
   * @return : decodeSucc：alloc output buf success, decodeErr:alloc output buf failed.
   */
  uint32_t AllocOutputBuffer(struct VpcInfo *vpc_input_info, int32_t *width, int32_t *height, int32_t *sub_sample);

  /*
   * @brief : config decode output
   * @param [in] VpcInfo& vpc_input_info : vpc input information
   * @param [in] int32_t& width : output width
   * @param [in] int32_t& height : output height
   * @return : decodeSucc：config output buf success, decodeErr:config output buf failed.
   */
  uint32_t ConfigVpcInputData(struct VpcInfo *vpc_input_info, int32_t *width, int32_t *height);
};

#endif  // SOFT_JPEGD_H
