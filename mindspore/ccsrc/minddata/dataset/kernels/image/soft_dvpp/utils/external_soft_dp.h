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

#ifndef EXTERNAL_SOFTDP_H
#define EXTERNAL_SOFTDP_H

#include <stdint.h>

struct SoftDpProcsessInfo {
  uint8_t *input_buffer;       // input buffer
  uint32_t input_buffer_size;  // input buffer size

  uint8_t *output_buffer;       // output buffer
  uint32_t output_buffer_size;  // output buffer size

  uint32_t output_width;   // output width
  uint32_t output_height;  // output height

  bool is_v_before_u;  // uv : true, uv : false
};

struct SoftDpCropInfo {
  uint32_t left;   // crop left boundary
  uint32_t right;  // crop right boundary
  uint32_t up;     // crop up boundary
  uint32_t down;   // crop down boundary
};

/*
 * @brief decode and resize image
 * @param [in] SoftDpProcsessInfo& soft_dp_process_info: soft decode process struct
 * @return success: return 0, fail: return error number
 */
uint32_t DecodeAndResizeJpeg(SoftDpProcsessInfo *soft_dp_process_info);

/*
 * @brief decode and crop and resize image
 * @param [in] SoftDpProcsessInfo& soft_dp_process_info: soft decode process struct
 * @param [in] SoftDpCropInfo& crop_info: user crop info
 * @return success: return 0, fail: return error number
 */
uint32_t DecodeAndCropAndResizeJpeg(SoftDpProcsessInfo *soft_dp_process_info, const SoftDpCropInfo &crop_info);

#endif  // EXTERNAL_SOFTDP_H
