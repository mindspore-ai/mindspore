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

#ifndef SOFT_DP_H
#define SOFT_DP_H

#include <cstdint>
#include "minddata/dataset/kernels/image/soft_dvpp/utils/external_soft_dp.h"

enum JpegdToVpcFormat {
  INPUT_VPC_UNKNOWN = -1,
  INPUT_YUV420_PLANNER = 1,  // 1
  INPUT_YUV422_PLANNER,      // 2
  INPUT_YUV444_PLANNER,      // 3
  INPUT_YUV400_PLANNER,      // 4
};

struct VpcInfo {
  uint8_t *addr;
  int32_t width;
  int32_t height;

  int32_t real_width;
  int32_t real_height;

  enum JpegdToVpcFormat format;
  bool is_v_before_u;
  bool is_fake420;

  VpcInfo()
      : addr(nullptr),
        width(0),
        height(0),
        real_width(0),
        real_height(0),
        format(INPUT_VPC_UNKNOWN),
        is_v_before_u(false),
        is_fake420(false) {}
};

#endif  // SOFT_DP_H
