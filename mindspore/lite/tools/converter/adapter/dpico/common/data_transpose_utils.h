/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef DPICO_COMMON_DATA_TRANSPOSE_UTILS_H
#define DPICO_COMMON_DATA_TRANSPOSE_UTILS_H

#include <vector>
#include "mindapi/base/format.h"
#include "mindapi/ir/tensor.h"
#include "mindapi/base/logging.h"
#include "include/errorcode.h"
#include "common/op_enum.h"
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NO_CHANGE;
using mindspore::lite::RET_OK;
using mindspore::lite::STATUS;
namespace mindspore {
namespace dpico {
inline const std::vector<int> kNH2NC = {0, 3, 1, 2};
inline const std::vector<int> kNC2NH = {0, 2, 3, 1};
enum FormatTransNodeType { kNCHW2NHWC, kNHWC2NCHW, kNONE };
struct TransTypePair {
  FormatTransNodeType pre_;
  FormatTransNodeType post_;
  TransTypePair() : pre_(kNONE), post_(kNONE) {}
};
template <typename T>
STATUS NHWC2NCHW(T *src_data, T *dst_data, std::vector<int32_t> shape) {
  if (shape.size() != kDims4) {
    MS_LOG(ERROR) << "The dim should be 4.";
    return RET_ERROR;
  }
  size_t batch = shape.at(0);
  size_t plane = shape.at(kAxis1) * shape.at(kAxis2);
  size_t channel = shape.at(kAxis3);
  for (size_t b = 0; b < batch; b++) {
    for (size_t p = 0; p < plane; p++) {
      for (size_t c = 0; c < channel; c++) {
        size_t src_idx = b * plane * channel + p * channel + c;
        size_t dst_idx = b * channel * plane + c * plane + p;
        dst_data[dst_idx] = src_data[src_idx];
      }
    }
  }
  return RET_OK;
}

template <typename T>
STATUS NCHW2NHWC(T *src_data, T *dst_data, std::vector<int32_t> shape) {
  if (shape.size() != kDims4) {
    MS_LOG(ERROR) << "The dim should be 4.";
    return RET_ERROR;
  }
  size_t batch = shape.at(0);
  size_t channel = shape.at(1);
  size_t plane = shape.at(kAxis2) * shape.at(kAxis3);
  for (size_t b = 0; b < batch; b++) {
    for (size_t c = 0; c < channel; c++) {
      for (size_t p = 0; p < plane; p++) {
        size_t src_idx = b * channel * plane + c * plane + p;
        size_t dst_idx = b * plane * channel + p * channel + c;
        dst_data[dst_idx] = src_data[src_idx];
      }
    }
  }
  return RET_OK;
}

STATUS TransFilterFormat(const mindspore::api::TensorPtr &tensor, mindspore::Format src_format,
                         mindspore::Format dst_format);

void TransposeMatrix(float *matrix, int row, int col);
}  // namespace dpico
}  // namespace mindspore
#endif  // DPICO_COMMON_DATA_TRANSPOSE_UTILS_H
