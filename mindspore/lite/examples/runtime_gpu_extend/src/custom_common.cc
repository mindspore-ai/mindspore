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

#include "src/custom_common.h"

namespace mindspore {
namespace custom_common {
int CheckInputs(const std::vector<mindspore::MSTensor> &inputs) {
  for (auto &input : inputs) {
    auto input_shape = input.Shape();
    if (std::find(input_shape.begin(), input_shape.end(), -1) != input_shape.end()) {
      return lite::RET_INFER_INVALID;
    }
  }
  return lite::RET_OK;
}

int CheckOutputs(const std::vector<mindspore::MSTensor> &outputs) {
  for (auto &output : outputs) {
    auto output_shape = output.Shape();
    if (std::find(output_shape.begin(), output_shape.end(), -1) != output_shape.end()) {
      return lite::RET_INFER_INVALID;
    }
  }
  return lite::RET_OK;
}

void PackNHWCToNHWC4(void *src, void *dst, bool src_is_fp16, bool dst_is_fp16, const GpuTensorInfo &tensor,
                     mindspore::DataType data_type) {
  auto src_fp16 = reinterpret_cast<float16_t *>(src);
  auto src_fp32 = reinterpret_cast<float32_t *>(src);
  auto src_int32 = reinterpret_cast<int32_t *>(src);
  auto dst_fp16 = reinterpret_cast<float16_t *>(dst);
  auto dst_fp32 = reinterpret_cast<float32_t *>(dst);
  auto dst_int32 = reinterpret_cast<int32_t *>(dst);
  for (int n = 0, src_idx = 0; n < tensor.N; n++) {
    for (int h = 0; h < tensor.H; ++h) {
      for (int w = 0; w < tensor.W; ++w) {
        for (int c = 0; c < tensor.C; ++c, ++src_idx) {
          int dst_idx = ((n * tensor.H + h) * tensor.W + w) * tensor.Slice * C4NUM + c;
          if (data_type == mindspore::DataType::kNumberTypeInt32) {
            dst_int32[dst_idx] = src_int32[src_idx];
          } else if (dst_is_fp16) {
            dst_fp16[dst_idx] = src_is_fp16 ? src_fp16[src_idx] : static_cast<float16_t>(src_fp32[src_idx]);
          } else {
            dst_fp32[dst_idx] = src_is_fp16 ? static_cast<float32_t>(src_fp16[src_idx]) : src_fp32[src_idx];
          }
        }
      }
    }
  }
  // scalar
  if (tensor.ElementsNum == 1) {
    if (dst_is_fp16) {
      dst_fp16[3] = dst_fp16[2] = dst_fp16[1] = dst_fp16[0];
    } else {
      dst_fp32[3] = dst_fp32[2] = dst_fp32[1] = dst_fp32[0];
    }
  }
}
}  // namespace custom_common
}  // namespace mindspore
