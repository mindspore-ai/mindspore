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

#include "src/litert/delegate/delegate_utils.h"
#include "nnacl/fp32/pack_fp32.h"
namespace mindspore::lite {
void PackNHWCToNCHWFp32(const void *src, void *dst, int batches, int plane, int channel) {
  int hw8 = plane / C8NUM * C8NUM;
  int batch = plane * channel;
  for (int n = 0; n < batches; n++) {
    const float *src_batch = (const float *)src + n * batch;
    float *dst_batch = reinterpret_cast<float *>(dst) + n * batch;
    int hw = 0;
    for (; hw < hw8; hw += C8NUM) {
      int c = 0;
#ifdef ENABLE_ARM64
      for (; c <= channel - C8NUM; c += C8NUM) {
        const float *src_ptr = src_batch + hw * channel + c;
        float *dst_ptr = dst_batch + c * plane + hw;
        Transpose8X8Fp32Arm64(src_ptr, dst_ptr, channel, plane);
      }
#endif
      for (; c < channel; c++) {
        const float *src_ptr = src_batch + hw * channel + c;
        float *dst_ptr = dst_batch + c * plane + hw;
        for (size_t i = 0; i < C8NUM; i++) {
          dst_ptr[i] = src_ptr[i * channel];
        }
      }
    }
    for (; hw < plane; hw++) {
      const float *src_ptr = src_batch + hw * channel;
      float *dst_ptr = dst_batch + hw;
      for (int i = 0; i < channel; i++) {
        dst_ptr[i * plane] = src_ptr[i];
      }
    }
  }
}

void PackNCHWToNHWCFp32(const void *src, void *dst, int batch, int plane, int channel) {
  return PackNHWCToNCHWFp32(src, dst, batch, channel, plane);
}

int MaskDataNHWC2NCHWBinary(int mask) {
  int mask_vec[COMM_SHAPE_SIZE];
  for (int i = 0; i < COMM_SHAPE_SIZE; ++i) {
    mask_vec[i] = (uint32_t)(mask) & (1 << i);
  }
  AssistDataNHWC2NCHW<int>(mask_vec, 1);
  int ret = 0;
  for (int i = 0; i < COMM_SHAPE_SIZE; ++i) {
    if (mask_vec[i]) {
      ret += 1 << i;
    }
  }
  return ret;
}

void BinaryMaskData2Bool(int src_mask, bool *dst_mask, size_t mask_size) {
  MS_ASSERT(dst_mask != nullptr);
  for (size_t i = 0; i < mask_size; ++i) {
    int mask_raw = (uint32_t)(src_mask) & (1 << i);
    dst_mask[i] = static_cast<bool>(mask_raw);
  }
}

bool IsSubGraphInputTensor(const std::vector<mindspore::MSTensor> &inputs, mindspore::MSTensor input) {
  return std::find(inputs.begin(), inputs.end(), input) != inputs.end();
}
}  // namespace mindspore::lite
