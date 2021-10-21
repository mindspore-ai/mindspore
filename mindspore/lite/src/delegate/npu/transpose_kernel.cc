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

#include "src/delegate/npu/transpose_kernel.h"
#include "src/delegate/npu/npu_converter_utils.h"
#include "src/delegate/npu/op/npu_op.h"
#include "nnacl/fp32/pack_fp32.h"
namespace mindspore {
#define C8NUM 8
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
      for (size_t i = 0; i < channel; i++) {
        dst_ptr[i * plane] = src_ptr[i];
      }
    }
  }
}

void PackNCHWToNHWCFp32(const void *src, void *dst, int batch, int plane, int channel) {
  return PackNHWCToNCHWFp32(src, dst, batch, channel, plane);
}

int TransposeNPUKernel::Execute() {
  std::vector<int> nh2nc_perm = {0, 3, 1, 2};
  std::vector<int> nc2nh_perm = {0, 2, 3, 1};
  if (perm_ != nh2nc_perm && perm_ != nc2nh_perm) {
    MS_LOG(ERROR) << "NPU transpose op only supports nhwc->nchw or nchw->nhwc.";
    return RET_ERROR;
  }
  auto shape = inputs()[0].Shape();
  if (shape.size() != NPU_SHAPE_SIZE) {
    MS_LOG(ERROR) << "NPU transpose op only supports input of 4 dims.";
    return RET_ERROR;
  }
  mindspore::MSTensor in_tensor = inputs()[0];
  mindspore::MSTensor out_tensor = outputs()[0];
  auto input = in_tensor.Data().get();
  MS_ASSERT(input);
  auto output = out_tensor.MutableData();
  MS_ASSERT(output);
  if (perm_ == nh2nc_perm) {
    PackNHWCToNCHWFp32(input, output, shape[NHWC_N], shape[NHWC_H] * shape[NHWC_W], shape[NHWC_C]);
  } else if (perm_ == nc2nh_perm) {
    PackNCHWToNHWCFp32(input, output, shape[NCHW_N], shape[NCHW_H] * shape[NCHW_W], shape[NCHW_C]);
  } else {
    MS_LOG(ERROR) << "NPU transpose op only supports nhwc->nchw or nchw->nhwc.";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore
