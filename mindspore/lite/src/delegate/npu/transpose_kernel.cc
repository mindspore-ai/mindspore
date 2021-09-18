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
namespace mindspore {
#define C8NUM 8
#ifdef ENABLE_ARM64
inline void Transpose8X8Fp32Arm64(const float *src_ptr, float *dst_ptr, int src_stride, int dst_stride) {
  size_t srcStride = src_stride * sizeof(float);
  size_t dstStride = dst_stride * sizeof(float);
  asm volatile(
    "mov x10, %[src_ptr]\n"
    "mov x11, %[dst_ptr]\n"

    "ld1 {v0.4s, v1.4s}, [x10], %[srcStride]\n"
    "ld1 {v2.4s, v3.4s}, [x10], %[srcStride]\n"

    "zip1 v8.4s, v0.4s, v2.4s\n"
    "zip2 v9.4s, v0.4s, v2.4s\n"
    "zip1 v12.4s, v1.4s, v3.4s\n"
    "zip2 v13.4s, v1.4s, v3.4s\n"

    "ld1 {v4.4s, v5.4s}, [x10], %[srcStride]\n"
    "ld1 {v6.4s, v7.4s}, [x10], %[srcStride]\n"

    "zip1 v10.4s, v4.4s, v6.4s\n"
    "zip2 v11.4s, v4.4s, v6.4s\n"
    "zip1 v14.4s, v5.4s, v7.4s\n"
    "zip2 v15.4s, v5.4s, v7.4s\n"

    "ld1 {v0.4s, v1.4s}, [x10], %[srcStride]\n"
    "ld1 {v2.4s, v3.4s}, [x10], %[srcStride]\n"

    "trn1 v16.2d, v8.2d, v10.2d\n"
    "trn2 v18.2d, v8.2d, v10.2d\n"
    "trn1 v20.2d, v9.2d, v11.2d\n"
    "trn2 v22.2d, v9.2d, v11.2d\n"

    "ld1 {v4.4s, v5.4s}, [x10], %[srcStride]\n"
    "ld1 {v6.4s, v7.4s}, [x10], %[srcStride]\n"

    "trn1 v24.2d, v12.2d, v14.2d\n"
    "trn2 v26.2d, v12.2d, v14.2d\n"
    "trn1 v28.2d, v13.2d, v15.2d\n"
    "trn2 v30.2d, v13.2d, v15.2d\n"

    "zip1 v8.4s, v0.4s, v2.4s\n"
    "zip2 v9.4s, v0.4s, v2.4s\n"
    "zip1 v12.4s, v1.4s, v3.4s\n"
    "zip2 v13.4s, v1.4s, v3.4s\n"

    "zip1 v10.4s, v4.4s, v6.4s\n"
    "zip2 v11.4s, v4.4s, v6.4s\n"
    "zip1 v14.4s, v5.4s, v7.4s\n"
    "zip2 v15.4s, v5.4s, v7.4s\n"

    "trn1 v17.2d, v8.2d, v10.2d\n"
    "trn2 v19.2d, v8.2d, v10.2d\n"
    "trn1 v21.2d, v9.2d, v11.2d\n"
    "trn2 v23.2d, v9.2d, v11.2d\n"

    "trn1 v25.2d, v12.2d, v14.2d\n"
    "trn2 v27.2d, v12.2d, v14.2d\n"
    "trn1 v29.2d, v13.2d, v15.2d\n"
    "trn2 v31.2d, v13.2d, v15.2d\n"

    "st1 {v16.4s, v17.4s}, [x11], %[dstStride]\n"
    "st1 {v18.4s, v19.4s}, [x11], %[dstStride]\n"
    "st1 {v20.4s, v21.4s}, [x11], %[dstStride]\n"
    "st1 {v22.4s, v23.4s}, [x11], %[dstStride]\n"
    "st1 {v24.4s, v25.4s}, [x11], %[dstStride]\n"
    "st1 {v26.4s, v27.4s}, [x11], %[dstStride]\n"
    "st1 {v28.4s, v29.4s}, [x11], %[dstStride]\n"
    "st1 {v30.4s, v31.4s}, [x11], %[dstStride]\n"

    :
    : [ dst_ptr ] "r"(dst_ptr), [ src_ptr ] "r"(src_ptr), [ srcStride ] "r"(srcStride), [ dstStride ] "r"(dstStride)
    : "x10", "x11", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
      "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30",
      "v31");
}
#endif

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
