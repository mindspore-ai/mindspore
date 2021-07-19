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

#ifndef MINDSPORE_LITE_EXAMPLES_RUNTIME_GPU_EXTEND_SRC_CUSTOM_COMMON_H
#define MINDSPORE_LITE_EXAMPLES_RUNTIME_GPU_EXTEND_SRC_CUSTOM_COMMON_H

#include <arm_neon.h>
#include <vector>
#include <iostream>
#include "include/api/types.h"
#include "include/errorcode.h"
#include "include/ms_tensor.h"
#include "include/api/data_type.h"
#include "include/registry/opencl_runtime_wrapper.h"

#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
#define C4NUM 4
namespace mindspore {
namespace custom_common {

template <typename SrcT, typename DstT>
void Broadcast2GpuShape(DstT *dst, const SrcT *src, int src_num) {
  if (src == nullptr || src_num <= 0) {
    return;
  }
  auto *N = dst;
  auto *H = dst + 1;
  auto *W = dst + 2;
  auto *C = dst + 3;
  if (src_num == 1) {  // 1 1 1 C
    *C = src[0];
  } else if (src_num == 2) {  // N 1 1 C
    *N = src[0];
    *C = src[1];
  } else if (src_num == 3) {  // N 1 W C
    *N = src[0];
    *W = src[1];
    *C = src[2];
  } else if (src_num == 4) {  // N H W C
    *N = src[0];
    *H = src[1];
    *W = src[2];
    *C = src[3];
  } else if (src_num > 4) {
    std::cerr << "GPU doesn't support ndim>=" << src_num;
  }
}

template <typename SrcT, typename DstT>
void Broadcast2GpuShape(DstT *dst, const SrcT *src, int src_num, DstT default_value) {
  for (int i = 0; i < 4; ++i) {
    dst[i] = default_value;
  }
  if (src == nullptr || src_num <= 0) {
    return;
  }
  Broadcast2GpuShape(dst, src, src_num);
}
#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
#define C4NUM 4
struct GpuTensorInfo {
  GpuTensorInfo() = default;
  explicit GpuTensorInfo(const MSTensor *tensor, registry::opencl::OpenCLRuntimeWrapper *opencl_run) {
    if (tensor == nullptr) {
      return;
    }
    auto shape_ori = tensor->Shape();
    int64_t shape[4];
    Broadcast2GpuShape(shape, shape_ori.data(), shape_ori.size(), 1l);
    N = shape[0];
    H = shape[1];
    W = shape[2];
    C = shape[3];
    Slice = UP_DIV(C, C4NUM);
    if (tensor->DataType() == mindspore::DataType::kNumberTypeFloat16) {
      FLT_size = sizeof(cl_half);
    } else {
      FLT_size = sizeof(cl_float);
    }
    FLT4_size = FLT_size * C4NUM;
    if (W * Slice <= opencl_run->GetMaxImage2DWidth()) {
      height = N * H;
      width = W * Slice;
    } else {
      height = N * H * W;
      width = Slice;
      if (height > opencl_run->GetMaxImage2DHeight()) {
        height = -1;
        width = -1;
      }
    }

    ElementsNum = N * H * W * C;
    Image2DSize = height * width * FLT4_size;
  }
  size_t N{1};
  size_t H{1};
  size_t W{1};
  size_t C{1};
  size_t Slice{};
  size_t width{};
  size_t height{};
  size_t FLT_size{4};
  size_t FLT4_size{16};
  size_t ElementsNum{};
  size_t Image2DSize{};
};
// verify that the inputs' shape is inferred successfully when inferring current node.
int CheckInputs(const std::vector<mindspore::MSTensor> &inputs);

// versify that the outputs' shape is inferred successfully when running current node.
int CheckOutputs(const std::vector<mindspore::MSTensor> &inputs);
void PackNHWCToNHWC4(void *src, void *dst, bool src_is_fp16, bool dst_is_fp16, const GpuTensorInfo &tensor,
                     mindspore::DataType data_type = mindspore::DataType::kNumberTypeFloat32);
}  // namespace custom_common
}  // namespace mindspore
#endif  // MINDSPORE_LITE_EXAMPLES_RUNTIME_GPU_EXTEND_SRC_CUSTOM_COMMON_H
