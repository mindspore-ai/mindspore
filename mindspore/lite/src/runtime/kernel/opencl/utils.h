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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_UTILS_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_UTILS_H_

#include <string>
#include <vector>
#include <set>
#include "CL/cl2.hpp"
#include "src/common/log_adapter.h"
#include "nnacl/op_base.h"
#include "src/lite_kernel.h"
#include "src/common/utils.h"
#include "src/runtime/kernel/opencl/opencl_kernel.h"

namespace mindspore::lite {
kernel::LiteKernel *GetOpenCLKernel(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                                    OpParameter *parameter, const InnerContext *ctx, const kernel::KernelKey &key);
}

namespace mindspore::kernel {

struct GpuTensorInfo;

// for fusion
extern const std::set<schema::PrimitiveType> ArithmeticPrimitives;
extern const std::set<schema::PrimitiveType> ArithmeticSelfPrimitives;
inline bool IsArithmetic(schema::PrimitiveType type) { return ArithmeticPrimitives.count(type); }
inline bool IsArithmeticSelf(schema::PrimitiveType type) { return ArithmeticSelfPrimitives.count(type); }

std::string GetActDefines();

int GetUpPow2(int n);

int GetMaxDivisor(int x, int divisor);

int GetMaxDivisorStrategy0(int x, int divisor);

int GetMaxDivisorStrategy1(int x, int divisor);

std::string CLErrorCode(cl_int error_code);

int WriteToBin(const std::string &file_path, void *data, size_t size);

int GetBroadcastGpuAxis(int ndim, int ori_axis);

void PackNHWCToNHWC4(void *src, void *dst, bool src_is_fp16, bool dst_is_fp16, const GpuTensorInfo &tensor);

int CheckParamLikeTensor(const std::string &kernel_name, const std::string &tensor_name, lite::Tensor *tensor,
                         TypeId expect_data_type, const std::vector<int> &expect_shape);

void StoreTmpWeight(lite::Tensor *tensor);
void FreeTmpWeight(void *tensor);

template <class T1, class T2>
void PackNCHWToNC4HW4(void *src, void *dst, int batch, int plane_in, int plane_out, int channel,
                      const std::function<T2(T1)> &to_dtype) {
  MS_ASSERT(src);
  MS_ASSERT(dst);
  int c4 = UP_DIV(channel, C4NUM);
  for (int b = 0; b < batch; b++) {
    int src_offset = b * plane_in * channel;
    int dst_offset = b * plane_out * c4;
    for (int c = 0; c < channel; c++) {
      int c4_block_num = c / C4NUM;
      int c4_block_rem = c % C4NUM;
      int src_c_offset = src_offset + c * plane_in;
      int dst_c_offset = dst_offset + c4_block_num * plane_out;
      for (int k = 0; k < plane_in; k++) {
        int src_kernel_offset = src_c_offset + k;
        int dst_kernel_offset = dst_c_offset + C4NUM * k + c4_block_rem;
        (static_cast<T2 *>(dst) + dst_kernel_offset)[0] = to_dtype((static_cast<T1 *>(src) + src_kernel_offset)[0]);
      }
    }
  }
}

template <class T>
std::vector<T> MatrixMultiply(const T A[], const T B[], int M, int N, int K) {
  std::vector<T> C(M * K);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < K; ++j) {
      float s = 0.0f;
      for (int k = 0; k < N; ++k) {
        s += A[i * N + k] * B[k * K + j];
      }
      C[i * K + j] = s;
    }
  }
  return C;
}

}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_UTILS_H_
