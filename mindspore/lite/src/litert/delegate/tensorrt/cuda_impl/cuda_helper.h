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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_CUDA_IMPL_CUDA_HELPER_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_CUDA_IMPL_CUDA_HELPER_H_

#include <cuda_runtime.h>
#include <algorithm>

class CudaHelper {
 public:
  int GetThreadNum() const;
  int GetThreadNum(const int block_size) const;
  int GetBlocksNum(const int total_threads) const;
  int GetBlocksNum(const int total_threads, const int block_size) const;
  static CudaHelper &GetInstance();

 private:
  CudaHelper();
  ~CudaHelper() = default;
  CudaHelper(const CudaHelper &) = delete;
  CudaHelper &operator=(const CudaHelper &) = delete;

  int max_blocks_;
  int threads_per_block_;
};

#define GET_BLOCKS(total_threads) CudaHelper::GetInstance().GetBlocksNum(total_threads)
#define GET_BLOCKS_CAL(total_threads, block_size) CudaHelper::GetInstance().GetBlocksNum(total_threads, block_size)

#define GET_THREADS CudaHelper::GetInstance().GetThreadNum()
#define GET_THREADS_CAL(block_size) CudaHelper::GetInstance().GetThreadNum(block_size)

#define CUDA_CHECK(ret)              \
  do {                               \
    cudaError_t cuda_ret = (ret);    \
    if ((cuda_ret) != cudaSuccess) { \
      return -1;                     \
    }                                \
  } while (0)

#define CUDA_CHECK_VOID(ret)         \
  do {                               \
    cudaError_t cuda_ret = (ret);    \
    if ((cuda_ret) != cudaSuccess) { \
      return;                        \
    }                                \
  } while (0)

#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_CUDA_IMPL_CUDA_HELPER_H_
