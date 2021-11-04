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

#include "src/delegate/tensorrt/cuda_impl/cuda_helper.h"

CudaHelper &CudaHelper::GetInstance() {
  static CudaHelper instance;
  return instance;
}
int CudaHelper::GetThreadNum() const { return threads_per_block_; }
int CudaHelper::GetBlocksNum(const int total_threads) const {
  return std::min(((total_threads - 1) / threads_per_block_) + 1, max_blocks_);
}

CudaHelper::CudaHelper() {
  int device_id = 0;
  (void)cudaGetDevice(&device_id);
  cudaDeviceProp prop;
  (void)cudaGetDeviceProperties(&prop, device_id);
  threads_per_block_ = prop.maxThreadsPerBlock;
  max_blocks_ = prop.multiProcessorCount;
}
