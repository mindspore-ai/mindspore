/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef AICPU_OPS_FSE_DECODE_KERNEL_H_
#define AICPU_OPS_FSE_DECODE_KERNEL_H_

#include <vector>
#include <random>
#include "common/kernel_base.h"

namespace aicpu {
class FSEDecodeKernel : public KernelBase {
 public:
  FSEDecodeKernel() : KernelBase("FSEDecodeKernel") {}
  ~FSEDecodeKernel() = default;

 protected:
  uint32_t ParseKernelParam() override;
  uint32_t DoCompute() override;
  bool CheckParams() const;
  uint32_t FSEDecodeTask();
  uint32_t FixedBitHalfDequantTask();
  uint32_t FixedBitFloatDequantTask();
  uint64_t Pop(const uint64_t *chunks, uint64_t bit_count);
  std::vector<int> output_shape_;
  int64_t input_shape_size_{0};

  int64_t dst_type_{0};
  uint64_t curr_chunk_{0};
  int64_t curr_chunk_index_{0};
  int64_t curr_bit_count_{0};
  int64_t table_log_{0};
};
}  // namespace aicpu
#endif  // AICPU_OPS_FSE_DECODE_KERNEL_H_
