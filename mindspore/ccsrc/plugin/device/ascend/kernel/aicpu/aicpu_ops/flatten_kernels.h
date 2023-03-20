/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef _AICPU_FLATTEN_KERNELS_H_
#define _AICPU_FLATTEN_KERNELS_H_

#include <vector>

#include "common/kernel_base.h"
#include "proto/aicpu_tensor.pb.h"

namespace aicpu {
namespace dataset {
struct MatrixInfo {
  std::vector<int> matrix_shape;
  ::aicpuops::DataType matrix_type;
};

class FlattenKernel : public KernelBase {
 public:
  FlattenKernel() : KernelBase("Flatten") {}
  ~FlattenKernel() = default;

 protected:
  MatrixInfo matrix_info_;
  size_t input_size_ = 0;
  uint32_t DoCompute() override;
  uint32_t ParseKernelParam() override;
};
}  // namespace dataset
}  // namespace aicpu
#endif
