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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_MATMUL_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_MATMUL_H_

#include <vector>
#include "src/runtime/kernel/opencl/opencl_kernel.h"
#include "src/common/utils.h"
#include "nnacl/matmul_parameter.h"

namespace mindspore::kernel {
class MatMulOpenCLKernel : public OpenCLKernel {
 public:
  using OpenCLKernel::OpenCLKernel;
  ~MatMulOpenCLKernel() override = default;

  int Run() override;
  int Prepare() override;
  int CheckSpecs() override;
  int InitWeights() override;
  int SetConstArgs() override;
  void SetGlobalLocal() override;
  int Tune() override { return lite::RET_OK; }
  int InitBias();
  int StoreConstData() override;

 protected:
  void *padWeight_{nullptr};
  bool enable_fp16_{false};
  bool transposeA{false};
  bool transposeB{true};
  int dims{};
  void *bias_{nullptr};
  int CO_{1};
  void *stored_weight_{nullptr};
  void *stored_bias_{nullptr};
  static constexpr int MAX_DIMS{4};  // max supported matmul dims
  bool act_weight_{false};
  std::vector<int> inShape{std::vector<int>(MAX_DIMS, 1)};
  std::vector<int> outShape{std::vector<int>(MAX_DIMS, 1)};

 private:
  int PadWeight(std::vector<int> weight_shape_4d, int ci, int co);
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_MATMUL_H_
