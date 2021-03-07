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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_POOLING_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_POOLING_H_

#include <vector>

#include "src/runtime/kernel/opencl/opencl_kernel.h"
#include "nnacl/fp32/pooling_fp32.h"

namespace mindspore::kernel {

class PoolingOpenCLKernel : public OpenCLKernel {
 public:
  PoolingOpenCLKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                      const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : OpenCLKernel(parameter, inputs, outputs, ctx), parameter_(reinterpret_cast<PoolingParameter *>(parameter)) {}
  ~PoolingOpenCLKernel() override = default;

  int Run() override;
  int Prepare() override;
  int CheckSpecs() override;
  void SetConstArgs() override;
  void SetGlobalLocal() override;

 private:
  PoolingParameter *parameter_;
};

}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_POOLING_H_
