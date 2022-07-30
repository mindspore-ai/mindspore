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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_GL_TO_CL_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_GL_TO_CL_H_

#include <vector>
#include "src/litert/lite_kernel.h"
#include "src/litert/kernel/opencl/opencl_kernel.h"

namespace mindspore::kernel {
class GLToCLOpenCLKernel : public OpenCLKernel {
 public:
  GLToCLOpenCLKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                     const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : OpenCLKernel(parameter, inputs, outputs, ctx) {
    out_mem_type_ = reinterpret_cast<OpenCLToFormatParameter *>(op_parameter_)->out_mem_type;
  }
  ~GLToCLOpenCLKernel() override = default;

  int Run() override;
  int Prepare() override;

  int CheckSpecs() override;
  int SetConstArgs() override;
  int SetGlobalLocal() override;
  int InferShape() override;
  int PreProcess() override;

 private:
  cl_mem_flags flags_{CL_MEM_READ_ONLY};
  cl_GLint mip_level_{0};
  size_t N_{1};
  size_t H_{1};
  size_t W_{1};
  size_t C_{1};
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_GL_TO_CL_H_
