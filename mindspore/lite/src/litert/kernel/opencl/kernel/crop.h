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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_CROP_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_CROP_H_

#include <vector>
#include <string>
#include "src/litert/kernel/opencl/opencl_kernel.h"
#include "nnacl/crop_parameter.h"

namespace mindspore::kernel {
class CropOpenCLKernel : public OpenCLKernel {
 public:
  using OpenCLKernel::OpenCLKernel;
  CropOpenCLKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                   const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : OpenCLKernel(parameter, inputs, outputs, ctx), crop_param_(reinterpret_cast<CropParameter *>(parameter)) {}
  ~CropOpenCLKernel() override = default;

  int Prepare() override;
  int CheckSpecsWithoutShape() override;
  int CheckSpecs() override;
  int SetConstArgs() override;
  int SetGlobalLocal() override;
  int Run() override;

 private:
  void RightShiftOffsetByAxis();

  CropParameter *crop_param_ = nullptr;
  GpuTensorInfo out_gpu_info_ = {};
  int offset_[COMM_SHAPE_SIZE] = {0};
};
}  // namespace mindspore::kernel
#endif
