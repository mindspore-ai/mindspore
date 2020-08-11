/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <string>
#include <set>

#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/kernel/opencl/kernel/leaky_relu.h"
#include "src/runtime/opencl/opencl_runtime.h"
#include "src/runtime/kernel/opencl/cl/fp32/leaky_relu.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_LeakyReLU;

namespace mindspore::kernel {

  int LeakyReluOpenCLKernel::Init() {
    if (inputs_[0]->shape().size() != 4) {
      MS_LOG(ERROR) << "leaky_relu only support dim=4, but your dim=" << inputs_[0]->shape().size();
    }
    std::set<std::string> build_options;
    std::string source = leaky_relu_source_fp32;
    std::string program_name = "LeakyRelu";
    std::string kernel_name = "LeakyRelu";
    auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
    ocl_runtime->LoadSource(program_name, source);
    ocl_runtime->BuildKernel(kernel_, program_name, kernel_name, build_options);

    MS_LOG(DEBUG) << kernel_name << " Init Done!";
    return RET_OK;
  }


    int LeakyReluOpenCLKernel::GetImageSize(size_t idx, std::vector<size_t> *img_size) {
      int H = inputs_[0]->shape()[1];
      int W = inputs_[0]->shape()[2];
      int C = inputs_[0]->shape()[3];

#ifdef ENABLE_FP16
      size_t img_dtype = CL_HALF_FLOAT;
#else
      size_t img_dtype = CL_FLOAT;
#endif

      img_size->clear();
      img_size->push_back(W * UP_DIV(C, C4NUM));
      img_size->push_back(H);
      img_size->push_back(img_dtype);
      return RET_OK;
    }

  int LeakyReluOpenCLKernel::Run() {
    auto param = reinterpret_cast<LeakyReluParameter *>(this->opParameter);
    MS_LOG(DEBUG) << this->Name() << " Running!";
    int N = inputs_[0]->shape()[0];
    int H = inputs_[0]->shape()[1];
    int W = inputs_[0]->shape()[2];
    int C = inputs_[0]->shape()[3];
    cl_int4 input_shape = {N, H, W, C};

    auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
    int arg_idx = 0;
    ocl_runtime->SetKernelArg(kernel_, arg_idx++, inputs_[0]->Data());
    ocl_runtime->SetKernelArg(kernel_, arg_idx++, outputs_[0]->Data());
    ocl_runtime->SetKernelArg(kernel_, arg_idx++, input_shape);
    ocl_runtime->SetKernelArg(kernel_, arg_idx++, param->alpha);

    std::vector<size_t> local = {1, 1};
    std::vector<size_t> global = {static_cast<size_t>(H), static_cast<size_t>(W)};
    ocl_runtime->RunKernel(kernel_, global, local, nullptr);
    return 0;
  }

  kernel::LiteKernel *OpenCLLeakyReluKernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                   const std::vector<lite::tensor::Tensor *> &outputs,
                                                   OpParameter *opParameter, const lite::Context *ctx,
                                                   const kernel::KernelKey &desc, const lite::Primitive *primitive) {
    auto *kernel = new LeakyReluOpenCLKernel(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs);
    if (inputs.size() == 0) {
      MS_LOG(ERROR) << "Input data size must must be greater than 0, but your size is " << inputs.size();
    }
    if (inputs[0]->shape()[0] > 1) {
      MS_LOG(ERROR) << "Init `leaky relu` kernel failed: Unsupported multi-batch.";
    }
    auto ret = kernel->Init();
    if (0 != ret) {
      MS_LOG(ERROR) << "Init `Leaky Relu` kernel failed!";
      delete kernel;
      return nullptr;
    }
    return kernel;
  }

  REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_LeakyReLU, OpenCLLeakyReluKernelCreator)
}  // namespace mindspore::kernel

