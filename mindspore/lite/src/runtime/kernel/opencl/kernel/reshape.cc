/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include <set>
#include <string>
#include "include/errorcode.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/kernel/reshape.h"
#include "src/runtime/kernel/opencl/cl/reshape.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Reshape;
using mindspore::schema::PrimitiveType_Squeeze;

namespace mindspore::kernel {

int ReshapeOpenCLKernel::Init() {
  std::string kernel_name = "reshape_NHWC4";
  if (out_tensors_[0]->shape().size() != 2 && out_tensors_[0]->shape().size() != 4) {
    MS_LOG(ERROR) << "Reshape output size should in 2,4";
    return RET_ERROR;
  }
#ifdef PROGRAM_WITH_IL
  kernel_ = ocl_runtime_->GetKernelFromBinary(kernel_name);
#else
  std::set<std::string> build_options;
  std::string source = reshape_source;
  std::string program_name = "reshape";
  ocl_runtime_->LoadSource(program_name, source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options);
#endif
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}

int ReshapeOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  auto in = Image2DInfo(in_tensors_.front());
  auto out = Image2DInfo(out_tensors_.front());

  std::vector<size_t> local = {};
  std::vector<size_t> global{out.width, out.height};
  cl_int4 src_size = {cl_int(in.C), cl_int(in.W), cl_int(in.H), cl_int(in.N)};
  cl_int4 dst_size = {cl_int(out.width), cl_int(out.height), cl_int(out.C), cl_int(out.C * out.W)};

  int arg_idx = 0;
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, in_tensors_[0]->data_c());
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, out_tensors_[0]->data_c());
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, src_size);
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, dst_size);
  ocl_runtime_->RunKernel(kernel_, global, local, nullptr);
  return RET_OK;
}

kernel::LiteKernel *OpenCLReshapeKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                               const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                               const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                               const mindspore::lite::PrimitiveC *primitive) {
  auto *kernel = new (std::nothrow) ReshapeOpenCLKernel(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel " << opParameter->name_ << " create failed.";
    free(opParameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Reshape, OpenCLReshapeKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Reshape, OpenCLReshapeKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Squeeze, OpenCLReshapeKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Squeeze, OpenCLReshapeKernelCreator)
}  // namespace mindspore::kernel
