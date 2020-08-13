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
#include "src/runtime/opencl/opencl_runtime.h"
#include "src/runtime/kernel/opencl/kernel/reshape.h"
#include "src/runtime/kernel/opencl/cl/fp16/reshape.cl.inc"
#include "src/runtime/kernel/opencl/cl/fp32/reshape.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Reshape;

namespace mindspore::kernel {

int ReshapeOpenCLKernel::Init() {
  std::string kernel_name = "reshape";
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();

#ifdef PROGRAM_WITH_IL
  ocl_runtime->CreateKernelFromIL(kernel_(), kernel_name);
#else
  std::set<std::string> build_options;
#ifdef ENABLE_FP16
  std::string source = reshape_source_fp16;
#else
  std::string source = reshape_source_fp32;
#endif
  std::string program_name = "reshape";
  ocl_runtime->LoadSource(program_name, source);
  ocl_runtime->BuildKernel(kernel_, program_name, kernel_name, build_options);
#endif
  out_tensors_[0]->SetFormat(schema::Format_NHWC);
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}

int ReshapeOpenCLKernel::ReSize() { return 0; }

int ReshapeOpenCLKernel::GetImageSize(size_t idx, std::vector<size_t> *img_size) {
  size_t im_dst_x, im_dst_y;
  std::vector<int> shapex = in_tensors_[0]->shape();
  int h = shapex[1];
  int w = shapex[2];
  int c = shapex[3];
  im_dst_x = UP_DIV(w * c, C4NUM);
  im_dst_y = h;
#ifdef ENABLE_FP16
  size_t img_dtype = CL_HALF_FLOAT;
#else
  size_t img_dtype = CL_FLOAT;
#endif
  img_size->clear();
  std::vector<size_t> vec{im_dst_x, im_dst_y, img_dtype};
  *img_size = vec;
  return RET_OK;
}

int ReshapeOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  std::vector<int> shapex = in_tensors_[0]->shape();
  int h = shapex[1];
  int w = shapex[2];
  int c = shapex[3];
  int c4 = UP_DIV(c, C4NUM);
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  // local size should less than MAX_GROUP_SIZE
  std::vector<size_t> local = {};
  std::vector<size_t> global = {(size_t)h, (size_t)w, (size_t)c4};
  cl_int4 size = {h, w, c4, 1};
  ocl_runtime->SetKernelArg(kernel_, 0, in_tensors_[0]->Data());
  ocl_runtime->SetKernelArg(kernel_, 1, out_tensors_[0]->Data());
  ocl_runtime->SetKernelArg(kernel_, 2, size);
  ocl_runtime->RunKernel(kernel_, global, local, nullptr);
  return RET_OK;
}

kernel::LiteKernel *OpenCLReshapeKernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                               const std::vector<lite::tensor::Tensor *> &outputs,
                                               OpParameter *opParameter, const lite::Context *ctx,
                                               const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  auto *kernel = new (std::nothrow) ReshapeOpenCLKernel(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel " << opParameter->name_ << " create failed.";
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
}  // namespace mindspore::kernel
