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

#include "src/runtime/kernel/opencl/kernel/pooling2d.h"
#include <string>
#include <set>
#include "include/errorcode.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "src/runtime/opencl/opencl_wrapper.h"
#include "src/runtime/opencl/opencl_runtime.h"
#include "src/runtime/kernel/opencl/image_format.h"
#ifndef PROGRAM_WITH_IL
#include "src/runtime/kernel/opencl/cl/fp32/max_pool2d.cl.inc"
#include "src/runtime/kernel/opencl/cl/fp32/avg_pool2d.cl.inc"
#endif

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_INVALID_OP_NAME;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Pooling;

namespace mindspore {
namespace kernel {
int PoolingOpenCLKernel::Init() {
  std::string kernel_name;
#ifndef PROGRAM_WITH_IL
  std::string source;
  std::string program_name;
#endif
  if (parameter_->max_pooling_) {
    kernel_name = "MaxPooling2d";
#ifndef PROGRAM_WITH_IL
    source = max_pool2d_source_fp32;
    program_name = "MaxPooling2d";
#endif
  } else if (parameter_->avg_pooling_) {
    kernel_name = "AvgPooling2d";
#ifndef PROGRAM_WITH_IL
    source = avg_pool2d_source_fp32;
    program_name = "AvgPooling2d";
#endif
  } else {
    MS_LOG(ERROR) << "Init `Pooling2d` kernel failed!";
    return RET_INVALID_OP_NAME;
  }
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();

#ifdef PROGRAM_WITH_IL
  ocl_runtime->CreateKernelFromIL(kernel_(), kernel_name);
#else
  if (mem_type_ == MEM_TYPE::BUF) {
    kernel_name += "_BUF";
  } else {
    kernel_name += "_IMG";
  }
  std::set<std::string> build_options;
  ocl_runtime->LoadSource(program_name, source);
  ocl_runtime->BuildKernel(kernel_, program_name, kernel_name, build_options);
#endif
  out_tensors_[0]->SetFormat(schema::Format_NHWC4);
  MS_LOG(DEBUG) << kernel_name << " Init Done!";

  return RET_OK;
}

std::vector<size_t> PoolingOpenCLKernel::InitGlobalSize() const {
  const size_t global_x = out_tensors_[0]->Height();
  const size_t global_y = out_tensors_[0]->Width();
  const size_t global_z = UP_DIV(out_tensors_[0]->Channel(), C4NUM);
  std::vector<size_t> global = {global_x, global_y, global_z};
  return global;
}

int PoolingOpenCLKernel::GetImageSize(size_t idx, std::vector<size_t> *img_size) {
  size_t CO4 = UP_DIV(out_tensors_[0]->Channel(), C4NUM);
  size_t im_dst_x, im_dst_y;
  im_dst_x = out_tensors_[0]->Width() * CO4;
  im_dst_y = out_tensors_[0]->Height();
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

int PoolingOpenCLKernel::InitBuffer() { return RET_OK; }

int PoolingOpenCLKernel::ReSize() { return RET_OK; }

int PoolingOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();

  // attribute
  int slices = UP_DIV(out_tensors_[0]->Channel(), C4NUM);
  cl_int4 input_shape = {in_tensors_[0]->Height(), in_tensors_[0]->Width(), in_tensors_[0]->Channel(), slices};
  cl_int4 output_shape = {out_tensors_[0]->Height(), out_tensors_[0]->Width(), out_tensors_[0]->Channel(), slices};
  cl_int2 stride = {parameter_->stride_h_, parameter_->stride_w_};
  cl_int2 kernel_size = {parameter_->window_h_, parameter_->window_w_};
  cl_int2 padding = {parameter_->pad_u_, parameter_->pad_l_};

  // binding parameters
  int arg_idx = 0;
  ocl_runtime->SetKernelArg(kernel_, arg_idx++, in_tensors_[0]->Data());
  ocl_runtime->SetKernelArg(kernel_, arg_idx++, out_tensors_[0]->Data());
  ocl_runtime->SetKernelArg(kernel_, arg_idx++, input_shape);
  ocl_runtime->SetKernelArg(kernel_, arg_idx++, output_shape);
  ocl_runtime->SetKernelArg(kernel_, arg_idx++, stride);
  ocl_runtime->SetKernelArg(kernel_, arg_idx++, kernel_size);
  ocl_runtime->SetKernelArg(kernel_, arg_idx++, padding);

  // set work group size
  std::vector<size_t> local_size;
  std::vector<size_t> global_size = InitGlobalSize();
  int max_work_group_size = ocl_runtime->GetKernelMaxWorkGroupSize(kernel_(), (*ocl_runtime->Device())());
  local_size = GetCommonLocalSize(global_size, max_work_group_size);
  global_size = GetCommonGlobalSize(local_size, global_size);

  // run opengl kernel
  ocl_runtime->RunKernel(kernel_, global_size, local_size, nullptr);
  return RET_OK;
}

kernel::LiteKernel *OpenCLPooling2dKernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                 const std::vector<lite::tensor::Tensor *> &outputs,
                                                 OpParameter *opParameter, const lite::Context *ctx,
                                                 const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  auto *kernel = new (std::nothrow)PoolingOpenCLKernel(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Create OpenCL Pooling kernel failed!";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (RET_OK != ret) {
    MS_LOG(ERROR) << "Init OpenCL Pooling kernel failed!";
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Pooling, OpenCLPooling2dKernelCreator)
}  // namespace kernel
}  // namespace mindspore
