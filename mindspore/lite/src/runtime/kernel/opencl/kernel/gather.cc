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
#include <cstring>
#include <string>
#include <algorithm>
#include <set>
#include <utility>
#include "src/kernel_registry.h"
#include "src/runtime/opencl/opencl_runtime.h"
#include "src/runtime/kernel/opencl/kernel/gather.h"
#include "src/runtime/kernel/opencl/cl/gather.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_Gather;

namespace mindspore::kernel {

int GatherOpenCLKernel::Init() {
  std::string kernel_name = "gather";
  auto in_format = op_format_;
  if (in_format != schema::Format_NHWC4 && in_format != schema::Format_NC4HW4) {
    MS_LOG(ERROR) << "input format(" << in_format << ") "
                  << "format not support!";
    return RET_ERROR;
  }
  in_ori_format_ = in_tensors_[0]->GetFormat();
  out_ori_format_ = out_tensors_[0]->GetFormat();
  in_tensors_[0]->SetFormat(op_format_);
  out_tensors_[0]->SetFormat(op_format_);
  if (in_format == schema::Format_NC4HW4) {
    kernel_name += "_NC4HW4";
  } else {
    kernel_name += "_NHWC4";
  }
  std::set<std::string> build_options;
  std::string source = gather_source;
  std::string program_name = "gather";
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  ocl_runtime->LoadSource(program_name, source);
  ocl_runtime->BuildKernel(kernel_, program_name, kernel_name, build_options);
  // init indices_data_
  auto indices_tensor = in_tensors_.at(1);
  int indices_num = indices_tensor->ElementsNum();
  bool isIndicesInt32 = indices_tensor->data_type() == kNumberTypeInt32;
  auto allocator = lite::opencl::OpenCLRuntime::GetInstance()->GetAllocator();
  if (!isIndicesInt32) {
    indices_data_ = reinterpret_cast<int32_t *>(allocator->Malloc(sizeof(int32_t) * indices_num));
    if (indices_data_ == nullptr) {
      MS_LOG(ERROR) << "Memory allocation failed";
      return RET_ERROR;
    }
  }
  return RET_OK;
}
int GatherOpenCLKernel::InitBuffer() {
  auto indices_tensor = in_tensors_.at(1);
  int indices_num = indices_tensor->ElementsNum();
  bool isIndicesInt32 = indices_tensor->data_type() == kNumberTypeInt32;
  if (!isIndicesInt32) {
    if (indices_tensor->data_type() == kNumberTypeInt64) {
      for (int i = 0; i < indices_num; i++) {
        indices_data_[i] = reinterpret_cast<int64_t *>(indices_tensor->data_c())[i];
      }
    } else if (indices_tensor->data_type() == kNumberTypeFloat32) {
      for (int i = 0; i < indices_num; i++) {
        indices_data_[i] = reinterpret_cast<float *>(indices_tensor->data_c())[i];
      }
    } else if (indices_tensor->data_type() == kNumberTypeFloat16) {
      for (int i = 0; i < indices_num; i++) {
        indices_data_[i] = reinterpret_cast<float16_t *>(indices_tensor->data_c())[i];
      }
    } else {
      MS_LOG(ERROR) << "Unsupported data type: " << indices_tensor->data_type();
      return RET_ERROR;
    }
  } else {
    indices_data_ = reinterpret_cast<int32_t *>(indices_tensor->data_c());
  }
  return RET_OK;
}
int GatherOpenCLKernel::ReSize() { return RET_OK; }
int GatherOpenCLKernel::GetImageSize(size_t idx, std::vector<size_t> *img_size) {
  size_t CO4 = UP_DIV(out_tensors_[0]->Channel(), C4NUM);
  size_t im_dst_x, im_dst_y;
  if (in_tensors_[0]->GetFormat() == schema::Format::Format_NHWC4) {
    im_dst_x = out_tensors_[0]->Width() * CO4;
    im_dst_y = out_tensors_[0]->Height();
  } else {
    im_dst_y = out_tensors_[0]->Batch() * out_tensors_[0]->Height() * CO4;
    im_dst_x = out_tensors_[0]->Width();
  }
  size_t img_dtype = CL_FLOAT;
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  auto enable_fp16_ = ocl_runtime->GetFp16Enable();
  if (enable_fp16_) {
    img_dtype = CL_HALF_FLOAT;
  }
  img_size->clear();
  std::vector<size_t> vec{im_dst_x, im_dst_y, img_dtype};
  *img_size = std::move(vec);
  return RET_OK;
}
int GatherOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running! ";
  auto param = reinterpret_cast<GatherParameter *>(this->op_parameter_);
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();

  if (InitBuffer() != RET_OK) {
    return RET_ERROR;
  }

  auto input_shape = in_tensors_[0]->shape();
  auto output_shape = out_tensors_[0]->shape();
  int indices_num = in_tensors_[1]->ElementsNum();
  size_t CO4 = UP_DIV(out_tensors_[0]->Channel(), C4NUM);
  size_t CI4 = UP_DIV(in_tensors_[0]->Channel(), C4NUM);
  cl_int4 src_size = {in_tensors_[0]->Width(), in_tensors_[0]->Height(), (cl_int)CI4, in_tensors_[0]->Batch()};
  cl_int4 dst_size = {(cl_int)out_tensors_[0]->Width(), (cl_int)out_tensors_[0]->Height(), (cl_int)CO4,
                      (cl_int)out_tensors_[0]->Batch()};
  std::vector<size_t> local = {1, 1, 1};
  std::vector<size_t> global = {(size_t)out_tensors_[0]->Width(), (size_t)out_tensors_[0]->Height(), CO4};
  int arg_cn = 0;
  ocl_runtime->SetKernelArg(kernel_, arg_cn++, in_tensors_[0]->data_c(), lite::opencl::MemType::IMG);
  ocl_runtime->SetKernelArg(kernel_, arg_cn++, indices_data_, lite::opencl::MemType::BUF);
  ocl_runtime->SetKernelArg(kernel_, arg_cn++, out_tensors_[0]->data_c(), lite::opencl::MemType::IMG);
  ocl_runtime->SetKernelArg(kernel_, arg_cn++, src_size);
  ocl_runtime->SetKernelArg(kernel_, arg_cn++, dst_size);
  ocl_runtime->SetKernelArg(kernel_, arg_cn++, indices_num);
  ocl_runtime->SetKernelArg(kernel_, arg_cn++, param->axis_);
  ocl_runtime->RunKernel(kernel_, global, local, nullptr);

  return RET_OK;
}

kernel::LiteKernel *OpenCLGatherKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                              const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                              const lite::Context *ctx, const kernel::KernelKey &desc,
                                              const mindspore::lite::PrimitiveC *primitive) {
  auto *kernel = new (std::nothrow) GatherOpenCLKernel(opParameter, inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Kernel " << opParameter->name_ << " new failed.";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Kernel " << opParameter->name_ << " init failed.";
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Gather, OpenCLGatherKernelCreator);
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Gather, OpenCLGatherKernelCreator);

}  // namespace mindspore::kernel
