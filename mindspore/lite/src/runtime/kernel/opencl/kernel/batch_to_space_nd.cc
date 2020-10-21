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
#include "src/runtime/kernel/opencl/kernel/batch_to_space_nd.h"
#include "src/runtime/kernel/opencl/cl/batch_to_space_nd.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_BatchToSpace;
using mindspore::schema::PrimitiveType_BatchToSpaceND;

namespace mindspore::kernel {

int BatchToSpaceNDOpenCLKernel::Init() {
  std::string kernel_name = "batch_to_space_nd";
  auto in_format = op_format_;
  if (in_tensors_[0]->shape().size() != 4 && out_tensors_[0]->shape().size() != 4) {
    MS_LOG(ERROR) << "input/output shape size must be 4, actual: " << in_tensors_[0]->shape().size() << ", "
                  << out_tensors_[0]->shape().size();
    return RET_ERROR;
  }
  if (in_format != schema::Format_NHWC4 && in_format != schema::Format_NC4HW4) {
    MS_LOG(ERROR) << "input format(" << in_format << ") "
                  << "format not support!";
    return RET_ERROR;
  }
  auto *param = reinterpret_cast<BatchToSpaceParameter *>(this->op_parameter_);
  if (param->block_shape_[0] < 1 || param->block_shape_[1] < 1) {
    MS_LOG(ERROR) << "block_sizes_ must > 1, actual " << param->block_shape_[0] << ", " << param->block_shape_[1];
    return RET_ERROR;
  }
  if (in_tensors_[0]->shape()[kNHWC_H] * param->block_shape_[0] <= (param->crops_[0] + param->crops_[1]) ||
      in_tensors_[0]->shape()[kNHWC_W] * param->block_shape_[1] <= (param->crops_[2] + param->crops_[3])) {
    MS_LOG(ERROR) << "crop shape error!";
    return RET_ERROR;
  }

  in_ori_format_ = in_tensors_[0]->GetFormat();
  out_ori_format_ = out_tensors_[0]->GetFormat();
  in_tensors_[0]->SetFormat(op_format_);
  out_tensors_[0]->SetFormat(op_format_);
#ifdef PROGRAM_WITH_IL
  kernel_ = ocl_runtime_->GetKernelFromBinary(kernel_name);
#else
  if (in_format == schema::Format_NC4HW4) {
    kernel_name += "_NC4HW4";
  } else {
    kernel_name += "_NHWC4";
  }
  std::set<std::string> build_options;
  std::string source = batch_to_space_nd_source;
  std::string program_name = "batch_to_space_nd";
  ocl_runtime_->LoadSource(program_name, source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options);
#endif
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}
int BatchToSpaceNDOpenCLKernel::InitBuffer() { return RET_OK; }
int BatchToSpaceNDOpenCLKernel::ReSize() { return RET_OK; }
int BatchToSpaceNDOpenCLKernel::GetImageSize(size_t idx, std::vector<size_t> *img_size) {
  size_t CO4 = UP_DIV(out_tensors_[0]->Channel(), C4NUM);
  size_t im_dst_x, im_dst_y;
  if (in_tensors_[0]->GetFormat() == schema::Format::Format_NHWC4) {
    im_dst_x = out_tensors_[0]->Width() * CO4;
    im_dst_y = out_tensors_[0]->Height() * out_tensors_[0]->Batch();
  } else {
    im_dst_y = out_tensors_[0]->Batch() * out_tensors_[0]->Height() * CO4;
    im_dst_x = out_tensors_[0]->Width();
  }
  size_t img_dtype = CL_FLOAT;
  auto enable_fp16_ = ocl_runtime_->GetFp16Enable();
  if (enable_fp16_) {
    img_dtype = CL_HALF_FLOAT;
  }
  img_size->clear();
  std::vector<size_t> vec{im_dst_x, im_dst_y, img_dtype};
  *img_size = std::move(vec);
  return RET_OK;
}
int BatchToSpaceNDOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running! ";
  auto param = reinterpret_cast<BatchToSpaceParameter *>(this->op_parameter_);

  size_t CO4 = UP_DIV(out_tensors_[0]->Channel(), C4NUM);
  size_t CI4 = UP_DIV(in_tensors_[0]->Channel(), C4NUM);
  cl_int4 src_size = {(cl_int)CI4, in_tensors_[0]->Width(), in_tensors_[0]->Height() * out_tensors_[0]->Batch(), 1};
  std::vector<int> out_shape = out_tensors_[0]->shape();
  cl_int4 dst_size = {(cl_int)CO4, out_shape[2], out_shape[1], out_shape[0]};
  cl_int2 block_size = {param->block_shape_[0], param->block_shape_[1]};
  cl_int4 paddings = {param->crops_[0], param->crops_[1], param->crops_[2], param->crops_[3]};
  std::vector<size_t> local = {1, 1, 1};
  std::vector<size_t> global = {(size_t)dst_size.s[0], (size_t)dst_size.s[1], (size_t)dst_size.s[2]};
  int arg_cn = 0;
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, in_tensors_[0]->data_c(), lite::opencl::MemType::IMG);
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, out_tensors_[0]->data_c(), lite::opencl::MemType::IMG);
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, src_size);
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, dst_size);
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, block_size);
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, paddings);
  ocl_runtime_->RunKernel(kernel_, global, local, nullptr);

  return RET_OK;
}

kernel::LiteKernel *OpenCLBatchToSpaceNDKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                      const std::vector<lite::Tensor *> &outputs,
                                                      OpParameter *opParameter, const lite::InnerContext *ctx,
                                                      const kernel::KernelKey &desc,
                                                      const mindspore::lite::PrimitiveC *primitive) {
  auto *kernel = new (std::nothrow) BatchToSpaceNDOpenCLKernel(opParameter, inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Kernel " << opParameter->name_ << " new failed.";
    free(opParameter);
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

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_BatchToSpaceND, OpenCLBatchToSpaceNDKernelCreator);
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_BatchToSpaceND, OpenCLBatchToSpaceNDKernelCreator);
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_BatchToSpace, OpenCLBatchToSpaceNDKernelCreator);
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_BatchToSpace, OpenCLBatchToSpaceNDKernelCreator);

}  // namespace mindspore::kernel
