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
#include "nnacl/fp32/common_func.h"
#include "src/kernel_registry.h"
#include "src/runtime/opencl/opencl_runtime.h"
#include "nnacl/fp32/matmul.h"
#include "src/runtime/kernel/opencl/kernel/matmul.h"
#ifndef PROGRAM_WITH_IL
#include "src/runtime/kernel/opencl/cl/matmul.cl.inc"
#endif

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_FullConnection;
using mindspore::schema::PrimitiveType_MatMul;

namespace mindspore::kernel {

int MatMulOpenCLKernel::Init() {
  std::string kernel_name = "MatMul";
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  enable_fp16_ = ocl_runtime->GetFp16Enable();
#ifdef PROGRAM_WITH_IL
  kernel_ = ocl_runtime->GetKernelFromBinary(kernel_name);
#else
  std::set<std::string> build_options;
  std::string source = matmul_source;
  std::string program_name = "MatMul";
  ocl_runtime->LoadSource(program_name, source);
  ocl_runtime->BuildKernel(kernel_, program_name, kernel_name, build_options);
#endif
  int ci, co;
  if (in_tensors_[1]->shape().size() == 2) {
    ci = in_tensors_[1]->shape()[1];
    co = in_tensors_[1]->shape()[0];
  } else {
    ci = in_tensors_[1]->shape()[3];
    co = in_tensors_[1]->shape()[0];
  }

  sizeCI = {ci, UP_DIV(ci, C4NUM)};
  sizeCO = {co, UP_DIV(co, C4NUM)};
  PadWeight();
  in_ori_format_ = in_tensors_[0]->GetFormat();
  in_tensors_[0]->SetFormat(schema::Format_NHWC4);
  out_ori_format_ = out_tensors_[0]->GetFormat();
  out_tensors_[0]->SetFormat(schema::Format_NHWC4);
  if (out_tensors_[0]->shape().size() == 2) {
    out_ori_format_ = schema::Format_NC;
    out_tensors_[0]->SetFormat(schema::Format_NC4);
    in_ori_format_ = schema::Format_NC;
    in_tensors_[0]->SetFormat(schema::Format_NC4);
  }
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}

int MatMulOpenCLKernel::ReSize() { return RET_OK; }

void MatMulOpenCLKernel::PadWeight() {
  auto allocator = lite::opencl::OpenCLRuntime::GetInstance()->GetAllocator();

  size_t dtype_size = enable_fp16_ ? sizeof(float16_t) : sizeof(float);
  padWeight_ = allocator->Malloc(sizeCI.s[1] * sizeCO.s[1] * C4NUM * C4NUM * dtype_size);
  padWeight_ = allocator->MapBuffer(padWeight_, CL_MAP_WRITE, nullptr, true);

  auto origin_weight = in_tensors_.at(kWeightIndex)->Data();
  int divCI = sizeCI.s[1];
  int divCO = sizeCO.s[1];
  int co = sizeCO.s[0];
  int index = 0;
  for (int i = 0; i < divCI; ++i) {
    for (int j = 0; j < divCO; ++j) {
      for (int k = 0; k < C4NUM; ++k) {
        for (int l = 0; l < C4NUM; ++l) {
          int src_x = i * C4NUM + l;
          int src_y = j * C4NUM + k;
          if (src_x < sizeCI.s[0] && src_y < sizeCO.s[0]) {
            if (enable_fp16_) {
              if (in_tensors_.at(kWeightIndex)->data_type() == kNumberTypeFloat32) {
                reinterpret_cast<uint16_t *>(padWeight_)[index++] =
                  Float32ToShort(reinterpret_cast<float *>(origin_weight)[src_y * sizeCI.s[0] + src_x]);
              } else {
                reinterpret_cast<uint16_t *>(padWeight_)[index++] =
                  reinterpret_cast<uint16_t *>(origin_weight)[src_y * sizeCI.s[0] + src_x];
              }
            } else {
              if (in_tensors_.at(kWeightIndex)->data_type() == kNumberTypeFloat16) {
                reinterpret_cast<float *>(padWeight_)[index++] =
                  ShortToFloat32(reinterpret_cast<uint16_t *>(origin_weight)[src_y * sizeCI.s[0] + src_x]);
              } else {
                reinterpret_cast<float *>(padWeight_)[index++] =
                  reinterpret_cast<float *>(origin_weight)[src_y * sizeCI.s[0] + src_x];
              }
            }
          } else {
            if (enable_fp16_) {
              reinterpret_cast<float16_t *>(padWeight_)[index++] = 0;
            } else {
              reinterpret_cast<float *>(padWeight_)[index++] = 0;
            }
          }
        }
      }
    }
  }

  size_t im_dst_x, im_dst_y;
  im_dst_x = divCO;
  im_dst_y = 1;
  size_t img_dtype = CL_FLOAT;
  if (enable_fp16_) {
    img_dtype = CL_HALF_FLOAT;
  }
  std::vector<size_t> img_size{im_dst_x, im_dst_y, img_dtype};
  bias_ = allocator->Malloc(im_dst_x * im_dst_y * C4NUM * dtype_size, img_size);
  bias_ = allocator->MapBuffer(bias_, CL_MAP_WRITE, nullptr, true);
  memset(bias_, 0x00, divCO * C4NUM * dtype_size);
  if (in_tensors_.size() >= 3) {
    if (in_tensors_[2]->data_type() == kNumberTypeFloat32 && enable_fp16_) {
      auto fdata = reinterpret_cast<float *>(in_tensors_[2]->Data());
      for (int i = 0; i < co; i++) {
        reinterpret_cast<uint16_t *>(bias_)[i] = Float32ToShort(fdata[i]);
      }
    } else {
      memcpy(bias_, in_tensors_[2]->Data(), co * dtype_size);
    }
  }
  allocator->UnmapBuffer(bias_);
}

int MatMulOpenCLKernel::GetImageSize(size_t idx, std::vector<size_t> *img_size) {
  size_t im_dst_x, im_dst_y;
  im_dst_x = sizeCO.s[1];
  im_dst_y = 1;
  size_t img_dtype = CL_FLOAT;
  if (enable_fp16_) {
    img_dtype = CL_HALF_FLOAT;
  }
  img_size->clear();
  std::vector<size_t> vec{im_dst_x, im_dst_y, img_dtype};
  *img_size = vec;
  return RET_OK;
}

int MatMulOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  // local size should less than MAX_GROUP_SIZE
  std::vector<size_t> local = {64, 4};
  std::vector<size_t> global = {UP_ROUND(sizeCO.s[1], local[0]), 4};
  int arg_count = 0;
  ocl_runtime->SetKernelArg(kernel_, arg_count++, in_tensors_[0]->Data());
  ocl_runtime->SetKernelArg(kernel_, arg_count++, padWeight_);
  ocl_runtime->SetKernelArg(kernel_, arg_count++, bias_);
  ocl_runtime->SetKernelArg(kernel_, arg_count++, out_tensors_[0]->Data());
  ocl_runtime->SetKernelArg(kernel_, arg_count++, sizeCI);
  ocl_runtime->SetKernelArg(kernel_, arg_count++, sizeCO);
  ocl_runtime->SetKernelArg(kernel_, arg_count++, hasBias_ ? 1 : 0);
  ocl_runtime->RunKernel(kernel_, global, local, nullptr);
  return RET_OK;
}

kernel::LiteKernel *OpenCLMatMulKernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                              const std::vector<lite::tensor::Tensor *> &outputs,
                                              OpParameter *opParameter, const lite::Context *ctx,
                                              const kernel::KernelKey &desc,
                                              const mindspore::lite::PrimitiveC *primitive) {
  bool hasBias = false;
  if (opParameter->type_ == PrimitiveType_FullConnection) {
    hasBias = (reinterpret_cast<MatMulParameter *>(opParameter))->has_bias_;
  }
  auto *kernel =
    new (std::nothrow) MatMulOpenCLKernel(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs, hasBias);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel " << opParameter->name_ << "is nullptr.";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_MatMul, OpenCLMatMulKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_FullConnection, OpenCLMatMulKernelCreator)
}  // namespace mindspore::kernel
