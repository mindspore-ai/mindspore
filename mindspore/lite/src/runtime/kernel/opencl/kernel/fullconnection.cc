/**
 * Copyright 2019 Huawei Technologies n., Ltd
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
#include <map>
#include "nnacl/fp32/common_func_fp32.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/kernel/fullconnection.h"
#include "src/runtime/kernel/opencl/utils.h"
#ifndef PROGRAM_WITH_IL
#include "src/runtime/kernel/opencl/cl/fullconnection.cl.inc"
#endif

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::opencl::ImageSize;
using mindspore::schema::ActivationType_RELU;
using mindspore::schema::ActivationType_RELU6;
using mindspore::schema::ActivationType_TANH;
using mindspore::schema::PrimitiveType_FullConnection;

namespace mindspore::kernel {

int FullConnectionOpenCLKernel::CheckSpecs() {
  if ((in_tensors_.size() != 2 && in_tensors_.size() != 3) || out_tensors_.size() != 1) {
    MS_LOG(ERROR) << "in size: " << in_tensors_.size() << ", out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  auto param = reinterpret_cast<MatMulParameter *>(op_parameter_);
  if (param->a_transpose_) {
    MS_LOG(ERROR) << "fullconnection only support a_transpose_=false yet.";
    return RET_ERROR;
  }
  auto out_gpu_info = GpuTensorInfo(out_tensors_[0]);
  if (out_gpu_info.H != 1 || out_gpu_info.W != 1) {
    MS_LOG(ERROR) << "fullconnection only support 2d output shape or 4d output but H=W=1";
    return RET_ERROR;
  }
  // for fusion: ActivationType_TANH
  if (param->act_type_ != ActType_No && param->act_type_ != ActType_Relu && param->act_type_ != ActType_Relu6 &&
      static_cast<schema::ActivationType>(param->act_type_) != ActivationType_TANH) {
    MS_LOG(ERROR) << "Unsupported activation type " << param->act_type_;
    return RET_ERROR;
  }
  // for fusion: ActivationType_TANH
  switch (static_cast<int>(param->act_type_)) {
    case ActType_No:
    case ActType_Relu:
    case ActType_Relu6:
    case ActivationType_TANH:
      break;
    default: {
      MS_LOG(ERROR) << "Unsupported activation type " << param->act_type_;
      return RET_ERROR;
    }
  }
  N_ = out_gpu_info.N;
  CO_ = out_gpu_info.C;
  auto intensor_shape = GpuTensorInfo(in_tensors_[0]);
  int input_nhw = intensor_shape.N * intensor_shape.H * intensor_shape.W;
  if (input_nhw < N_) {
    MS_LOG(ERROR) << "Unsupported fullconnection shape";
  }
  if (!in_tensors_.at(kWeightIndex)->IsConst()) {
    weight_var_ = true;
    if (!param->b_transpose_) {
      MS_LOG(ERROR) << "If fullconnection input weight is not constant, b_transpose_ should be true.";
      return RET_ERROR;
    }
    if (in_tensors_.at(kWeightIndex)->shape().size() != 2) {
      MS_LOG(ERROR) << "If fullconnection input weight is not constant, it should be 2d.";
      return RET_ERROR;
    }
    if (intensor_shape.C != in_tensors_.at(kWeightIndex)->shape()[1]) {
      MS_LOG(ERROR)
        << "If fullconnection input weight is not constant, input channel should equal to weight in_channel.";
      return RET_ERROR;
    }
  }
  if (in_tensors_.size() == 3 && !in_tensors_.at(2)->IsConst()) {
    MS_LOG(ERROR) << "FullConnection don't support non-constant bias yet.";
    return RET_ERROR;
  }
  CI_remainder_ = input_nhw / N_;
  return RET_OK;
}

int FullConnectionOpenCLKernel::Prepare() {
  auto param = reinterpret_cast<MatMulParameter *>(op_parameter_);
  transposeA = param->a_transpose_;
  transposeB = param->b_transpose_;
  enable_fp16_ = ocl_runtime_->GetFp16Enable();

  std::string kernel_name = "FullConnection";
  if (weight_var_) {
    kernel_name = "FullConnectionWeightVar";
  }
#ifdef PROGRAM_WITH_IL
  kernel_ = ocl_runtime_->GetKernelFromBinary(kernel_name);
#else
  std::string source = fullconnection_source;
  std::string program_name = "FullConnection";
  ocl_runtime_->LoadSource(program_name, GetActDefines() + source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name);
#endif
  auto ret = InitWeights();
  if (ret != RET_OK) {
    return ret;
  }
  SetConstArgs();
  SetGlobalLocal();
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}

int FullConnectionOpenCLKernel::InitWeights() {
  if (!weight_var_) {
    auto ret = InitFilter();
    if (ret != RET_OK) {
      return ret;
    }
  }
  return InitBias();
}  // namespace mindspore::kernel

int FullConnectionOpenCLKernel::InitFilter() {
  auto ret = DequantWeight();
  if (ret != RET_OK) {
    return ret;
  }
  auto allocator = ocl_runtime_->GetAllocator();
  auto intensor_shape = GpuTensorInfo(in_tensors_[0]);
  int co4 = UP_DIV(CO_, C4NUM);
  int nhw_remainder = intensor_shape.N * intensor_shape.H * intensor_shape.W / N_;
  size_t dtype_size = enable_fp16_ ? sizeof(uint16_t) : sizeof(float);
  padWeight_ = allocator->Malloc(nhw_remainder * intensor_shape.Slice * co4 * C4NUM * C4NUM * dtype_size);
  padWeight_ = allocator->MapBuffer(padWeight_, CL_MAP_WRITE, nullptr, true);
  auto padWeightFp32 = reinterpret_cast<float *>(padWeight_);
  auto padWeightFp16 = reinterpret_cast<float16_t *>(padWeight_);
  memset(padWeight_, 0x00, nhw_remainder * intensor_shape.Slice * co4 * C4NUM * C4NUM * dtype_size);
  auto originWeightFp32 = reinterpret_cast<float *>(in_tensors_.at(kWeightIndex)->data_c());
  auto originWeightFp16 = reinterpret_cast<float16_t *>(in_tensors_.at(kWeightIndex)->data_c());
  bool isModelFp16 = in_tensors_.at(kWeightIndex)->data_type() == kNumberTypeFloat16;

  // pad weight
  // HWCICO -> (HWCI4)(CO4)(4 from CO)(4 from CI)
  // if tranposeB, COHWCI -> (HWCI4)(CO4)(4 from CO)(4 from CI)
  int index = 0;
  for (int nhw = 0; nhw < nhw_remainder; nhw++) {
    for (int i = 0; i < intensor_shape.Slice; ++i) {
      for (int j = 0; j < co4; ++j) {
        for (int k = 0; k < C4NUM; ++k) {
          for (int l = 0; l < C4NUM; ++l) {
            int src_ci = i * C4NUM + l;
            int src_co = j * C4NUM + k;
            if (src_ci < intensor_shape.C && src_co < CO_) {
              int originId = (nhw * intensor_shape.C + src_ci) * CO_ + src_co;
              if (transposeB) {
                originId = src_co * intensor_shape.C * nhw_remainder + nhw * intensor_shape.C + src_ci;
              }
              if (enable_fp16_) {
                if (!isModelFp16) {
                  padWeightFp16[index++] = originWeightFp32[originId];
                } else {
                  padWeightFp16[index++] = originWeightFp16[originId];
                }
              } else {
                if (!isModelFp16) {
                  padWeightFp32[index++] = originWeightFp32[originId];
                } else {
                  padWeightFp32[index++] = originWeightFp16[originId];
                }
              }
            } else {
              index++;
            }
          }
        }
      }
    }
  }
  allocator->UnmapBuffer(padWeight_);
  FreeDequantedWeight();
  return RET_OK;
}

int FullConnectionOpenCLKernel::InitBias() {
  // pad FC Bias
  auto allocator = ocl_runtime_->GetAllocator();
  int co4 = UP_DIV(CO_, C4NUM);
  size_t dtype_size = enable_fp16_ ? sizeof(uint16_t) : sizeof(float);
  size_t im_dst_x, im_dst_y;
  im_dst_x = co4;
  im_dst_y = 1;
  size_t img_dtype = CL_FLOAT;
  if (enable_fp16_) {
    img_dtype = CL_HALF_FLOAT;
  }
  ImageSize img_size{im_dst_x, im_dst_y, img_dtype};
  bias_ = allocator->Malloc(img_size);
  bias_ = allocator->MapBuffer(bias_, CL_MAP_WRITE, nullptr, true);
  memset(bias_, 0x00, co4 * C4NUM * dtype_size);
  if (in_tensors_.size() == 3) {
    if (in_tensors_[2]->data_type() == kNumberTypeFloat32 && enable_fp16_) {
      for (int i = 0; i < CO_; i++) {
        reinterpret_cast<float16_t *>(bias_)[i] = reinterpret_cast<float *>(in_tensors_[2]->data_c())[i];
      }
    } else if (in_tensors_[2]->data_type() == kNumberTypeFloat16 && !enable_fp16_) {
      for (int i = 0; i < CO_; i++) {
        reinterpret_cast<float *>(bias_)[i] = reinterpret_cast<float16_t *>(in_tensors_[2]->data_c())[i];
      }
    } else {
      memcpy(bias_, in_tensors_[2]->data_c(), CO_ * dtype_size);
    }
  }
  allocator->UnmapBuffer(bias_);
  return RET_OK;
}

void FullConnectionOpenCLKernel::SetGlobalLocal() {
  local_size_ = {32, 4, 1};
  size_t CO = CO_;
  size_t N = N_;
  global_size_ = {UP_DIV(CO, C4NUM), 4, N};
  AlignGlobalLocal(global_size_, local_size_);
}

void FullConnectionOpenCLKernel::SetConstArgs() {
  if (!weight_var_) {
    ocl_runtime_->SetKernelArg(kernel_, 2, padWeight_, lite::opencl::MemType::BUF);
  }
  int arg_count = 3;
  ocl_runtime_->SetKernelArg(kernel_, arg_count++, bias_);
  ocl_runtime_->SetKernelArg(kernel_, arg_count++, N_);
  auto intensor_shape = GpuTensorInfo(in_tensors_[0]);
  int CI4 = CI_remainder_ * intensor_shape.Slice;
  ocl_runtime_->SetKernelArg(kernel_, arg_count++, CI4);
  ocl_runtime_->SetKernelArg(kernel_, arg_count++, UP_DIV(CO_, C4NUM));
  auto in_shape_info = GpuTensorInfo(in_tensors_[0]);
  cl_int2 in_img_shape = {static_cast<int>(in_shape_info.height), static_cast<int>(in_shape_info.width)};
  ocl_runtime_->SetKernelArg(kernel_, arg_count++, in_img_shape);
  auto *param = reinterpret_cast<MatMulParameter *>(op_parameter_);
  ocl_runtime_->SetKernelArg(kernel_, arg_count, static_cast<cl_int>(param->act_type_));
}

int FullConnectionOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  int arg_count = 0;
  ocl_runtime_->SetKernelArg(kernel_, arg_count++, in_tensors_[0]->data_c());
  ocl_runtime_->SetKernelArg(kernel_, arg_count++, out_tensors_[0]->data_c());
  if (weight_var_) {
    ocl_runtime_->SetKernelArg(kernel_, arg_count++, in_tensors_[1]->data_c());
  }
  ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_);
  return RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_FullConnection, OpenCLKernelCreator<FullConnectionOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_FullConnection, OpenCLKernelCreator<FullConnectionOpenCLKernel>)
}  // namespace mindspore::kernel
