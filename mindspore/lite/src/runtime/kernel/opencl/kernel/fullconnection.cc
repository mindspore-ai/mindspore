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
#include "nnacl/fp32/common_func.h"
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
using mindspore::schema::PrimitiveType_FullConnection;

namespace mindspore::kernel {

int FullConnectionOpenCLKernel::Init() {
  // deleted soon
  return CheckSpecs();
}

int FullConnectionOpenCLKernel::CheckSpecs() {
  auto param = reinterpret_cast<MatMulParameter *>(op_parameter_);
  transposeA = param->a_transpose_;
  if (transposeA) {
    MS_LOG(ERROR) << "fullconnection only support a_transpose_=false yet.";
    return RET_ERROR;
  }
  transposeB = param->b_transpose_;
  enable_fp16_ = ocl_runtime_->GetFp16Enable();
  if ((in_tensors_[0]->shape().size() != 4 && in_tensors_[0]->shape().size() != 2) ||
      (out_tensors_[0]->shape().size() != 4 && out_tensors_[0]->shape().size() != 2)) {
    MS_LOG(ERROR) << "fullconnection only support input output shape size = 2 or 4";
    return RET_ERROR;
  }
  switch (param->act_type_) {
    case ActType_No:
      break;
    case ActType_Relu:
      activation_min_ = 0.f;
      break;
    case ActType_Relu6:
      activation_min_ = 0.f;
      activation_max_ = 6.f;
      break;
    default:
      MS_LOG(ERROR) << "Unsupported activation type " << param->act_type_;
      return RET_ERROR;
  }
  return RET_OK;
}

int FullConnectionOpenCLKernel::Prepare() {
  std::string kernel_name = "FullConnection_NHWC4";
  inShape = Image2DInfo(in_tensors_[0]);
  outShape = Image2DInfo(out_tensors_[0]);
#ifdef PROGRAM_WITH_IL
  kernel_ = ocl_runtime_->GetKernelFromBinary(kernel_name);
#else
  std::set<std::string> build_options;
  std::string source = fullconnection_source;
  std::string program_name = "FullConnection";
  ocl_runtime_->LoadSource(program_name, source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options);
#endif
  InitWeights();
  SetConstArgs();
  SetGlobalLocal();
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}

int FullConnectionOpenCLKernel::InitWeights() {
  auto allocator = ocl_runtime_->GetAllocator();
  int ci = inShape.C;
  int ci4 = UP_DIV(ci, C4NUM);
  int co = outShape.C;
  int co4 = UP_DIV(co, C4NUM);
  int h = inShape.H;
  int w = inShape.W;

  size_t dtype_size = enable_fp16_ ? sizeof(uint16_t) : sizeof(float);
  padWeight_ = allocator->Malloc(h * w * ci4 * co4 * C4NUM * C4NUM * dtype_size);
  padWeight_ = allocator->MapBuffer(padWeight_, CL_MAP_WRITE, nullptr, true);
  auto padWeightFp32 = reinterpret_cast<float *>(padWeight_);
  auto padWeightFp16 = reinterpret_cast<float16_t *>(padWeight_);
  memset(padWeight_, 0x00, h * w * ci4 * co4 * C4NUM * C4NUM * dtype_size);
  auto originWeightFp32 = reinterpret_cast<float *>(in_tensors_.at(kWeightIndex)->data_c());
  auto originWeightFp16 = reinterpret_cast<float16_t *>(in_tensors_.at(kWeightIndex)->data_c());
  bool isModelFp16 = in_tensors_.at(kWeightIndex)->data_type() == kNumberTypeFloat16;

  // pad weight
  // HWCICO -> (HWCI4)(CO4)(4 from CO)(4 from CI)
  // if tranposeB, COHWCI -> (HWCI4)(CO4)(4 from CO)(4 from CI)
  int index = 0;
  for (int hh = 0; hh < h; hh++) {
    for (int ww = 0; ww < w; ww++) {
      int baseHW = hh * w + ww;
      for (int i = 0; i < ci4; ++i) {
        for (int j = 0; j < co4; ++j) {
          for (int k = 0; k < C4NUM; ++k) {
            for (int l = 0; l < C4NUM; ++l) {
              int src_ci = i * C4NUM + l;
              int src_co = j * C4NUM + k;
              if (src_ci < ci && src_co < co) {
                int originId = baseHW * ci * co + src_ci * co + src_co;
                if (transposeB) {
                  originId = src_co * ci * h * w + baseHW * ci + src_ci;
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
  }
  allocator->UnmapBuffer(padWeight_);

  // pad FC Bias
  size_t im_dst_x, im_dst_y;
  im_dst_x = co4;
  im_dst_y = 1;
  size_t img_dtype = CL_FLOAT;
  if (enable_fp16_) {
    img_dtype = CL_HALF_FLOAT;
  }
  std::vector<size_t> img_size{im_dst_x, im_dst_y, img_dtype};
  bias_ = allocator->Malloc(im_dst_x * im_dst_y * C4NUM * dtype_size, img_size);
  bias_ = allocator->MapBuffer(bias_, CL_MAP_WRITE, nullptr, true);
  memset(bias_, 0x00, co4 * C4NUM * dtype_size);
  if (in_tensors_.size() >= 3) {
    if (in_tensors_[2]->data_type() == kNumberTypeFloat32 && enable_fp16_) {
      for (int i = 0; i < co; i++) {
        reinterpret_cast<float16_t *>(bias_)[i] = reinterpret_cast<float *>(in_tensors_[2]->data_c())[i];
      }
    } else if (in_tensors_[2]->data_type() == kNumberTypeFloat16 && !enable_fp16_) {
      for (int i = 0; i < co; i++) {
        reinterpret_cast<float *>(bias_)[i] = reinterpret_cast<float16_t *>(in_tensors_[2]->data_c())[i];
      }
    } else {
      memcpy(bias_, in_tensors_[2]->data_c(), co * dtype_size);
    }
  }
  allocator->UnmapBuffer(bias_);
  return RET_OK;
}

void FullConnectionOpenCLKernel::SetGlobalLocal() {
  std::vector<size_t> local = {32, 4, 1};
  std::vector<size_t> global = {UP_DIV(outShape.C, C4NUM), 4, outShape.N};
  AlignGlobalLocal(global, local);
}

void FullConnectionOpenCLKernel::SetConstArgs() {
  int arg_count = 2;
  cl_int4 in_shape = {static_cast<int>(inShape.N), static_cast<int>(inShape.H), static_cast<int>(inShape.W),
                      static_cast<int>(inShape.C)};
  cl_int2 out_shape = {static_cast<int>(outShape.N), static_cast<int>(outShape.C)};
  ocl_runtime_->SetKernelArg(kernel_, arg_count++, padWeight_, lite::opencl::MemType::BUF);
  ocl_runtime_->SetKernelArg(kernel_, arg_count++, bias_);
  ocl_runtime_->SetKernelArg(kernel_, arg_count++, in_shape);
  ocl_runtime_->SetKernelArg(kernel_, arg_count++, out_shape);
  ocl_runtime_->SetKernelArg(kernel_, arg_count++, activation_min_);
  ocl_runtime_->SetKernelArg(kernel_, arg_count++, activation_max_);
}

int FullConnectionOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  int arg_count = 0;
  ocl_runtime_->SetKernelArg(kernel_, arg_count++, in_tensors_[0]->data_c());
  ocl_runtime_->SetKernelArg(kernel_, arg_count++, out_tensors_[0]->data_c());
  ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr);
  return RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_FullConnection, OpenCLKernelCreator<FullConnectionOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_FullConnection, OpenCLKernelCreator<FullConnectionOpenCLKernel>)
}  // namespace mindspore::kernel
