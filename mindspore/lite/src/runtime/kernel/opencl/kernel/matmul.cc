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
#include "src/runtime/kernel/opencl/kernel/matmul.h"
#ifndef PROGRAM_WITH_IL
#include "src/runtime/kernel/opencl/cl/matmul.cl.inc"
#endif

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_MatMul;

namespace mindspore::kernel {

int MatMulOpenCLKernel::CheckSpecs() {
  if (in_tensors_.size() != 2 || out_tensors_.size() != 1) {
    MS_LOG(ERROR) << "in size: " << in_tensors_.size() << ", out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  auto param = reinterpret_cast<MatMulParameter *>(op_parameter_);
  transposeA = param->a_transpose_;
  if (transposeA) {
    MS_LOG(ERROR) << "matmul only support a_transpose_=false yet.";
    return mindspore::lite::RET_ERROR;
  }
  transposeB = param->b_transpose_;
  enable_fp16_ = ocl_runtime_->GetFp16Enable();
  if (in_tensors_[0]->shape().size() != out_tensors_[0]->shape().size() || in_tensors_[0]->shape().size() < 2 ||
      in_tensors_[0]->shape().size() > 4) {
    MS_LOG(ERROR) << "matmul only support input shape size= 2, 3 or 4.";
    return mindspore::lite::RET_ERROR;
  }
  if (!in_tensors_.at(kWeightIndex)->IsConst()) {
    MS_LOG(ERROR) << "Matmul don't support non-constant filter yet.";
    return RET_ERROR;
  }
  return RET_OK;
}

int MatMulOpenCLKernel::Prepare() {
  std::string kernel_name = "MatMul_NHWC4";
  dims = in_tensors_[0]->shape().size();
  for (int i = 0; i < dims; i++) {
    inShape[MAX_DIMS - dims + i] = in_tensors_[0]->shape()[i];
    outShape[MAX_DIMS - dims + i] = out_tensors_[0]->shape()[i];
  }
  std::map<int, std::string> dims2str = {{2, "_2d"}, {3, "_4d"}, {4, "_4d"}};
  kernel_name += dims2str[dims];
#ifdef PROGRAM_WITH_IL
  kernel_ = ocl_runtime_->GetKernelFromBinary(kernel_name);
#else
  std::string source = matmul_source;
  std::string program_name = "MatMul";
  ocl_runtime_->LoadSource(program_name, source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name);
#endif
  auto ret = InitWeights();
  if (ret != RET_OK) {
    return ret;
  }
  SetConstArgs();
  SetGlobalLocal();
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return mindspore::lite::RET_OK;
}

int MatMulOpenCLKernel::InitWeights() {
  // ABMCI @ ABCICO = ABMCO
  auto ret = DequantWeight();
  if (ret != RET_OK) {
    return ret;
  }
  auto allocator = ocl_runtime_->GetAllocator();
  int ci = inShape[3];
  int ci4 = UP_DIV(ci, C4NUM);
  int co = outShape[3];
  int co4 = UP_DIV(co, C4NUM);
  int a = inShape[0];
  int b = inShape[1];

  size_t dtype_size = enable_fp16_ ? sizeof(uint16_t) : sizeof(float);
  padWeight_ = allocator->Malloc(a * b * ci4 * co4 * C4NUM * C4NUM * dtype_size);
  padWeight_ = allocator->MapBuffer(padWeight_, CL_MAP_WRITE, nullptr, true);
  auto padWeightFp32 = reinterpret_cast<float *>(padWeight_);
  auto padWeightFp16 = reinterpret_cast<float16_t *>(padWeight_);
  memset(padWeight_, 0x00, a * b * ci4 * co4 * C4NUM * C4NUM * dtype_size);
  auto originWeightFp32 = reinterpret_cast<float *>(in_tensors_.at(kWeightIndex)->data_c());
  auto originWeightFp16 = reinterpret_cast<float16_t *>(in_tensors_.at(kWeightIndex)->data_c());
  bool isModelFp16 = in_tensors_.at(kWeightIndex)->data_type() == kNumberTypeFloat16;

  // pad weight
  // ABCICO -> AB(CI4)(CO4)(4 from CO)(4 from CI)
  // if tranposeB, ABCOCI -> AB(CI4)(CO4)(4 from CO)(4 from CI)
  int index = 0;
  for (int aa = 0; aa < a; aa++) {
    for (int bb = 0; bb < b; bb++) {
      int baseAB = (aa * b + bb) * ci * co;
      for (int i = 0; i < ci4; ++i) {
        for (int j = 0; j < co4; ++j) {
          for (int k = 0; k < C4NUM; ++k) {
            for (int l = 0; l < C4NUM; ++l) {
              int src_ci = i * C4NUM + l;
              int src_co = j * C4NUM + k;
              if (src_ci < ci && src_co < co) {
                int originId = baseAB + src_ci * co + src_co;
                if (transposeB) {
                  originId = baseAB + src_co * ci + src_ci;
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
  FreeDequantedWeight();
  return RET_OK;
}

void MatMulOpenCLKernel::SetGlobalLocal() {
  // local size should less than MAX_GROUP_SIZE
  local_size_ = {32, 4, 1};
  global_size_ = {UP_DIV(static_cast<size_t>(outShape[3]), C4NUM),
                  4 * static_cast<size_t>(outShape[0]) * static_cast<size_t>(outShape[1]),
                  static_cast<size_t>(outShape[2])};
  AlignGlobalLocal(global_size_, local_size_);
}

void MatMulOpenCLKernel::SetConstArgs() {
  int arg_count = 2;
  cl_int4 in_shape = {inShape[0], inShape[1], inShape[2], inShape[3]};
  cl_int4 out_shape = {outShape[0], outShape[1], outShape[2], outShape[3]};
  ocl_runtime_->SetKernelArg(kernel_, arg_count++, padWeight_, lite::opencl::MemType::BUF);
  ocl_runtime_->SetKernelArg(kernel_, arg_count++, in_shape);
  ocl_runtime_->SetKernelArg(kernel_, arg_count++, out_shape);
}

int MatMulOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  int arg_count = 0;
  ocl_runtime_->SetKernelArg(kernel_, arg_count++, in_tensors_[0]->data_c());
  ocl_runtime_->SetKernelArg(kernel_, arg_count++, out_tensors_[0]->data_c());
  ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_);
  return mindspore::lite::RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_MatMul, OpenCLKernelCreator<MatMulOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_MatMul, OpenCLKernelCreator<MatMulOpenCLKernel>)
}  // namespace mindspore::kernel
