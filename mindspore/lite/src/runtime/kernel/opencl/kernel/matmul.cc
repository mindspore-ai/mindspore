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
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/kernel/matmul.h"
#include "src/runtime/kernel/opencl/kernel/strassen.h"

#ifndef PROGRAM_WITH_IL
#include "src/runtime/kernel/opencl/cl/matmul.cl.inc"
#endif

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_MatMul;

namespace mindspore::kernel {

bool IsUseStrassenMatmul(const std::vector<lite::Tensor *> &in_tensors_) {
  if (in_tensors_.at(0)->shape().size() == 2) {
    auto shape0 = in_tensors_.at(0)->shape();
    auto shape1 = in_tensors_.at(1)->shape();
    if (in_tensors_.at(1)->IsConst() && (shape0[0] == shape0[1]) && (shape1[0] == shape1[1]) &&
        (shape0[0] == shape1[0]) && (shape0[0] % 8 == 0)) {
      return true;
    } else {
      return false;
    }
  } else {
    return false;
  }
}

int MatMulOpenCLKernel::CheckSpecs() {
  if (in_tensors_.size() != 2 || out_tensors_.size() != 1) {
    MS_LOG(ERROR) << "in size: " << in_tensors_.size() << ", out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  auto param = reinterpret_cast<MatMulParameter *>(op_parameter_);
  transposeA = param->a_transpose_;
  if (transposeA) {
    MS_LOG(ERROR) << "matmul only support a_transpose_=false yet.";
    return RET_ERROR;
  }
  transposeB = param->b_transpose_;
  act_weight_ = !in_tensors_[1]->IsConst();
  enable_fp16_ = ocl_runtime_->GetFp16Enable();
  if (in_tensors_[0]->shape().size() != out_tensors_[0]->shape().size() || in_tensors_[0]->shape().size() < 2 ||
      in_tensors_[0]->shape().size() > 4) {
    MS_LOG(ERROR) << "matmul only support input shape size= 2, 3 or 4.";
    return RET_ERROR;
  }
  return RET_OK;
}

int MatMulOpenCLKernel::Prepare() {
  std::string kernel_name = "MatMul";
  if (act_weight_) {
    if (transposeB) {
      kernel_name = "MatMulActWeightTransposeB";
    } else {
      kernel_name = "MatMulActWeight";
    }
  }
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
  SetConstArgs();
  SetGlobalLocal();
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}

int MatMulOpenCLKernel::InitWeights() {
  if (!in_tensors_[1]->IsConst()) {
    return RET_OK;
  }
  // ABMCI @ ABCICO = ABMCO
  auto ret = DequantWeight();
  if (ret != RET_OK) {
    return ret;
  }
  auto allocator = ocl_runtime_->GetAllocator();
  auto weight_shape = in_tensors_[1]->shape();
  int weight_ndim = weight_shape.size();
  std::vector<int> weight_shape_4d(MAX_DIMS, 1);
  for (int i = 0; i < weight_ndim; i++) {
    weight_shape_4d[MAX_DIMS - weight_ndim + i] = weight_shape[i];
  }
  auto param = reinterpret_cast<MatMulParameter *>(op_parameter_);
  transposeB = param->b_transpose_;
  enable_fp16_ = ocl_runtime_->GetFp16Enable();
  int ci, co;
  if (transposeB) {
    ci = weight_shape_4d[3];
    co = weight_shape_4d[2];
  } else {
    ci = weight_shape_4d[2];
    co = weight_shape_4d[3];
  }
  int ci4 = UP_DIV(ci, C4NUM);
  int co4 = UP_DIV(co, C4NUM);
  int a = weight_shape_4d[0];
  int b = weight_shape_4d[1];

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
  global_size_ = {1, 1, 1};
  global_size_ = {UP_DIV(static_cast<size_t>(outShape[3]), C4NUM),
                  4 * static_cast<size_t>(outShape[0]) * static_cast<size_t>(outShape[1]),
                  static_cast<size_t>(outShape[2])};
  AlignGlobalLocal(global_size_, local_size_);
}

void MatMulOpenCLKernel::SetConstArgs() {
  int arg_count = 2;
  cl_int4 in_shape = {inShape[0], inShape[1], inShape[2], inShape[3]};
  cl_int4 out_shape = {outShape[0], outShape[1], outShape[2], outShape[3]};
  if (act_weight_) {
    arg_count++;
  } else {
    ocl_runtime_->SetKernelArg(kernel_, arg_count++, padWeight_, lite::opencl::MemType::BUF);
  }
  ocl_runtime_->SetKernelArg(kernel_, arg_count++, in_shape);
  ocl_runtime_->SetKernelArg(kernel_, arg_count++, out_shape);
}

int MatMulOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  int arg_count = 0;
  ocl_runtime_->SetKernelArg(kernel_, arg_count++, in_tensors_[0]->data_c());
  ocl_runtime_->SetKernelArg(kernel_, arg_count++, out_tensors_[0]->data_c());
  if (act_weight_) {
    ocl_runtime_->SetKernelArg(kernel_, arg_count++, in_tensors_[1]->data_c());
  }
  ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_);
  return RET_OK;
}

kernel::LiteKernel *OpenCLMatMulKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                              const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                              const lite::InnerContext *ctx, const kernel::KernelKey &desc) {
  kernel::OpenCLKernel *kernel = nullptr;
  bool infer_shape_done = opParameter->infer_flag_;
  if (infer_shape_done && IsUseStrassenMatmul(inputs)) {
    MS_LOG(DEBUG) << "use_matmul_strassen";
    kernel = new (std::nothrow) StrassenOpenCLKernel(opParameter, inputs, outputs, ctx);
  } else {
    kernel = new (std::nothrow) MatMulOpenCLKernel(opParameter, inputs, outputs, ctx);
  }
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel " << opParameter->name_ << "is nullptr.";
    free(opParameter);
    return nullptr;
  }
  if (!infer_shape_done) {
    MS_LOG(WARNING) << "kernel don't infer shape yet!";
    return kernel;
  }
  if (kernel->CheckSpecs() != RET_OK || kernel->OpenCLKernel::CheckSpecs() != RET_OK) {
    MS_LOG(ERROR) << "Check " << opParameter->name_ << " specification failed!";
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_MatMul, OpenCLMatMulKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_MatMul, OpenCLMatMulKernelCreator)

}  // namespace mindspore::kernel
