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
#include "src/runtime/opencl/opencl_runtime.h"
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
  kernel_name += "_" + std::string(EnumNameFormat(op_format_));
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  auto param = reinterpret_cast<MatMulParameter *>(op_parameter_);
  transposeA = param->a_transpose_;
  if (transposeA) {
    MS_LOG(ERROR) << "matmul only support a_transpose_=false yet.";
    return RET_ERROR;
  }
  transposeB = param->b_transpose_;
  enable_fp16_ = ocl_runtime->GetFp16Enable();
  if (in_tensors_[0]->shape().size() != out_tensors_[0]->shape().size() ||
      (in_tensors_[0]->shape().size() != 2 && in_tensors_[0]->shape().size() != 4)) {
    MS_LOG(ERROR) << "matmul only support input shape size=2 or 4.";
    return RET_ERROR;
  }
  dims = in_tensors_[0]->shape().size();
  for (int i = 0; i < dims; i++) {
    inShape[MAX_DIMS - dims + i] = in_tensors_[0]->shape()[i];
    outShape[MAX_DIMS - dims + i] = out_tensors_[0]->shape()[i];
  }
  std::map<int, std::string> dims2str = {{2, "_2d"}, {4, "_4d"}};
  kernel_name += dims2str[dims];
#ifdef PROGRAM_WITH_IL
  kernel_ = ocl_runtime->GetKernelFromBinary(kernel_name);
#else
  std::set<std::string> build_options;
  std::string source = matmul_source;
  std::string program_name = "MatMul";
  ocl_runtime->LoadSource(program_name, source);
  ocl_runtime->BuildKernel(kernel_, program_name, kernel_name, build_options);
#endif

  PadWeight();
  in_ori_format_ = in_tensors_[0]->GetFormat();
  out_ori_format_ = out_tensors_[0]->GetFormat();
  in_tensors_[0]->SetFormat(op_format_);
  out_tensors_[0]->SetFormat(op_format_);
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}

int MatMulOpenCLKernel::ReSize() { return RET_OK; }

void MatMulOpenCLKernel::PadWeight() {
  // ABMCI @ ABCICO = ABMCO
  auto allocator = lite::opencl::OpenCLRuntime::GetInstance()->GetAllocator();
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
  auto originWeightFp32 = reinterpret_cast<float *>(in_tensors_.at(kWeightIndex)->MutableData());
  auto originWeightFp16 = reinterpret_cast<float16_t *>(in_tensors_.at(kWeightIndex)->MutableData());
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
        reinterpret_cast<float16_t *>(bias_)[i] = reinterpret_cast<float *>(in_tensors_[2]->MutableData())[i];
      }
    } else if (in_tensors_[2]->data_type() == kNumberTypeFloat16 && !enable_fp16_) {
      for (int i = 0; i < co; i++) {
        reinterpret_cast<float *>(bias_)[i] = reinterpret_cast<float16_t *>(in_tensors_[2]->MutableData())[i];
      }
    } else {
      memcpy(bias_, in_tensors_[2]->MutableData(), co * dtype_size);
    }
  }
  allocator->UnmapBuffer(bias_);
}

int MatMulOpenCLKernel::GetImageSize(size_t idx, std::vector<size_t> *img_size) {
  size_t im_dst_x, im_dst_y;
  auto out_shape = out_tensors_[0]->shape();
  int n = 1, h = 1, w = 1, c = 1;
  if (dims == 2) {
    n = out_shape[0];
    c = out_shape[1];
  } else if (dims == 4) {
    n = out_shape[0];
    h = out_shape[1];
    w = out_shape[2];
    c = out_shape[3];
  }
  if (op_format_ == schema::Format_NHWC4) {
    im_dst_x = w * UP_DIV(c, C4NUM);
    im_dst_y = n * h;
  } else if (op_format_ == schema::Format_NC4HW4) {
    im_dst_x = w;
    im_dst_y = n * UP_DIV(c, C4NUM) * h;
  } else {
    MS_LOG(ERROR) << "not support op format:" << EnumNameFormat(op_format_);
    return RET_ERROR;
  }
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
  std::vector<size_t> local = {32, 4, 1};
  std::vector<size_t> global = {UP_DIV(static_cast<size_t>(outShape[3]), C4NUM),
                                4 * static_cast<size_t>(outShape[0]) * static_cast<size_t>(outShape[1]),
                                static_cast<size_t>(outShape[2])};
  int arg_count = 0;
  cl_int4 in_shape = {inShape[0], inShape[1], inShape[2], inShape[3]};
  cl_int4 out_shape = {outShape[0], outShape[1], outShape[2], outShape[3]};
  ocl_runtime->SetKernelArg(kernel_, arg_count++, in_tensors_[0]->MutableData());
  ocl_runtime->SetKernelArg(kernel_, arg_count++, padWeight_, lite::opencl::MemType::BUF);
  ocl_runtime->SetKernelArg(kernel_, arg_count++, bias_);
  ocl_runtime->SetKernelArg(kernel_, arg_count++, out_tensors_[0]->MutableData());
  ocl_runtime->SetKernelArg(kernel_, arg_count++, in_shape);
  ocl_runtime->SetKernelArg(kernel_, arg_count++, out_shape);
  ocl_runtime->SetKernelArg(kernel_, arg_count++, hasBias_ ? 1 : 0);
  ocl_runtime->RunKernel(kernel_, global, local, nullptr);
  return RET_OK;
}

kernel::LiteKernel *OpenCLMatMulKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                              const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                              const lite::InnerContext *ctx, const kernel::KernelKey &desc,
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
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_MatMul, OpenCLMatMulKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_FullConnection, OpenCLMatMulKernelCreator)
}  // namespace mindspore::kernel
