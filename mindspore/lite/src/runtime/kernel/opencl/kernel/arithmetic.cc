/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/opencl/kernel/arithmetic.h"
#include <set>
#include <vector>
#include <string>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/utils.h"
#ifndef PROGRAM_WITH_IL
#include "src/runtime/kernel/opencl/cl/fp32/arithmetic_buffer.cl.inc"
#include "src/runtime/kernel/opencl/cl/fp32/arithmetic_image2d.cl.inc"
#endif

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;

namespace mindspore::kernel {

std::vector<size_t> ArithmeticOpenCLKernel::InitGlobalSize() const {
  const size_t global_x = out_tensors_[0]->Width();
  const size_t global_y = out_tensors_[0]->Height();
  const size_t global_z = UP_ROUND_DIV(out_tensors_[0]->Channel(), 4);
  std::vector<size_t> global = {global_x, global_y, global_z};
  return global;
}

void ArithmeticOpenCLKernel::Image2dGetWorkGroupSize() {
  size_t H = out_tensors_[0]->Batch() * out_tensors_[0]->Height();
  size_t W = out_tensors_[0]->Width() * UP_DIV(out_tensors_[0]->Channel(), C4NUM);
  local_size_ = {16, 16};
  global_size_ = {W, H};
}

void ArithmeticOpenCLKernel::BufferGetWorkGroupSize() {
  uint32_t element_num = out_tensors_[0]->ElementsC4Num();
  global_size_ = {element_num};
}

int ArithmeticOpenCLKernel::GetImageSize(size_t idx, std::vector<size_t> *img_size) {
  size_t CO4 = UP_DIV(out_tensors_[0]->Channel(), C4NUM);
  int H = out_tensors_[0]->Batch() * out_tensors_[0]->Height();
  int W = out_tensors_[0]->Width() * CO4;
  size_t im_dst_x, im_dst_y;
  if (in_tensors_[0]->GetFormat() == schema::Format_NHWC4) {
    im_dst_x = W;
    im_dst_y = H;
  } else {
    im_dst_y = out_tensors_[0]->Batch() * out_tensors_[0]->Height() * CO4;
    im_dst_x = out_tensors_[0]->Width();
  }
#ifdef ENABLE_FP16
  size_t img_dtype = CL_HALF_FLOAT;
#else
  size_t img_dtype = CL_FLOAT;
#endif
  img_size->clear();
  std::vector<size_t> vec{im_dst_x, im_dst_y, img_dtype};
  *img_size = vec;
  return 0;
}

int ArithmeticOpenCLKernel::Init() {
  runtime_ = lite::opencl::OpenCLRuntime::GetInstance();
  std::string kernel_name;

  if (in_tensors_[1]->TensorType() == schema::NodeType_ValueNode && in_tensors_[1]->Data() != nullptr) {
    element_flag_ = false;
    kernel_name = "BoardcastArith";
  } else {
    element_flag_ = true;
    switch (op_parameter_->type_) {
      case PrimitiveType_Mul:
        kernel_name = "ElementMul";
        break;
      case PrimitiveType_Add:
        kernel_name = "ElementAdd";
        break;
      case PrimitiveType_Sub:
        kernel_name = "ElementSub";
        break;
      case PrimitiveType_Div:
        kernel_name = "ElementDiv";
        break;
      default:
        MS_LOG(ERROR) << "Error Operator type " << op_parameter_->type_;
        break;
    }
  }

#ifdef PROGRAM_WITH_IL
  runtime_->CreateKernelFromIL(kernel_(), kernel_name);
#else
  std::string program_name = "Arithmetic";
  std::set<std::string> build_options;
  std::string source = arithmetic_image2d_source_fp32;
  runtime_->LoadSource(program_name, source);
  runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options);
#endif
  out_tensors_[0]->SetFormat(schema::Format_NHWC4);
  Image2dGetWorkGroupSize();
  return 0;
}

int ArithmeticOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();

  uint32_t element_num = out_tensors_[0]->ElementsC4Num();
  int arg_idx = 0;

  ocl_runtime->SetKernelArg(kernel_, arg_idx++, in_tensors_[0]->Data());
  if (element_flag_) {
    runtime_->SetKernelArg(kernel_, arg_idx++, in_tensors_[1]->Data());
  } else {
    float value = static_cast<float *>(in_tensors_[1]->Data())[0];
    switch (op_parameter_->type_) {
      case PrimitiveType_Mul:
        weight_ = value;
        break;
      case PrimitiveType_Add:
        bias_ = value;
        break;
      case PrimitiveType_Sub:
        bias_ = -1 * value;
        break;
      case PrimitiveType_Div:
        weight_ = 1 / value;
        break;
      default:
        MS_LOG(ERROR) << "Error Operator type " << op_parameter_->type_;
        break;
    }
    ocl_runtime->SetKernelArg(kernel_, arg_idx++, weight_);
    ocl_runtime->SetKernelArg(kernel_, arg_idx++, bias_);
  }
  ocl_runtime->SetKernelArg(kernel_, arg_idx++, out_tensors_[0]->Data());
  int H = out_tensors_[0]->Batch() * out_tensors_[0]->Height();
  int W = out_tensors_[0]->Width() * UP_DIV(out_tensors_[0]->Channel(), C4NUM);
  cl_int2 output_shape{W, H};
  ocl_runtime->SetKernelArg(kernel_, arg_idx++, output_shape);
  ocl_runtime->RunKernel(kernel_, global_size_, local_size_, nullptr);
  return 0;
}

kernel::LiteKernel *OpenCLArithmeticKernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                  const std::vector<lite::tensor::Tensor *> &outputs,
                                                  OpParameter *opParameter, const lite::Context *ctx,
                                                  const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  auto *kernel =
    new (std::nothrow) ArithmeticOpenCLKernel(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Create OpenCL Arithmetic kernel failed!";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (0 != ret) {
    MS_LOG(ERROR) << "Init kernel failed, name: Arithmetic";
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Mul, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Add, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Sub, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Div, OpenCLArithmeticKernelCreator)
}  // namespace mindspore::kernel
