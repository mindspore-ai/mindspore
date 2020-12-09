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
#include "nnacl/fp32/common_func_fp32.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/utils.h"
#ifndef PROGRAM_WITH_IL
#include "src/runtime/kernel/opencl/cl/arithmetic.cl.inc"
#endif

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::opencl::MemType;
using mindspore::schema::ActivationType_NO_ACTIVATION;
using mindspore::schema::ActivationType_RELU;
using mindspore::schema::ActivationType_RELU6;
using mindspore::schema::PrimitiveType_Eltwise;

namespace mindspore::kernel {

int ArithmeticOpenCLKernel::CheckSpecs() {
  if (in_tensors_.size() != 2 || out_tensors_.size() != 1) {
    MS_LOG(ERROR) << "in size: " << in_tensors_.size() << ", out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  auto *param = reinterpret_cast<const ArithmeticParameter *>(op_parameter_);
  if (param->broadcasting_ && out_tensors_[0]->shape()[0] > 1) {
    MS_LOG(ERROR) << "Broadcasting don't support  N > 1";
    return RET_ERROR;
  }
  if (!IsArithmetic(Type())) {
    MS_LOG(ERROR) << "UnSupported Operator: " << schema::EnumNamePrimitiveType(Type());
    return RET_ERROR;
  }
  if (!(param->activation_type_ == ActivationType_NO_ACTIVATION || param->activation_type_ == ActivationType_RELU ||
        param->activation_type_ == ActivationType_RELU6)) {
    MS_LOG(ERROR) << "Unsupported activation type " << param->activation_type_;
    return RET_ERROR;
  }
  return RET_OK;
}

void ArithmeticOpenCLKernel::SetGlobalLocal() {
  if (element_flag_) {
    local_size_ = {};
    auto out_shape = out_tensors_[0]->shape();
    if (out_shape.size() == 2) {
      size_t H = out_shape[0];
      size_t W = UP_DIV(out_shape[1], C4NUM);
      global_size_ = {W, H};
    } else {
      size_t H = out_shape[0] * out_shape[1];
      size_t W = out_shape[2] * UP_DIV(out_shape[3], C4NUM);
      global_size_ = {W, H};
    }
  } else {
    local_size_ = {};
    auto out_shape = GetNHWCShape(out_tensors_[0]->shape());
    global_size_ = {static_cast<size_t>(UP_DIV(out_shape[3], C4NUM)), static_cast<size_t>(out_shape[2]),
                    static_cast<size_t>(out_shape[1] * out_shape[0])};
  }
  AlignGlobalLocal(global_size_, local_size_);
}

int ArithmeticOpenCLKernel::InitWeights() {
  auto fp16_enable = ocl_runtime_->GetFp16Enable();
  auto data_size = fp16_enable ? sizeof(float16_t) : sizeof(float);
  for (auto in_tensor_ : in_tensors_) {
    auto nhwc_shape = GetNHWCShape(in_tensor_->shape());
    inputs_nhwc_shapes_.push_back(nhwc_shape);
    if (!in_tensor_->IsConst()) {
      inputs_weight_ptrs_.push_back(nullptr);
    } else {
      auto allocator = ocl_runtime_->GetAllocator();
      std::vector<size_t> img_size = GetImage2dShapeFromNHWC(nhwc_shape, schema::Format_NHWC4);
      int pack_weight_size = img_size[0] * img_size[1] * C4NUM;
      int plane = nhwc_shape[1] * nhwc_shape[2];
      int channel = nhwc_shape[3];
      int batch = nhwc_shape[0];
      img_size.push_back(fp16_enable ? CL_HALF_FLOAT : CL_FLOAT);
      if (!fp16_enable) {
        float *weight = new (std::nothrow) float[pack_weight_size];
        if (weight == nullptr) {
          MS_LOG(ERROR) << "Malloc buffer failed!";
          return RET_ERROR;
        }
        memset(weight, 0x00, pack_weight_size * data_size);
        if (in_tensor_->data_type() == kNumberTypeFloat32) {
          std::function<float(float)> to_dtype = [](float x) -> float { return x; };
          PackNHWCToNHWC4<float, float>(in_tensor_->data_c(), weight, batch, plane, channel, to_dtype);
        } else if (in_tensor_->data_type() == kNumberTypeFloat16) {
          std::function<float(float16_t)> to_dtype = [](float16_t x) -> float { return static_cast<float>(x); };
          PackNHWCToNHWC4<float16_t, float>(in_tensor_->data_c(), weight, batch, plane, channel, to_dtype);
        }
        if (batch * plane * channel == 1) {
          // scalar
          weight[3] = weight[2] = weight[1] = weight[0];
        }
        auto weight_ptr_ = allocator->Malloc(pack_weight_size, img_size, weight);
        inputs_weight_ptrs_.push_back(weight_ptr_);
        delete[] weight;
      } else {
        float16_t *weight = new (std::nothrow) float16_t[pack_weight_size];
        if (weight == nullptr) {
          MS_LOG(ERROR) << "Malloc buffer failed!";
          return RET_ERROR;
        }
        memset(weight, 0x00, pack_weight_size * data_size);
        if (in_tensor_->data_type() == kNumberTypeFloat32) {
          std::function<float16_t(float)> to_dtype = [](float x) -> float16_t { return static_cast<float16_t>(x); };
          PackNHWCToNHWC4<float, float16_t>(in_tensor_->data_c(), weight, batch, plane, channel, to_dtype);
        } else if (in_tensor_->data_type() == kNumberTypeFloat16) {
          std::function<float16_t(float16_t)> to_dtype = [](float16_t x) -> float16_t { return x; };
          PackNHWCToNHWC4<float16_t, float16_t>(in_tensor_->data_c(), weight, batch, plane, channel, to_dtype);
        }
        if (batch * plane * channel == 1) {
          // scalar
          weight[3] = weight[2] = weight[1] = weight[0];
        }
        auto weight_ptr_ = allocator->Malloc(pack_weight_size, img_size, weight);
        inputs_weight_ptrs_.push_back(weight_ptr_);
        delete[] weight;
      }
    }
  }
  return RET_OK;
}

void ArithmeticOpenCLKernel::SetConstArgs() {
  int arg_idx = 3;
  if (!element_flag_) {
    cl_int4 input0_shape = {inputs_nhwc_shapes_[0][0], inputs_nhwc_shapes_[0][1], inputs_nhwc_shapes_[0][2],
                            UP_DIV(inputs_nhwc_shapes_[0][3], C4NUM)};
    ocl_runtime_->SetKernelArg(kernel_, arg_idx++, input0_shape);
    cl_int4 input1_shape = {inputs_nhwc_shapes_[1][0], inputs_nhwc_shapes_[1][1], inputs_nhwc_shapes_[1][2],
                            UP_DIV(inputs_nhwc_shapes_[1][3], C4NUM)};
    ocl_runtime_->SetKernelArg(kernel_, arg_idx++, input1_shape);
    auto out_shape = GetNHWCShape(out_tensors_[0]->shape());
    cl_int4 output_shape{out_shape[0], out_shape[1], out_shape[2], UP_DIV(out_shape[3], C4NUM)};
    ocl_runtime_->SetKernelArg(kernel_, arg_idx++, output_shape);
    int broadcastC_flag = 0;  // do not need broadcast in C4
    if (inputs_nhwc_shapes_[0][3] == 1 && inputs_nhwc_shapes_[1][3] != 1) {
      broadcastC_flag = 1;  // BroadCast C4 in input0
    } else if (inputs_nhwc_shapes_[0][3] != 1 && inputs_nhwc_shapes_[1][3] == 1) {
      broadcastC_flag = 2;  // BroadCast C4 in input1
    }
    ocl_runtime_->SetKernelArg(kernel_, arg_idx++, broadcastC_flag);
  } else {
    cl_int2 output_shape{static_cast<int>(global_range_[0]), static_cast<int>(global_range_[1])};
    ocl_runtime_->SetKernelArg(kernel_, arg_idx++, output_shape);
  }
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, activation_min_);
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, activation_max_);
}

int ArithmeticOpenCLKernel::Prepare() {
  lite::STATUS error_code = RET_OK;
#ifdef PROGRAM_WITH_IL
  kernel_ = ocl_runtime_->GetKernelFromBinary(kernel_name_);
#else

  auto *param = reinterpret_cast<const ArithmeticParameter *>(op_parameter_);
  element_flag_ = !param->broadcasting_;
  kernel_name_ = param->broadcasting_ ? "BroadcastNHWC4" : "Element";
  kernel_name_ += schema::EnumNamePrimitiveType(Type());
  if (param->activation_type_ == ActivationType_RELU) {
    activation_min_ = 0.f;
  } else if (param->activation_type_ == ActivationType_RELU6) {
    activation_min_ = 0.f;
    activation_max_ = 6.f;
  }

  std::string program_name = "Arithmetic";
  std::string source = arithmetic_source;
  ocl_runtime_->LoadSource(program_name, source);
  error_code = ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name_);
#endif
  if (error_code != RET_OK) {
    return error_code;
  }

  SetGlobalLocal();
  InitWeights();
  SetConstArgs();
  MS_LOG(DEBUG) << kernel_name_ << " Init Done!";
  return RET_OK;
}

int ArithmeticOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";

  int arg_idx = 0;
  auto input_0_ptr = inputs_weight_ptrs_[0] == nullptr ? in_tensors_[0]->data_c() : inputs_weight_ptrs_[0];
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, input_0_ptr);
  auto input_1_ptr = inputs_weight_ptrs_[1] == nullptr ? in_tensors_[1]->data_c() : inputs_weight_ptrs_[1];
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, input_1_ptr);
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, out_tensors_[0]->data_c());
  ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_);
  return RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Mul, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Add, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Sub, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Div, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_LogicalAnd, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_LogicalOr, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Maximum, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Minimum, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_FloorDiv, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_FloorMod, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_SquaredDifference, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Equal, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_NotEqual, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Less, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_LessEqual, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Greater, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_GreaterEqual, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Eltwise, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Mul, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Add, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Sub, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Div, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_LogicalAnd, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_LogicalOr, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Maximum, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Minimum, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_FloorDiv, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_FloorMod, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_SquaredDifference, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Equal, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_NotEqual, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Less, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_LessEqual, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Greater, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_GreaterEqual, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Eltwise, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
}  // namespace mindspore::kernel
