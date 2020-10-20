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
#include "nnacl/fp32/common_func.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/utils.h"
#ifndef PROGRAM_WITH_IL
#include "src/runtime/kernel/opencl/cl/arithmetic.cl.inc"
#endif

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_Eltwise;

namespace mindspore::kernel {

ArithmeticOpenCLKernel::~ArithmeticOpenCLKernel() {}

std::vector<size_t> ArithmeticOpenCLKernel::InitGlobalSize() const {
  const size_t global_x = out_tensors_[0]->Width();
  const size_t global_y = out_tensors_[0]->Height();
  const size_t global_z = UP_ROUND_DIV(out_tensors_[0]->Channel(), 4);
  std::vector<size_t> global = {global_x, global_y, global_z};
  return global;
}

void ArithmeticOpenCLKernel::Image2dGetWorkGroupSize() {
  local_size_ = {16, 16};
  if (out_tensors_[0]->shape().size() == 2) {
    size_t H = out_tensors_[0]->shape()[0];
    size_t W = UP_DIV(out_tensors_[0]->shape()[1], C4NUM);
    global_size_ = {W, H};
    return;
  }
  if (out_tensors_[0]->GetFormat() == schema::Format_NC4HW4) {
    size_t H = out_tensors_[0]->Batch() * out_tensors_[0]->Height() * UP_DIV(out_tensors_[0]->Channel(), C4NUM);
    size_t W = out_tensors_[0]->Width();
    global_size_ = {W, H};
  } else if (out_tensors_[0]->GetFormat() == schema::Format_NHWC4) {
    size_t H = out_tensors_[0]->Batch() * out_tensors_[0]->Height();
    size_t W = out_tensors_[0]->Width() * UP_DIV(out_tensors_[0]->Channel(), C4NUM);
    global_size_ = {W, H};
  } else if (out_tensors_[0]->GetFormat() == schema::Format_NC4) {
    size_t H = out_tensors_[0]->Batch();
    size_t W = UP_DIV(out_tensors_[0]->Channel(), C4NUM);
    global_size_ = {W, H};
  } else {
    MS_LOG(ERROR) << "Unsupport data format " << out_tensors_[0]->GetFormat();
  }
}

void ArithmeticOpenCLKernel::BufferGetWorkGroupSize() {
  uint32_t element_num = out_tensors_[0]->ElementsC4Num();
  global_size_ = {element_num};
}

int ArithmeticOpenCLKernel::GetImageSize(size_t idx, std::vector<size_t> *img_size) {
  size_t im_dst_x, im_dst_y;
  if (out_tensors_[0]->shape().size() == 2) {
    im_dst_x = UP_DIV(out_tensors_[0]->shape()[1], C4NUM);
    im_dst_y = out_tensors_[0]->shape()[0];
  } else {
    if (out_tensors_[0]->GetFormat() == schema::Format_NC4HW4) {
      im_dst_x = out_tensors_[0]->Width();
      im_dst_y = out_tensors_[0]->Batch() * out_tensors_[0]->Height() * UP_DIV(out_tensors_[0]->Channel(), C4NUM);
    } else if (out_tensors_[0]->GetFormat() == schema::Format_NHWC4) {
      im_dst_x = out_tensors_[0]->Width() * UP_DIV(out_tensors_[0]->Channel(), C4NUM);
      im_dst_y = out_tensors_[0]->Batch() * out_tensors_[0]->Height();
    } else if (out_tensors_[0]->GetFormat() == schema::Format_NC4) {
      im_dst_y = out_tensors_[0]->Batch();
      im_dst_x = UP_DIV(out_tensors_[0]->Channel(), C4NUM);
    } else {
      MS_LOG(ERROR) << "Unsupport data format " << out_tensors_[0]->GetFormat();
      return RET_ERROR;
    }
  }

  size_t img_dtype = CL_FLOAT;
  if (in_tensors_[0]->data_type() == kNumberTypeFloat16) {
    img_dtype = CL_HALF_FLOAT;
  } else if (in_tensors_[0]->data_type() == kNumberTypeFloat32) {
    img_dtype = CL_FLOAT;
  } else {
    MS_LOG(ERROR) << "Unsupport data type " << in_tensors_[0]->data_type();
  }
  img_size->clear();
  std::vector<size_t> vec{im_dst_x, im_dst_y, img_dtype};
  *img_size = vec;
  return RET_OK;
}

int ArithmeticOpenCLKernel::InitBuffer() {
  auto fp16_enable = ocl_runtime_->GetFp16Enable();
  auto data_size = fp16_enable ? sizeof(float16_t) : sizeof(float);
  for (auto in_tensor_ : in_tensors_) {
    auto nhwc_shape = GetNHWCShape(in_tensor_->shape());
    inputs_nhwc_shapes_.push_back(nhwc_shape);
    if (in_tensor_->category() != lite::Tensor::Category::CONST || in_tensor_->data_c() == nullptr) {
      inputs_weight_ptrs_.push_back(nullptr);
    } else {
      auto allocator = ocl_runtime_->GetAllocator();
      std::vector<size_t> img_size = GetImage2dShapeFromNHWC(nhwc_shape, op_format_);
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
        if (op_format_ == schema::Format_NHWC4) {
          if (in_tensor_->data_type() == kNumberTypeFloat32) {
            std::function<float(float)> to_dtype = [](float x) -> float { return x; };
            PackNHWCToNHWC4<float, float>(in_tensor_->data_c(), weight, batch, plane, channel, to_dtype);
          } else if (in_tensor_->data_type() == kNumberTypeFloat16) {
            std::function<float(float16_t)> to_dtype = [](float16_t x) -> float { return static_cast<float>(x); };
            PackNHWCToNHWC4<float16_t, float>(in_tensor_->data_c(), weight, batch, plane, channel, to_dtype);
          }
        } else if (op_format_ == schema::Format_NC4HW4) {
          if (in_tensor_->data_type() == kNumberTypeFloat32) {
            std::function<float(float)> to_dtype = [](float x) -> float { return x; };
            PackNHWCToNC4HW4<float, float>(in_tensor_->data_c(), weight, batch, plane, channel, to_dtype);
          } else if (in_tensor_->data_type() == kNumberTypeFloat16) {
            std::function<float(float16_t)> to_dtype = [](float16_t x) -> float { return static_cast<float>(x); };
            PackNHWCToNC4HW4<float16_t, float>(in_tensor_->data_c(), weight, batch, plane, channel, to_dtype);
          }
        }
        if (batch * plane * channel == 1) {
          // scalar
          weight[3] = weight[2] = weight[1] = weight[0];
        }
        auto weight_ptr_ = allocator->CreateImageFromHost(weight, pack_weight_size, img_size);
        inputs_weight_ptrs_.push_back(weight_ptr_);
        delete[] weight;
      } else {
        float16_t *weight = new (std::nothrow) float16_t[pack_weight_size];
        if (weight == nullptr) {
          MS_LOG(ERROR) << "Malloc buffer failed!";
          return RET_ERROR;
        }
        memset(weight, 0x00, pack_weight_size * data_size);
        if (op_format_ == schema::Format_NHWC4) {
          if (in_tensor_->data_type() == kNumberTypeFloat32) {
            std::function<float16_t(float)> to_dtype = [](float x) -> float16_t { return static_cast<float16_t>(x); };
            PackNHWCToNHWC4<float, float16_t>(in_tensor_->data_c(), weight, batch, plane, channel, to_dtype);
          } else if (in_tensor_->data_type() == kNumberTypeFloat16) {
            std::function<float16_t(float16_t)> to_dtype = [](float16_t x) -> float16_t { return x; };
            PackNHWCToNHWC4<float16_t, float16_t>(in_tensor_->data_c(), weight, batch, plane, channel, to_dtype);
          }
        } else if (op_format_ == schema::Format_NC4HW4) {
          if (in_tensor_->data_type() == kNumberTypeFloat32) {
            std::function<float16_t(float)> to_dtype = [](float x) -> float16_t { return static_cast<float16_t>(x); };
            PackNHWCToNC4HW4<float, float16_t>(in_tensor_->data_c(), weight, batch, plane, channel, to_dtype);
          } else if (in_tensor_->data_type() == kNumberTypeFloat16) {
            std::function<float16_t(float16_t)> to_dtype = [](float16_t x) -> float16_t { return x; };
            PackNHWCToNC4HW4<float16_t, float16_t>(in_tensor_->data_c(), weight, batch, plane, channel, to_dtype);
          }
        }
        if (batch * plane * channel == 1) {
          // scalar
          weight[3] = weight[2] = weight[1] = weight[0];
        }
        auto weight_ptr_ = allocator->CreateImageFromHost(weight, pack_weight_size, img_size);
        inputs_weight_ptrs_.push_back(weight_ptr_);
        delete[] weight;
      }
    }
  }
  return RET_OK;
}

int ArithmeticOpenCLKernel::Init() {
  std::string kernel_name;

  const ArithmeticParameter *arithmetic_parameter = reinterpret_cast<const ArithmeticParameter *>(op_parameter_);

  if (arithmetic_parameter->broadcasting_) {
    element_flag_ = false;
    if (op_format_ == schema::Format_NHWC4) {
      kernel_name = "BroadcastNHWC4";
    } else {
      kernel_name = "BroadcastNC4HW4";
      MS_LOG(ERROR) << "Don't support BroadcastNC4HW4 yet";
      return RET_ERROR;
    }
  } else {
    kernel_name = "Element";
  }

  switch (op_parameter_->type_) {
    case PrimitiveType_Mul:
      kernel_name += "Mul";
      break;
    case PrimitiveType_Add:
      kernel_name += "Add";
      break;
    case PrimitiveType_Sub:
      kernel_name += "Sub";
      break;
    case PrimitiveType_Div:
      kernel_name += "Div";
      break;
    case PrimitiveType_LogicalAnd:
      kernel_name += "And";
      break;
    case PrimitiveType_LogicalOr:
      kernel_name += "Or";
      break;
    case PrimitiveType_Maximum:
      kernel_name += "Max";
      break;
    case PrimitiveType_Minimum:
      kernel_name += "Min";
      break;
    case PrimitiveType_FloorDiv:
      kernel_name += "FloorDiv";
      break;
    case PrimitiveType_FloorMod:
      kernel_name += "FloorMod";
      break;
    case PrimitiveType_SquaredDifference:
      kernel_name += "SquaredDifference";
      break;
    case PrimitiveType_Equal:
      kernel_name += "Equal";
      break;
    case PrimitiveType_NotEqual:
      kernel_name += "NotEqual";
      break;
    case PrimitiveType_Less:
      kernel_name += "Less";
      break;
    case PrimitiveType_LessEqual:
      kernel_name += "LessEqual";
      break;
    case PrimitiveType_Greater:
      kernel_name += "Greater";
      break;
    case PrimitiveType_GreaterEqual:
      kernel_name += "GreaterEqual";
      break;
    default:
      MS_LOG(ERROR) << "Error Operator type " << op_parameter_->type_;
      return RET_ERROR;
  }

  switch (arithmetic_parameter->activation_type_) {
    case schema::ActivationType_NO_ACTIVATION:
      break;
    case schema::ActivationType_RELU:
      activation_min_ = 0.f;
      break;
    case schema::ActivationType_RELU6:
      activation_min_ = 0.f;
      activation_max_ = 6.f;
      break;
    default:
      MS_LOG(ERROR) << "Unsupported activation type " << arithmetic_parameter->activation_type_;
      return RET_ERROR;
  }

  lite::STATUS error_code = RET_OK;
#ifdef PROGRAM_WITH_IL
  kernel_ = ocl_runtime_->GetKernelFromBinary(kernel_name);
#else
  if (out_mem_type_ == OpenCLMemType::IMG) {
    kernel_name += "_IMG";
  } else {
    kernel_name += "_BUF";
  }
  std::string program_name = "Arithmetic";
  std::set<std::string> build_options;
  std::string source = arithmetic_source;
  ocl_runtime_->LoadSource(program_name, source);
  error_code = ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options);
#endif
  if (error_code != RET_OK) {
    return error_code;
  }

  auto format = schema::Format::Format_NHWC4;
  if (arithmetic_parameter->ndim_ == 2) {
    format = schema::Format::Format_NC4;
  }
  in_ori_format_ = in_tensors_[0]->GetFormat();
  out_ori_format_ = out_tensors_[0]->GetFormat();
  in_tensors_[0]->SetFormat(format);
  if (element_flag_ && in_tensors_[1]->category() != lite::Tensor::Category::CONST) {
    in_tensors_[1]->SetFormat(format);
  }
  out_tensors_[0]->SetFormat(format);
  Image2dGetWorkGroupSize();
  InitBuffer();
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
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
    ocl_runtime_->SetKernelArg(kernel_, arg_idx++, activation_min_);
    ocl_runtime_->SetKernelArg(kernel_, arg_idx++, activation_max_);
    ocl_runtime_->RunKernel(kernel_,
                            {static_cast<size_t>(UP_DIV(out_shape[3], C4NUM)), static_cast<size_t>(out_shape[2]),
                             static_cast<size_t>(out_shape[1] * out_shape[0])},
                            {}, nullptr);
  } else {
    cl_int2 output_shape{static_cast<int>(global_size_[0]), static_cast<int>(global_size_[1])};
    ocl_runtime_->SetKernelArg(kernel_, arg_idx++, output_shape);
    ocl_runtime_->SetKernelArg(kernel_, arg_idx++, activation_min_);
    ocl_runtime_->SetKernelArg(kernel_, arg_idx++, activation_max_);
    ocl_runtime_->RunKernel(kernel_, global_size_, local_size_, nullptr);
  }
  return RET_OK;
}

kernel::LiteKernel *OpenCLArithmeticKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                  const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                                  const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                                  const mindspore::lite::PrimitiveC *primitive) {
  auto *kernel =
    new (std::nothrow) ArithmeticOpenCLKernel(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Create OpenCL Arithmetic kernel failed!";
    free(opParameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
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
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_LogicalAnd, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_LogicalOr, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Maximum, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Minimum, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_FloorDiv, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_FloorMod, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_SquaredDifference, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Equal, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_NotEqual, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Less, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_LessEqual, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Greater, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_GreaterEqual, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Eltwise, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Mul, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Add, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Sub, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Div, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_LogicalAnd, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_LogicalOr, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Maximum, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Minimum, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_FloorDiv, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_FloorMod, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_SquaredDifference, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Equal, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_NotEqual, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Less, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_LessEqual, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Greater, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_GreaterEqual, OpenCLArithmeticKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Eltwise, OpenCLArithmeticKernelCreator)
}  // namespace mindspore::kernel
