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

ArithmeticOpenCLKernel::~ArithmeticOpenCLKernel() {
  if (weight_ptr_ != nullptr) {
    auto allocator = ocl_runtime_->GetAllocator();
    allocator->Free(weight_ptr_);
    weight_ptr_ = nullptr;
  }
}

std::vector<size_t> ArithmeticOpenCLKernel::InitGlobalSize() const {
  const size_t global_x = out_tensors_[0]->Width();
  const size_t global_y = out_tensors_[0]->Height();
  const size_t global_z = UP_ROUND_DIV(out_tensors_[0]->Channel(), 4);
  std::vector<size_t> global = {global_x, global_y, global_z};
  return global;
}

void ArithmeticOpenCLKernel::Image2dGetWorkGroupSize() {
  local_size_ = {16, 16};
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
  const ArithmeticParameter *arithmetic_parameter = reinterpret_cast<const ArithmeticParameter *>(op_parameter_);
  if (!arithmetic_parameter->broadcasting_) {
    if (in_tensors_[1]->category() == lite::Tensor::Category::CONST && in_tensors_[1]->data_c() != nullptr) {
      auto allocator = ocl_runtime_->GetAllocator();
      std::vector<size_t> img_size;
      GetImageSize(0, &img_size);
      int pack_weight_size = in_tensors_[1]->ElementsC4Num();
      int plane = in_tensors_[1]->Height() * in_tensors_[1]->Width();
      int channel = in_tensors_[1]->Channel();
      int batch = in_tensors_[1]->Batch();

      if (in_tensors_[0]->GetFormat() == in_tensors_[1]->GetFormat()) {
        if (in_tensors_[0]->data_type() == in_tensors_[1]->data_type()) {
          weight_ptr_ =
            allocator->CreateImageFromHost(in_tensors_[1]->data_c(), in_tensors_[1]->ElementsNum(), img_size);
        } else {
          MS_LOG(ERROR) << "Unsupport data type transpose from " << in_tensors_[1]->data_type() << "to "
                        << in_tensors_[0]->data_type();
          return RET_ERROR;
        }
      } else if (in_tensors_[0]->GetFormat() == schema::Format_NC4HW4) {
        if (in_tensors_[1]->GetFormat() == schema::Format_NHWC) {
          if (in_tensors_[0]->data_type() == kNumberTypeFloat32) {
            float *weight = new (std::nothrow) float[pack_weight_size];
            if (weight == nullptr) {
              MS_LOG(ERROR) << "Malloc buffer failed!";
              return RET_ERROR;
            }
            std::function<float(float)> to_dtype = [](float x) -> float { return x; };
            PackNHWCToNC4HW4<float, float>(in_tensors_[1]->data_c(), weight, batch, plane, channel, to_dtype);
            weight_ptr_ = allocator->CreateImageFromHost(weight, in_tensors_[1]->ElementsNum(), img_size);
            delete[] weight;
          } else if (in_tensors_[0]->data_type() == kNumberTypeFloat16) {
            float16_t *weight = new (std::nothrow) float16_t[pack_weight_size];
            if (weight == nullptr) {
              MS_LOG(ERROR) << "Malloc buffer failed!";
              return RET_ERROR;
            }
            std::function<float16_t(float)> to_dtype = [](float x) -> float16_t { return static_cast<float16_t>(x); };
            PackNHWCToNC4HW4<float, float16_t>(in_tensors_[1]->data_c(), weight, batch, plane, channel, to_dtype);
            weight_ptr_ = allocator->CreateImageFromHost(weight, in_tensors_[1]->ElementsNum(), img_size);
            delete[] weight;
          } else {
            MS_LOG(ERROR) << "Unsupport data type transpose from " << in_tensors_[1]->data_type() << "to "
                          << in_tensors_[0]->data_type();
            return RET_ERROR;
          }
        } else {
          MS_LOG(ERROR) << "Unsupport format transpose from " << in_tensors_[1]->GetFormat() << "to "
                        << in_tensors_[0]->GetFormat();
          return RET_ERROR;
        }
      } else if (in_tensors_[0]->GetFormat() == schema::Format_NHWC4) {
        if (in_tensors_[1]->GetFormat() == schema::Format_NHWC) {
          if (in_tensors_[0]->data_type() == kNumberTypeFloat32) {
            float *weight = new (std::nothrow) float[pack_weight_size];
            if (weight == nullptr) {
              MS_LOG(ERROR) << "Malloc buffer failed!";
              return RET_ERROR;
            }
            std::function<float(float)> to_dtype = [](float x) -> float { return x; };
            PackNHWCToNHWC4<float, float>(in_tensors_[1]->data_c(), weight, batch, plane, channel, to_dtype);
            weight_ptr_ = allocator->CreateImageFromHost(weight, in_tensors_[1]->ElementsNum(), img_size);
            delete[] weight;
          } else if (in_tensors_[0]->data_type() == kNumberTypeFloat16) {
            float16_t *weight = new (std::nothrow) float16_t[pack_weight_size];
            if (weight == nullptr) {
              MS_LOG(ERROR) << "Malloc buffer failed!";
              return RET_ERROR;
            }
            std::function<float16_t(float)> to_dtype = [](float x) -> float16_t { return static_cast<float16_t>(x); };
            PackNHWCToNHWC4<float, float16_t>(in_tensors_[1]->data_c(), weight, batch, plane, channel, to_dtype);
            weight_ptr_ = allocator->CreateImageFromHost(weight, in_tensors_[1]->ElementsNum(), img_size);
            delete[] weight;
          } else {
            MS_LOG(ERROR) << "Unsupport data type transpose from " << in_tensors_[1]->data_type() << "to "
                          << in_tensors_[0]->data_type();
            return RET_ERROR;
          }
        } else {
          MS_LOG(ERROR) << "Unsupport format transpose from " << in_tensors_[1]->GetFormat() << "to "
                        << in_tensors_[0]->GetFormat();
          return RET_ERROR;
        }
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
    kernel_name = "Broadcast";
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
  return RET_OK;
}

int ArithmeticOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";

  int arg_idx = 0;
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, in_tensors_[0]->data_c());
  if (element_flag_) {
    void *weight = weight_ptr_ == nullptr ? in_tensors_[1]->data_c() : weight_ptr_;
    ocl_runtime_->SetKernelArg(kernel_, arg_idx++, weight);
  } else {
    float weight = 0.f;
    if (in_tensors_[1]->data_type() == kNumberTypeFloat32) {
      weight = static_cast<float *>(in_tensors_[1]->data_c())[0];
    } else if (in_tensors_[1]->data_type() == kNumberTypeFloat16) {
      weight = static_cast<float>(static_cast<float16_t *>(in_tensors_[1]->data_c())[0]);
    } else {
      MS_LOG(ERROR) << "Unsupport data type " << in_tensors_[1]->data_type();
      return RET_ERROR;
    }
    ocl_runtime_->SetKernelArg(kernel_, arg_idx++, weight);
  }
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, out_tensors_[0]->data_c());

  int H = 0;
  int W = 0;
  if (out_tensors_[0]->GetFormat() == schema::Format_NC4HW4) {
    H = out_tensors_[0]->Batch() * out_tensors_[0]->Height() * UP_DIV(out_tensors_[0]->Channel(), C4NUM);
    W = out_tensors_[0]->Width();
  } else if (out_tensors_[0]->GetFormat() == schema::Format_NHWC4) {
    H = out_tensors_[0]->Batch() * out_tensors_[0]->Height();
    W = out_tensors_[0]->Width() * UP_DIV(out_tensors_[0]->Channel(), C4NUM);
  } else if (out_tensors_[0]->GetFormat() == schema::Format_NC4) {
    H = out_tensors_[0]->Batch();
    W = UP_DIV(out_tensors_[0]->Channel(), C4NUM);
  } else {
    MS_LOG(ERROR) << "Error output type " << out_tensors_[0]->GetFormat();
    return RET_ERROR;
  }
  cl_int2 output_shape{W, H};
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, output_shape);
  ocl_runtime_->RunKernel(kernel_, global_size_, local_size_, nullptr);
  return RET_OK;
}

kernel::LiteKernel *OpenCLBiasAddKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                               const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                               const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                               const lite::PrimitiveC *primitive);

kernel::LiteKernel *OpenCLArithmeticKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                  const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                                  const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                                  const mindspore::lite::PrimitiveC *primitive) {
  const ArithmeticParameter *arithmetic_parameter = reinterpret_cast<const ArithmeticParameter *>(opParameter);
  if (arithmetic_parameter->broadcasting_) {
    for (size_t i = 0; i < arithmetic_parameter->ndim_; i++) {
      if (arithmetic_parameter->in_shape1_[i] != 0 && arithmetic_parameter->in_shape1_[i] != 1) {
        return OpenCLBiasAddKernelCreator(inputs, outputs, opParameter, ctx, desc, primitive);
      }
    }
  }
  auto *kernel =
    new (std::nothrow) ArithmeticOpenCLKernel(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Create OpenCL Arithmetic kernel failed!";
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
