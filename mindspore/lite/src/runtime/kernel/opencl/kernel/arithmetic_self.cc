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
#include <cstring>
#include <algorithm>
#include <set>
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/kernel/arithmetic_self.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "src/runtime/kernel/opencl/cl/arithmeticself.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_Abs;
using mindspore::schema::PrimitiveType_Ceil;
using mindspore::schema::PrimitiveType_Cos;
using mindspore::schema::PrimitiveType_Exp;
using mindspore::schema::PrimitiveType_Floor;
using mindspore::schema::PrimitiveType_Log;
using mindspore::schema::PrimitiveType_LogicalNot;
using mindspore::schema::PrimitiveType_Neg;
using mindspore::schema::PrimitiveType_Round;
using mindspore::schema::PrimitiveType_Rsqrt;
using mindspore::schema::PrimitiveType_Sin;
using mindspore::schema::PrimitiveType_Sqrt;
using mindspore::schema::PrimitiveType_Square;

namespace mindspore::kernel {

int ArithmeticSelfOpenCLKernel::GetImageSize(size_t idx, std::vector<size_t> *img_size) {
  size_t CO4 = UP_DIV(out_tensors_[0]->Channel(), C4NUM);
  size_t im_dst_x, im_dst_y;
  if (in_tensors_[0]->GetFormat() == schema::Format_NHWC4) {
    im_dst_x = out_tensors_[0]->Width() * CO4;
    im_dst_y = out_tensors_[0]->Height() * out_tensors_[0]->Batch();
  } else {
    im_dst_y = out_tensors_[0]->Batch() * out_tensors_[0]->Height() * CO4;
    im_dst_x = out_tensors_[0]->Width();
  }
  size_t img_dtype = CL_FLOAT;
  auto enable_fp16_ = ocl_runtime_->GetFp16Enable();
  if (enable_fp16_) {
    img_dtype = CL_HALF_FLOAT;
  }
  img_size->clear();
  std::vector<size_t> vec{im_dst_x, im_dst_y, img_dtype};
  *img_size = vec;
  return RET_OK;
}

void ArithmeticSelfOpenCLKernel::GetKernelName(std::string *kernel_name, ArithmeticSelfParameter *param) {
  switch (param->op_parameter_.type_) {
    case PrimitiveType_Abs:
      kernel_name[0] += "_ElementAbs";
      break;
    case PrimitiveType_Cos:
      kernel_name[0] += "_ElementCos";
      break;
    case PrimitiveType_Exp:
      kernel_name[0] += "_ElementExp";
      break;
    case PrimitiveType_Log:
      kernel_name[0] += "_ElementLog";
      break;
    case PrimitiveType_Square:
      kernel_name[0] += "_ElementSquare";
      break;
    case PrimitiveType_Sqrt:
      kernel_name[0] += "_ElementSqrt";
      break;
    case PrimitiveType_Rsqrt:
      kernel_name[0] += "_ElementRsqrt";
      break;
    case PrimitiveType_Sin:
      kernel_name[0] += "_ElementSin";
      break;
    case PrimitiveType_LogicalNot:
      kernel_name[0] += "_ElementLogicalNot";
      break;
    case PrimitiveType_Floor:
      kernel_name[0] += "_ElementFloor";
      break;
    case PrimitiveType_Ceil:
      kernel_name[0] += "_ElementCeil";
      break;
    case PrimitiveType_Round:
      kernel_name[0] += "_ElementRound";
    case PrimitiveType_Neg:
      kernel_name[0] += "_ElementNeg";
      break;
    default:
      break;
  }
}

int ArithmeticSelfOpenCLKernel::Init() {
  if (in_tensors_[0]->shape().size() != 4) {
    MS_LOG(ERROR) << " only support dim = 4 ";
    return RET_ERROR;
  }
  auto param = reinterpret_cast<ArithmeticSelfParameter *>(this->op_parameter_);

  auto in_format = op_format_;
  if (in_format != schema::Format_NHWC4 && in_format != schema::Format_NC4HW4) {
    MS_LOG(ERROR) << "input format(" << in_format << ") "
                  << "format not support!";
    return RET_ERROR;
  }
  in_ori_format_ = in_tensors_[0]->GetFormat();
  in_tensors_[0]->SetFormat(op_format_);
  out_ori_format_ = out_tensors_[0]->GetFormat();
  out_tensors_[0]->SetFormat(op_format_);

  std::string kernel_name = "ArithmeticSelf";
  GetKernelName(&kernel_name, param);
  if (in_format == schema::Format_NC4HW4) {
    kernel_name += "_NC4HW4";
  } else if (in_format == schema::Format_NHWC4) {
    kernel_name += "_NHWC4";
  }
  MS_LOG(DEBUG) << "execute kernel name : " << kernel_name;
  std::set<std::string> build_options;
  std::string source = arithmeticself_source;
  std::string program_name = "ArithmeticSelf";
  ocl_runtime_->LoadSource(program_name, source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options);

  return RET_OK;
}

int ArithmeticSelfOpenCLKernel::ReSize() { return RET_OK; }

void ArithmeticSelfGetWorkGroup(const std::vector<size_t> &global, std::vector<size_t> *local, int max_size) {
  const int max_divider = 8;
  const int max_x = 4, max_y = 8;
  int x = std::min(GetMaxDivisorStrategy1(global[0], max_divider), max_x);
  int yz = max_size / x;
  int y = std::min(std::min(GetMaxDivisorStrategy1(global[1], max_divider), yz), max_y);
  int z = std::min(yz / y, static_cast<int>(UP_DIV(global[2], 2)));

  local->clear();
  local->push_back(x);
  local->push_back(y);
  local->push_back(z);
}

int ArithmeticSelfOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running! ";

  auto output_shape = out_tensors_[0]->shape();
  cl_int4 output_shape_ = {output_shape[0], output_shape[1], output_shape[2], UP_DIV(output_shape[3], C4NUM)};

  uint32_t OH = output_shape[0] * output_shape[1];  // N*H
  uint32_t OW = output_shape[2];
  uint32_t OC = UP_DIV(output_shape[3], C4NUM);

  const std::vector<size_t> &max_global = ocl_runtime_->GetWorkItemSize();
  std::vector<size_t> local = {1, 1, 1};  // init local
  std::vector<size_t> global = {OH, OW, OC};
  ArithmeticSelfGetWorkGroup(global, &local, max_global[0]);

  int arg_cn = 0;
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, in_tensors_[0]->data_c());
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, out_tensors_[0]->data_c());
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, output_shape_);

  ocl_runtime_->RunKernel(kernel_, global, local, nullptr);

  return RET_OK;
}

kernel::LiteKernel *OpenCLArithmeticSelfKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                      const std::vector<lite::Tensor *> &outputs,
                                                      OpParameter *opParameter, const lite::InnerContext *ctx,
                                                      const kernel::KernelKey &desc,
                                                      const mindspore::lite::PrimitiveC *primitive) {
  auto *kernel = new (std::nothrow) ArithmeticSelfOpenCLKernel(opParameter, inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << " new ArithmeticSelfOpenCLKernel failed ";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << " Init kernel failed, name: ArithmeticSelf ";
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Abs, OpenCLArithmeticSelfKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Ceil, OpenCLArithmeticSelfKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Cos, OpenCLArithmeticSelfKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Exp, OpenCLArithmeticSelfKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Floor, OpenCLArithmeticSelfKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Log, OpenCLArithmeticSelfKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_LogicalNot, OpenCLArithmeticSelfKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Round, OpenCLArithmeticSelfKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Rsqrt, OpenCLArithmeticSelfKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Sin, OpenCLArithmeticSelfKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Neg, OpenCLArithmeticSelfKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Sqrt, OpenCLArithmeticSelfKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Square, OpenCLArithmeticSelfKernelCreator)

}  // namespace mindspore::kernel
