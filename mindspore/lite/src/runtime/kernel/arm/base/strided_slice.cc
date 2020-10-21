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

#include "src/runtime/kernel/arm/base/strided_slice.h"
#include <vector>
#include "nnacl/strided_slice.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_StridedSlice;

namespace mindspore::kernel {
namespace {
constexpr size_t kMultiInputsSize = 4;
constexpr size_t kBeginsIndex = 1;
constexpr size_t kEndsIndex = 2;
constexpr size_t kStridesInex = 3;
}  // namespace
int StridedSliceCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }

  return ReSize();
}

int StridedSliceCPUKernel::ReSize() {
  auto input = in_tensors_.at(0);
  auto parameter = reinterpret_cast<StridedSliceParameter *>(op_parameter_);
  MS_ASSERT(input);
  MS_ASSERT(parameter);
  parameter->data_type = input->data_type() == kNumberTypeInt8 ? kDataTypeInt8 : kDataTypeFloat;
  auto input_shape = input->shape();
  for (size_t i = 0; i < input_shape.size(); ++i) {
    parameter->in_shape_[i] = input_shape[i];
  }
  parameter->in_shape_length_ = static_cast<int>(input_shape.size());
  return RET_OK;
}

int StridedSliceCPUKernel::HandleMultiInputs() {
  if (in_tensors_.size() != kMultiInputsSize) {
    MS_LOG(ERROR) << "Inputs size should be " << kMultiInputsSize << ", got " << in_tensors_.size();
    return RET_ERROR;
  }
  auto param = reinterpret_cast<StridedSliceParameter *>(op_parameter_);
  if (param == nullptr) {
    MS_LOG(ERROR) << "StridedSliceParamater cast nullptr";
    return RET_ERROR;
  }
  auto begins = in_tensors_.at(kBeginsIndex);
  MS_ASSERT(begins != nullptr);
  int axis_num = begins->ElementsNum();
  if (axis_num > DIMENSION_6D) {
    MS_LOG(ERROR) << "StridedSlice supports max dimension " << DIMENSION_6D << ", input begins dim is " << axis_num;
    return RET_ERROR;
  }
  memcpy(param->begins_, begins->MutableData(), axis_num * sizeof(int));

  auto ends = in_tensors_.at(kEndsIndex);
  MS_ASSERT(ends != nullptr);
  MS_ASSERT(axis_num == ends->ElementsNum());
  memcpy(param->ends_, ends->MutableData(), axis_num * sizeof(int));

  auto strides = in_tensors_.at(kStridesInex);
  MS_ASSERT(strides != nullptr);
  MS_ASSERT(axis_num == strides->ElementsNum());
  memcpy(param->strides_, strides->MutableData(), axis_num * sizeof(int));

  param->num_axes_ = axis_num;
  return RET_OK;
}

int StridedSliceCPUKernel::Run() {
  auto input = in_tensors_.at(0);
  auto output = out_tensors_.at(0);
  MS_ASSERT(input);
  MS_ASSERT(output);
  if (in_tensors().size() == kMultiInputsSize) {
    auto ret = HandleMultiInputs();
    if (ret != RET_OK) {
      return ret;
    }
  }
  auto ret = DoStridedSlice(input->MutableData(), output->MutableData(),
                            reinterpret_cast<StridedSliceParameter *>(op_parameter_));
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "StridedSlice error error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuStridedSliceKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                 const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                                 const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                                 const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(desc.type == schema::PrimitiveType_StridedSlice);
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "opParameter null pointer dereferencing.";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) StridedSliceCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "New kernel fails.";
    free(opParameter);
    return nullptr;
  }

  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_StridedSlice, CpuStridedSliceKernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_StridedSlice, CpuStridedSliceKernelCreator)
}  // namespace mindspore::kernel
