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

#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/fp32/pad.h"
#include "include/errorcode.h"
#include "src/runtime/kernel/arm/opclib/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Pad;

namespace mindspore::kernel {
namespace {
constexpr int kInputNum = 1;
constexpr int kOutputNum = 1;
constexpr int kInputRank = 4;
constexpr int kPaddingsSize = 8;
}  // namespace

int PadCPUKernel::CheckInputsOutputsParams() {
  if (inputs_.size() != kInputNum || outputs_.size() != kOutputNum) {
    MS_LOG(ERROR) << "Pad input size should be " << kInputNum << ", got " << inputs_.size() << ", output size should be"
                  << kOutputNum << ", got " << outputs_.size();
    return RET_ERROR;
  }

  auto input = inputs_.at(0);
  auto output = outputs_.at(0);
  if (input == nullptr || output == nullptr) {
    MS_LOG(ERROR) << "Pad input or output nullptr";
    return RET_NULL_PTR;
  }

  auto rank = input->shape().size();
  if (rank != kInputRank) {
    MS_LOG(ERROR) << "Pad input rank should be " << kInputRank << ", got " << rank;
    return RET_ERROR;
  }

  if (paddings_size_ != kPaddingsSize) {
    MS_LOG(ERROR) << "Pad op paddings size should be 2*input_rank: " << 2 * rank << " but got " << paddings_size_;
    return RET_ERROR;
  }

  for (auto pad : paddings_) {
    if (pad < 0) {
      MS_LOG(ERROR) << "Pad op paddings should be >= 0, but got " << pad;
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int PadCPUKernel::MaybeConvertInputLayout() {
  auto input = inputs_.at(0);
  auto input_format = input->GetFormat();
  if (input_format != exec_format_) {
    auto input_type = input->data_type();
    layout_convertor_ = LayoutTransform(input_type, input_format, exec_format_);
    if (layout_convertor_ == nullptr) {
      MS_LOG(ERROR) << "Pad lack layout convertor from " << input_format << " to " << exec_format_;
      return RET_ERROR;
    }
    exec_input_data_ = reinterpret_cast<float *>(malloc(input->DataSize() * sizeof(float)));
    if (exec_input_data_ == nullptr) {
      MS_LOG(ERROR) << "Pad malloc failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int PadCPUKernel::Init() {
  auto ret = CheckInputsOutputsParams();
  if (ret != RET_OK) {
    return ret;
  }

  ret = MaybeConvertInputLayout();
  if (ret != RET_OK) {
    return ret;
  }

  auto output = outputs_.at(0);
  output->SetFormat(exec_format_);

  return RET_OK;
}

int PadImpl(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto padKernel = reinterpret_cast<PadCPUKernel *>(cdata);
  int error_code = padKernel->RunImpl(task_id);
  if (error_code != OPCLIB_OK) {
    MS_LOG(ERROR) << "Pad Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int PadCPUKernel::RunImpl(int task_id) {
  auto input = inputs_.at(0);
  auto output = outputs_.at(0);

  auto input_data = reinterpret_cast<float *>(input->Data());
  auto output_data = reinterpret_cast<float *>(output->Data());
  auto input_shape = input->shape().data();
  auto output_shape = output->shape().data();
  if (exec_input_data_ != nullptr) {
    Pad(exec_input_data_, output_data, input_shape, output_shape, paddings_.data(), task_id, context_->threadNum);
  } else {
    Pad(input_data, output_data, input_shape, output_shape, paddings_.data(), task_id, context_->threadNum);
  }

  return RET_OK;
}

int PadCPUKernel::Run() {
  auto output = outputs_.at(0);
  int output_size = output->DataSize();

  auto output_data = reinterpret_cast<float *>(output->Data());
  // todo parallel memset to save time
  memset(output_data, 0, output_size * sizeof(float));

  auto input = inputs_.at(0);
  if (exec_input_data_ != nullptr) {
    if (layout_convertor_ == nullptr) {
      return RET_NULL_PTR;
    }
    layout_convertor_(inputs_.at(0), exec_input_data_, input->Batch(), input->Height() * input->Width(),
                      input->Channel());
  }

  int error_code = LiteBackendParallelLaunch(PadImpl, this, context_->threadNum);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Pad run error, error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuPadFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                            const std::vector<lite::tensor::Tensor *> &outputs,
                                            OpParameter *opParameter, const lite::Context *ctx,
                                            const kernel::KernelKey &desc) {
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Pad opParameter nullptr";
    return nullptr;
  }
  MS_ASSERT(desc.type == PrimitiveType_Pad);
  auto *kernel = new (std::nothrow) PadCPUKernel(opParameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new PadCPUKernel failed.";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, PrimitiveType_Pad, CpuPadFp32KernelCreator)
}  // namespace mindspore::kernel

