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
#include "src/runtime/kernel/arm/nnacl/errorcode.h"
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
}  // namespace

int PadCPUKernel::Init() {
  if (in_tensors_.size() != kInputNum || out_tensors_.size() != kOutputNum) {
    MS_LOG(ERROR) << "Pad input size should be " << kInputNum << ", got " << in_tensors_.size()
                  << ", output size should be" << kOutputNum << ", got " << out_tensors_.size();
    return RET_ERROR;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int PadCPUKernel::ReSize() {
  auto input = in_tensors_.at(0);
  auto output = out_tensors_.at(0);
  if (input == nullptr || output == nullptr) {
    MS_LOG(ERROR) << "Pad input or output nullptr";
    return RET_NULL_PTR;
  }

  auto rank = input->shape().size();
  if (rank > DEFAULT_PAD_NDIMS) {
    MS_LOG(ERROR) << "Pad input rank should <= " << DEFAULT_PAD_NDIMS << ", got " << rank;
    return RET_ERROR;
  }

  for (int i = 0; i < rank; i++) {
    in_[DEFAULT_PAD_NDIMS - rank + i] = input->shape()[i];
    out_[DEFAULT_PAD_NDIMS - rank + i] = output->shape()[i];
  }
  return RET_OK;
}

int PadImpl(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto padKernel = reinterpret_cast<PadCPUKernel *>(cdata);
  int error_code = padKernel->RunImpl(task_id);
  if (error_code != NNACL_OK) {
    MS_LOG(ERROR) << "Pad Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int PadCPUKernel::RunImpl(int task_id) {
  auto input = in_tensors_.at(0);
  auto output = out_tensors_.at(0);

  auto input_data = reinterpret_cast<float *>(input->Data());
  auto output_data = reinterpret_cast<float *>(output->Data());

  Pad(input_data, output_data, in_, out_, pad_param_->paddings_, task_id, context_->thread_num_);

  return RET_OK;
}

int PadCPUKernel::Run() {
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << prepare_ret;
    return prepare_ret;
  }
  auto output = out_tensors_.at(0);
  int output_size = output->DataSize();

  auto output_data = reinterpret_cast<float *>(output->Data());
  // todo parallel memset to save time
  memset(output_data, 0, output_size * sizeof(float));

  int error_code = LiteBackendParallelLaunch(PadImpl, this, context_->thread_num_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Pad run error, error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
