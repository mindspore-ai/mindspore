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

#include "src/runtime/kernel/arm/fp32/one_hot.h"
#include "src/runtime/kernel/arm/nnacl/fp32/one_hot.h"
#include "schema/model_generated.h"
#include "src/runtime/runtime_api.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_OneHot;

namespace mindspore::kernel {
namespace {
constexpr size_t kInputNum = 4;
constexpr size_t kOutputNum = 1;
}  // namespace

int OneHotCPUKernel::Init() {
  // indices depth on_value off_value
  if (in_tensors_.size() != kInputNum || out_tensors_.size() != kOutputNum) {
    MS_LOG(ERROR) << "OneHot input size should be " << kInputNum << ", got " << in_tensors_.size()
                  << ", output size should be" << kOutputNum << ", got " << out_tensors_.size();
    return RET_ERROR;
  }
  if (context_ == nullptr) {
    MS_LOG(ERROR) << "OneHot context nullptr";
    return RET_NULL_PTR;
  }
  thread_num_ = context_->thread_num_;

  auto param = reinterpret_cast<OneHotParameter *>(op_parameter_);
  if (param == nullptr) {
    MS_LOG(ERROR) << "OneHot op_parameter_ nullptr";
    return RET_NULL_PTR;
  }
  axis_ = param->axis_;

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int OneHotCPUKernel::ReSize() {
  auto indices = in_tensors_.at(0);
  if (indices == nullptr) {
    MS_LOG(ERROR) << "OneHot inputs[0] indices nullptr";
    return RET_NULL_PTR;
  }
  auto indices_shape = indices->shape();
  const int indices_rank = static_cast<int>(indices_shape.size());
  if (axis_ < 0) {
    axis_ += indices_rank + 1;
  }

  outer_size_ = 1;
  for (size_t i = 0; i < static_cast<size_t>(axis_); i++) {
    outer_size_ *= indices_shape[i];
  }
  inner_size_ = indices->ElementsNum() / outer_size_;

  return RET_OK;
}

int RunOneHot(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto onehot_kernel = reinterpret_cast<OneHotCPUKernel *>(cdata);
  if (onehot_kernel == nullptr) {
    MS_LOG(ERROR) << "cast OneHotCPUKernel failed";
    return RET_ERROR;
  }
  auto error_code = onehot_kernel->OneHotImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "RunOneHot error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int OneHotCPUKernel::OneHotImpl(int task_id) {
  auto indices_data = static_cast<int *>(in_tensors_.at(0)->Data());
  auto output = out_tensors_.at(0);
  if (output == nullptr) {
    MS_LOG(ERROR) << "OneHot output nullptr";
    return RET_NULL_PTR;
  }
  auto output_data = static_cast<float *>(output->Data());

  auto ret = GetParams();
  if (ret != RET_OK) {
    return ret;
  }
  auto one_hot_param = reinterpret_cast<OneHotParameter *>(op_parameter_);

  ret = OneHot(indices_data, output_data, one_hot_param, task_id, thread_num_);
  return ret;
}

int OneHotCPUKernel::GetParams() {
  auto one_hot_param = reinterpret_cast<OneHotParameter *>(op_parameter_);
  if (one_hot_param == nullptr) {
    MS_LOG(ERROR) << "cast OneHotParameter nullptr";
    return RET_NULL_PTR;
  }

  auto depth_tensor = in_tensors_.at(1);
  if (depth_tensor == nullptr) {
    MS_LOG(ERROR) << "OneHot inputs[1] depth nullptr";
    return RET_NULL_PTR;
  }
  const int *depth = static_cast<int *>(depth_tensor->Data());
  if (depth == nullptr) {
    return RET_NULL_PTR;
  }
  one_hot_param->depth_ = *depth;

  auto on_value_tensor = in_tensors_.at(2);
  if (on_value_tensor == nullptr) {
    MS_LOG(ERROR) << "OneHot inputs[2] on_value nullptr";
    return RET_NULL_PTR;
  }
  const float *on_value = static_cast<float *>(on_value_tensor->Data());
  if (on_value == nullptr) {
    return RET_NULL_PTR;
  }
  one_hot_param->on_value_ = *on_value;

  auto off_value_tensor = in_tensors_.at(3);
  if (off_value_tensor == nullptr) {
    MS_LOG(ERROR) << "OneHot inputs[3] off_value nullptr";
    return RET_NULL_PTR;
  }
  const float *off_value = static_cast<float *>(off_value_tensor->Data());
  if (off_value == nullptr) {
    return RET_NULL_PTR;
  }
  one_hot_param->off_value_ = *off_value;

  one_hot_param->outer_size_ = outer_size_;
  one_hot_param->inner_size_ = inner_size_;

  return RET_OK;
}

int OneHotCPUKernel::Run() {
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << prepare_ret;
    return prepare_ret;
  }
  int error_code = LiteBackendParallelLaunch(RunOneHot, this, context_->thread_num_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "OneHot function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuOneHotFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                               const std::vector<lite::tensor::Tensor *> &outputs,
                                               OpParameter *opParameter, const lite::Context *ctx,
                                               const kernel::KernelKey &desc,
                                               const mindspore::lite::PrimitiveC *primitive) {
  if (opParameter != nullptr) {
    MS_LOG(ERROR) << "OneHot opParameter nullptr.";
    return nullptr;
  }
  if (desc.type != schema::PrimitiveType_OneHot) {
    MS_LOG(ERROR) << "OneHot desc type should be " << schema::PrimitiveType_OneHot << " got " << desc.type;
    return nullptr;
  }
  auto *kernel = new (std::nothrow) OneHotCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "OneHot new kernel failed.";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_OneHot, CpuOneHotFp32KernelCreator)
}  // namespace mindspore::kernel
