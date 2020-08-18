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

#include "src/runtime/kernel/arm/fp32/transpose.h"
#include <vector>
#include "src/runtime/kernel/arm/nnacl/transpose.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_OP_EXECUTE_FAILURE;
using mindspore::schema::PrimitiveType_Transpose;

namespace mindspore::kernel {
namespace {
constexpr int kTransposeInputNum = 1;
constexpr int kTransposeOutputNum = 1;
}  // namespace
int TransposeCPUKernel::Init() {
  TransposeParameter *param = reinterpret_cast<TransposeParameter *>(this->op_parameter_);
  num_unit_ = static_cast<int>(in_tensors_[kInputIndex]->shape().at(param->perm_[kNHWC_H]));
  thread_h_num_ = MSMIN(thread_num_, num_unit_);
  thread_h_stride_ = UP_DIV(num_unit_, thread_h_num_);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int TransposeCPUKernel::ReSize() {
  auto &inTensor = in_tensors_.front();
  auto &outTensor = out_tensors_.front();
  auto param = reinterpret_cast<TransposeParameter *>(op_parameter_);
  auto in_shape = inTensor->shape();
  auto out_shape = outTensor->shape();
  param->strides_[param->num_axes_ - 1] = 1;
  param->out_strides_[param->num_axes_ - 1] = 1;
  param->data_size_ = inTensor->Size();
  for (int i = param->num_axes_ - 2; i >= 0; i--) {
    param->strides_[i] = in_shape[i + 1] * param->strides_[i + 1];
    param->out_strides_[i] = out_shape[i + 1] * param->out_strides_[i + 1];
  }
  return RET_OK;
}

int TransposeCPUKernel::TransposeParallel(int task_id) {
  int num_unit_thread = MSMIN(thread_h_stride_, num_unit_ - task_id * thread_h_stride_);
  if (num_unit_thread <= 0) {
    return RET_OK;
  }
  int thread_offset = task_id * thread_h_stride_;
  TransposeParameter *param = reinterpret_cast<TransposeParameter *>(this->op_parameter_);
  auto ret =
    DoTranspose(in_data_, out_data_, in_shape_, out_shape_, param, thread_offset, thread_offset + num_unit_thread);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Transpose error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int TransposeRun(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto g_kernel = reinterpret_cast<TransposeCPUKernel *>(cdata);
  auto ret = g_kernel->TransposeParallel(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "TransposeRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_OP_EXECUTE_FAILURE;
  }
  return RET_OK;
}

int TransposeCPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << ret;
    return ret;
  }
  MS_ASSERT(in_tensors_.size() == TransposeInputNum);
  MS_ASSERT(out_tensors_.size() == TransposeOutputNum);
  auto &in_tensor = in_tensors_.front();
  auto &out_tensor = out_tensors_.front();
  if (in_tensor == nullptr || out_tensor == nullptr) {
    MS_LOG(ERROR) << "null pointer dreferencing.";
    return RET_ERROR;
  }
  in_data_ = reinterpret_cast<float *>(in_tensor->Data());
  out_data_ = reinterpret_cast<float *>(out_tensor->Data());
  in_shape_ = const_cast<int *>(in_tensor->shape().data());
  out_shape_ = const_cast<int *>(out_tensor->shape().data());

  ret = LiteBackendParallelLaunch(TransposeRun, this, thread_h_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Tranpose error error_code[" << ret << "]";
    return ret;
  }
  return ret;
}  // namespace mindspore::kernel

kernel::LiteKernel *CpuTransposeFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                  const std::vector<lite::tensor::Tensor *> &outputs,
                                                  OpParameter *opParameter, const lite::Context *ctx,
                                                  const kernel::KernelKey &desc,
                                                  const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(desc.type == schema::PrimitiveType_Transpose);
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "desc type is not Transpose";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) TransposeCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "New kernel fails.";
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Transpose, CpuTransposeFp32KernelCreator)
}  // namespace mindspore::kernel
