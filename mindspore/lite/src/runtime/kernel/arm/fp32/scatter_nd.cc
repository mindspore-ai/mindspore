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

#include "src/runtime/kernel/arm/fp32/scatter_nd.h"
#include <string.h>
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ScatterND;

namespace mindspore::kernel {
namespace {
constexpr int kScatterNDInputNum = 3;
constexpr int kScatterNDOutputNum = 1;
constexpr int kScatterShapeIndex = 0;
constexpr int kScatterIndicesIndex = 1;
constexpr int kScatterUpdateIndex = 2;
}  // namespace
int ScatterNDCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ScatterNDCPUKernel::ReSize() {
  auto shape = in_tensors_.at(kScatterShapeIndex);
  auto indices = in_tensors_.at(kScatterIndicesIndex);
  auto update = in_tensors_.at(kScatterUpdateIndex);

  update_ptr_ = reinterpret_cast<float *>(update->Data());
  output_ptr_ = reinterpret_cast<float *>(out_tensors_.at(0)->Data());

  // check indices shape
  auto shape_rank = shape->ElementsNum();
  auto shape_data = reinterpret_cast<int *>(shape->Data());
  auto indice_unit_rank = indices->shape().back();
  if (indice_unit_rank > shape_rank) {
    MS_LOG(ERROR) << "Value of last dimension of indices is greater than shape rank.";
    return RET_ERROR;
  }

  if (indices->shape().size() < 2) {
    MS_LOG(ERROR) << "Indices dimension smaller than 2.";
    return RET_ERROR;
  }

  // check consistency of the shape indices and shape
  auto update_rank = static_cast<int>(update->shape().size());
  auto indices_shape = indices->shape();
  if (update_rank != indices->shape().size() - 1 + shape_rank - indice_unit_rank) {
    MS_LOG(ERROR) << "Update, shape rank and indices rank inconsistent.";
    return RET_ERROR;
  }
  // check update shape
  auto update_shape = update->shape();
  for (size_t i = 0; i < indices_shape.size() - 1; i++) {
    if (update_shape[i] != indices_shape[i]) {
      MS_LOG(ERROR) << "Value of " << i << " th dimension of indices is not equal to that of update.";
      return RET_ERROR;
    }
  }
  for (size_t i = 0; i < shape->ElementsNum() - (indices_shape.size() - 1); i++) {
    if (update_shape[i + indices_shape.size() - 1] != shape_data[i + indices_shape.size() - 1]) {
      MS_LOG(ERROR) << "Value of " << i + indices_shape.size() - 1
                    << " th dimension of indices is not equal to the corresbonding dimension of shape.";
      return RET_ERROR;
    }
  }
  // todo check indeices out of range
  // for (size_t i = 0; i < static_cast<size_t>(indice_unit_rank); i++) {}

  // calculate unit_size_
  unit_size_ = 1;
  for (int i = indices_shape.size() - 1; i < update_rank; i++) {
    unit_size_ *= update_shape[i];
  }

  // calculate offsets
  int out_stride = 1;
  out_strides_.push_back(1);
  for (int i = indice_unit_rank - 2; i >= 0; i--) {
    out_stride *= shape_data[i + 1];
    out_strides_.push_back(out_stride);
  }

  num_unit_ = 1;
  num_unit_ *= update_shape[indices_shape.size() - 2];
  for (int i = indices_shape.size() - 3; i >= 0; i--) {
    num_unit_ *= update_shape[i];
  }

  int *indices_ptr = reinterpret_cast<int *>(indices->Data());
  for (int i = 0; i < num_unit_; i++) {
    int tmp_stride = 0;
    for (int j = 0; j < indice_unit_rank; j++) {
      tmp_stride += indices_ptr[i * indice_unit_rank + j] * out_strides_[j] * unit_size_;
    }
    output_unit_offsets_.push_back(tmp_stride);
  }

  thread_n_num_ = MSMIN(op_parameter_->thread_num_, num_unit_);
  thread_n_stride_ = UP_DIV(num_unit_, thread_n_num_);
  return RET_OK;
}

int ScatterNDCPUKernel::ScatterND(int task_id) {
  int num_unit_thread = MSMIN(thread_n_stride_, num_unit_ - task_id * thread_n_stride_);
  if (num_unit_thread <= 0) {
    return RET_OK;
  }
  int offset = task_id * thread_n_stride_;
  MS_LOG(ERROR) << "offset " << offset << std::endl;
  auto ret = DoScatterND(output_ptr_, update_ptr_ + offset * unit_size_, output_unit_offsets_.data() + offset,
                         unit_size_, num_unit_thread);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ScatterND error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ScatterNDRun(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto g_kernel = reinterpret_cast<ScatterNDCPUKernel *>(cdata);
  auto ret = g_kernel->ScatterND(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ScatterNDRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ScatterNDCPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << ret;
    return ret;
  }
  ret = LiteBackendParallelLaunch(ScatterNDRun, this, thread_n_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ScatterND error error_code[" << ret << "]";
    return RET_ERROR;
  }

  return RET_OK;
}

kernel::LiteKernel *CpuScatterNDFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                  const std::vector<lite::tensor::Tensor *> &outputs,
                                                  OpParameter *opParameter, const lite::Context *ctx,
                                                  const kernel::KernelKey &desc,
                                                  const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(desc.type == schema::PrimitiveType_ScatterND);
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "desc type is not scatterND";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) ScatterNDCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "New kernel fails.";
    return nullptr;
  }

  auto ret = kernel->Init();
  if (ret != 0) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }

  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ScatterND, CpuScatterNDFp32KernelCreator)
}  // namespace mindspore::kernel
