/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/arm/fp32/scatter_nd_update_fp32.h"
#include <cstring>
#include "src/runtime/kernel/arm/fp32/scatter_nd_fp32.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ScatterNdUpdate;

namespace mindspore::kernel {
namespace {
constexpr int kScatterUpdateInputIndex = 0;
constexpr int kScatterIndicesIndex = 1;
constexpr int kScatterUpdateIndex = 2;
constexpr size_t kScatterIndicesDims = 2;
}  // namespace
int ScatterNdUpdateCPUKernel::Init() {
  CHECK_LESS_RETURN(in_tensors_.size(), DIMENSION_3D);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ScatterNdUpdateCPUKernel::ReSize() {
  auto input = in_tensors_.at(kScatterUpdateInputIndex);
  auto indices = in_tensors_.at(kScatterIndicesIndex);
  auto update = in_tensors_.at(kScatterUpdateIndex);
  auto output = out_tensors_.front();

  output_ptr_ = reinterpret_cast<float *>(output->MutableData());
  MS_ASSERT(output_ptr_ != nullptr);

  // check indices shape
  int input_rank = static_cast<int>(input->shape().size());
  int indice_unit_rank = indices->shape().back();
  if (indice_unit_rank > input_rank) {
    MS_LOG(ERROR) << "Value of last dimension of indices is greater than input rank.";
    return RET_ERROR;
  }

  if (indices->shape().size() < kScatterIndicesDims) {
    MS_LOG(ERROR) << "Indices dimension smaller than 2.";
    return RET_ERROR;
  }

  // check consistency of the shape indices and shape
  int update_rank = static_cast<int>(update->shape().size());
  auto indices_shape = indices->shape();
  auto update_shape = update->shape();
  unit_size_ = 1;
  for (int i = indices_shape.size() - 1; i < update_rank; i++) {
    unit_size_ *= update_shape.at(i);
  }

  // calculate offsets
  int out_stride = 1;
  out_strides_.push_back(1);
  for (int i = indice_unit_rank - 2; i >= 0; i--) {
    out_stride *= input->shape()[i + 1];
    out_strides_.push_back(out_stride);
  }
  std::reverse(out_strides_.begin(), out_strides_.end());

  num_unit_ = 1;
  num_unit_ *= update_shape.at(indices_shape.size() - 2);
  for (int i = indices_shape.size() - 3; i >= 0; i--) {
    num_unit_ *= update_shape.at(i);
  }

  int *indices_ptr = reinterpret_cast<int *>(indices->MutableData());
  MS_ASSERT(indices_ptr != nullptr);
  output_unit_offsets_.clear();
  for (int i = 0; i < num_unit_; i++) {
    int tmp_stride = 0;
    for (int j = 0; j < indice_unit_rank; j++) {
      tmp_stride += indices_ptr[i * indice_unit_rank + j] * out_strides_.at(j) * unit_size_;
    }
    output_unit_offsets_.push_back(tmp_stride);
  }

  thread_n_num_ = MSMIN(op_parameter_->thread_num_, num_unit_);
  if (thread_n_num_ == 0) {
    return RET_ERROR;
  }
  thread_n_stride_ = UP_DIV(num_unit_, thread_n_num_);
  return RET_OK;
}

int ScatterNdUpdateCPUKernel::ScatterNdUpdate(int task_id) {
  int num_unit_thread = MSMIN(thread_n_stride_, num_unit_ - task_id * thread_n_stride_);
  if (num_unit_thread <= 0) {
    return RET_OK;
  }
  int offset = task_id * thread_n_stride_;
  auto ret = DoScatterND(output_ptr_, update_ptr_ + offset * unit_size_, output_unit_offsets_.data() + offset,
                         unit_size_, num_unit_thread);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ScatterNdUpdate error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  in_tensors_.at(kScatterUpdateInputIndex)->IncRefCount();
  return RET_OK;
}

int ScatterNdUpdateRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto g_kernel = reinterpret_cast<ScatterNdUpdateCPUKernel *>(cdata);
  MS_ASSERT(g_kernel != nullptr);
  auto ret = g_kernel->ScatterNdUpdate(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ScatterNdUpdateRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ScatterNdUpdateCPUKernel::Run() {
  auto in_tensor = in_tensors().front();
  auto out_tensor = out_tensors().front();
  if (in_tensor->allocator() == nullptr || in_tensor->allocator() != out_tensor->allocator() ||
      op_parameter_->is_train_session_) {
    memcpy(out_tensor->data(), in_tensor->data(), in_tensor->Size());
  } else {
    out_tensor->FreeData();
    out_tensor->ResetRefCount();
    in_tensor->allocator()->IncRefCount(in_tensor->data(), out_tensor->ref_count());
    out_tensor->set_data(in_tensor->data());
    out_tensor->set_own_data(in_tensor->own_data());
    output_ptr_ = reinterpret_cast<float *>(out_tensor->data());
  }
  auto indices = in_tensors_.at(kScatterIndicesIndex);
  if (!indices->IsConst() && ReSize() != RET_OK) {
    MS_LOG(ERROR) << "ScatterNdUpdate resize failed.";
    return RET_ERROR;
  }
  auto update = in_tensors_.at(kScatterUpdateIndex);
  update_ptr_ = reinterpret_cast<float *>(update->MutableData());
  MS_ASSERT(update_ptr_ != nullptr);

  auto ret = ParallelLaunch(this->ms_context_, ScatterNdUpdateRun, this, thread_n_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ScatterNdUpdate error error_code[" << ret << "]";
    return RET_ERROR;
  }

  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ScatterNdUpdate, LiteKernelCreator<ScatterNdUpdateCPUKernel>)
}  // namespace mindspore::kernel
