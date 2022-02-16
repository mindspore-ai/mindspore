/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
}  // namespace
int ScatterNdUpdateCPUKernel::Prepare() {
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
  auto input_shape = input->shape();
  auto indices_shape = indices->shape();
  auto update_shape = update->shape();
  int input_rank = static_cast<int>(input->shape().size());
  int indices_rank = static_cast<int>(indices->shape().size());
  int update_rank = static_cast<int>(update->shape().size());
  int indice_unit_rank = indices->shape().back();
  auto output = out_tensors_.front();
  output_ptr_ = output->data();

  // check indices shape
  MS_CHECK_TRUE_MSG(indices_rank >= DIMENSION_2D, RET_ERROR, "The rank of indices must be greater equal than 2.");
  MS_CHECK_TRUE_MSG(indice_unit_rank <= input_rank, RET_ERROR,
                    "The value of indices' last dimension must be less equal than the input rank.");
  MS_CHECK_TRUE_MSG(update_rank == indices_rank - 1 + input_rank - indice_unit_rank, RET_ERROR,
                    "The rank of update is illegal.");
  // check consistency of the shape indices and shape
  for (int i = 0; i < update_rank; i++) {
    if (i < indices_rank - 1) {
      MS_CHECK_TRUE_MSG(update_shape[i] == indices_shape[i], RET_ERROR, "the shape of update tensor is illegal.");
    }
    if (i >= indice_unit_rank) {
      MS_CHECK_TRUE_MSG(update_shape[i] == input_shape[i], RET_ERROR, "the shape of update tensor is illegal.");
    }
  }

  // calculate unit_size
  param_->unit_size = 1;
  for (int i = indices_shape.size() - 1; i < update_rank; i++) {
    param_->unit_size *= update_shape.at(i);
  }

  // calculate offsets
  int out_stride = 1;
  std::vector<int> out_strides;
  out_strides.push_back(1);
  for (int i = indice_unit_rank - C2NUM; i >= 0; i--) {
    out_stride *= input_shape[i + 1];
    out_strides.push_back(out_stride);
  }
  std::reverse(out_strides.begin(), out_strides.end());

  param_->num_unit = 1;
  param_->num_unit *= update_shape.at(indices_shape.size() - C2NUM);
  for (int i = indices_shape.size() - C3NUM; i >= 0; i--) {
    param_->num_unit *= update_shape.at(i);
  }

  int *indices_ptr = reinterpret_cast<int *>(indices->MutableData());
  MS_ASSERT(indices_ptr != nullptr);
  output_unit_offsets_.clear();
  for (int i = 0; i < param_->num_unit; i++) {
    int tmp_stride = 0;
    for (int j = 0; j < indice_unit_rank; j++) {
      tmp_stride += indices_ptr[i * indice_unit_rank + j] * out_strides.at(j) * param_->unit_size;
    }
    output_unit_offsets_.push_back(tmp_stride);
  }
  return RET_OK;
}

int ScatterNdUpdateCPUKernel::ScatterNdUpdate(int task_id) {
  void *update_data = in_tensors_[kScatterUpdateIndex]->data();
  auto output_tensor = out_tensors_[kOutputIndex];
  void *output_data = output_tensor->data();
  CHECK_NULL_RETURN(update_data);
  CHECK_NULL_RETURN(output_data);
  param_->data_type_len = output_tensor->data_type() == kNumberTypeFloat16 ? FP16_DATA_TYPE_LEN : sizeof(float);
  auto ret = DoScatterND(output_data, update_data, output_unit_offsets_.data(), param_, task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DoScatterND failed, ret: " << ret;
    return RET_ERROR;
  }
  in_tensors_.at(kScatterUpdateInputIndex)->IncRefCount();
  return RET_OK;
}

int ScatterNdUpdateRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto kernel = static_cast<ScatterNdUpdateCPUKernel *>(cdata);
  CHECK_NULL_RETURN(kernel);
  return kernel->ScatterNdUpdate(task_id);
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
    output_ptr_ = out_tensor->data();
  }
  auto indices = in_tensors_.at(kScatterIndicesIndex);
  if (!indices->IsConst() && ReSize() != RET_OK) {
    MS_LOG(ERROR) << "ScatterNdUpdate resize failed.";
    return RET_ERROR;
  }

  auto ret = ParallelLaunch(ms_context_, ScatterNdUpdateRun, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ScatterNdUpdate error error_code[" << ret << "]";
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ScatterNdUpdate, LiteKernelCreator<ScatterNdUpdateCPUKernel>)
#ifdef ENABLE_FP16
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_ScatterNdUpdate, LiteKernelCreator<ScatterNdUpdateCPUKernel>)
#endif
}  // namespace mindspore::kernel
