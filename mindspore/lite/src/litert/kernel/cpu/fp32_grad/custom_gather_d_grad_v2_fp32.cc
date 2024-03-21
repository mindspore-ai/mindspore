/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include <functional>
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "nnacl/custom_gather_d_grad_v2_parameter.h"
#include "src/litert/kernel/cpu/fp32_grad/custom_gather_d_grad_v2_fp32.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NOT_SUPPORT;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
namespace {
constexpr size_t with_dim_index_idx = 2;
constexpr size_t with_dim_grad_idx = 3;
constexpr size_t without_dim_index_idx = 1;
constexpr size_t without_dim_grad_idx = 2;

size_t get_element_num(const std::vector<int> &shape) {
  return std::accumulate(shape.begin(), shape.end(), static_cast<std::size_t>(1), std::multiplies<int>());
}

void GatherDGradCopyTask(size_t cur, std::vector<size_t> *pos, float *input, int *index, const int &dim, float *output,
                         const std::vector<int> &output_shape, const std::vector<size_t> &out_cargo_size,
                         const std::vector<size_t> &input_cargo_size) {
  for (int i = 0; i < output_shape[cur]; ++i) {
    (*pos)[cur] = i;
    if (cur == output_shape.size() - 1) {
      int input_offset = 0;
      int out_offset = 0;
      // out offset
      for (size_t j = 0; j < output_shape.size(); ++j) {
        out_offset += (*pos)[j] * out_cargo_size[j];
      }
      // input offset
      int cur_index = (*pos)[dim];
      (*pos)[dim] = index[out_offset];
      for (size_t j = 0; j < output_shape.size(); ++j) {
        input_offset += (*pos)[j] * input_cargo_size[j];
      }
      // do copy
      input[input_offset] += output[out_offset];
      (*pos)[dim] = cur_index;
    } else {
      // CopyTask
      GatherDGradCopyTask(cur + 1, pos, input, index, dim, output, output_shape, out_cargo_size, input_cargo_size);
    }
  }
}
}  // namespace

CustomGatherDGradV2CPUKernel::~CustomGatherDGradV2CPUKernel() {}

int CustomGatherDGradV2CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C3NUM);
  if (in_tensors_.size() == C3NUM) {
    index_idx_ = without_dim_index_idx;
    grad_idx_ = without_dim_grad_idx;
  } else if (in_tensors_.size() == C4NUM) {
    index_idx_ = with_dim_index_idx;
    grad_idx_ = with_dim_grad_idx;
  } else {
    MS_LOG(ERROR) << "in tensors size exceed max input 4.";
    return RET_ERROR;
  }
  CHECK_LESS_RETURN(out_tensors_.size(), C1NUM);
  if (InitParamter() != RET_OK) {
    MS_LOG(ERROR) << "Init Built-in CustomGatherGradV2 Parameter failed." << name_;
    return RET_ERROR;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int CustomGatherDGradV2CPUKernel::InitParamter() {
  auto param = reinterpret_cast<CustomGatherGradV2Parameter *>(op_parameter_);
  if (in_tensors_.size() == C3NUM) {
    axis_ = param->dim;
  }
  return RET_OK;
}

int CustomGatherDGradV2CPUKernel::ReSize() {
  index_shape_ = in_tensors_[index_idx_]->shape();
  grad_shape_ = in_tensors_[grad_idx_]->shape();
  output_shape_ = out_tensors_[0]->shape();
  if (grad_shape_.size() != index_shape_.size() || output_shape_.size() != index_shape_.size()) {
    MS_LOG(ERROR) << "For '" << name_ << "', the dimension of grad and output must be the equal to the "
                  << "dimension of index: " << index_shape_.size()
                  << ", but got the dimension of grad: " << grad_shape_.size()
                  << ", the dimension of output: " << output_shape_.size();
    return RET_ERROR;
  }

  return RET_OK;
}

int CustomGatherDGradV2CPUKernel::Run() {
  auto *index = reinterpret_cast<int *>(in_tensors_[index_idx_]->data());
  auto *grad = reinterpret_cast<float *>(in_tensors_[grad_idx_]->data());
  auto out = reinterpret_cast<float *>(out_tensors_[0]->data());
  if (in_tensors_.size() == C4NUM) {
    axis_ = reinterpret_cast<int *>(in_tensors_[dim_idx]->data())[0];
  }
  int output_rank = output_shape_.size();
  if (axis_ >= output_rank || axis_ < -output_rank) {
    MS_LOG(ERROR) << "For '" << name_ << "', the value of 'dim' must be in [" << -output_rank << ", " << output_rank
                  << "), but got: " << axis_;
  }
  if (axis_ < 0) {
    axis_ = axis_ + output_rank;
  }

  // check index
  size_t index_size = get_element_num(index_shape_);
  int max_index = output_shape_[axis_];
  for (size_t i = 0; i < index_size; ++i) {
    if (index[i] >= max_index || index[i] < -max_index) {
      MS_LOG(ERROR) << "For '" << name_ << "', the value of 'index' must be in [" << -max_index << ", " << max_index
                    << "), but got: " << index[i];
    }
    if (index[i] < 0) {
      index[i] = max_index + index[i];
    }
  }
  auto out_size = get_element_num(output_shape_);
  (void)memset(out, 0, out_size * sizeof(float));

  // out_cargo_size
  std::vector<size_t> out_cargo_size = std::vector<size_t>(output_shape_.size(), 1);
  for (int i = static_cast<int>(out_cargo_size.size()) - 2; i >= 0; --i) {
    out_cargo_size[i] = output_shape_[i + 1] * out_cargo_size[i + 1];
  }
  // grad_cargo_size
  std::vector<size_t> grad_cargo_size = std::vector<size_t>(grad_shape_.size(), 1);
  for (int i = static_cast<int>(grad_cargo_size.size()) - 2; i >= 0; --i) {
    grad_cargo_size[i] = grad_shape_[i + 1] * grad_cargo_size[i + 1];
  }

  // copy task
  std::vector<size_t> pos(index_shape_.size(), 0);
  GatherDGradCopyTask(0, &pos, out, index, axis_, grad, index_shape_, grad_cargo_size, out_cargo_size);
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimType_Inner_CustomGatherDGradV2,
           LiteKernelCreator<CustomGatherDGradV2CPUKernel>)
}  // namespace mindspore::kernel
