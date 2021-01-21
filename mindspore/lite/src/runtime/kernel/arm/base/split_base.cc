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
#include "src/runtime/kernel/arm/base/split_base.h"
#include <vector>
#include "src/runtime/kernel/arm/fp32/split_fp32.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "include/context.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Split;

namespace mindspore::kernel {
int SplitBaseCPUKernel::Init() {
  auto split_dim = param->split_dim_;
  param->split_dim_ = split_dim >= 0 ? split_dim : in_tensors_.front()->shape().size() + split_dim;
  return RET_OK;
}

int SplitBaseCPUKernel::ReSize() {
  auto in_tensor = in_tensors_.front();
  auto input_shape = in_tensor->shape();

  MS_ASSERT(param);
  MS_ASSERT(input_shape.size() >= 1 && input_shape.size() <= SPLIT_STRIDES_SIZE);
  param->strides_[input_shape.size() - 1] = 1;
  for (int i = input_shape.size() - 2; i >= 0; i--) {
    param->strides_[i] = param->strides_[i + 1] * input_shape.at(i + 1);
  }

  MS_ASSERT(static_cast<size_t>(param->split_dim_) < input_shape.size());
  param->split_count_ =
    param->strides_[0] * input_shape.at(0) / (input_shape.at(param->split_dim_) * param->strides_[param->split_dim_]);
  param->n_dims_ = input_shape.size();

  if (param->split_sizes_[0] == 0) {
    MS_ASSERT(param->num_split_ > 0 && static_cast<int>(param->num_split_) <= input_shape[param->split_dim_]);
    if (input_shape[param->split_dim_] % param->num_split_ != 0) {
      MS_LOG(ERROR) << "Default split size is not usable.";
      return RET_ERROR;
    }
    int split_size = input_shape.at(param->split_dim_) / param->num_split_;
    for (int i = 0; i < param->num_split_; i++) {
      param->split_sizes_[i] = split_size;
    }
  }

  if (param->split_sizes_[param->num_split_ - 1] == -1) {
    int split_shape_end = input_shape.at(param->split_dim_);
    for (int i = 0; i < param->num_split_ - 1; i++) {
      split_shape_end -= param->split_sizes_[i];
    }
    param->split_sizes_[param->num_split_ - 1] = split_shape_end;
  }

  num_unit_ = param->split_count_ * param->num_split_;
  thread_n_num_ = MSMIN(thread_count_, num_unit_);
  if (thread_n_num_ != 0) {
    thread_n_stride_ = UP_DIV(num_unit_, thread_n_num_);
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
