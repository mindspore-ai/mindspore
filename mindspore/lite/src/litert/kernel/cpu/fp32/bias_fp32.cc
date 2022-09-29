/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/fp32/bias_fp32.h"
#include <vector>
#include "nnacl/fp32/bias_add.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_BiasAdd;

namespace mindspore::kernel {
int BiasAddRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  CHECK_NULL_RETURN(cdata);
  auto kernel = reinterpret_cast<BiasCPUKernel *>(cdata);
  auto ret = kernel->DoExecute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "BatchnormRun error task_id[" << task_id << "] error_code[" << ret << "]";
  }
  return ret;
}

int BiasCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  if (!InferShapeDone()) {
    return RET_OK;
  }

  return ReSize();
}

int BiasCPUKernel::ReSize() {
  auto in_dims = in_tensors_.at(0)->shape();
  auto bias_dims = in_tensors_.at(1)->shape();
  if (bias_dims.empty() || in_dims.empty() || in_dims.size() < bias_dims.size()) {
    MS_LOG(ERROR) << "inTensors' shape are invalid.";
    return RET_ERROR;
  }
  size_t dim_offset = in_dims.size() - bias_dims.size();
  inner_num_ = 1;
  for (size_t i = 0; i < bias_dims.size(); ++i) {
    if (in_dims[i + dim_offset] != bias_dims[i]) {
      MS_LOG(ERROR) << "inTensors' shape cannot match.";
      return RET_ERROR;
    }
    MS_CHECK_FALSE_MSG(INT_MUL_OVERFLOW(bias_dims[i], inner_num_), RET_ERROR, "mul overflow.");
    inner_num_ *= bias_dims[i];
  }
  outer_num_ = 1;
  for (size_t i = 0; i < dim_offset; ++i) {
    MS_CHECK_FALSE_MSG(INT_MUL_OVERFLOW(in_dims[i], outer_num_), RET_ERROR, "mul overflow.");
    outer_num_ *= in_dims[i];
  }
  MS_CHECK_FALSE_MSG(INT_MUL_OVERFLOW(inner_num_, outer_num_), RET_ERROR, "mul overflow.");
  total_num_ = inner_num_ * outer_num_;
  return ChooseThreadCuttingStrategy();
}

int BiasCPUKernel::ChooseThreadCuttingStrategy() {
  split_points_.clear();
  int64_t block_size = 1;

  if (UpdateThreadNumPass(TC_PTYPE(PrimitiveType_BiasAdd), 2, 1, total_num_) != RET_OK) {  // load 2, store 1
    return RET_ERROR;
  }
  block_size = total_num_ / thread_num_;
  int64_t remain_data = total_num_ - block_size * thread_num_;
  int64_t split_point = 0;
  while (split_point < total_num_) {
    split_points_.push_back(split_point);
    split_point += block_size;
    if (remain_data > 0) {
      ++split_point;
      --remain_data;
    }
  }
  if (inner_num_ >= C64NUM && block_size / inner_num_ >= C6NUM) {
    batch_priority_ = true;
  } else {
    batch_priority_ = false;
  }
  return RET_OK;
}

int BiasCPUKernel::Run() {
  auto ret = ParallelLaunch(this->ms_context_, BiasAddRun, this, split_points_.size());
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "BiasAddRun error error_code[" << ret << "]";
  }
  return ret;
}

int BiasCPUKernel::DoExecute(int task_id) {
  auto input = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());
  auto bias = reinterpret_cast<float *>(in_tensors_.at(1)->MutableData());
  auto output = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());
  CHECK_NULL_RETURN(input);
  CHECK_NULL_RETURN(bias);
  CHECK_NULL_RETURN(output);
  int64_t block_start = split_points_[task_id];
  int64_t block_end = total_num_;
  if (static_cast<size_t>(task_id + 1) < split_points_.size()) {
    block_end = split_points_[task_id + 1];
  }
  BiasAddOpt(input, bias, output, block_start, block_end, inner_num_, batch_priority_);
  return lite::RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_BiasAdd, LiteKernelCreator<BiasCPUKernel>)
}  // namespace mindspore::kernel
