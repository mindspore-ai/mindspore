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

#include "src/litert/kernel/cpu/fp32/softmax_fp32.h"
#include <cstring>
#include "nnacl/errorcode.h"
#include "nnacl/fp32/softmax_fp32.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Softmax;

namespace mindspore::kernel {
int SoftmaxCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  auto ret = SoftmaxBaseCPUKernel::Prepare();
  if (ret != RET_OK) {
    return ret;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int SoftmaxCPUKernel::ReSize() {
  auto ret = SoftmaxBaseCPUKernel::ReSize();
  if (ret != RET_OK) {
    return ret;
  }
  auto n_dim = softmax_param_->n_dim_;
  auto axis = softmax_param_->axis_;
  auto in_shape = in_tensors_.front()->shape();
  int out_plane_size = 1;
  MS_CHECK_TRUE_RET(axis > 0 && static_cast<size_t>(axis) < in_shape.size(), RET_ERROR);
  for (int i = 0; i < axis; ++i) {
    out_plane_size *= in_shape.at(i);
  }
  int in_plane_size = 1;
  for (int i = axis + 1; i < n_dim; i++) {
    in_plane_size *= in_shape.at(i);
  }
  in_plane_size_ = in_plane_size;
  out_plane_size_ = out_plane_size;
  if (in_plane_size_ > 1) {
    if (sum_data_ != nullptr) {
      free(sum_data_);
    }
    CHECK_LESS_RETURN(MAX_MALLOC_SIZE, out_plane_size_ * in_plane_size_ * sizeof(float));
    sum_data_ = reinterpret_cast<float *>(malloc(out_plane_size * in_plane_size * sizeof(float)));
    if (sum_data_ == nullptr) {
      MS_LOG(ERROR) << "malloc data for softmax fail!";
      return RET_ERROR;
    }
  }

  if (UpdateThreadNumPass(
        TC_PTYPE(softmax_param_->op_parameter_.type_), softmax_param_->input_shape_[softmax_param_->axis_],
        softmax_param_->input_shape_[softmax_param_->axis_], out_tensors_.at(0)->ElementsNum()) != RET_OK) {
    return RET_ERROR;
  }
  return RET_OK;
}

int SoftmaxCPUKernel::DoSoftmaxLastAxis(int task_id) {
  int unit = UP_DIV(out_plane_size_, thread_num_);
  if (INT_MUL_OVERFLOW(task_id, unit)) {
    MS_LOG(ERROR) << "int mul overflow.";
    return RET_ERROR;
  }
  int begin = task_id * unit;
  int end = MSMIN(begin + unit, out_plane_size_);
  int channel = softmax_param_->input_shape_[softmax_param_->axis_];
  if (INT_MUL_OVERFLOW(begin, channel)) {
    MS_LOG(ERROR) << "int mul overflow.";
    return RET_ERROR;
  }
  int offset = begin * channel;
  auto input_ptr = reinterpret_cast<float *>(in_tensors_.at(kInputIndex)->MutableData());
  auto output_ptr = reinterpret_cast<float *>(out_tensors_.at(kOutputIndex)->MutableData());
  int ret = SoftmaxLastAxis(input_ptr + offset, output_ptr + offset, end - begin, channel);
  if (ret != NNACL_OK) {
    MS_LOG(ERROR) << "do SoftmaxLastAxis failed. " << this->name();
    return RET_ERROR;
  }
  return RET_OK;
}

int SoftmaxLastAxisRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  CHECK_NULL_RETURN(cdata);
  auto kernel = reinterpret_cast<SoftmaxCPUKernel *>(cdata);
  auto ret = kernel->DoSoftmaxLastAxis(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DoSoftmaxLastAxis error task_id: " << task_id << ", ret: " << ret;
  }
  return ret;
}

int SoftmaxCPUKernel::Run() {
  int ret = RET_OK;
  if (in_plane_size_ == 1) {
    ret = ParallelLaunch(this->ms_context_, SoftmaxLastAxisRun, this, thread_num_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "SoftmaxCPUKernel ParallelLaunch failed, ret: " << ret;
    }
  } else {
    MS_ASSERT(sum_data_);
    MS_ASSERT(softmax_param_);
    auto input_ptr = reinterpret_cast<float *>(in_tensors_.at(kInputIndex)->data());
    MS_ASSERT(input_ptr);
    auto output_ptr = reinterpret_cast<float *>(out_tensors_.at(kOutputIndex)->data());
    MS_ASSERT(output_ptr);
    Softmax(input_ptr, output_ptr, sum_data_, softmax_param_);
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Softmax, LiteKernelCreator<SoftmaxCPUKernel>)
}  // namespace mindspore::kernel
