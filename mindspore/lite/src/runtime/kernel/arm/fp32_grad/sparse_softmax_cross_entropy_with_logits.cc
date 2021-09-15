/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include <cmath>
#include "src/kernel_registry.h"
#include "nnacl/softmax_parameter.h"
#include "nnacl/fp32/softmax_fp32.h"
#include "nnacl/fp32_grad/softmax.h"
#include "src/runtime/kernel/arm/fp32_grad/sparse_softmax_cross_entropy_with_logits.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_SparseSoftmaxCrossEntropyWithLogits;

namespace mindspore::kernel {
int SparseSoftmaxCrossEntropyWithLogitsCPUKernel::ReSize() { return RET_OK; }

int SparseSoftmaxCrossEntropyWithLogitsCPUKernel::ForwardPostExecute(const int *labels, const float *losses,
                                                                     float *output) const {
  float total_loss = 0;
  MS_CHECK_GT(param->batch_size_, 0, RET_ERROR);
  for (int i = 0; i < param->batch_size_; ++i) {
    if (labels[i] < 0) {
      MS_LOG(ERROR) << "label value must >= 0";
      return RET_ERROR;
    }
    size_t label = labels[i];
    if (label > param->number_of_classes_) {
      MS_LOG(ERROR) << "error label input!";
      return RET_ERROR;
    } else {
      total_loss -= logf(losses[i * param->number_of_classes_ + label]);
    }
  }
  output[0] = total_loss / static_cast<float>(param->batch_size_);
  return RET_OK;
}

int SparseSoftmaxCrossEntropyWithLogitsCPUKernel::GradPostExecute(const int *labels, const float *losses,
                                                                  float *grads) const {
  MS_CHECK_GT(param->batch_size_, 0, RET_ERROR);
  size_t row_start = 0;
  for (int i = 0; i < param->batch_size_; ++i) {
    if (labels[i] < 0) {
      MS_LOG(ERROR) << "label value must >= 0";
      return RET_ERROR;
    }
    size_t label = labels[i];
    if (label > param->number_of_classes_) {
      MS_LOG(ERROR) << "error label input!";
      return RET_ERROR;
    } else {
      for (size_t j = 0; j < param->number_of_classes_; ++j) {
        size_t index = row_start + j;
        if (j == label) {
          grads[index] = (losses[index] - 1) / static_cast<float>(param->batch_size_);
        } else {
          grads[index] = losses[index] / static_cast<float>(param->batch_size_);
        }
      }
    }
    row_start += param->number_of_classes_;
  }
  return RET_OK;
}

int SparseSoftmaxCrossEntropyWithLogitsCPUKernel::Execute(int task_id) {
  auto sce_param = reinterpret_cast<SoftmaxCrossEntropyParameter *>(op_parameter_);
  auto ins = reinterpret_cast<float *>(in_tensors_.at(0)->data());
  CHECK_NULL_RETURN(ins);
  auto labels = reinterpret_cast<int *>(in_tensors_.at(1)->data());
  CHECK_NULL_RETURN(labels);
  float *out = reinterpret_cast<float *>(out_tensors_.at(0)->data());
  CHECK_NULL_RETURN(out);
  size_t data_size = in_tensors_.at(0)->ElementsNum();
  float *losses = static_cast<float *>(workspace());
  CHECK_NULL_RETURN(losses);
  float *sum_data = losses + data_size;
  int length = sm_params_.input_shape_[sm_params_.axis_];
  int stride = UP_DIV(outter_size_, threads_);
  int count = MSMIN(stride, outter_size_ - stride * task_id);
  if (count <= 0) return RET_OK;
  switch (stage_) {
    case 0:
      SoftMaxP1(ins, losses, sum_data, task_id * stride, count, length, inner_size_);
      break;
    case 1:
      SoftMaxP2(ins, losses, sum_data, task_id * stride, count, length, inner_size_);
      break;
    case 2:
      if (sce_param->is_grad_) {
        GradPostExecute(labels, losses, out);
      } else {
        ForwardPostExecute(labels, losses, out);
      }
      break;
  }
  return RET_OK;
}

int SparseSoftmaxCrossEntropyWithLogitsRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  CHECK_NULL_RETURN(cdata);
  auto sparse_kernel = reinterpret_cast<SparseSoftmaxCrossEntropyWithLogitsCPUKernel *>(cdata);
  auto error_code = sparse_kernel->Execute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "SparseSoftmaxCrossEntropyWithLogitsRun error task_id[" << task_id << "] error_code[" << error_code
                  << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SparseSoftmaxCrossEntropyWithLogitsCPUKernel::Run() {
  int axis = sm_params_.axis_;
  int n_dim = sm_params_.n_dim_;
  const int *input_shape = sm_params_.input_shape_;
  int inner_size = 1;
  int outter_size = 1;
  CHECK_NULL_RETURN(in_tensors_.at(0));
  size_t data_size = in_tensors_.at(0)->ElementsNum();
  float *losses = static_cast<float *>(workspace());
  CHECK_NULL_RETURN(losses);
  float *sum_data = losses + data_size;
  std::fill(losses, losses + data_size, 0.f);
  std::fill(sum_data, sum_data + sm_params_.input_shape_[0], 0.f);
  for (int i = 0; i < axis; i++) {
    outter_size *= input_shape[i];
  }
  for (int i = axis + 1; i < n_dim; i++) {
    inner_size *= input_shape[i];
  }
  inner_size_ = inner_size;
  outter_size_ = outter_size;
  const std::vector<int> threads = {op_parameter_->thread_num_, op_parameter_->thread_num_, 1};
  for (int stage = 0; stage < static_cast<int>(threads.size()); stage++) {
    stage_ = stage;
    threads_ = threads.at(stage);
    int error_code = ParallelLaunch(this->ms_context_, SparseSoftmaxCrossEntropyWithLogitsRun, this, threads_);
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "SparseSoftmaxCrossEntropyWithLogits function error error_code[" << error_code << "]";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int SparseSoftmaxCrossEntropyWithLogitsCPUKernel::Init() {
  CHECK_LESS_RETURN(in_tensors_.size(), 2);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_.at(0));
  CHECK_NULL_RETURN(in_tensors_.at(1));
  CHECK_NULL_RETURN(out_tensors_.at(0));
  auto dims = in_tensors_.at(0)->shape();
  param->n_dim_ = 2;
  param->number_of_classes_ = dims.at(1);
  param->batch_size_ = dims.at(0);
  for (unsigned int i = 0; i < dims.size(); i++) {
    param->input_shape_[i] = dims.at(i);
  }
  if (this->in_tensors_.size() != 2) {
    MS_LOG(ERROR) << "sparse softmax entropy loss should have two inputs";
    return RET_ERROR;
  }
  auto *in0 = in_tensors_.front();
  if (in0 == nullptr) {
    MS_LOG(ERROR) << "sparse softmax etropy loss in0 have no data";
    return RET_ERROR;
  }
  size_t data_size = in_tensors_.at(0)->ElementsNum();
  set_workspace_size((data_size + dims.at(0)) * sizeof(float));
  sm_params_.n_dim_ = 2;
  sm_params_.element_size_ = static_cast<int>(data_size);
  sm_params_.axis_ = 1;
  for (size_t i = 0; i < dims.size(); i++) {
    sm_params_.input_shape_[i] = dims.at(i);
  }
  return RET_OK;
}
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SparseSoftmaxCrossEntropyWithLogits,
           LiteKernelCreator<SparseSoftmaxCrossEntropyWithLogitsCPUKernel>)
}  // namespace mindspore::kernel
