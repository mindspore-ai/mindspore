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

#include "src/litert/kernel/cpu/fp32_grad/sgd.h"
#include <string>
#include <algorithm>
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_SGD;

namespace mindspore::kernel {
int SgdCPUKernel::ReSize() { return RET_OK; }

int DoSgd(float *weight, float *accumulate, float *gradient, float learning_rate, float dampening, float moment,
          bool nesterov, float weight_decay, int start, int end) {
  if (weight_decay > 0.f) {
    for (int i = start; i < end; ++i) {
      gradient[i] += weight[i] * weight_decay;
    }
  }
  if (moment > 0.f) {
    if (nesterov) {
      for (int i = start; i < end; ++i) {
        accumulate[i] = accumulate[i] * moment + gradient[i] * (1.f - dampening);
        weight[i] -= (accumulate[i] * moment + gradient[i]) * learning_rate;
      }
    } else {
      for (int i = start; i < end; ++i) {
        accumulate[i] = accumulate[i] * moment + gradient[i] * (1.f - dampening);
        weight[i] -= accumulate[i] * learning_rate;
      }
    }
  } else {
    for (int i = start; i < end; ++i) {
      weight[i] -= gradient[i] * learning_rate;
    }
  }
  return RET_OK;
}

int DoSgdInit(float *weight, float *accumulate, float *gradient, float learning_rate, float moment, bool nesterov,
              float weight_decay, int start, int end) {
  std::copy(&(gradient[start]), &(gradient[end]), &(accumulate[start]));
  if (weight_decay > 0.f) {
    for (int i = start; i < end; ++i) {
      accumulate[i] += weight[i] * weight_decay;
    }
  }
  if (moment > 0.f) {
    if (nesterov) {
      for (int i = start; i < end; ++i) {
        weight[i] -= (accumulate[i] * moment + accumulate[i]) * learning_rate;
      }
    } else {
      for (int i = start; i < end; ++i) {
        weight[i] -= accumulate[i] * learning_rate;
      }
    }
  } else {
    for (int i = start; i < end; ++i) {
      weight[i] -= accumulate[i] * learning_rate;
    }
  }
  return RET_OK;
}

int SgdCPUKernel::DoExecute(int task_id) {
  auto weight = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());
  CHECK_NULL_RETURN(weight);
  auto accumulate = reinterpret_cast<float *>(in_tensors_.at(3)->MutableData());
  CHECK_NULL_RETURN(accumulate);
  float learning_rate = lr_;
  auto gradient = reinterpret_cast<float *>(in_tensors_.at(1)->MutableData());
  CHECK_NULL_RETURN(gradient);
  CHECK_NULL_RETURN(in_tensors_.at(4)->MutableData());
  float moment = reinterpret_cast<float *>(in_tensors_.at(4)->MutableData())[0];
  int length = in_tensors_.at(0)->ElementsNum();

  int stride = UP_DIV(length, thread_count_);
  int count = MSMIN(stride, length - stride * task_id);
  count = (count < 0) ? 0 : count;
  int start = stride * task_id;
  int end = start + count;

  DoSgd(weight, accumulate, gradient, learning_rate, sgd_param_->dampening_, moment, sgd_param_->use_nesterov_,
        sgd_param_->weight_decay_, start, end);

  return RET_OK;
}

int SgdCPUKernel::ExecuteInit(int task_id) {
  auto weight = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());
  CHECK_NULL_RETURN(weight);
  auto accumulate = reinterpret_cast<float *>(in_tensors_.at(3)->MutableData());
  CHECK_NULL_RETURN(accumulate);
  float learning_rate = lr_;
  auto gradient = reinterpret_cast<float *>(in_tensors_.at(1)->MutableData());
  CHECK_NULL_RETURN(gradient);
  CHECK_NULL_RETURN(in_tensors_.at(4)->MutableData());
  float moment = reinterpret_cast<float *>(in_tensors_.at(4)->MutableData())[0];
  auto stat = reinterpret_cast<float *>(in_tensors_.at(5)->MutableData());
  CHECK_NULL_RETURN(stat);
  int length = in_tensors_.at(0)->ElementsNum();
  MS_CHECK_GT(thread_count_, 0, RET_ERROR);
  int stride = UP_DIV(length, thread_count_);
  int count = MSMIN(stride, length - stride * task_id);

  int start = stride * task_id;
  int end = start + count;

  if (count > 0) {
    (void)DoSgdInit(weight, accumulate, gradient, learning_rate, moment, sgd_param_->use_nesterov_,
                    sgd_param_->weight_decay_, start, end);
    sgd_stat_ = 0.0f;
  }
  return RET_OK;
}

int SgdRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto sgd_kernel = reinterpret_cast<SgdCPUKernel *>(cdata);
  CHECK_NULL_RETURN(sgd_kernel);
  auto error_code = RET_OK;
  if (sgd_kernel->get_optimizer_mode() == WeightUpdateMode::VIRTUAL_BATCH) {
    error_code = sgd_kernel->ExecuteVirtualBatch(task_id);
  } else if (sgd_kernel->get_optimizer_mode() == WeightUpdateMode::ACCUMULATE_GRADS) {
    error_code = sgd_kernel->ExecuteVirtualBatch(task_id);
  } else {
    error_code = sgd_kernel->DoExecute(task_id);
  }
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "SGD run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SgdRunInit(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto sgd_kernel = reinterpret_cast<SgdCPUKernel *>(cdata);
  CHECK_NULL_RETURN(sgd_kernel);
  auto error_code = RET_OK;
  if (sgd_kernel->get_optimizer_mode() == WeightUpdateMode::VIRTUAL_BATCH) {
    error_code = sgd_kernel->ExecuteVirtualBatch(task_id);
  } else {
    error_code = sgd_kernel->ExecuteInit(task_id);
  }
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "SGD run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SgdCPUKernel::Run() {
  auto stat = reinterpret_cast<float *>(in_tensors_.at(5)->MutableData());
  CHECK_NULL_RETURN(stat);
  sgd_stat_ = *stat;
  auto error_code = RET_OK;
  if (sgd_stat_ > 0.0f) {
    error_code = ParallelLaunch(this->ms_context_, SgdRunInit, this, thread_count_);
    *stat = sgd_stat_;
  } else {
    error_code = ParallelLaunch(this->ms_context_, SgdRun, this, thread_count_);
  }
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "SGD function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SgdCPUKernel::Prepare() {
  CHECK_NULL_RETURN(sgd_param_);
  CHECK_LESS_RETURN(in_tensors_.size(), 6);
  CHECK_NULL_RETURN(in_tensors_.at(0));
  CHECK_NULL_RETURN(in_tensors_.at(1));
  CHECK_NULL_RETURN(in_tensors_.at(2));
  CHECK_NULL_RETURN(in_tensors_.at(3));
  CHECK_NULL_RETURN(in_tensors_.at(4));
  CHECK_NULL_RETURN(in_tensors_.at(5));
  if (sgd_param_->dampening_ < 0.0f) {
    MS_LOG(ERROR) << "dampening should be at least 0.0";
    return RET_ERROR;
  }

  if (sgd_param_->use_nesterov_ && sgd_param_->dampening_ > 0.0f) {
    MS_LOG(ERROR) << "If use nesterov, dampening must be equal to 0.0";
    return RET_ERROR;
  }
  auto ret = OptimizerKernel::Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Failed to initialize Sgd Kernel";
    return RET_ERROR;
  }
  return RET_OK;
}

std::vector<int> SgdCPUKernel::GetOptimizerParamsIdxs() const {
  std::vector<int> indices = {4};
  return indices;
}

int SgdCPUKernel::OptimizerStep() {
  auto weight = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());

  auto accumulate = reinterpret_cast<float *>(in_tensors_.at(3)->MutableData());
  CHECK_NULL_RETURN(accumulate);
  float learning_rate = lr_;
  auto stat = reinterpret_cast<float *>(in_tensors_.at(5)->MutableData());
  CHECK_NULL_RETURN(stat);
  CHECK_NULL_RETURN(in_tensors_.at(4)->MutableData());
  float moment = reinterpret_cast<float *>(in_tensors_.at(4)->MutableData())[0];
  size_t length = in_tensors_.at(0)->ElementsNum();

  if (grad_sum_ != nullptr && valid_grad_sum_) {
    size_t start = 0;
    size_t end = length;
    if (*stat > 0) {
      DoSgd(weight, accumulate, grad_sum_, learning_rate, sgd_param_->dampening_, moment, sgd_param_->use_nesterov_,
            sgd_param_->weight_decay_, start, end);
    } else {
      (void)DoSgdInit(weight, accumulate, grad_sum_, learning_rate, moment, sgd_param_->use_nesterov_,
                      sgd_param_->weight_decay_, start, end);
      *stat = 0.0f;
    }
    std::fill(grad_sum_, grad_sum_ + length, 0);
    OptimizerKernel::OptimizerStep();
  }

  return RET_OK;
}

kernel::LiteKernel *CpuSgdFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                            const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                            const lite::InnerContext *ctx, const kernel::KernelKey &desc) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_SGD);
  auto *kernel = new (std::nothrow) SgdCPUKernel(opParameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new SgdCPUKernel failed!";
    free(opParameter);
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SGD, CpuSgdFp32KernelCreator)
}  // namespace mindspore::kernel
