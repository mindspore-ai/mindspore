
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

#include "src/runtime/kernel/arm/fp32_grad/sgd.h"
#include <algorithm>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_SGD;

namespace mindspore::kernel {

int SgdCPUKernel::ReSize() { return RET_OK; }

int DoSgd(float *weight, float *accumulate, float *gradient, float learning_rate, float dampening, float moment,
          bool nesterov, int start, int end) {
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

int DoSgdInit(float *weight, float *accumulate, float *gradient, float *stat, float learning_rate, float dampening,
              float moment, bool nesterov, int start, int end) {
  std::copy(&(gradient[start]), &(gradient[end]), &(accumulate[start]));
  if (nesterov) {
    for (int i = start; i < end; ++i) {
      weight[i] -= (accumulate[i] * moment + gradient[i]) * learning_rate;
    }
  } else {
    for (int i = start; i < end; ++i) {
      weight[i] -= accumulate[i] * learning_rate;
    }
  }
  *stat = 1.0f;
  return RET_OK;
}

int SgdCPUKernel::Execute(int task_id) {
  auto weight = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());
  auto accumulate = reinterpret_cast<float *>(in_tensors_.at(3)->MutableData());
  float learning_rate = lr_;
  auto gradient = reinterpret_cast<float *>(in_tensors_.at(1)->MutableData());
  float moment = reinterpret_cast<float *>(in_tensors_.at(4)->MutableData())[0];
  int length = in_tensors_.at(0)->ElementsNum();

  int stride = UP_DIV(length, thread_count_);
  int count = MSMIN(stride, length - stride * task_id);
  count = (count < 0) ? 0 : count;
  int start = stride * task_id;
  int end = start + count;

  DoSgd(weight, accumulate, gradient, learning_rate, sgd_param_->dampening_, moment, sgd_param_->use_nesterov_, start,
        end);

  return RET_OK;
}

int SgdCPUKernel::ExecuteInit(int task_id) {
  auto weight = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());
  auto accumulate = reinterpret_cast<float *>(in_tensors_.at(3)->MutableData());
  float learning_rate = lr_;
  auto gradient = reinterpret_cast<float *>(in_tensors_.at(1)->MutableData());
  float moment = reinterpret_cast<float *>(in_tensors_.at(4)->MutableData())[0];
  auto stat = reinterpret_cast<float *>(in_tensors_.at(5)->MutableData());
  int length = in_tensors_.at(0)->ElementsNum();

  int stride = UP_DIV(length, thread_count_);
  int count = MSMIN(stride, length - stride * task_id);

  int start = stride * task_id;
  int end = start + count;

  if (count > 0) {
    DoSgdInit(weight, accumulate, gradient, stat, learning_rate, sgd_param_->dampening_, moment,
              sgd_param_->use_nesterov_, start, end);
  }
  return RET_OK;
}

int SgdRun(void *cdata, int task_id) {
  auto sgd_kernel = reinterpret_cast<SgdCPUKernel *>(cdata);
  auto error_code = RET_OK;
  if (sgd_kernel->get_optimizer_mode() == OptimizerKernel::WeightUpdateMode::VIRTUAL_BATCH) {
    error_code = sgd_kernel->ExecuteVirtualBatch(task_id);
  } else {
    error_code = sgd_kernel->Execute(task_id);
  }
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "SGD run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SgdRunInit(void *cdata, int task_id) {
  auto sgd_kernel = reinterpret_cast<SgdCPUKernel *>(cdata);
  auto error_code = RET_OK;
  if (sgd_kernel->get_optimizer_mode() == OptimizerKernel::WeightUpdateMode::VIRTUAL_BATCH) {
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
  auto error_code = RET_OK;
  if (*stat > 0.0f) {
    error_code = ParallelLaunch(this->context_->thread_pool_, SgdRunInit, this, thread_count_);
  } else {
    error_code = ParallelLaunch(this->context_->thread_pool_, SgdRun, this, thread_count_);
  }
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "SGD function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SgdCPUKernel::Init() {
  if (sgd_param_->dampening_ < 0.0f) {
    MS_LOG(ERROR) << "dampening should be at least 0.0";
    return RET_ERROR;
  }

  if (sgd_param_->use_nesterov_ && sgd_param_->dampening_ > 0.0f) {
    MS_LOG(ERROR) << "If use nesterov, dampening must equal to 0.0";
    return RET_ERROR;
  }
  auto ret = OptimizerKernel::Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Failed to initialize Sgd Kernel";
    return RET_ERROR;
  }
  return RET_OK;
}

int SgdCPUKernel::OptimizerStep() {
  auto weight = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());
  auto accumulate = reinterpret_cast<float *>(in_tensors_.at(3)->MutableData());
  float learning_rate = lr_;
  auto stat = reinterpret_cast<float *>(in_tensors_.at(5)->MutableData());
  float moment = reinterpret_cast<float *>(in_tensors_.at(4)->MutableData())[0];
  size_t length = in_tensors_.at(0)->ElementsNum();

  if (grad_sum_ != nullptr && valid_grad_sum_) {
    size_t start = 0;
    size_t end = length;
    if (*stat > 0) {
      DoSgd(weight, accumulate, grad_sum_, learning_rate, sgd_param_->dampening_, moment, sgd_param_->use_nesterov_,
            start, end);
    } else {
      DoSgdInit(weight, accumulate, grad_sum_, stat, learning_rate, sgd_param_->dampening_, moment,
                sgd_param_->use_nesterov_, start, end);
    }
    std::fill(grad_sum_, grad_sum_ + length, 0);
    OptimizerKernel::OptimizerStep();
  }

  return RET_OK;
}

kernel::LiteKernel *CpuSgdFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                            const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                            const lite::InnerContext *ctx, const kernel::KernelKey &desc) {
  MS_ASSERT(desc.type == schema::PrimitiveType_SGD);
  auto *kernel = new (std::nothrow) SgdCPUKernel(opParameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new SgdCPUKernel failed!";
    free(opParameter);
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SGD, CpuSgdFp32KernelCreator)
}  // namespace mindspore::kernel
