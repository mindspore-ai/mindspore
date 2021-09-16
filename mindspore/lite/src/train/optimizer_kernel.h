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
#ifndef MINDSPORE_LITE_SRC_TRAIN_OPTIMIZER_KERNEL_H_
#define MINDSPORE_LITE_SRC_TRAIN_OPTIMIZER_KERNEL_H_
#include <vector>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <string>
#include <iostream>
#include "src/lite_kernel.h"
#include "include/ms_tensor.h"
#include "include/errorcode.h"
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_OUT_OF_TENSOR_RANGE;

static __attribute__((always_inline)) inline bool MS_ISNAN(float var) {
  volatile float d = var;
  return d != d;
}

namespace mindspore::kernel {
enum class WeightUpdateMode { NORMAL, VIRTUAL_BATCH, ACCUMULATE_GRADS };

class OptimizerKernel : public InnerKernel {
 public:
  OptimizerKernel() = default;
  OptimizerKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                  const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx, int lr_idx, int grad_idx)
      : InnerKernel(parameter, inputs, outputs, ctx), lr_idx_(lr_idx), grad_idx_(grad_idx) {}
  ~OptimizerKernel() = default;

  WeightUpdateMode get_optimizer_mode() { return weight_update_mod_; }

  int Init() override {
    default_lr_ = reinterpret_cast<float *>(in_tensors_.at(lr_idx_)->MutableData())[0];
    lr_ = default_lr_;
    return RET_OK;
  }

  int SetLearningRate(float lr) {
    lr_ = lr;
    return RET_OK;
  }

  float GetLearningRate() { return lr_; }

  virtual std::vector<int> GetOptimizerParamsIdxs() const {
    std::vector<int> indices;
    return indices;
  }

  std::vector<tensor::MSTensor *> GetOptimizerParams() const {
    std::vector<tensor::MSTensor *> params;
    auto indices = GetOptimizerParamsIdxs();
    indices.push_back(lr_idx_);
    for (size_t ix = 0; ix < indices.size(); ix++) {
      auto param = in_tensors_.at(indices[ix]);
      if (param->data() == nullptr) {
        continue;
      }
      params.push_back(param);
    }
    return params;
  }

  bool SetOptimizerParams(tensor::MSTensor *param) {
    if (param == nullptr) {
      return false;
    }
    bool found = false;
    auto indices = GetOptimizerParamsIdxs();
    indices.push_back(lr_idx_);
    for (size_t ix = 0; ix < indices.size(); ix++) {
      if (param->tensor_name() == in_tensors_.at(indices[ix])->tensor_name() && param->ElementsNum() == 1 &&
          (param->data_type() == kNumberTypeFloat32 || param->data_type() == kNumberTypeFloat)) {
        auto value = static_cast<float *>(param->MutableData())[0];
        static_cast<float *>(in_tensors_.at(indices[ix])->MutableData())[0] = value;
        if (lr_idx_ == indices[ix]) {
          lr_ = value;
        }
        found = true;
        break;
      }
    }
    return found;
  }

  lite::Tensor *GetGradients() {
    lite::Tensor *grad_sum_tensor = nullptr;
    if (grad_sum_ != nullptr) {
      auto shape = in_tensors_.at(grad_idx_)->shape();
      grad_sum_tensor = new (std::nothrow) lite::Tensor(kNumberTypeFloat, shape);
      if (grad_sum_tensor == nullptr) {
        MS_LOG(ERROR) << "failed to allocate grad sum tensor";
        return nullptr;
      }
      grad_sum_tensor->set_tensor_name(in_tensors_.at(grad_idx_)->tensor_name());
      grad_sum_tensor->set_data(static_cast<void *>(grad_sum_));
      grad_sum_tensor->set_own_data(false);
    }
    return grad_sum_tensor;
  }

  int RestoreDefaultLearningRate() {
    SetLearningRate(default_lr_);
    return RET_OK;
  }

  int SetOptimizerMode(WeightUpdateMode mod) {
    if (mod == WeightUpdateMode::VIRTUAL_BATCH || mod == WeightUpdateMode::ACCUMULATE_GRADS) {
      if (grad_sum_ != nullptr) {
        ms_context_->allocator->Free(grad_sum_);
        grad_sum_ = nullptr;
      }
      size_t size = in_tensors_.at(grad_idx_)->Size();
      size_t elem_num = in_tensors_.at(grad_idx_)->ElementsNum();
      grad_sum_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(size));
      if (grad_sum_ == nullptr) {
        MS_LOG(ERROR) << "failed to malloc grad sum tensor, size=" << size;
        return RET_ERROR;
      }
      valid_grad_sum_ = false;
      std::fill(grad_sum_, grad_sum_ + elem_num, 0);
      weight_update_mod_ = mod;
    } else {
      if (grad_sum_ != nullptr) {
        OptimizerStep();
        ms_context_->allocator->Free(grad_sum_);
        grad_sum_ = nullptr;
      }
    }
    return RET_OK;
  }

  int ExecuteVirtualBatch(int task_id) {
    auto gradient = reinterpret_cast<float *>(in_tensors_.at(grad_idx_)->MutableData());
    int length = in_tensors_.at(grad_idx_)->ElementsNum();

    int stride = UP_DIV(length, ms_context_->thread_num_);
    int count = MSMIN(stride, length - stride * task_id);
    int start = stride * task_id;
    int end = start + count;
    for (int i = start; i < end; ++i) {
      grad_sum_[i] += gradient[i];
    }
    valid_grad_sum_ = true;
    return RET_OK;
  }

  virtual int OptimizerStep() {
    valid_grad_sum_ = false;
    return RET_OK;
  }

  int Eval() override {
    if (weight_update_mod_ != WeightUpdateMode::ACCUMULATE_GRADS) {
      OptimizerStep();
    }
    return InnerKernel::Eval();
  }

  int PreProcess() override {
    auto ret = InnerKernel::PreProcess();
    if (ret != RET_OK) {
      return ret;
    }

    auto ctx = static_cast<const lite::InnerContext *>(this->ms_context_);
    if (ctx->IsCpuFloat16Enabled()) {
      auto t = in_tensors_.at(grad_idx_);
      auto gradient = reinterpret_cast<float *>(t->data());
      int length = in_tensors_.at(grad_idx_)->ElementsNum();

      for (int i = 0; i < length; ++i) {
        if (MS_ISNAN(gradient[i]) || std::isinf(gradient[i])) {
          MS_LOG(INFO) << "optimizer grad is nan or inf";
          return RET_OUT_OF_TENSOR_RANGE;
        }
      }

      auto is_scale = t->IsScale();
      auto scale = t->get_scale();
      if (is_scale) {
        t->set_scale(1.0f / scale);
        for (int i = 0; i < length; ++i) {
          gradient[i] *= (1.0f / scale);
        }
      }
    }
    return RET_OK;
  }
  int set_grad_sum_valid() {
    valid_grad_sum_ = true;
    return RET_OK;
  }

 protected:
  float default_lr_ = 0.0f;
  float lr_ = 0.0f;
  int lr_idx_ = 0;
  int grad_idx_ = 0;
  float *grad_sum_ = nullptr;
  bool valid_grad_sum_ = false;

 private:
  WeightUpdateMode weight_update_mod_ = WeightUpdateMode::NORMAL;
};

}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_TRAIN_OPTIMIZER_KERNEL_H_
