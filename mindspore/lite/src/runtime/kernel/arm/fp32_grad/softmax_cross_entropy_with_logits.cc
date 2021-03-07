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

#include <math.h>
#include "src/kernel_registry.h"
#include "nnacl/softmax_parameter.h"
#include "nnacl/fp32/softmax_fp32.h"
#include "src/runtime/kernel/arm/fp32_grad/softmax_cross_entropy_with_logits.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_SoftmaxCrossEntropyWithLogits;

namespace mindspore::kernel {

int SoftmaxCrossEntropyWithLogitsCPUKernel::Init() { return ReSize(); }

void SoftmaxCrossEntropyWithLogitsCPUKernel::ForwardPostExecute(const float *labels, const float *logits, float *grads,
                                                                float *output2) const {
  float eps = 1e-6;
  if (grads != nullptr) {
    for (int i = 0; i < param_->batch_size_; ++i) {
      float loss = 0.f;
      for (size_t j = 0; j < param_->number_of_classes_; ++j) {
        float logit =
          -logf(logits[i * param_->number_of_classes_ + j] <= 0.0 ? eps : logits[i * param_->number_of_classes_ + j]);
        grads[i * param_->number_of_classes_ + j] =
          (logits[i * param_->number_of_classes_ + j] - labels[i * param_->number_of_classes_ + j]);
        loss += labels[i * param_->number_of_classes_ + j] * logit;
      }
      output2[i] = loss;
    }
  } else {
    for (int i = 0; i < param_->batch_size_; ++i) {
      float loss = 0.f;
      for (size_t j = 0; j < param_->number_of_classes_; ++j) {
        float logit =
          -logf(logits[i * param_->number_of_classes_ + j] <= 0.0 ? eps : logits[i * param_->number_of_classes_ + j]);
        loss += labels[i * param_->number_of_classes_ + j] * logit;
      }
      output2[i] = loss;
    }
  }
}

int SoftmaxCrossEntropyWithLogitsCPUKernel::Execute(int task_id) {
  auto ins = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());
  auto labels = reinterpret_cast<float *>(in_tensors_.at(1)->MutableData());
  float *out = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());
  float *grads = nullptr;
  if (IsTrain() && out_tensors_.size() > 1) {
    grads = reinterpret_cast<float *>(out_tensors_.at(1)->MutableData());
  }
  size_t data_size = in_tensors_.at(0)->ElementsNum();

  MS_ASSERT(out != nullptr);
  MS_ASSERT(labels != nullptr);
  MS_ASSERT(ins != nullptr);
  float *losses_ = static_cast<float *>(workspace());
  float *sum_data_ = losses_ + data_size;
  std::fill(losses_, losses_ + data_size, 0);
  std::fill(sum_data_, sum_data_ + sm_params_.input_shape_[0], 0);
  Softmax(ins, losses_, sum_data_, &sm_params_);
  ForwardPostExecute(labels, losses_, grads, out);
  return RET_OK;
}

int SoftmaxCrossEntropyWithLogitsRun(void *cdata, int task_id) {
  auto softmax_kernel = reinterpret_cast<SoftmaxCrossEntropyWithLogitsCPUKernel *>(cdata);
  auto error_code = softmax_kernel->Execute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "SoftmaxCrossEntropy error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SoftmaxCrossEntropyWithLogitsCPUKernel::Run() {
  int error_code = ParallelLaunch(this->context_->thread_pool_, SoftmaxCrossEntropyWithLogitsRun, this, 1);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "SoftmaxCrossEntropy function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SoftmaxCrossEntropyWithLogitsCPUKernel::ReSize() {
  auto dims = in_tensors_.at(0)->shape();
  param_->n_dim_ = 2;
  param_->number_of_classes_ = dims.at(1);
  param_->batch_size_ = dims.at(0);
  for (unsigned int i = 0; i < dims.size(); i++) param_->input_shape_[i] = dims.at(i);
  if (this->in_tensors_.size() != 2) {
    MS_LOG(ERROR) << "softmax entropy loss should have two inputs";
    return RET_ERROR;
  }
  auto *in0 = in_tensors_.front();
  if (in0 == nullptr) {
    MS_LOG(ERROR) << "softmax etropy loss in0 have no data";
    return RET_ERROR;
  }

  size_t data_size = in_tensors_.at(0)->ElementsNum();
  set_workspace_size((data_size + dims.at(0)) * sizeof(float));
  sm_params_.n_dim_ = 2;
  sm_params_.element_size_ = data_size;
  sm_params_.axis_ = 1;
  for (size_t i = 0; i < dims.size(); i++) sm_params_.input_shape_[i] = dims.at(i);

  return RET_OK;
}

kernel::LiteKernel *CpuSoftmaxCrossEntropyFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                            const std::vector<lite::Tensor *> &outputs,
                                                            OpParameter *opParameter, const lite::InnerContext *ctx,
                                                            const kernel::KernelKey &desc) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_SoftmaxCrossEntropyWithLogits);
  auto *kernel = new (std::nothrow) SoftmaxCrossEntropyWithLogitsCPUKernel(opParameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new SoftmaxCrossEntropyWithLogitsCPUKernel failed";
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SoftmaxCrossEntropyWithLogits,
           CpuSoftmaxCrossEntropyFp32KernelCreator)
}  // namespace mindspore::kernel
