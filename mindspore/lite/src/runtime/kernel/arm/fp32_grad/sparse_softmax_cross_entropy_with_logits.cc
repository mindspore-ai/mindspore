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

#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/nnacl/softmax_parameter.h"
#include "src/runtime/kernel/arm/nnacl/fp32/softmax.h"
#include "src/runtime/kernel/arm/fp32_grad/sparse_softmax_cross_entropy_with_logits.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_SoftmaxCrossEntropy;

namespace mindspore::kernel {

int SparseSoftmaxCrossEntropyWithLogitsCPUKernel::ReSize() { return RET_OK; }

void SparseSoftmaxCrossEntropyWithLogitsCPUKernel::ForwardPostExecute(const int *labels, const float *losses,
                                                                      float *output) const {
  float total_loss = 0;
  for (int i = 0; i < param->batch_size_; ++i) {
    if (labels[i] < 0) {
      MS_LOG(EXCEPTION) << "label value must >= 0";
    }
    size_t label = labels[i];
    if (label > param->number_of_classes_) {
      MS_LOG(EXCEPTION) << "error label input!";
    } else {
      total_loss -= logf(losses[i * param->number_of_classes_ + label]);
    }
  }
  output[0] = total_loss / param->batch_size_;
}

void SparseSoftmaxCrossEntropyWithLogitsCPUKernel::GradPostExecute(const int *labels, const float *losses,
                                                                   float *output) const {
  size_t row_start = 0;
  for (int i = 0; i < param->batch_size_; ++i) {
    if (labels[i] < 0) {
      MS_LOG(EXCEPTION) << "label value must >= 0";
    }
    size_t label = labels[i];
    if (label > param->number_of_classes_) {
      MS_LOG(EXCEPTION) << "error label input!";
    }
    for (size_t j = 0; j < param->number_of_classes_; ++j) {
      size_t index = row_start + j;
      if (j == label) {
        output[index] = (losses[index] - 1) / param->batch_size_;
      } else {
        output[index] = losses[index] / param->batch_size_;
      }
    }
    row_start += param->number_of_classes_;
  }
}

int SparseSoftmaxCrossEntropyWithLogitsCPUKernel::Run() {
  auto ins = reinterpret_cast<float *>(inputs_.at(0)->Data());
  auto labels = reinterpret_cast<int *>(inputs_.at(1)->Data());
  auto out = reinterpret_cast<float *>(outputs_.at(1)->Data());
  float *grads = NULL;
  if (is_train()) {  // outputs_.size() > 1)
    grads = reinterpret_cast<float *>(outputs_.at(0)->Data());
  }
  size_t data_size = inputs_.at(0)->ElementsNum();
  float *losses = new (std::nothrow) float[data_size];
  MS_ASSERT(losses != nullptr);
  std::fill(losses, losses + data_size, 0);

  MS_ASSERT(out != nullptr);
  MS_ASSERT(labels != nullptr);
  MS_ASSERT(ins != nullptr);

  SoftmaxParameter sm_params;
  sm_params.n_dim_ = param->n_dim_;
  sm_params.element_size_ = data_size;
  sm_params.axis_ = 0;
  for (int i = 0; i < 4; i++)  // softmax has only 4 params in shape
    sm_params.input_shape_[i] = param->input_shape_[i];
  float sum_data[sm_params.input_shape_[sm_params.axis_]] = {0};
  std::fill(sum_data, sum_data + sm_params.input_shape_[sm_params.axis_], 0);
  Softmax(ins, losses, sum_data, &sm_params);

  if (is_train()) {
    GradPostExecute(labels, losses, grads);
  } else {
    ForwardPostExecute(labels, losses, out);
  }
  return RET_OK;
}

int SparseSoftmaxCrossEntropyWithLogitsCPUKernel::Init() {
  if (context_->infer_shape_interrupt_ && !context_->running_) {
    SetNeedReInit();
    return RET_OK;
  }
  auto dims = inputs_[0]->shape();
  param->n_dim_ = 2;
  param->number_of_classes_ = dims[1];
  param->batch_size_ = dims[0];
  for (unsigned int i = 0; i < dims.size(); i++) param->input_shape_[i] = dims[i];
  if (2 != this->inputs_.size()) {
    MS_LOG(ERROR) << "softmax entropy loss should have two inputs";
    return RET_ERROR;
  }
  auto *in0 = inputs_.front();
  if (in0 == nullptr) {
    MS_LOG(ERROR) << "softmax etropy loss in0 have no data";
    return RET_ERROR;
  }

  return RET_OK;
}

kernel::LiteKernel *CpuSoftmaxCrossEntropyFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                            const std::vector<lite::tensor::Tensor *> &outputs,
                                                            OpParameter *opParameter, const lite::Context *ctx,
                                                            const kernel::KernelKey &desc,
                                                            const lite::Primitive *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_SoftmaxCrossEntropy);
  auto *kernel =
    new (std::nothrow) SparseSoftmaxCrossEntropyWithLogitsCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  MS_ASSERT(kernel != nullptr);
  auto ret = kernel->Init();
  if (RET_OK != ret) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SoftmaxCrossEntropy, CpuSoftmaxCrossEntropyFp32KernelCreator)
}  // namespace mindspore::kernel
