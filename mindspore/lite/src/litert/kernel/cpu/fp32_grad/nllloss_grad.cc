/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/fp32_grad/nllloss_grad.h"

#include <vector>

#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/fp32_grad/nllloss_grad_fp32.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_NLLLossGrad;

namespace mindspore::kernel {
namespace {
constexpr size_t kLogitsIndex = 0;
constexpr size_t kLossGradIndex = 1;
constexpr size_t kLabelsIndex = 2;
constexpr size_t kWeightsIndex = 3;
constexpr size_t kTotalWeightIndex = 4;
constexpr size_t kDxIndex = 0;
}  // namespace

int NLLLossGradCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C5NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), C1NUM);
  for (size_t i = 0; i < C5NUM; i++) {
    CHECK_NULL_RETURN(in_tensors_[i]);
  }
  CHECK_NULL_RETURN(out_tensors_[kDxIndex]);

  const auto logits_shape = in_tensors_[kLogitsIndex]->shape();
  batch_ = logits_shape[0];
  class_num_ = logits_shape[1];
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int NLLLossGradCPUKernel::ReSize() { return RET_OK; }

int NLLLossGradCPUKernel::Run() {
  const auto *logits = reinterpret_cast<float *>(in_tensors_[kLogitsIndex]->data());
  const auto *loss_grad = reinterpret_cast<float *>(in_tensors_[kLossGradIndex]->data());
  const auto *labels = reinterpret_cast<int *>(in_tensors_[kLabelsIndex]->data());
  const auto *weight = reinterpret_cast<float *>(in_tensors_[kWeightsIndex]->data());
  const auto *total_weight = reinterpret_cast<float *>(in_tensors_[kTotalWeightIndex]->data());
  auto *logits_grad = reinterpret_cast<float *>(out_tensors_[kDxIndex]->data());
  CHECK_NULL_RETURN(logits);
  CHECK_NULL_RETURN(loss_grad);
  CHECK_NULL_RETURN(labels);
  CHECK_NULL_RETURN(weight);
  CHECK_NULL_RETURN(total_weight);
  CHECK_NULL_RETURN(logits_grad);

  int ret = NLLLossGrad(logits, loss_grad, labels, weight, total_weight, logits_grad, batch_, class_num_,
                        nllloss_param_->reduction_type_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "NLLLossGrad Run error: error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_NLLLossGrad, LiteKernelCreator<NLLLossGradCPUKernel>)
}  // namespace mindspore::kernel
