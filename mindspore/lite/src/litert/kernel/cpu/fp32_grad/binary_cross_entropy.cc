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

#include "src/litert/kernel/cpu/fp32_grad/binary_cross_entropy.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/fp32_grad/binary_cross_entropy.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_BinaryCrossEntropy;

namespace mindspore::kernel {
BinaryCrossEntropyCPUKernel::~BinaryCrossEntropyCPUKernel() {
  if (tmp_loss_ != nullptr) {
    free(tmp_loss_);
    tmp_loss_ = nullptr;
  }
}

int BinaryCrossEntropyCPUKernel::ReSize() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_.at(0));
  CHECK_NULL_RETURN(in_tensors_.at(1));
  if (in_tensors_.size() == C3NUM) {
    weight_defined_ = true;
    CHECK_NULL_RETURN(in_tensors_.at(C2NUM));
  }
  CHECK_NULL_RETURN(out_tensors_.at(0));
  CHECK_NULL_RETURN(op_parameter_);

  auto param_ = reinterpret_cast<BinaryCrossEntropyParameter *>(op_parameter_);
  CHECK_NULL_RETURN(param_);
  if (tmp_loss_ != nullptr) {
    free(tmp_loss_);
    tmp_loss_ = nullptr;
  }
  size_t input_size = in_tensors_.at(0)->ElementsNum();
  tmp_loss_ = reinterpret_cast<float *>(malloc(input_size * sizeof(float)));
  if (tmp_loss_ == nullptr) {
    MS_LOG(ERROR) << "malloc tmp_loss_ for BinaryCrossEntropy op failed";
    return RET_ERROR;
  }

  return RET_OK;
}

int BinaryCrossEntropyCPUKernel::DoExecute(int task_id) {
  auto logits = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());
  CHECK_NULL_RETURN(logits);
  auto labels = reinterpret_cast<float *>(in_tensors_.at(1)->MutableData());
  CHECK_NULL_RETURN(labels);
  auto *out = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());
  CHECK_NULL_RETURN(out);

  auto param_ = reinterpret_cast<BinaryCrossEntropyParameter *>(op_parameter_);
  int reduction = param_->reduction;
  size_t input_size = in_tensors_.at(0)->ElementsNum();
  if (weight_defined_) {
    weight_ = reinterpret_cast<float *>(in_tensors_.at(C2NUM)->MutableData());
    CHECK_NULL_RETURN(weight_);
  }
  BinaryCrossEntropy(input_size, reduction, logits, labels, weight_, out, tmp_loss_, weight_defined_);
  return RET_OK;
}

int BinaryCrossEntropyRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  CHECK_NULL_RETURN(cdata);
  auto bin_crs_ent_kernel = reinterpret_cast<BinaryCrossEntropyCPUKernel *>(cdata);
  auto error_code = bin_crs_ent_kernel->DoExecute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "BinaryCrossEntropy error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int BinaryCrossEntropyCPUKernel::Run() {
  int error_code = ParallelLaunch(this->ms_context_, BinaryCrossEntropyRun, this, 1);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "SigmoidCrossEntropyWithLogits function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int BinaryCrossEntropyCPUKernel::Prepare() { return ReSize(); }

kernel::LiteKernel *CpuBinaryCrossEntropyFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                           const std::vector<lite::Tensor *> &outputs,
                                                           OpParameter *opParameter, const lite::InnerContext *ctx,
                                                           const kernel::KernelKey &desc) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_BinaryCrossEntropy);
  auto *kernel = new (std::nothrow) BinaryCrossEntropyCPUKernel(opParameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new SigmoidCrossEntropyWithLogits failed";
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_BinaryCrossEntropy, CpuBinaryCrossEntropyFp32KernelCreator)
}  // namespace mindspore::kernel
