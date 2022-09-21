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

#include "src/litert/kernel/cpu/fp32_grad/sigmoid_cross_entropy_with_logits_grad.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_SigmoidCrossEntropyWithLogitsGrad;

namespace mindspore::kernel {
int SigmoidCrossEntropyWithLogitsGradCPUKernel::ReSize() {
  CHECK_NULL_RETURN(op_parameter_);
  CHECK_LESS_RETURN(in_tensors_.size(), 3);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_.at(0));
  CHECK_NULL_RETURN(in_tensors_.at(1));
  CHECK_NULL_RETURN(in_tensors_.at(2));
  CHECK_NULL_RETURN(out_tensors_.at(0));
  return RET_OK;
}

int SigmoidCrossEntropyWithLogitsGradCPUKernel::DoExecute(int task_id) {
  auto logits = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());
  CHECK_NULL_RETURN(logits);
  auto labels = reinterpret_cast<float *>(in_tensors_.at(1)->MutableData());
  CHECK_NULL_RETURN(labels);
  auto dloss = reinterpret_cast<float *>(in_tensors_.at(2)->MutableData());
  CHECK_NULL_RETURN(dloss);
  auto *out = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());
  CHECK_NULL_RETURN(out);
  const float zero = 0.0f;
  const float one = 1.0f;

  size_t tensor_len = in_tensors_.at(0)->ElementsNum();
  for (uint64_t i = 0; i < tensor_len; ++i) {
    if (logits[i] >= zero) {
      out[i] = (one / (one + expf(-logits[i])) - labels[i]) * dloss[i];
    } else {
      const float exp_val = expf(logits[i]);
      out[i] = (exp_val / (one + exp_val) - labels[i]) * dloss[i];
    }
  }
  return RET_OK;
}

int SigmoidCrossEntropyWithLogitsGradRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  CHECK_NULL_RETURN(cdata);
  auto sig_crs_ent_kernel = reinterpret_cast<SigmoidCrossEntropyWithLogitsGradCPUKernel *>(cdata);
  CHECK_NULL_RETURN(sig_crs_ent_kernel);
  auto error_code = sig_crs_ent_kernel->DoExecute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "SigmoidCrossEntropyWithLogitsGrad error task_id[" << task_id << "] error_code[" << error_code
                  << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SigmoidCrossEntropyWithLogitsGradCPUKernel::Run() {
  int error_code = ParallelLaunch(this->ms_context_, SigmoidCrossEntropyWithLogitsGradRun, this, 1);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "SigmoidCrossEntropyWithLogitsGrad function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SigmoidCrossEntropyWithLogitsGradCPUKernel::Prepare() { return RET_OK; }

kernel::LiteKernel *CpuSigmoidCrossEntropyWithLogitsGradFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                                          const std::vector<lite::Tensor *> &outputs,
                                                                          OpParameter *opParameter,
                                                                          const lite::InnerContext *ctx,
                                                                          const kernel::KernelKey &desc) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_SigmoidCrossEntropyWithLogitsGrad);
  auto *kernel = new (std::nothrow) SigmoidCrossEntropyWithLogitsGradCPUKernel(opParameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new SigmoidCrossEntropyWithLogitsGradWithLogitsCPUKernel failed";
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SigmoidCrossEntropyWithLogitsGrad,
           CpuSigmoidCrossEntropyWithLogitsGradFp32KernelCreator)
}  // namespace mindspore::kernel
