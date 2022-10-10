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

#include "src/runtime/kernel/cpu/fp32_grad/binary_cross_entropy_grad.h"
#include "src/runtime/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/fp32_grad/binary_cross_entropy_grad.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_BinaryCrossEntropyGrad;

namespace mindspore::kernel {
int BinaryCrossEntropyGradCPUKernel::ReSize() {
  CHECK_LESS_RETURN(in_tensors_.size(), C3NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), C1NUM);
  CHECK_NULL_RETURN(in_tensors_.at(C0NUM));
  CHECK_NULL_RETURN(in_tensors_.at(C1NUM));
  CHECK_NULL_RETURN(in_tensors_.at(C2NUM));
  if (in_tensors_.size() == C4NUM) {
    weight_defined_ = true;
    CHECK_NULL_RETURN(in_tensors_.at(C3NUM));
  }
  CHECK_NULL_RETURN(out_tensors_.at(0));
  CHECK_NULL_RETURN(op_parameter_);
  auto param_ = reinterpret_cast<BinaryCrossEntropyGradParameter *>(op_parameter_);
  CHECK_NULL_RETURN(param_);

  return RET_OK;
}

int BinaryCrossEntropyGradCPUKernel::DoExecute(int task_id) {
  auto input_x = reinterpret_cast<float *>(in_tensors_.at(C0NUM)->MutableData());
  CHECK_NULL_RETURN(input_x);
  auto input_y = reinterpret_cast<float *>(in_tensors_.at(C1NUM)->MutableData());
  CHECK_NULL_RETURN(input_y);
  auto dloss = reinterpret_cast<float *>(in_tensors_.at(C2NUM)->MutableData());
  CHECK_NULL_RETURN(dloss);
  if (weight_defined_) {
    weight_ = reinterpret_cast<float *>(in_tensors_.at(C3NUM)->MutableData());
    CHECK_NULL_RETURN(weight_);
  }
  auto *out = reinterpret_cast<float *>(out_tensors_.at(C0NUM)->MutableData());
  CHECK_NULL_RETURN(out);

  auto param_ = reinterpret_cast<BinaryCrossEntropyGradParameter *>(op_parameter_);
  int reduction = param_->reduction;
  size_t input_size = in_tensors_.at(0)->ElementsNum();
  BinaryCrossEntropyGrad(input_size, reduction, input_x, input_y, weight_, dloss, out, weight_defined_);
  return RET_OK;
}

int BinaryCrossEntropyGradRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  CHECK_NULL_RETURN(cdata);
  auto bin_crs_ent_kernel = reinterpret_cast<BinaryCrossEntropyGradCPUKernel *>(cdata);
  auto error_code = bin_crs_ent_kernel->DoExecute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "BinaryCrossEntropyGrad error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int BinaryCrossEntropyGradCPUKernel::Run() {
  int error_code = ParallelLaunch(this->ms_context_, BinaryCrossEntropyGradRun, this, 1);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "BinaryCrossEntropyGrad function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int BinaryCrossEntropyGradCPUKernel::Prepare() { return ReSize(); }

kernel::LiteKernel *CpuBinaryCrossEntropyGradFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                               const std::vector<lite::Tensor *> &outputs,
                                                               OpParameter *opParameter, const lite::Context *ctx,
                                                               const kernel::KernelKey &desc) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_BinaryCrossEntropyGrad);
  auto *kernel = new (std::nothrow)
    BinaryCrossEntropyGradCPUKernel(opParameter, inputs, outputs, static_cast<const lite::InnerContext *>(ctx));
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new BinaryCrossEntropyGrad failed";
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_BinaryCrossEntropyGrad, CpuBinaryCrossEntropyGradFp32KernelCreator)
}  // namespace mindspore::kernel
