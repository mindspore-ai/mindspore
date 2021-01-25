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

#include "src/runtime/kernel/arm/fp32_grad/softmax_grad.h"
#include <string.h>
#include <vector>
#include "nnacl/fp32_grad/softmax_grad.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int SoftmaxGradCPUKernel::Init() {
  param = reinterpret_cast<SoftmaxParameter *>(op_parameter_);
  auto in_shape = in_tensors_.at(0)->shape();
  auto in_dims = in_shape.size();
  int ele_size = 1;
  param->n_dim_ = in_dims;
  for (size_t i = 0; i < in_dims; i++) {
    param->input_shape_[i] = in_shape.at(i);
    ele_size *= in_shape.at(i);
  }
  param->element_size_ = ele_size;

  auto axis = param->axis_;
  if ((axis < -1) || (axis > param->n_dim_)) {
    MS_LOG(ERROR) << "SoftmaxGrad axis is invalid!";
    return RET_ERROR;
  } else if (axis == -1) {
    axis = param->axis_ = (in_dims - 1);
  }

  inner_size_ = 1;
  for (size_t i = axis + 1; i < in_dims; i++) {
    inner_size_ *= in_shape.at(i);
  }
  set_workspace_size(inner_size_ * (1 + in_shape.at(axis)) * sizeof(float));
  return RET_OK;
}

int SoftmaxGradCPUKernel::ReSize() { return RET_OK; }

int SoftmaxGradCPUKernel::Execute(int task_id) {
  auto input_ptr = reinterpret_cast<float *>(in_tensors_.at(kInputIndex)->MutableData());
  auto yt_ptr = reinterpret_cast<float *>(in_tensors_.at(1)->MutableData());
  auto output_ptr = reinterpret_cast<float *>(out_tensors_.at(kOutputIndex)->MutableData());
  float *sum_data_ = static_cast<float *>(workspace());
  float *sum_mul_ = sum_data_ + inner_size_;
  SoftmaxGrad(input_ptr, yt_ptr, output_ptr, sum_data_, sum_mul_, reinterpret_cast<SoftmaxParameter *>(op_parameter_));

  return RET_OK;
}

int SoftmaxGradRun(void *cdata, int task_id) {
  auto softmax_kernel = reinterpret_cast<SoftmaxGradCPUKernel *>(cdata);
  auto error_code = softmax_kernel->Execute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "softmax_kernel SoftmaxGradRun task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SoftmaxGradCPUKernel::Run() {
  int error_code = ParallelLaunch(this->context_->thread_pool_, SoftmaxGradRun, this, 1);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "SoftmaxGradRun function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuSoftmaxGradFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                    const std::vector<lite::Tensor *> &outputs,
                                                    OpParameter *opParameter, const lite::InnerContext *ctx,
                                                    const kernel::KernelKey &desc) {
  MS_ASSERT(opParameter != nullptr);
  auto *kernel = new (std::nothrow) SoftmaxGradCPUKernel(opParameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new SoftmaxGradCPUKernel fail!";
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
}  // namespace mindspore::kernel
