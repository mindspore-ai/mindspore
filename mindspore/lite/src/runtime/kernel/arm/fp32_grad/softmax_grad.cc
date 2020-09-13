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
#include "include/errorcode.h"

// using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
// using mindspore::schema::PrimitiveType_SoftMaxGrad;

namespace mindspore::kernel {
int SoftmaxGradCPUKernel::Init() {
  // auto input_tensor =in_tensors_.at(0);

  param = reinterpret_cast<SoftmaxParameter *>(op_parameter_);
  auto in_shape = in_tensors_.at(0)->shape();
  auto in_dims = in_shape.size();
  int ele_size = 1;
  param->n_dim_ = in_dims;
  for (size_t i = 0; i < in_dims; i++) {
    param->input_shape_[i] = in_shape[i];
    ele_size *= in_shape[i];
  }
  param->element_size_ = ele_size;

  // malloc tmp buffer
  auto axis = param->axis_;
  if ((axis < -1) || (axis > param->n_dim_)) {
    MS_LOG(ERROR) << "SoftmaxGrad axis is invalid!";
  } else if (axis == -1) {
    axis = param->axis_ = (in_dims - 1);
  }

  int inner_size = 1;
  for (size_t i = axis + 1; i < in_dims; i++) {
    inner_size *= in_shape[i];
  }

  sum_data_ = new (std::nothrow) float[inner_size];
  MS_ASSERT(sum_data_ != nullptr);
  sum_mul_ = new (std::nothrow) float[inner_size * in_shape[axis]];
  MS_ASSERT(sum_mul_ != nullptr);
  return RET_OK;
}

int SoftmaxGradCPUKernel::ReSize() { return RET_OK; }

int SoftmaxGradCPUKernel::Run() {
  // auto input_ptr = reinterpret_cast<float *>(in_tensors_.at(kInputIndex)->MutableData());
  auto input_ptr = reinterpret_cast<float *>(in_tensors_.at(kInputIndex)->MutableData());
  auto yt_ptr = reinterpret_cast<float *>(in_tensors_.at(1)->MutableData());
  auto output_ptr = reinterpret_cast<float *>(out_tensors_.at(kOutputIndex)->MutableData());
  SoftmaxGrad(input_ptr, yt_ptr, output_ptr, sum_data_, sum_mul_, reinterpret_cast<SoftmaxParameter *>(op_parameter_));
  return RET_OK;
}

kernel::LiteKernel *CpuSoftmaxGradFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                    const std::vector<lite::Tensor *> &outputs,
                                                    OpParameter *opParameter, const lite::Context *ctx,
                                                    const kernel::KernelKey &desc,
                                                    const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(opParameter != nullptr);
  // MS_ASSERT(desc.type == schema::PrimitiveType_SoftMaxGrad);
  auto *kernel = new (std::nothrow) SoftmaxGradCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new SoftmaxGradCPUKernel fail!";
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

// REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SoftMaxGrad, CpuSoftmaxGradFp32KernelCreator)
}  // namespace mindspore::kernel
