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

#include "src/runtime/kernel/arm/fp32_grad/pooling_grad.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "nnacl/fp32/pooling.h"
#include "nnacl/fp32_grad/pooling_grad.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_PoolingGrad;

namespace mindspore::kernel {
int PoolingGradCPUKernel::Init() {
  PoolingParameter *pool_param = reinterpret_cast<PoolingParameter *>(opParameter);

  auto in_shape = inputs_.at(0)->shape();
  int input_h = in_shape.at(1);
  int input_w = in_shape.at(2);

  if (pool_param->global_) {
    pool_param->window_w_ = input_w;
    pool_param->window_h_ = input_h;
  }

  // Emir -- here I assume we get the outputshape in the output tensor
  auto *out_tensor = outputs_.front();
  auto out_shape = out_tensor->shape();

  out_tensor->set_shape(out_shape);
  out_tensor->set_data_type(inputs_.at(0)->data_type());
  return RET_OK;
}

int PoolingGradCPUKernel::ReSize() { return RET_OK; }

int PoolingGradCPUKernel::Run() {
  PoolingParameter *pool_param = reinterpret_cast<PoolingParameter *>(opParameter);
  auto input_ptr = reinterpret_cast<float *>(inputs_.at(0)->Data());
  auto output_ptr = reinterpret_cast<float *>(outputs_.at(0)->Data());

  if (pool_param->pool_mode_ == PoolMode_MaxPool) {
    auto ind = reinterpret_cast<int *>(inputs_.at(1)->Data());
    MaxPoolingGrad(input_ptr, ind, output_ptr, pool_param);
  } else {
    AvgPoolingGrad(input_ptr, output_ptr, pool_param);
  }
  return RET_OK;
}

kernel::LiteKernel *CpuPoolingGradFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                    const std::vector<lite::tensor::Tensor *> &outputs,
                                                    OpParameter *opParameter, const lite::Context *ctx,
                                                    const kernel::KernelKey &desc,
                                                    const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_PoolingGrad);

  auto *kernel = new (std::nothrow) PoolingGradCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new PoolingGradCPUKernel fail!";
    return nullptr;
  }

  auto ret = kernel->Init();
  if (RET_OK != ret) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_PoolingGrad, CpuPoolingGradFp32KernelCreator)
}  // namespace mindspore::kernel
