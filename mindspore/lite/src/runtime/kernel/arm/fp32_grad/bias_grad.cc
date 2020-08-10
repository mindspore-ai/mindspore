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

#include <vector>
#include "src/runtime/kernel/arm/fp32_grad/bias_grad.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_BiasGrad;

namespace mindspore::kernel {
int BiasGradCPUKernel::InferShape() {
  if (1 != this->inputs_.size()) {
    MS_LOG(ERROR) << "BiasGrad should have one input";
    return RET_ERROR;
  }
  if (1 != this->outputs_.size()) {
    MS_LOG(ERROR) << "BiasGrad should have one output";
    return RET_ERROR;
  }
  auto *in0 = inputs_.front();
  auto *out = outputs_.front();
  MS_ASSERT(in0 != nullptr);
  MS_ASSERT(out != nullptr);
  auto inshape = in0->shape();
  int ndim = inshape.size();
  for (int i = 0; i < ndim - 1; i++) {
    inshape[i] = 1;
  }
  out->set_shape(inshape);
  out->set_data_type(in0->data_type());
  return RET_OK;
}

int BiasGradCPUKernel::Init() {
  MS_ASSERT(InferShape() == RET_OK);

  auto dims = inputs_[0]->shape();
  bias_param->ndim_ = dims.size();
  for (unsigned int i = 0; i < bias_param->ndim_; i++) {
    bias_param->in_shape0_[i] = dims[i];
    bias_param->out_shape_[i] = 1;  // 1 dimension for N,H,W,
  }
  bias_param->out_shape_[bias_param->ndim_ - 1] = dims[bias_param->ndim_ - 1];
  for (int i = bias_param->ndim_; i < 4; i++) {
    bias_param->in_shape0_[i] = 0;
    bias_param->out_shape_[i] = 0;
  }
  return RET_OK;
}

int BiasGradCPUKernel::ReSize() { return 0; }

int BiasGradCPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare failed.";
    return RET_ERROR;
  }
  auto in = reinterpret_cast<float *>(inputs_.at(0)->Data());
  auto out = reinterpret_cast<float *>(outputs_.at(0)->Data());
  // size_t data_size = inputs_.at(0)->ElementsNum();

  size_t nhw_size = 1;
  size_t channels = bias_param->in_shape0_[bias_param->ndim_ - 1];  // C in NHWC
  for (unsigned int i = 0; i < bias_param->ndim_ - 1; i++) nhw_size *= bias_param->in_shape0_[i];

  size_t total_size = channels * nhw_size;
  for (size_t c = 0; c < channels; ++c) {
    out[c] = 0;
    for (size_t offset = 0; offset < total_size; offset += channels) {
      out[c] += in[offset + c];
    }
  }

  return RET_OK;
}

kernel::LiteKernel *CpuBiasGradFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                 const std::vector<lite::tensor::Tensor *> &outputs,
                                                 OpParameter *opParameter, const lite::Context *ctx,
                                                 const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_BiasGrad);
  auto *kernel =
    new (std::nothrow) BiasGradCPUKernel(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs, ctx, primitive);
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_BiasGrad, CpuBiasGradFp32KernelCreator)
}  // namespace mindspore::kernel
