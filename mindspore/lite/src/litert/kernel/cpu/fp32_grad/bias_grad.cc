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
#include "src/litert/kernel/cpu/fp32_grad/bias_grad.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_BiasAddGrad;

namespace mindspore::kernel {
int BiasGradCPUKernel::ReSize() {
  auto dims = in_tensors_[0]->shape();
  bias_param->ndim_ = dims.size();
  for (unsigned int i = 0; i < bias_param->ndim_; i++) {
    bias_param->in_shape0_[i] = dims[i];
    bias_param->out_shape_[i] = 1;  // 1 dimension for N,H,W,
  }
  bias_param->out_shape_[bias_param->ndim_ - 1] = dims[bias_param->ndim_ - 1];
  for (auto i = bias_param->ndim_; i < DIMENSION_4D; i++) {
    bias_param->in_shape0_[i] = 0;
    bias_param->out_shape_[i] = 0;
  }
  return RET_OK;
}

int BiasGradCPUKernel::Prepare() {
  CHECK_NULL_RETURN(bias_param);
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_[0]);
  CHECK_NULL_RETURN(out_tensors_[0]);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int BiasGradCPUKernel::DoExecute(int task_id) {
  auto in = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());
  auto out = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());
  CHECK_NULL_RETURN(in);
  CHECK_NULL_RETURN(out);
  size_t nhw_size = 1;
  size_t channels = bias_param->in_shape0_[bias_param->ndim_ - 1];  // C in NHWC
  for (unsigned int i = 0; i < bias_param->ndim_ - 1; i++) {
    nhw_size *= static_cast<size_t>(bias_param->in_shape0_[i]);
  }

  size_t total_size = channels * nhw_size;
  for (size_t c = 0; c < channels; ++c) {
    float tmp = 0.0f;
    for (size_t offset = 0; offset < total_size; offset += channels) {
      tmp += in[offset + c];
    }
    out[c] = tmp;
  }

  return RET_OK;
}

int BiasGradRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  CHECK_NULL_RETURN(cdata);
  auto bias_kernel = reinterpret_cast<BiasGradCPUKernel *>(cdata);
  auto error_code = bias_kernel->DoExecute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "bias error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int BiasGradCPUKernel::Run() {
  int error_code = ParallelLaunch(this->ms_context_, BiasGradRun, this, 1);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "bias function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuBiasGradFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                 const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                                 const lite::InnerContext *ctx, const kernel::KernelKey &desc) {
  MS_ASSERT(desc.type == schema::PrimitiveType_BiasAddGrad);
  auto *kernel = new (std::nothrow) BiasGradCPUKernel(opParameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new BiasGradCPUKernel fail!";
    free(opParameter);
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_BiasAddGrad, CpuBiasGradFp32KernelCreator)
}  // namespace mindspore::kernel
