/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "src/litert/kernel/cpu/fp16_grad/bias_fp16_grad.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_BiasAddGrad;

namespace mindspore::kernel {
constexpr static int kMaxDim = 4;

int BiasGradCPUKernelFp16::ReSize() {
  CHECK_NULL_RETURN(in_tensors_[C0NUM]);
  CHECK_NULL_RETURN(out_tensors_[C0NUM]);
  auto dims = in_tensors_[C0NUM]->shape();
  bias_param->ndim_ = dims.size();
  for (unsigned int i = 0; i < bias_param->ndim_; i++) {
    bias_param->in_shape0_[i] = dims[i];
    bias_param->out_shape_[i] = C1NUM;  // 1 dimension for N,H,W,
  }
  bias_param->out_shape_[bias_param->ndim_ - C1NUM] = dims[bias_param->ndim_ - C1NUM];
  for (auto i = bias_param->ndim_; i < kMaxDim; i++) {
    bias_param->in_shape0_[i] = 0;
    bias_param->out_shape_[i] = 0;
  }
  return RET_OK;
}

int BiasGradCPUKernelFp16::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C1NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), C1NUM);
  CHECK_NULL_RETURN(bias_param);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int BiasGradCPUKernelFp16::DoExecute(int task_id) {
  auto in = reinterpret_cast<float16_t *>(in_tensors_.at(C0NUM)->data());
  auto out = reinterpret_cast<float16_t *>(out_tensors_.at(C0NUM)->data());
  CHECK_NULL_RETURN(in);
  CHECK_NULL_RETURN(out);
  size_t nhw_size = 1;
  size_t channels = bias_param->in_shape0_[bias_param->ndim_ - 1];  // C in NHWC
  for (unsigned int i = 0; i < bias_param->ndim_ - 1; i++) {
    nhw_size *= bias_param->in_shape0_[i];
  }

  size_t total_size = channels * nhw_size;
  for (size_t c = 0; c < channels; ++c) {
    float sum = 0;
    for (size_t offset = 0; offset < total_size; offset += channels) {
      sum += in[offset + c];
    }
    out[c] = (float16_t)sum;
  }

  return RET_OK;
}

int BiasGradFp16Run(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  CHECK_NULL_RETURN(cdata);
  auto bias_kernel = reinterpret_cast<BiasGradCPUKernelFp16 *>(cdata);
  auto error_code = bias_kernel->DoExecute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "bias error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int BiasGradCPUKernelFp16::Run() {
  int error_code = ParallelLaunch(this->ms_context_, BiasGradFp16Run, this, 1);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "bias function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_BiasAddGrad, LiteKernelCreator<BiasGradCPUKernelFp16>)
}  // namespace mindspore::kernel
