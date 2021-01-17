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
#include "nnacl/fp32/pooling_fp32.h"
#include "nnacl/fp32_grad/pooling_grad.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_PoolingGrad;

namespace mindspore::kernel {
int PoolingGradCPUKernel::ReSize() {
  PoolingParameter *pool_param = reinterpret_cast<PoolingParameter *>(op_parameter_);

  auto in_shape = in_tensors_.at(0)->shape();
  auto out_shape = in_tensors_.at(1)->shape();

  if (pool_param->pool_mode_ == PoolMode_AvgPool) {
    out_shape = in_tensors_.at(2)->shape();
  }

  int input_h = in_shape.at(1);
  int input_w = in_shape.at(2);

  if (pool_param->global_) {
    pool_param->window_w_ = input_w;
    pool_param->window_h_ = input_h;
  }

  pool_param->input_h_ = in_shape[kNHWC_H];
  pool_param->input_w_ = in_shape[kNHWC_W];
  pool_param->input_batch_ = in_shape[kNHWC_N];
  pool_param->input_channel_ = in_shape[kNHWC_C];
  pool_param->output_h_ = out_shape[kNHWC_H];
  pool_param->output_w_ = out_shape[kNHWC_W];
  pool_param->output_batch_ = out_shape[kNHWC_N];
  pool_param->output_channel_ = out_shape[kNHWC_C];

  return RET_OK;
}

int PoolingGradCPUKernel::Init() { return ReSize(); }

int PoolingGradCPUKernel::Execute(int task_id) {
  PoolingParameter *pool_param = reinterpret_cast<PoolingParameter *>(op_parameter_);
  auto input_ptr = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());
  auto output_ptr = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());

  if (pool_param->pool_mode_ == PoolMode_MaxPool) {
    auto dx_ptr = reinterpret_cast<float *>(in_tensors_.at(1)->MutableData());
    auto dy_ptr = reinterpret_cast<float *>(in_tensors_.at(2)->MutableData());
    MaxPoolingGrad(input_ptr, dx_ptr, dy_ptr, output_ptr, pool_param, task_id);
  } else {
    input_ptr = reinterpret_cast<float *>(in_tensors_.at(2)->MutableData());
    AvgPoolingGrad(input_ptr, output_ptr, pool_param, task_id);
  }
  return RET_OK;
}

int PoolingGradImpl(void *cdata, int task_id) {
  MS_ASSERT(cdata != nullptr);
  auto pooling = reinterpret_cast<PoolingGradCPUKernel *>(cdata);
  auto error_code = pooling->Execute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Pooling Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int PoolingGradCPUKernel::Run() {
  int error_code = ParallelLaunch(this->context_->thread_pool_, PoolingGradImpl, this, 1);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "pooling error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuPoolingGradFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                    const std::vector<lite::Tensor *> &outputs,
                                                    OpParameter *opParameter, const lite::InnerContext *ctx,
                                                    const kernel::KernelKey &desc,
                                                    const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_PoolingGrad);

  auto *kernel = new (std::nothrow) PoolingGradCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new PoolingGradCPUKernel fail!";
    free(opParameter);
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
