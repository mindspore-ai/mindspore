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

#include "src/runtime/kernel/arm/fp32/power_fp32.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_PowFusion;

namespace mindspore::kernel {
int PowerCPUKernel::Init() { return RET_OK; }

int PowerCPUKernel::ReSize() { return RET_OK; }

int PowerImpl(void *cdata, int task_id) {
  auto kernel = reinterpret_cast<PowerCPUKernel *>(cdata);
  auto ret = kernel->RunImpl(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PowerImpl error: " << ret;
    return ret;
  }
  return RET_OK;
}

int PowerCPUKernel::Run() {
  auto ret = ParallelLaunch(this->context_->thread_pool_, PowerImpl, this, thread_count_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PowerCPUKernel error: " << ret;
    return RET_ERROR;
  }
  return RET_OK;
}

int PowerCPUKernel::RunImpl(int task_id) {
  auto x_addr = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());
  MS_ASSERT(x_addr);
  auto output_addr = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());
  MS_ASSERT(output_addr);
  auto size = in_tensors_.at(0)->ElementsNum();
  int stride = UP_DIV(size, thread_count_);
  int len = MSMIN(stride, size - stride * task_id);
  if (len <= 0) {
    return RET_OK;
  }
  float *exp_addr = nullptr;
  bool broadcast = true;
  MS_ASSERT(in_tensors_.size() == 2);
  exp_addr = reinterpret_cast<float *>(in_tensors_[1]->data_c());
  MS_ASSERT(exp_addr != nullptr);
  broadcast = in_tensors_[0]->shape() == in_tensors_[1]->shape() ? false : true;

  float *cur_exp = nullptr;
  if (broadcast) {
    cur_exp = exp_addr;
  } else {
    cur_exp = exp_addr + stride * task_id;
  }
  Power(x_addr + stride * task_id, cur_exp, output_addr + stride * task_id, len, scale_, shift_, broadcast);
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_PowFusion, LiteKernelCreator<PowerCPUKernel>)
}  // namespace mindspore::kernel
