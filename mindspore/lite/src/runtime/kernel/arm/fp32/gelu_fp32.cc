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

#include "src/runtime/kernel/arm/fp32/gelu_fp32.h"
#include "src/runtime/kernel/arm/base/gelu_base.h"
#include "nnacl/fp32/gelu_fp32.h"
#include "nnacl/gelu_parameter.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
// using mindspore::schema::PrimitiveType_GeLU;

namespace mindspore::kernel {

int GeLUCPUKernel::Init() {
  auto ret = GeLUBaseCPUKernel::Init();
  if (ret != RET_OK) {
    return ret;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }

  return ReSize();
}

int GeLUCPUKernel::ReSize() { return RET_OK; }

int GeLUCPUKernel::GeLU(int task_id) {
  int64_t real_dst_count = MSMIN(elements_num_ - task_id * count_unit_, count_unit_);
  if (real_dst_count <= 0) {
    return lite::RET_OK;
  }
  float *cur_input_data = input_ptr_ + task_id * count_unit_;
  float *cur_output_data = output_ptr_ + task_id * count_unit_;
  auto ret = DoGeLU(cur_input_data, cur_output_data, real_dst_count, gelu_param_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "GeLU error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int GeLURun(void *cdata, int task_id) {
  auto g_kernel = reinterpret_cast<GeLUCPUKernel *>(cdata);
  auto ret = g_kernel->GeLU(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "GeLURun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int GeLUCPUKernel::Run() {
  auto in_tensor = in_tensors_.front();
  auto out_tensor = out_tensors_.front();
  input_ptr_ = reinterpret_cast<float *>(in_tensor->MutableData());
  output_ptr_ = reinterpret_cast<float *>(out_tensor->MutableData());
  elements_num_ = out_tensor->ElementsNum();
  count_unit_ = thread_count_ > 1 ? UP_DIV(elements_num_, thread_count_) : elements_num_;
  auto ret = ParallelLaunch(this->context_->thread_pool_, GeLURun, this, thread_count_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Scale error error_code[" << ret << "]";
    return RET_ERROR;
  }

  return RET_OK;
}
}  // namespace mindspore::kernel
