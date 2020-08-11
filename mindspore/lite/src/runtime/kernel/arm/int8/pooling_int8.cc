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

#include "src/runtime/kernel/arm/int8/pooling_int8.h"
#include "src/runtime/kernel/arm/nnacl/int8/pooling_int8.h"
#include "src/runtime/kernel/arm/nnacl/fp32/cast.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int PoolingInt8CPUKernel::Init() {
  if (context_->infer_shape_interrupt_ && !context_->running_) {
    set_need_reinit();
    return RET_OK;
  }
  auto ret = PoolingBaseCPUKernel::Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PoolingBase Init failed.";
    return RET_ERROR;
  }
  ret = SetQuantParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set pooling quant param failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int PoolingInt8CPUKernel::ReSize() {
  FreeQuantParam();
  auto ret = PoolingBaseCPUKernel::Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PoolingBase Init failed.";
    return RET_ERROR;
  }
  SetQuantParam();
  ret = SetQuantParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set pooling quant param failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int PoolingInt8CPUKernel::RunImpl(int task_id) {
  auto input_data = reinterpret_cast<int8_t *>(in_tensors_.at(kInputIndex)->Data());
  auto output_data = reinterpret_cast<int8_t *>(out_tensors_.at(kOutputIndex)->Data());
  if (pooling_param_->max_pooling_) {
    MaxPoolingInt8(input_data, output_data, pooling_param_, task_id);
  } else {
    AvgPoolingInt8(input_data, output_data, pooling_param_, task_id);
  }
  return RET_OK;
}

int PoolingInt8Impl(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto pooling = reinterpret_cast<PoolingInt8CPUKernel *>(cdata);
  auto error_code = pooling->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "PoolingInt8 Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int PoolingInt8CPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare failed.";
    return RET_ERROR;
  }
  int error_code = LiteBackendParallelLaunch(PoolingInt8Impl, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "poolingInt8 error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
