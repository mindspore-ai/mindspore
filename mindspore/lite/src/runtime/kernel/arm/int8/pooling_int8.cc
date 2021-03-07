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
#include "nnacl/int8/pooling_int8.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"
#include "src/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::schema::PrimitiveType_AvgPoolFusion;
using mindspore::schema::PrimitiveType_MaxPoolFusion;

namespace mindspore::kernel {
int PoolingInt8CPUKernel::Init() {
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
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int PoolingInt8CPUKernel::ReSize() {
  FreeQuantParam();
  auto ret = PoolingBaseCPUKernel::ReSize();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PoolingBase Init failed.";
    return ret;
  }

  ret = SetQuantParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set pooling quant param failed.";
    return ret;
  }
  return RET_OK;
}

int PoolingInt8CPUKernel::RunImpl(int task_id) {
  auto input_data = reinterpret_cast<int8_t *>(in_tensors_.at(kInputIndex)->MutableData());
  MS_ASSERT(input_data);
  auto output_data = reinterpret_cast<int8_t *>(out_tensors_.at(kOutputIndex)->MutableData());
  MS_ASSERT(output_data);
  MS_ASSERT(pooling_param_);
  if (pooling_param_->pool_mode_ == PoolMode_MaxPool) {
    if (pooling_param_->quantize_) {
      MaxPoolingWithQuantInt8(input_data, output_data, pooling_param_, task_id);
    } else {
      MaxPoolingOptInt8(input_data, output_data, pooling_param_, task_id);
    }
  } else {
    auto ret = AvgPoolingOptInt8(input_data, output_data, pooling_param_, task_id);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "AvgPooling run failed.";
      return ret;
    }
  }
  return RET_OK;
}

int PoolingInt8Impl(void *cdata, int task_id) {
  auto pooling = reinterpret_cast<PoolingInt8CPUKernel *>(cdata);
  auto error_code = pooling->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "PoolingInt8 Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int PoolingInt8CPUKernel::Run() {
  int error_code = ParallelLaunch(this->context_->thread_pool_, PoolingInt8Impl, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "poolingInt8 error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_AvgPoolFusion, LiteKernelCreator<PoolingInt8CPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_MaxPoolFusion, LiteKernelCreator<PoolingInt8CPUKernel>)
}  // namespace mindspore::kernel
