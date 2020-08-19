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

#include "src/runtime/kernel/arm/fp32/pooling.h"
#include "nnacl/fp32/pooling.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"
#include "nnacl/op_base.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Pooling;

namespace mindspore::kernel {
int PoolingCPUKernel::Init() {
  auto ret = PoolingBaseCPUKernel::Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PoolingBase Init failed.";
    return RET_ERROR;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int PoolingCPUKernel::ReSize() {
  auto ret = PoolingBaseCPUKernel::ReSize();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PoolingBase ReSize fai1!ret: " << ret;
    return ret;
  }
  return RET_OK;
}

int PoolingCPUKernel::RunImpl(int task_id) {
  auto input_ptr = reinterpret_cast<float *>(in_tensors_.at(kInputIndex)->Data());
  auto output_ptr = reinterpret_cast<float *>(out_tensors_.at(kOutputIndex)->Data());
  if (pooling_param_->max_pooling_) {
    switch (pooling_param_->act_type_) {
      case ActType_Relu:
        MaxPoolingRelu(input_ptr, output_ptr, pooling_param_, task_id);
        break;
      case ActType_Relu6:
        MaxPoolingRelu6(input_ptr, output_ptr, pooling_param_, task_id);
        break;
      default:
        MaxPooling(input_ptr, output_ptr, pooling_param_, task_id);
    }
  } else {
    switch (pooling_param_->act_type_) {
      case ActType_Relu:
        AvgPoolingRelu(input_ptr, output_ptr, pooling_param_, task_id);
        break;
      case ActType_Relu6:
        AvgPoolingRelu6(input_ptr, output_ptr, pooling_param_, task_id);
        break;
      default:
        AvgPooling(input_ptr, output_ptr, pooling_param_, task_id);
    }
  }
  return RET_OK;
}

int PoolingImpl(void *cdata, int task_id) {
  auto pooling = reinterpret_cast<PoolingCPUKernel *>(cdata);
  auto error_code = pooling->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Pooling Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int PoolingCPUKernel::Run() {
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << prepare_ret;
    return prepare_ret;
  }
  int error_code = ParallelLaunch(THREAD_POOL_DEFAULT, PoolingImpl, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "pooling error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
