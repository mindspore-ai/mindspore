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

#include "src/runtime/kernel/arm/fp32/pooling_fp32.h"
#include <float.h>
#include "nnacl/fp32/pooling_fp32.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"
#include "nnacl/op_base.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_AvgPoolFusion;
using mindspore::schema::PrimitiveType_MaxPoolFusion;

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
  auto input_ptr = reinterpret_cast<float *>(in_tensors_.at(kInputIndex)->MutableData());
  auto output_ptr = reinterpret_cast<float *>(out_tensors_.at(kOutputIndex)->MutableData());
  float minf = -FLT_MAX;
  float maxf = FLT_MAX;
  if (pooling_param_->act_type_ == ActType_Relu) {
    minf = 0.f;
  } else if (pooling_param_->act_type_ == ActType_Relu6) {
    minf = 0.f;
    maxf = 6.f;
  }
  if (pooling_param_->pool_mode_ == PoolMode_MaxPool) {
    MaxPooling(input_ptr, output_ptr, pooling_param_, task_id, minf, maxf);
  } else {
    auto ret = AvgPooling(input_ptr, output_ptr, pooling_param_, task_id, minf, maxf);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "AcgPooling run failed.";
      return ret;
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
  int error_code = ParallelLaunch(this->context_->thread_pool_, PoolingImpl, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "pooling error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_AvgPoolFusion, LiteKernelCreator<PoolingCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_MaxPoolFusion, LiteKernelCreator<PoolingCPUKernel>)
}  // namespace mindspore::kernel
