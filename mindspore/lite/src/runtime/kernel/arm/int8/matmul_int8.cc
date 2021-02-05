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

#include "src/runtime/kernel/arm/int8/matmul_int8.h"
#include "nnacl/int8/matmul_int8.h"
#include "nnacl/common_func.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"
#include "src/kernel_registry.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_MatMul;

namespace mindspore::kernel {
int MatmulInt8CPUKernel::Init() {
  InitParameter();

  auto ret = MatmulBaseInt8CPUKernel::Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ParallelLaunch failed";
    return ret;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }

  return ReSize();
}

int MatmulInt8CPUKernel::ReSize() {
  int batch = 1;
  auto x_shape = in_tensors_.at(0)->shape();
  auto o_shape = out_tensors_.at(0)->shape();
  MS_ASSERT(x_shape.size() >= 2);
  for (size_t i = 0; i < x_shape.size() - 2; ++i) {
    batch *= x_shape[i];
  }
  param_->batch = batch;
  MS_ASSERT(o_shape.size() >= 2);
  param_->row_ = o_shape[o_shape.size() - 2];
  param_->col_ = o_shape[o_shape.size() - 1];
  param_->deep_ = param_->a_transpose_ ? x_shape[x_shape.size() - 2] : x_shape[x_shape.size() - 1];

  auto ret = MatmulBaseInt8CPUKernel::ReSize();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "MatmulBaseInt8CPUKernel failed";
    return ret;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_MatMul, LiteKernelCreator<MatmulInt8CPUKernel>)
}  // namespace mindspore::kernel
