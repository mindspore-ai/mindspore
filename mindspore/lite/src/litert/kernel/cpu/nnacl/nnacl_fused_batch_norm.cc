/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "nnacl/nnacl_fused_batch_norm.h"
#include "nnacl/nnacl_manager.h"
#include "include/errorcode.h"
#include "nnacl/fp32/batchnorm_fp32.h"
#include "nnacl/kernel/fused_batch_norm.h"

using mindspore::schema::PrimitiveType_FusedBatchNorm;

namespace mindspore::nnacl {
int FusedBatchNormKernel::Eval() {
  auto ret = LiteKernel::Eval();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Inner kernel eval in nnacl fused batchnorm error.";
    return ret;
  }
  return FusedBatchNormEval(kernel_);
}

int FusedBatchNormKernel::Run() {
  reinterpret_cast<FusedBatchNormStruct *>(kernel_)->train_mode_ = IsTrain();
  return NNACLKernel::Run();
}

NNACL_KERNEL(PrimitiveType_FusedBatchNorm, kNumberTypeFloat32, NNACLOpt<FusedBatchNormKernel>)
NNACL_KERNEL(PrimitiveType_FusedBatchNorm, kNumberTypeFloat16, NNACLOpt<FusedBatchNormKernel>)
}  // namespace mindspore::nnacl
