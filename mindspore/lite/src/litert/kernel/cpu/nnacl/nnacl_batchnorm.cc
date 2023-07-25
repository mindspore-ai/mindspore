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

#include "nnacl/nnacl_batchnorm.h"
#include "nnacl/nnacl_manager.h"
#include "include/errorcode.h"
#include "nnacl/fp32/batchnorm_fp32.h"

using mindspore::schema::PrimitiveType_BatchNorm;

namespace mindspore::nnacl {
int BatchNormKernel::SetupVirtualBatch(int virtual_batch_multiplier, int momentum) {
  CHECK_NULL_RETURN(kernel_);
  BatchNormSetupVirtualBatch(kernel_, virtual_batch_multiplier, momentum);
  return RET_OK;
}

NNACL_KERNEL(PrimitiveType_BatchNorm, kNumberTypeFloat32, NNACLOpt<BatchNormKernel>)
NNACL_KERNEL(PrimitiveType_BatchNorm, kNumberTypeFloat16, NNACLOpt<BatchNormKernel>)
}  // namespace mindspore::nnacl
