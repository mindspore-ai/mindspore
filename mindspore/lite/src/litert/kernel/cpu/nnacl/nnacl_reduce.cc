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

#include "nnacl/nnacl_reduce.h"
#include "nnacl/nnacl_manager.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/kernel/reduce.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ReduceFusion;

namespace mindspore::nnacl {
int ReduceKernel::Run() {
  ReduceStruct *reduce = reinterpret_cast<ReduceStruct *>(kernel_);
  CHECK_NULL_RETURN(reduce);

  if (!reduce->only_copy_) {
    return NNACLKernel::Run();
  }

  return NNACLKernel::OptimizeDataCopy();
}

NNACL_KERNEL(PrimitiveType_ReduceFusion, kNumberTypeFloat32, NNACLOpt<ReduceKernel>)
NNACL_KERNEL(PrimitiveType_ReduceFusion, kNumberTypeInt32, NNACLOpt<ReduceKernel>)
NNACL_KERNEL(PrimitiveType_ReduceFusion, kNumberTypeBool, NNACLOpt<ReduceKernel>)
}  // namespace mindspore::nnacl
