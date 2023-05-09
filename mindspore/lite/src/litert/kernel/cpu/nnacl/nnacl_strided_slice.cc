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

#include "nnacl/nnacl_strided_slice.h"
#include "nnacl/nnacl_manager.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/kernel/strided_slice.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_StridedSlice;

namespace mindspore::nnacl {
int StridedSliceKernel::Run() {
  StridedSliceStruct *strided_slice = reinterpret_cast<StridedSliceStruct *>(kernel_);
  CHECK_NULL_RETURN(strided_slice);

  if (!strided_slice->soft_copy_mode_) {
    return NNACLKernel::Run();
  }

  return NNACLKernel::OptimizeDataCopy();
}

NNACL_KERNEL(PrimitiveType_StridedSlice, kNumberTypeFloat32, NNACLOpt<StridedSliceKernel>)
NNACL_KERNEL(PrimitiveType_StridedSlice, kNumberTypeFloat16, NNACLOpt<StridedSliceKernel>)
NNACL_KERNEL(PrimitiveType_StridedSlice, kNumberTypeInt64, NNACLOpt<StridedSliceKernel>)
NNACL_KERNEL(PrimitiveType_StridedSlice, kNumberTypeInt32, NNACLOpt<StridedSliceKernel>)
NNACL_KERNEL(PrimitiveType_StridedSlice, kNumberTypeInt8, NNACLOpt<StridedSliceKernel>)
}  // namespace mindspore::nnacl
