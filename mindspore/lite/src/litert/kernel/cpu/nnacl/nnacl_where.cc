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

#include "nnacl/nnacl_where.h"
#include "nnacl/nnacl_manager.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Where;

namespace mindspore::nnacl {
int WhereKernel::PreProcess() {
  if (in_tensors_.size() == Num3) {
    return LiteKernel::PreProcess();
  }
  return RET_OK;
}

int WhereKernel::Run() {
  int ret = NNACLKernel::Run();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "NNACL run where op failed. Kernel: " << name() << ", ret: " << ret;
    return ret;
  }

  for (auto *output : this->out_tensors()) {
    output->ResetRefCount();
  }
  return RET_OK;
}

NNACL_KERNEL(PrimitiveType_Where, kNumberTypeBool, NNACLOpt<WhereKernel>)
NNACL_KERNEL(PrimitiveType_Where, kNumberTypeInt32, NNACLOpt<WhereKernel>)
NNACL_KERNEL(PrimitiveType_Where, kNumberTypeFloat16, NNACLOpt<WhereKernel>)
NNACL_KERNEL(PrimitiveType_Where, kNumberTypeFloat32, NNACLOpt<WhereKernel>)
}  // namespace mindspore::nnacl
