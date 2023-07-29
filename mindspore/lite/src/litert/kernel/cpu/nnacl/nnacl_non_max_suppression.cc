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

#include "nnacl/nnacl_non_max_suppression.h"
#include "nnacl/nnacl_manager.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_NonMaxSuppression;

namespace mindspore::nnacl {

int NonMaxSuppressionKernel::PreProcess() { return RET_OK; }

int NonMaxSuppressionKernel::Run() {
  int ret = NNACLKernel::Run();
  for (auto *output : this->out_tensors()) {
    output->ResetRefCount();
  }
  return ret;
}

NNACL_KERNEL(PrimitiveType_NonMaxSuppression, kNumberTypeFloat32, NNACLOpt<NonMaxSuppressionKernel>)
}  // namespace mindspore::nnacl
