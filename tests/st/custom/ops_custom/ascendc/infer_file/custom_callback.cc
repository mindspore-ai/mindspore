/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include <vector>
#include "acl/acl_base.h"

typedef struct aclOpExecutor aclOpExecutor;
typedef struct aclTensor aclTensor;
typedef struct aclScalar aclScalar;
typedef struct aclIntArray aclIntArray;
typedef struct aclFloatArray aclFloatArray;
typedef struct aclBoolArray aclBoolArray;
typedef struct aclTensorList aclTensorList;
typedef struct aclScalarList aclScalarList;

extern "C" int aclnnMulGetWorkSpaceSize(void *func_ptr, std::vector<void *> inputs, std::vector<void *> outputs,
                                        uint64_t *workspace_size, aclOpExecutor **executor) {
  using FuncType = int (*)(aclTensor *, aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **);
  auto func = reinterpret_cast<FuncType>(func_ptr);
  aclTensor *input0 = static_cast<aclTensor *>(inputs[0]);
  aclTensor *input1 = static_cast<aclTensor *>(inputs[1]);
  aclTensor *output = static_cast<aclTensor *>(outputs[0]);
  return func(input0, input1, output, workspace_size, executor);
}



