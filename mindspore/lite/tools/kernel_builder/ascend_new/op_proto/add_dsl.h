/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef GE_OP_ADD_DSL_H
#define GE_OP_ADD_DSL_H
#include "graph/operator_reg.h"
namespace ge {
REG_OP(AddDSL)
  .INPUT(x1, TensorType({DT_FLOAT16}))
  .INPUT(x2, TensorType({DT_FLOAT16}))
  .OUTPUT(y, TensorType({DT_FLOAT16}))
  .OP_END_FACTORY_REG(AddDSL)
}
#endif  // GE_OP_ADD_DSL_H
