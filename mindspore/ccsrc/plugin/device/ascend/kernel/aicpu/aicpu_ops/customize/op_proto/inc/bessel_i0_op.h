/**
 * Copyright (c) 2022-2022 Huawei Technologies Co., Ltd.  All rights reserved.
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

#ifndef GE_OP_BESSEL_I0_H
#define GE_OP_BESSEL_I0_H
#include "op_proto_macro.h"
namespace ge {
REG_CUST_OP(BesselI0)
  .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
  .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
  .CUST_OP_END_FACTORY_REG(BesselI0)
}
#endif  // GE_OP_BESSEL_I0_H
