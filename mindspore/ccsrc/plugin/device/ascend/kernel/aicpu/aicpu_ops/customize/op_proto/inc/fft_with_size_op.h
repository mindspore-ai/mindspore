/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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

#ifndef CUSTOMIZE_OP_PROTO_INC_FFT_WITH_SIZE_OP_H
#define CUSTOMIZE_OP_PROTO_INC_FFT_WITH_SIZE_OP_H

#include "op_proto_macro.h"

namespace ge {
REG_CUST_OP(FFTWithSize)
  .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT, DT_COMPLEX128, DT_COMPLEX64}))
  .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT, DT_COMPLEX128, DT_COMPLEX64}))
  .REQUIRED_ATTR(signal_nadim, Int)
  .REQUIRED_ATTR(inverse, Bool)
  .ATTR(signal_sizes, ListInt, {})
  .ATTR(norm, String, "backward")
  .ATTR(onesided, Bool, true)
  .CUST_OP_END_FACTORY_REG(FFTWithSize)
}  // namespace ge
#endif