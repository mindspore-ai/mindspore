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

#ifndef CUSTOMIZE_OP_PROTO_INC_SEGMENT_MIN_OP_H
#define CUSTOMIZE_OP_PROTO_INC_SEGMENT_MIN_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
* @brief: Computes the minimum along segments of a tensor.
* Computes a tensor such that output[i]=(data[i]) where max is over j such that segment_ids[j] == i.
* If the min is empty for a given segment ID i, output[i] = 0

* @par Inputs:
* Two inputs, include:
* @li x:A Tensor of type float16, float32, int32,int8,uint8.
* @li segment_ids:should be the size of the first dimension
        must sorted and need not cover all values in the full range of valid values
        must be positive intege

* @par Outputs:
* y:A Tensor with same type as "x" . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator SegmentMin.
*/

REG_CUST_OP(SegmentMin)
  .INPUT(x, TensorType::RealNumberType())
  .INPUT(segment_ids, TensorType::IndexNumberType())
  .OUTPUT(y, TensorType::RealNumberType())
  .CUST_OP_END_FACTORY_REG(SegmentMin)
}  // namespace ge
#endif