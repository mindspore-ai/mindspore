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

#ifndef CUSTOMIZE_OP_PROTO_INC_ARG_MAX_OP_H
#define CUSTOMIZE_OP_PROTO_INC_ARG_MAX_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
*@brief Returns the index with the largest value across axes of a tensor. \n

*@par Inputs:
* Two inputs, including:
*@li x: A multi-dimensional Tensor of type float16, float32, or int16.
*@li dimension: A Scalar of type int32, specifying the index with the largest value. \n

*@par Attributes:
*dtype: The output type, either "int32" or "int64". Defaults to "int64". \n

*@par Outputs:
*y: A multi-dimensional Tensor of type int32 or int64, specifying the index with the largest value. The dimension is one
less than that of "x". \n

*@attention Constraints:
*@li x: If there are multiple maximum values, the index of the first maximum value is used.
*@li The value range of "dimension" is [-dims, dims - 1]. "dims" is the dimension length of "x". \n

*@par Third-party framework compatibility
* Compatible with TensorFlow operator ArgMax.
*/
REG_CUST_OP(ArgMaxV2)
  .INPUT(x, TensorType::NumberType())
  .INPUT(dimension, TensorType::IndexNumberType())
  .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
  .ATTR(dtype, Type, DT_INT64)
  .CUST_OP_END_FACTORY_REG(ArgMaxV2)
}  // namespace ge
#endif