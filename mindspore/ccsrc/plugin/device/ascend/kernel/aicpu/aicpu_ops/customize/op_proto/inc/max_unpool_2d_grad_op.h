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

#ifndef CUSTOMIZE_OP_PROTO_INC_MAX_UNPOOL_2D_GRAD_OP_H
#define CUSTOMIZE_OP_PROTO_INC_MAX_UNPOOL_2D_GRAD_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
* @brief Performs the backpropagation of MaxUnpool2DGrad . \n

* @par Inputs:
* Three inputs, including:
* @li x: An 4d tensor. Supported type: float, double, int32,
* uint8, int16, int8, int64, uint16, half, uint32, uint64.
* Must set the format, supported format list ["NCHW, NHWC"]
* @li grads: An 4d tensor. Supported type: float, double, int32,
* uint8, int16, int8, int64, uint16, half, uint32, uint64.
* Must set the format, supported format list ["NCHW, NHWC"]
* @li An 4d tensor of type uint16 or int64. Must set the format, supported format list ["NCHW, NHWC"] \n

* @par Attributes:
* @li ksize: A required list of int8, int16, int32, or int64 values,
* specifying the size of the window for each dimension of the input tensor.
* No default value.
* @li strides: A required list of int8, int16, int32, or int64 values,
* specifying the stride of the sliding window for each dimension of
* the input tensor. No default value.
* @li pads:A required list of int8, int16, int32, or int64 values,
* a data to calculate when padding_mode is "CALCULATED".
* @li data_format: An optional string. Defaults to "NCHW" .
* @li output_shape: A required tuple or list of type int32. \n

* @par Outputs:
* y: A Tensor. Has the same type and format as input "x" . \n

* @attention Constraints:
* @li "ksize" is a list that has length 4: ksize[0] = ksize[1] = 1 or ksize[0] = ksize[3] = 1,
* @li "strides" is a list that has length 4: strides[0] = strides[1] = 1 or strides[0] = strides[3] = 1
* @li "pads" is a list that has specify the number of implicit zero pad in
* each dimension and the corresponding function dimension is the same as stride. \n

* @see MaxUnpool2D
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator MaxUnpool2DGrad.
*/
REG_CUST_OP(MaxUnpool2DGrad)
  .INPUT(x, TensorType::RealNumberType())
  .INPUT(grads, TensorType::RealNumberType())
  .INPUT(argmax, TensorType::IndexNumberType())
  .OUTPUT(y, TensorType::RealNumberType())
  .REQUIRED_ATTR(ksize, ListInt)
  .REQUIRED_ATTR(strides, ListInt)
  .REQUIRED_ATTR(pads, ListInt)
  .ATTR(data_format, String, "NCHW")
  .ATTR(output_shape, ListInt, {})
  .CUST_OP_END_FACTORY_REG(MaxUnpool2DGrad)
}  // namespace ge
#endif