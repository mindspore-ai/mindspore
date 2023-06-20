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

#ifndef CUSTOMIZE_OP_PROTO_INC_MAX_POOL3_D_GRAD_WITH_ARGMAX_OP_H
#define CUSTOMIZE_OP_PROTO_INC_MAX_POOL3_D_GRAD_WITH_ARGMAX_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
 * @brief Computes gradients of the MaxPool3DWithArgmax function
 *
 * @par Inputs:
 *  Three input:
 *  x: An 5D tensor. Supported type: RealNumberType. Format as NCDHW.
 *  grads: Gradient tensor(NDC1HWC0) of RealNumberType
 *  argmax: An 5D tensor. Supported type: int32/int64.
 * @par Attributes:
 *  @li ksize: A required list of int32 values,
 *   specifying the size of the window for each dimension of the input tensor.
 *   No default value.
 *  @li strides: A required list of int32 values,
 *   specifying the stride of the sliding window for each dimension of
 *   the input tensor. No default value.
 *  @li pads: A required 3*2-dimension-list of int32 values.
 *   specifying the pad of three dimension of input, implement with 0.
 *  @li dilation: dilation of kernel. default value is {1,1,1,1,1}.
 *  @li ceil_mode: default value is false.
 *  @li data_format: the format of torch input, default value is "NCDHW".
 *  @li dtype: the function of this field is to determine the type of
 *   output tensor, "0" is the default value, represents float32. Only "0" and
 *   "1" for float16 is supported.
 * @par Outputs:
 *  y: Result tensor of RealNumberType.
 */
REG_CUST_OP(MaxPool3DGradWithArgmax)
  .INPUT(x, TensorType::RealNumberType())
  .INPUT(grads, TensorType::RealNumberType())
  .INPUT(argmax, TensorType::IndexNumberType())
  .OUTPUT(y, TensorType::RealNumberType())
  .REQUIRED_ATTR(ksize, ListInt)
  .REQUIRED_ATTR(strides, ListInt)
  .REQUIRED_ATTR(pads, ListInt)
  .REQUIRED_ATTR(dilation, ListInt)
  .ATTR(ceil_mode, Bool, false)
  .ATTR(data_format, String, "NCDHW")
  .ATTR(dtype, Int, 0)
  .CUST_OP_END_FACTORY_REG(MaxPool3DGradWithArgmax)
}  // namespace ge
#endif