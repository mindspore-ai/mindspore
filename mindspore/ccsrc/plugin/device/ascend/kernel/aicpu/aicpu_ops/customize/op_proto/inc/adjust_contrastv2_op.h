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

#ifndef CUSTOMIZE_OP_PROTO_INC_ADJUST_CONTRASTV2_OP_H
#define CUSTOMIZE_OP_PROTO_INC_ADJUST_CONTRASTV2_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
* @brief Adjust the contrast of one or more images . \n

* @par Inputs:
* Input images is a tensor of at least 3 dimensions. The last 3 dimensions are
interpreted as '[height, width, channels]'. Inputs include:
* @li images:A Tensor of type float. Images to adjust. At least 3-D. The format
must be NHWC.
* @li scale:A Tensor of type float. A float multiplier for adjusting contrast . \n

* @par Outputs:
* y:A Tensor of type float. The format must be NHWC. \n

* @attention Constraints:
* Input images is a tensor of at least 3 dimensions. The last dimension is
interpreted as channels, and must be three . \n

* @par Third-party framework compatibility
* Compatible with tensorflow AdjustContrast operator.
*/

REG_CUST_OP(AdjustContrast)
  .INPUT(images, TensorType({DT_FLOAT16, DT_FLOAT}))
  .INPUT(contrast_factor, TensorType({DT_FLOAT}))
  .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
  .CUST_OP_END_FACTORY_REG(AdjustContrast)
}  // namespace ge
#endif