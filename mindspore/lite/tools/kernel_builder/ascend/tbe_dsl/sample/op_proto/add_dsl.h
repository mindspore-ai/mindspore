/**
 * Copyright (C)  2020. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file add_dsl.h
 *
 * @brief
 *
 * @version 1.0
 *
 */

#ifndef GE_OPS_OP_PROTO_ADDDSL_H_
#define GE_OPS_OP_PROTO_ADDDSL_H_
#include "graph/operator_reg.h"
namespace ge {
REG_OP(AddDsl)
  .INPUT(x1, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE,
                         DT_COMPLEX128, DT_COMPLEX64, DT_STRING}))
  .INPUT(x2, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE,
                         DT_COMPLEX128, DT_COMPLEX64, DT_STRING}))
  .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE,
                         DT_COMPLEX128, DT_COMPLEX64, DT_STRING}))
  .OP_END_FACTORY_REG(AddDsl)
}

#endif  // GE_OPS_OP_PROTO_ADDDSL_H_
