/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_CUST_RANDOM_OPS_H_
#define MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_CUST_RANDOM_OPS_H_

#include "graph/operator.h"
#include "graph/operator_reg.h"
#include "transform/graph_ir/custom_op_proto/op_proto_macro.h"

/* clang-format off */

namespace ge {
REG_CUST_OP(LogNormalReverse)
  .INPUT(input, TensorType({DT_FLOAT, DT_FLOAT16}))
  .OUTPUT(output, TensorType({DT_FLOAT, DT_FLOAT16}))
  .REQUIRED_ATTR(mean, Float)
  .REQUIRED_ATTR(std, Float)
  .CUST_OP_END_FACTORY_REG(LogNormalReverse)

REG_CUST_OP(Dropout2D)
  .INPUT(x, TensorType({DT_BOOL, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8,
                        DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8}))
  .OUTPUT(y, TensorType({DT_BOOL, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8,
                         DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8}))
  .OUTPUT(mask, TensorType({DT_BOOL}))
  .REQUIRED_ATTR(keep_prob, Float)
  .CUST_OP_END_FACTORY_REG(Dropout2D)

REG_CUST_OP(Randperm)
  .INPUT(n, TensorType({DT_INT32, DT_INT64}))
  .OUTPUT(output, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64,
                              DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
  .ATTR(max_length, Int, 1)
  .ATTR(pad, Int, 1)
  .ATTR(dtype, Type, DT_INT32)
  .CUST_OP_END_FACTORY_REG(Randperm)

REG_CUST_OP(Gamma)
  .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
  .INPUT(alpha, TensorType({DT_FLOAT}))
  .INPUT(beta, TensorType({DT_FLOAT}))
  .INPUT(seed, TensorType({DT_INT64}))  // seed_adapter convert attr to input
  .INPUT(seed2, TensorType({DT_INT64}))  // seed_adapter convert attr to input
  .REQUIRED_ATTR(seed, Int)
  .REQUIRED_ATTR(seed2, Int)
  .OUTPUT(output, TensorType({DT_FLOAT}))
  .CUST_OP_END_FACTORY_REG(Gamma)

REG_CUST_OP(Multinomial)
    .INPUT(logits, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(num_samples, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(dtype, Type, DT_INT64)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .CUST_OP_END_FACTORY_REG(Multinomial)

REG_CUST_OP(RandomCategorical)
  .INPUT(logits, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
  .INPUT(num_samples, TensorType({DT_INT32, DT_INT64}))
  .INPUT(seed, TensorType({DT_INT32, DT_INT64}))
  .OUTPUT(y, TensorType({DT_INT16, DT_INT32, DT_INT64}))
  .CUST_OP_END_FACTORY_REG(RandomCategorical)

REG_CUST_OP(RandomPoisson)
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .INPUT(rate, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, \
        DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, \
        DT_INT32, DT_INT64}))
    .ATTR(dtype, Type, DT_INT64)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .CUST_OP_END_FACTORY_REG(RandomPoisson)

REG_CUST_OP(RandomShuffle)
    .INPUT(x, TensorType({DT_INT64, DT_INT32, DT_UINT16, DT_INT16,
        DT_UINT8, DT_INT8, DT_UINT64, DT_UINT32, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64,
        DT_COMPLEX128, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_INT64, DT_INT32, DT_UINT16, DT_INT16,
        DT_UINT8, DT_INT8, DT_UINT64, DT_UINT32, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64,
        DT_COMPLEX128, DT_BOOL}))
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .CUST_OP_END_FACTORY_REG(RandomShuffle)

REG_CUST_OP(StandardLaplace)
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .ATTR(dtype, Type, DT_FLOAT)
    .CUST_OP_END_FACTORY_REG(StandardLaplace)

REG_CUST_OP(RandomChoiceWithMask)
    .INPUT(x, TensorType({DT_BOOL}))
    .OUTPUT(index, TensorType({DT_INT32}))
    .OUTPUT(mask, TensorType({DT_BOOL}))
    .ATTR(count, Int, 0)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .CUST_OP_END_FACTORY_REG(RandomChoiceWithMask)

REG_CUST_OP(RandomUniformInt)
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .INPUT(min, TensorType({DT_INT32, DT_INT64}))
    .INPUT(max, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .CUST_OP_END_FACTORY_REG(RandomUniformInt)

REG_CUST_OP(Igamma)
  .INPUT(a, TensorType({DT_DOUBLE, DT_FLOAT}))
  .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT}))
  .OUTPUT(z, TensorType({DT_DOUBLE, DT_FLOAT}))
  .CUST_OP_END_FACTORY_REG(Igamma)

REG_CUST_OP(Poisson)
  .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
  .INPUT(mean, TensorType({DT_FLOAT}))
  .INPUT(seed, TensorType({DT_INT64}))  // seed_adapter convert attr to input
  .INPUT(seed2, TensorType({DT_INT64}))  // seed_adapter convert attr to input
  .REQUIRED_ATTR(seed, Int)
  .REQUIRED_ATTR(seed2, Int)
  .OUTPUT(output, TensorType({DT_INT32}))
  .CUST_OP_END_FACTORY_REG(Poisson)
}  // namespace ge
#endif  // MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_CUST_RANDOM_OPS_H_
