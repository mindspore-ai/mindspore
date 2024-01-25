/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

/*!
 * \file lookup_ops_shape_fns.h
 * \brief
 */
#ifndef CUSTOMIZE_OP_PROTO_UTIL_LOOKUP_OPS_SHAPE_FNS_H_
#define CUSTOMIZE_OP_PROTO_UTIL_LOOKUP_OPS_SHAPE_FNS_H_

#include <vector>
#include "graph/tensor.h"
#include "graph/inference_context.h"
#include "graph/operator.h"

namespace ge {
/**
 * Validate table resource handle
 * @param keys keys of the shape
 * @param handleData vector of handle data
 * @param output_shape_and_type shape and type that created
 * @param is_lookup if is lookup
 * @return status whether this operation success
 */
graphStatus ValidateTableResourceHandle(Shape keys, std::vector<ShapeAndType> handleData,
                                        ShapeAndType &output_shape_and_type, bool is_lookup, const ge::Operator &op);

/**
 * Validate table resource handle
 * @param op op context
 * @param keys keys of the shape
 * @param handleData vector of handle data
 * @param output_shape_and_type shape and type that created
 * @param is_lookup if is lookup
 * @return status whether this operation success
 */
graphStatus ValidateTableResourceHandle(const Operator &op, Shape &keys, const DataType &key_dtype,
                                        const DataType &value_dtype, const bool &is_lookup,
                                        ShapeAndType &output_shape_and_type);
}  // namespace ge

#endif  // CUSTOMIZE_OP_PROTO_UTIL_LOOKUP_OPS_SHAPE_FNS_H_
