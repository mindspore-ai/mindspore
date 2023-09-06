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
 * \file images_ops_shape_fns.h
 * \brief
 */
#ifndef CUSTOMIZE_OP_PROTO_UTIL_IMAGES_OPS_SHAPE_FNS_H_
#define CUSTOMIZE_OP_PROTO_UTIL_IMAGES_OPS_SHAPE_FNS_H_

#include <string>

#include "common_shape_fns.h"

namespace ge {
/**
 * ColorspaceShapeFn, infereshape function of colorspace op
 * @param op, Operators that need to reason about shape
 * @param output_name, the name of output
 * @return status whether infer shape success
 */
graphStatus ColorspaceShapeFn(Operator &op, const std::string &output_name);

/**
 * ResizeShapeFn, infereshape function of image resize op
 * @param op, Operators that need to reason about shape
 * @param input_name, the name of input
 * @param size_input_name, the name of size input name
 * @param output_name, the name of output
 * @return status whether infer shape success
 */
graphStatus ResizeShapeFn(Operator &op, const std::string &input_name, const std::string &size_input_name,
                          const std::string &output_name);

/**
 * SetOutputToSizedImage, set output shape of size image op
 * @param op, Operators that need to set output shape
 * @param batch_dim, the dim of batch
 * @param size_input_name, the name of size input
 * @param channel_dim, the dim of channel
 * @param output_name, the name of output
 * @return status whether set output shape success
 */
graphStatus SetOutputToSizedImage(Operator &op, const int64_t batch_dim, const std::string &size_input_name,
                                  const int64_t channel_dim, const std::string &output_name);

/**
 * EncodeImageShapeFn, infereshape function of EncodeImage op
 * @param op, Operators that need to reason about shape
 * @return status whether infer shape success
 */
graphStatus EncodeImageShapeFn(Operator &op);

/**
 * DecodeImageShapeFn, infereshape function of DecodeImage op
 * @param op, Operators that need to reason about shape
 * @return status whether infer shape success
 */
graphStatus DecodeImageShapeFn(Operator &op);

/**
 * EncodeImageShapeFn, infereshape function of EncodeImage op
 * @param inputs, the list of impu dims
 * @param unknown_dim_val, the definithion of UNKNOWN_DIM
 * @return status whether infer shape success
 */
bool DimsAllEqualOrUnknown(std::initializer_list<int64_t> &&inputs, int64_t unknown_dim_val = UNKNOWN_DIM);

}  // namespace ge

#endif  // CUSTOMIZE_OP_PROTO_UTIL_IMAGES_OPS_SHAPE_FNS_H_
