/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
 * \file reduce_infer_util.h
 * \brief
 */
#ifndef CUSTOMIZE_OP_PROTO_UTIL_REDUCE_INFER_UTIL_H_
#define CUSTOMIZE_OP_PROTO_UTIL_REDUCE_INFER_UTIL_H_

#include <string>
#include <vector>
#include <map>
#include <algorithm>

using namespace std;
using namespace ge;
namespace reduce_ops {

/*
 * only do infer shape for reduce with input_shape/axes, when keepdims = true
 * param[in] input_shape: Shape input shape
 * param[in] reduce_axes: reduce axes list
 * param[in] output_shape: Shape output shape
 * return bool:
 *        true:infer success false:infer failed
 */
bool DoReduceInfershapeWithAxesKeepdims(const Shape &input_shape, std::vector<int64_t> &reduce_axes,
                                        Shape &output_shape);

/*
 * only do infer shape for reduce with input_shape/axes, when keepdims = false
 * param[in] input_shape: Shape input shape
 * param[in] reduce_axes: reduce axes list
 * param[in] output_shape: Shape output shape
 * return bool:
 *        true:infer success false:infer failed
 */
bool DoReduceInfershapeWithAxesNoKeepdims(const Shape &input_shape, std::vector<int64_t> &reduce_axes,
                                          Shape &output_shape);

/*
 * only do infer shape for reduce with input_shape, axes and keepdims
 * param[in] input_shape: Shape input shape
 * param[in] keep_dims: bool
 * param[in] reduce_axes: reduce axes list
 * param[in] output_shape: Shape output shape
 * return bool:
 *        true:infer success false:infer failed
 */
bool DoReduceInfershapeWithAxes(const Shape &input_shape, const bool keep_dims, std::vector<int64_t> &reduce_axes,
                                Shape &output_shape);

/*
 * only do infer range for reduce
 * param[in] tensordesc_input_x: TensorDesc of input tensor
 * param[in] tensordesc_output: TensorDesc of output tensor
 * param[in] reduce_axes: reduce axes list
 * param[in] keep_dims: bool
 * return bool:
 *        true:infer success false:infer failed
 */
bool DoReduceInferRangeWithAxes(TensorDesc &tensordesc_input_x, TensorDesc &tensordesc_output,
                                std::vector<int64_t> &reduce_axes, bool keep_dims);

/*
 * get const value from const node to vector const_values
 * param[in] op: op desc get from by ge
 * param[in] const_input_idx: the input idx for const node
 * param[in] const_values: the const value
 * return bool:
 *        true:infer success false:infer failed
 */
bool GetConstData(const Operator &op, const int64_t const_input_idx, std::vector<int64_t> &const_values);

/*
 * infer shape and range for reduce, when the axes is not const
 * param[in] op: op desc get from by ge
 * param[in] tensordesc_input_x: TensorDesc of input tensor
 * param[in] tensordesc_output: TensorDesc of output tensor
 * param[in] axes_shape: the axes shape
 * param[in] keep_dims: bool
 * return bool:
 *        true:get value success false:no not get the const value
 */
bool DoReduceInferShapeWithoutAxes(const Operator &op, TensorDesc &tensordesc_input_x, TensorDesc &tensordesc_output,
                                   const Shape &axes_shape, bool keep_dims);

/*
 * reduce infershape function, when axes is input
 * param[in] op: op desc get from by ge
 * param[in] input_x_idx: the input tensor idx int64
 * param[in] output_idx: the output tensor idx int64
 * param[in] input_axes_idx: the input const idx
 * param[in] keep_dims: bool
 * return bool:
 *        true:infer success false:infer failed
 */
bool CommonReduceInferWithInputAxes(Operator &op, const int64_t input_x_idx, const int64_t output_idx,
                                    const std::string &axes_name, bool keep_dims);
bool CommonReduceInferWithAttrAxes(Operator &op, const int64_t input_x_idx, const int64_t output_idx,
                                   vector<int64_t> attr_axes, bool keep_dims);
}  // namespace reduce_ops

#endif  // CUSTOMIZE_OP_PROTO_UTIL_REDUCE_INFER_UTIL_H_
