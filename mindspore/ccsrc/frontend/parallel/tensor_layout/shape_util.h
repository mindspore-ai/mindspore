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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_TENSOR_LAYOUT_SHAPE_UTIL_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_TENSOR_LAYOUT_SHAPE_UTIL_H_

#include "frontend/parallel/status.h"
#include "frontend/parallel/device_matrix.h"

namespace mindspore {
namespace parallel {
/*
 * compute the accumulating product of all the values in shape from left to right,
 * the accumulating results are saved in shape_accum from left to right
 *
 * given a shape = [d_n-1, d_n-2, ..., d_0](d_i > 0, i=0,1,...,n-1, elements of shape must be larger than zero),
 * then *shape_accum = [d_n-1, d_n-1 * d_n-2, d_n-1 * d_n-2 * d_n-3, ..., d_n-1 * d_n-2 * ... *d_0]
 *
 * example:
 * shape = [2, 8, 32]
 * shape_accum = [2, 2 * 8, 2 * 8 * 32]
 *
 */
Status ShapeToAccumulateProduct(const Shape &shape, Shape *shape_accum);

/*
 * compute the accumulating product of all the values in shape from right to left,
 * the accumulating results are saved in shape_accum from right to left
 *
 * given a shape = [d_n-1, d_n-2, ..., d_0](d_i > 0, i=0,1,...,n-1, elements of shape must be larger than zero),
 * then *shape_accum = [d_n-1 * d_n-2 * ... *d_0, d_n-2 * d_n-3 * ... *d_0, ..., d_0]
 *
 * example:
 * shape = [2, 8, 32]
 * shape_accum = [2 * 8 * 32, 8 * 32, 32]
 *
 */
Status ShapeToAccumulateProductReverse(const Shape &shape, Shape *shape_accum);

/*
 * compute the original shape from the accumulating product shape_accum,
 * elements of shape_accum is saved from left to right,
 * given shape_accum = [accum_n-1, accum_n-2, accum_n-3, ..., accum_0]
 * (accum_i > 0, i=0,1,...,n-1, elements of shape_accum must be larger than zero),
 * (accum_i-1 % accum_i == 0, i=1,...,n-1)
 * then *shape = [accum_n-2/accum_n-1, accum_n-3/accum_n-2, ..., accum_0/accum_1]
 *
 * example:
 * shape_accum = [2, 2 * 8, 2 * 8 * 32]
 * shape = [2, 8, 32]
 *
 */
Status AccumulateProductToShape(const Shape &shape_accum, Shape *shape);

/*
 * compute the original shape from the accumulating product shape_accum,
 * elements of shape_accum is saved from right to left,
 * given shape_accum_reverse = [accum_n-1, accum_n-2, accum_n-3, ..., accum_0]
 * (accum_i > 0, i=0,1,...,n-1, elements of shape_accum must be larger than zero),
 * (accum_i % accum_i-1 == 0, i=1,...,n-1)
 * then *shape = [accum_n-1/accum_n-2, accum_n-2/accum_n-1, ..., accum_1/accum_0]
 *
 * example:
 * shape_accum_reverse = [2 * 8 * 32, 8 * 32, 32]
 * shape = [2, 8, 32]
 *
 */
Status AccumulateProductReverseToShape(const Shape &shape_accum_reverse, Shape *shape);

/*
 * given two accumulate product in1_accum and in2_accum, compute the union of in1_accum and in2_accum,
 * results are saved in out.
 * i.e. *out_accum = in1_accum U in2_accum
 * elements of out are saved in increasing order
 *
 * example1:
 * in1_accum = [2, 8]
 * in2_accum = [4, 8]
 * out_accum = [2, 4, 8]
 *
 * example2:
 * in1_accum = [2, 4, 16]
 * in2_accum = [8, 16]
 * out_accum = [2, 4, 8, 16]
 */
Status UnifyAccumulateProduct(const Shape &in1_accum, const Shape &in2_accum, Shape *out_accum);

/*
 * given two shape in1 = [din1_n-1, din1_n-2, ..., din1_0] and in2 = [din2_m-1, din2_m-2, ..., din2_m]
 * size = din1_n-1 * din1n-2 * ... * din1_0 = din2_m-1 * din2_m-2 * ... * din2_0
 * find *out = [dout_k-1, dout_k-2, ..., dout_0], s.t. dout_k-1 * dout_k-2 * ... * dout_0 = size and
 * suppose in1_accum, in2_accum, and *out_accum is the ShapeToAccumulateProduct result of in1, in2, and *out
 * then for each din1_i in in1_accum, din1_i is in *out_accumulate,
 * for each din2_i in in2_accum, din2_i is in *out_accumulate
 *
 * example:
 * in1 = [8, 4]
 * in2 = [2, 16]
 * out = [2, 4, 4]
 */
Status UnifyShape(const Shape &in1, const Shape &in2, Shape *out);

/*
 * given two accumulate product in reverse order of in and expand,
 * in_accum_reverse = [din_n-1, din_n-2, ..., din_0] and expand_pos_reverse = [dexp_n-1, dexp_n-2, ..., dexp_0],
 * i.e. in_accum_reverse is the ShapeToAccumulateProductReverse result of a shape in,
 * expand_accum_reverse is the ShapeToAccumulateProductReverse result of a shape expand,
 * compute the accumulate product in reverse order out_accum_reverse = [dout_k-1, dout_k-2, ..., dout_0],
 * s.t. elements in out_accum_reverse are union of elements in in_accum_reverse and expand_accum_reverse
 * (out_accum_reverse = in_accum_reverse U expand_accum_reverse), and
 * out_accum_reverse is the ShapeToAccumulateProductReverse result of shape expand,
 * i.e.  dout_i > 0, i=0,1,...,k-1, elements of out_accum_reverse must be larger than zero,
 * dout_i-1 % dout_i == 0, i=1,...,k-1
 *
 * example1:
 * in_accum_reverse = [2 * 8 * 32, 8 * 32, 32]
 * expand_accum_reverse = [2 * 8 * 32, 32, 8]
 * out_accum_reverse = [2 * 8 * 4 * 8, 8 * 4 * 8, 4 * 8, 8]
 *
 * example2:
 * in_accum_reverse = [2 * 8 * 32, 8 * 32, 32]
 * expand_accum_reverse = [2 * 4 * 8, 4 * 8, 8]
 * out_accum_reverse = [2 * 4 * 2 * 4 * 8, 4 * 2 * 4 * 8, 2 * 4 * 8, 4 * 8, 8]
 */
Status ExpandAccumulateProduct(const Shape &in_accum_reverse, const Shape &expand_accum_reverse,
                               Shape *out_accum_reverse);

/*
 * given a shape in = [din_n-1, din_n-2, ..., d_0], and the expand shape expand= [dexp_m-1, dexp_m-2, ..., dexp_0],
 * compute the expended shape out = [dout_k-1, dout_k-2, ..., dout_0],
 * s.t. dout_k-1 * dout_k-2 * ...* dout_0 = din_n-1 * din_n-2 * ... * d_0
 * suppose in_accum_reverse is the ShapeToAccumulateProductReverse result of in,
 * expand_accum_reverse is the ShapeToAccumulateProductReverse result of expand,
 * out_accum_reverse is the ShapeToAccumulateProductReverse result of out,
 * then out_accum_reverse is the union of in_accum_reverse and expand_accum_reverse
 * (out_accum_reverse = in_accum_reverse U expand_accum_reverse)
 *
 * example1:
 * in = [2, 8, 32]
 * expand = [16, 4, 8]
 * out = [2, 8, 4, 8]
 *
 * example2:
 * in = [2, 8, 32]
 * expand = [2, 4, 8]
 * out = [2, 4, 2, 4, 8]
 */
Status ExpandShape(const Shape &in, const Shape &expand, Shape *out);
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_TENSOR_LAYOUT_SHAPE_UTIL_H_
