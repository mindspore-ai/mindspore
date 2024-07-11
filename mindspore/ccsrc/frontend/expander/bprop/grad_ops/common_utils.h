/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_FRONTEND_EXPANDER_BPROP_GRAD_OPS_COMMON_UTILS_H_
#define MINDSPORE_CCSRC_FRONTEND_EXPANDER_BPROP_GRAD_OPS_COMMON_UTILS_H_

#include <cmath>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "mindspore/core/ops/dynamic_broadcast_gradient_args.h"
#include "frontend/expander/bprop/bprop_irbuilder.h"
#include "include/common/expander/core/node.h"

namespace mindspore::expander::bprop {
constexpr size_t i0 = 0;
constexpr size_t i1 = 1;
constexpr size_t i2 = 2;
constexpr size_t i3 = 3;
constexpr size_t i4 = 4;
constexpr size_t i5 = 5;
constexpr size_t i6 = 6;
constexpr size_t i7 = 7;
constexpr size_t i8 = 8;
constexpr size_t i9 = 9;
constexpr size_t i10 = 10;
inline const auto pi = std::acos(-1.0);
inline const auto log_2 = std::log(2.0);
inline const auto log_pi = std::log(pi);

using mindspore::ops::BroadcastGradientArgsInferValue;

NodePtrList ReturnZeros(BpropBuilder *ib);
// normalize the axis to [0, rank)
int64_t NormalizeAxis(int64_t axis, size_t rank);

std::vector<int64_t> GetTransposeAxis(const std::vector<int64_t> &x_shape, int64_t axis);

std::vector<int64_t> TupleDiv(const std::vector<int64_t> &x, const std::vector<int64_t> &y);

std::vector<int64_t> ReduceShape(const std::vector<int64_t> &x, const std::vector<int64_t> &axis,
                                 bool skip_mode = false);

int64_t CheckRange(int64_t idx, int64_t dim_size);

NodePtrList BinopGradCommon(BpropBuilder *ib, const NodePtr &x, const NodePtr &y, const NodePtr &dx, const NodePtr &dy,
                            size_t shift = 0UL);
NodePtrList MatMulExtBroadCastGrad(BpropBuilder *ib, const NodePtr &x, const NodePtr &y, const NodePtr &dx,
                                   const NodePtr &dy, size_t ignore_offset = 0UL);

std::vector<int64_t> Range(int64_t start, int64_t stop, int64_t step = 1);
std::vector<int64_t> Range(int64_t stop);

template <typename T>
std::vector<T> operator+(std::vector<T> const &m, std::vector<T> const &n) {
  std::vector<T> v;                             // initialized vector v
  v.reserve(m.size() + n.size());               // reverse function used in v
  (void)v.insert(v.end(), m.begin(), m.end());  // insert func used in vec m.
  (void)v.insert(v.end(), n.begin(), n.end());  // insert func used in vec n.
  return v;                                     // return the vector v
}

int64_t GetIntValue(const NodePtr &node);
std::vector<int64_t> GetIntList(const ValuePtr &value);
std::vector<int64_t> GetIntList(const NodePtr &node);

NodePtr GetEps(BpropBuilder *ib, const TypePtr &type);
std::vector<int64_t> GenerateInverseIndex(const std::vector<int64_t> &x_shp, int64_t axis_v, int64_t batch_dims = 0);
std::vector<int64_t> GenerateShapeIndex(const std::vector<int64_t> &out_shp, const std::vector<int64_t> &ind_shp,
                                        int64_t axis_v, int64_t batch_dims = 0);
std::vector<int64_t> RegenerateOutputShape(const std::vector<int64_t> &x_shp, const std::vector<int64_t> &ind_shp,
                                           int64_t axis_v, int64_t batch_dims = 0);
std::vector<int64_t> InvertPermutation(const std::vector<int64_t> &perm);
std::vector<int64_t> GetTransposition(int64_t axis, int64_t rank);

NodePtr SumGrad(Emitter *ib, const NodePtr &x, const NodePtr &axis, const NodePtr &dout, bool keep_dims = false,
                bool skip_mode = false);
NodePtr MinOrMaxGrad(BpropBuilder *ib, const NodePtr &x, const NodePtr &axis, const NodePtr &keep_dims,
                     const NodePtr &out, const NodePtr &dout);
std::pair<ShapeVector, ShapeVector> SplitShapeIndex(const ShapeVector &input_shape, const ShapeVector &axis);
NodePtr ArgminOrArgmaxGrad(BpropBuilder *ib, const NodePtr &x, const NodePtr &axis, const NodePtr &keep_dims,
                           const NodePtr &out, const NodePtr &dout, const bool is_max);
NodePtr MeidanDimGrad(BpropBuilder *ib, const NodePtr &x, const NodePtr &axis, const NodePtr &keep_dims,
                      const NodePtr &out, const NodePtr &dout);
inline NodePtr ReduceCommonOpGrad(BpropBuilder *ib, const NodePtr &x, const NodePtr &axis, const NodePtr &keep_dims,
                                  const NodePtr &out, const NodePtr &dout, int64_t dout_index, int64_t indices_index);
TypeId PromoteBinaryDtype(TypeId t1, TypeId t2);
NodePtr LGamma(BpropBuilder *ib, const NodePtr &x);
bool CheckType(const TypePtr &check_type, const std::set<TypePtr> &template_types);
ShapeVector PoolToNHWC(const ShapeVector &v);
ShapeVector ConvToNHWC(const ShapeVector &v);
ShapeVector GetShapeByRange(const ShapeVector &v, int64_t begin = 0, int64_t end = -1);
NodePtr MatrixTranspose(BpropBuilder *ib, const NodePtr &x);
NodePtr MatrixTransposeExt(BpropBuilder *ib, const NodePtr &x);
NodePtr Adjoint(BpropBuilder *ib, const NodePtr &x);
}  // namespace mindspore::expander::bprop
#endif  // MINDSPORE_CCSRC_FRONTEND_EXPANDER_BPROP_GRAD_OPS_COMMON_UTILS_H_
