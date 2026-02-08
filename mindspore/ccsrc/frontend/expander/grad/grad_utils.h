/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_FRONTEND_EXPANDER_GRAD_GRAD_UTILS_H_
#define MINDSPORE_CCSRC_FRONTEND_EXPANDER_GRAD_GRAD_UTILS_H_

#include <cmath>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "frontend/expander/bprop/common_utils.h"
#include "mindspore/ops/infer/dynamic_broadcast_gradient_args.h"
#include "frontend/expander/bprop/bprop_irbuilder.h"
#include "mindspore/ccsrc/include/utils/expander/node.h"

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
constexpr size_t i11 = 11;
constexpr size_t i12 = 12;
constexpr size_t i13 = 13;
constexpr size_t i14 = 14;
constexpr size_t i15 = 15;
constexpr size_t i16 = 16;
constexpr size_t i17 = 17;
constexpr size_t i18 = 18;
constexpr size_t i19 = 19;
constexpr size_t i20 = 20;
constexpr size_t i21 = 21;
constexpr size_t i22 = 22;
constexpr size_t i23 = 23;
constexpr size_t i24 = 24;
constexpr size_t i25 = 25;
constexpr size_t i26 = 26;
constexpr size_t i27 = 27;
constexpr size_t i28 = 28;
constexpr size_t i29 = 29;
constexpr size_t i30 = 30;
constexpr size_t i31 = 31;
constexpr size_t i32 = 32;
constexpr size_t i33 = 33;
constexpr size_t i34 = 34;
constexpr size_t i35 = 35;

inline const auto pi = std::acos(-1.0);
inline const auto log_2 = std::log(2.0);
inline const auto log_pi = std::log(pi);

NodePtrList ReturnZeros(BpropBuilder *ib);
// normalize the axis to [0, rank)
int64_t NormalizeAxis(int64_t axis, size_t rank);

std::vector<int64_t> ReduceShapeTupleDiv(const std::vector<int64_t> &x, const std::vector<int64_t> &y);

std::vector<int64_t> ReduceShape(const std::vector<int64_t> &x, const std::vector<int64_t> &axis,
                                 bool skip_mode = false);

int64_t CheckRange(int64_t idx, int64_t dim_size);

NodePtrList BinopGradCommon(BpropBuilder *ib, const NodePtr &x, const NodePtr &y, const NodePtr &dx, const NodePtr &dy,
                            size_t shift = 0UL);

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
NodePtrList GetUnsqueezeTensor(Emitter *ib, const NodePtr &input, const NodePtr &axis, bool keep_dims,
                               const NodePtrList &outputs);
NodePtr LogSumExpGrad(Emitter *ib, const NodePtr &input, const NodePtr &dim, bool keepdim, const NodePtr &out,
                      const NodePtr &dout);
NodePtr InplacePutGrad(Emitter *ib, const NodePtr &index, const NodePtr &source, bool accumulate, const NodePtr &dout,
                       const NodePtr &type);
NodePtr VarGrad(BpropBuilder *ib, const NodePtr &x, const NodePtr &axis_node, const NodePtr &dout,
                const NodePtr &correction, const NodePtr &keepdim);
NodePtr MinOrMaxGrad(Emitter *ib, const NodePtr &x, const NodePtr &axis, bool keep_dims, const NodePtr &out,
                     const NodePtr &dout);
std::pair<ShapeVector, ShapeVector> SplitShapeIndex(const ShapeVector &input_shape, const ShapeVector &axis);
NodePtr ArgminOrArgmaxGrad(BpropBuilder *ib, const NodePtr &x, const NodePtr &axis, const NodePtr &keep_dims,
                           const NodePtr &out, const NodePtr &dout, const bool is_max,
                           const bool is_minmax_dim = false);
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
NodePtr VectorNormGrad(BpropBuilder *ib, const NodePtr &input_node, const NodePtr &p, const NodePtr &dim_node,
                       const NodePtr &keepdim, const NodePtr &out_node, const NodePtr &dout_node);
std::optional<float> GetAlpha(const NodePtr &alpha);
}  // namespace mindspore::expander::bprop
#endif  // MINDSPORE_CCSRC_FRONTEND_EXPANDER_GRAD_GRAD_UTILS_H_
