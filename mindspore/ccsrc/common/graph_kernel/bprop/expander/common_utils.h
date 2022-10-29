/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_BPROP_EXPANDER_COMMON_UTILS_H_
#define MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_BPROP_EXPANDER_COMMON_UTILS_H_

#include <vector>
#include <utility>
#include "common/graph_kernel/bprop/expander/node.h"
#include "common/graph_kernel/bprop/bprop_irbuilder.h"

namespace mindspore::expander::bprop {
std::vector<std::vector<int64_t>> BroadcastGradientArgs(const std::vector<int64_t> &x_shape,
                                                        const std::vector<int64_t> &y_shape);

std::vector<int64_t> GetTransposeAxis(const std::vector<int64_t> &x_shape, int64_t axis);

std::vector<int64_t> TupleDiv(const std::vector<int64_t> &x, const std::vector<int64_t> &y);

std::vector<int64_t> ReduceShape(const std::vector<int64_t> &x, const std::vector<int64_t> &axis);

std::vector<int64_t> GetAxisList(const ValuePtr &value);

NodePtrList BinopGradCommon(const BpropIRBuilder *ib, const NodePtr &x, const NodePtr &y, const NodePtr &dx,
                            const NodePtr &dy);

NodePtrList BinopGradCommonWithShift(const BpropIRBuilder *ib, const NodePtr &x, const NodePtr &y, const NodePtr &dx,
                                     const NodePtr &dy, size_t shift);

std::vector<int64_t> Range(int64_t start, int64_t stop, int64_t step = 1);
std::vector<int64_t> Range(int64_t stop);

template <typename T>
std::vector<T> operator+(std::vector<T> const &m, std::vector<T> const &n) {
  std::vector<T> v;                       // initialized vector v
  v.reserve(m.size() + n.size());         // reverse function used in v
  v.insert(v.end(), m.begin(), m.end());  // insert func used in vec m.
  v.insert(v.end(), n.begin(), n.end());  // insert func used in vec n.
  return v;                               // return the vector v
}

NodePtr GetEps(const BpropIRBuilder *ib, const TypePtr &type);
NodePtrList BinopGatherCommon(const BpropIRBuilder *ib);
std::vector<int64_t> GenerateInverseIndex(const std::vector<int64_t> &x_shp, int64_t axis_v);
std::vector<int64_t> GenerateShapeIndex(const std::vector<int64_t> &out_shp, const std::vector<int64_t> &ind_shp,
                                        int64_t axis_v);
std::vector<int64_t> RegenerateOutputShape(const std::vector<int64_t> &x_shp, const std::vector<int64_t> &ind_shp,
                                           int64_t axis_v);
std::vector<int64_t> GetTupleIntFromValueNode(const NodePtr &node);
int64_t GetIntFromValueNode(const NodePtr &node);
std::vector<int64_t> TileShape(const std::vector<int64_t> &multiples, const std::vector<int64_t> &shapex);
std::vector<int64_t> InvertPermutation(const std::vector<int64_t> &perm);
std::vector<int64_t> GetTransposition(int64_t axis, int64_t rank);

NodePtr SumGrad(const BpropIRBuilder *ib, const NodePtr &x, const std::vector<int64_t> &axis, const NodePtr &dout);
NodePtr MinOrMaxGrad(const BpropIRBuilder *ib, const NodePtr &x, const std::vector<int64_t> &axis, const NodePtr &out,
                     const NodePtr &dout);
std::pair<ShapeVector, ShapeVector> SplitShapeIndex(const ShapeVector &input_shape, ShapeVector axis);
ShapeVector GetAxisValue(const NodePtr &axis);
NodePtr ArgminOrArgmaxGrad(const BpropIRBuilder *ib, const NodePtr &x, const int64_t &axis, const bool &keep_dims,
                           const NodePtr &out, const NodePtr &dout, const bool is_max);
TypeId PromoteBinaryDtype(TypeId t1, TypeId t2);
}  // namespace mindspore::expander::bprop
#endif  // MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_BPROP_EXPANDER_COMMON_UTILS_H_
