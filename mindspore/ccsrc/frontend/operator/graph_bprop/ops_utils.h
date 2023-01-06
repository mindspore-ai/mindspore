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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPERATOR_GRAPH_BPROP_OPS_UTILS_H_
#define MINDSPORE_CCSRC_FRONTEND_OPERATOR_GRAPH_BPROP_OPS_UTILS_H_

#include <string>
#include <vector>
#include "frontend/operator/graph_bprop/utils.h"
#include "ops/core_ops.h"

namespace mindspore {
namespace graph_bprop {
// Ops.
AnfNodePtr Add();
AnfNodePtr Mul(const FuncGraphPtr &fg);
AnfNodePtr Mod();
AnfNodePtr MatMul(const FuncGraphPtr &fg, bool transpose_a = false, bool transpose_b = false);
AnfNodePtr Conj();
AnfNodePtr ReluGrad();
AnfNodePtr GeLUGrad();
AnfNodePtr MakeTuple();
AnfNodePtr TensorShape();
AnfNodePtr Shape();
AnfNodePtr RowTensorGetValues();
AnfNodePtr RowTensorGetIndices();
AnfNodePtr RowTensorGetDenseShape();
AnfNodePtr MakeRowTensor();
AnfNodePtr Cast(const FuncGraphPtr &fg);
AnfNodePtr ReduceProd(const FuncGraphPtr &fg);
AnfNodePtr ExpandDims(const FuncGraphPtr &fg);
AnfNodePtr Range(const FuncGraphPtr &fg);
AnfNodePtr TensorScatterUpdate(const FuncGraphPtr &fg);
AnfNodePtr InvertPermutation(const FuncGraphPtr &fg);
AnfNodePtr Transpose(const FuncGraphPtr &fg);
AnfNodePtr ZerosLike();
AnfNodePtr Neg();
AnfNodePtr Conv2DBackpropInput(const FuncGraphPtr &fg, const PrimitivePtr &primal);
AnfNodePtr Conv2DBackpropFilter(const FuncGraphPtr &fg, const PrimitivePtr &primal);
AnfNodePtr LayerNormGrad(const FuncGraphPtr &fg, const ValuePtr &begin_norm_axis, const ValuePtr &begin_params_axis);
AnfNodePtr ReduceSum(const FuncGraphPtr &fg, bool keep_dims = false, bool skip_mode = false);
AnfNodePtr Reshape(const FuncGraphPtr &fg);
AnfNodePtr DynamicBroadcastGradientArgs();
AnfNodePtr MaxPoolGrad(const FuncGraphPtr &fg, const PrimitivePtr &primal);
AnfNodePtr BatchNormGrad(const FuncGraphPtr &fg, const PrimitivePtr &primal);
AnfNodePtr DType();
AnfNodePtr BiasAddGrad(const string &format);

// Common methods.
AnfNodePtr ZerosLikeFunction(const FuncGraphPtr &fg, const AnfNodePtr &input);
AnfNodePtr GetAttr(const FuncGraphPtr &fg, const AnfNodePtr &node, const std::string &attr);
AnfNodePtr TupleGetItem(const FuncGraphPtr &fg, const AnfNodePtr &output, int64_t idx);
AnfNodePtr NewNode(const FuncGraphPtr &fg, const std::vector<AnfNodePtr> &inputs, bool need_infer = false,
                   bool infer_value = false);
AnfNodePtr DynSize(const FuncGraphPtr &fg, const AnfNodePtr &node, const TypePtr &dtype = kInt64);
AnfNodePtr DynInvertPermutation(const FuncGraphPtr &fg, const AnfNodePtr &perm);
AnfNodePtr DynBinopGradCommon(const FuncGraphPtr &fg, const AnfNodePtr &x, const AnfNodePtr &y, const AnfNodePtr &dx,
                              const AnfNodePtr &dy);
AnfNodePtr BinopGradCommon(const FuncGraphPtr &fg, const AnfNodePtr &x, const AnfNodePtr &y, const AnfNodePtr &dx,
                           const AnfNodePtr &dy);
AnfNodePtr ReduceSumWithCast(const FuncGraphPtr &fg, const AnfNodePtr &dx, const ShapeVector &axis);
AnfNodePtr SumGradReduceAxis(const FuncGraphPtr &fg, const AnfNodePtr &x, const AnfNodePtr &rx, bool keep_dims = false);
bool IsSequenceValueUnknown(const FuncGraphPtr &fg, const AnfNodePtr &shape_node);
AnfNodePtr BroadcastGradientArgs(const FuncGraphPtr &fg, const AnfNodePtr &x_shape_node,
                                 const AnfNodePtr &y_shape_node);
ValuePtr GetPadModStr(const ValuePtr &value, bool upper = false);
}  // namespace graph_bprop
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPERATOR_GRAPH_BPROP_OPS_UTILS_H_
