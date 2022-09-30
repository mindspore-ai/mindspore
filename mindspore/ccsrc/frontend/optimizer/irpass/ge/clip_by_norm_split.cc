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

#include "frontend/optimizer/irpass/ge/clip_by_norm_split.h"

#include <string>
#include <vector>
#include <memory>
#include <algorithm>

#include "pybind_api/ir/tensor_py.h"
#include "pipeline/pynative/base.h"
#include "pipeline/jit/static_analysis/prim.h"
#include "include/common/utils/python_adapter.h"
#include "mindspore/core/mindapi/ir/common.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"

namespace mindspore {
namespace opt {
namespace irpass {
namespace {
constexpr char kOpsFunctionName[] = "mindspore.ops";

ShapeVector GetOutputInferShape(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto shape = common::AnfAlgo::GetOutputInferShape(node, 0);
  return shape;
}

void CheckCNodeInputSize(const CNodePtr &cnode, size_t input_tensor_size) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto real_input_tensor_num = common::AnfAlgo::GetInputTensorNum(cnode);
  if (real_input_tensor_num != input_tensor_size) {
    MS_LOG(EXCEPTION) << "The input tensor size[" << real_input_tensor_num
                      << "] of node [" + cnode->DebugString() + "] is not equal to " << input_tensor_size
                      << trace::DumpSourceLines(cnode);
  }
}

std::vector<int64_t> InferBroadcastShape(const std::vector<int64_t> &x_shape, const std::vector<int64_t> &y_shape,
                                         const std::string &op_name, const std::string &op_x_name,
                                         const std::string &op_y_name) {
  if (x_shape == y_shape) {
    return x_shape;
  }
  auto x_length = static_cast<int64_t>(x_shape.size());
  auto y_length = static_cast<int64_t>(y_shape.size());
  auto length = x_length < y_length ? x_length : y_length;
  std::vector<int64_t> broadcast_shape;
  if (x_length == length) {
    (void)std::copy(y_shape.begin(), y_shape.end() - length, std::back_inserter(broadcast_shape));
  } else {
    (void)std::copy(x_shape.begin(), x_shape.end() - length, std::back_inserter(broadcast_shape));
  }
  for (int64_t i = -length; i < 0; ++i) {
    if (x_shape[LongToSize(x_length + i)] == 1) {
      broadcast_shape.push_back(y_shape[LongToSize(y_length + i)]);
    } else if (y_shape[LongToSize(y_length + i)] == 1) {
      broadcast_shape.push_back(x_shape[LongToSize(x_length + i)]);
    } else if (x_shape[x_length + i] == y_shape[LongToSize(y_length + i)]) {
      broadcast_shape.push_back(x_shape[LongToSize(x_length + i)]);
    } else if ((x_shape[x_length + i] == abstract::Shape::kShapeDimAny) ||
               (y_shape[y_length + i] == abstract::Shape::kShapeDimAny)) {
      MS_EXCEPTION(ValueError) << "For '" << op_name << "', input dynamic shape args is not supported.";
    } else {
      MS_EXCEPTION(ValueError) << "For '" << op_name << "', the two input '" << op_x_name << "' and '" << op_y_name
                               << "' can not broadcast";
    }
  }
  return broadcast_shape;
}

PrimitivePtr GetPrimitiveFromPyAdapter(const py::object &prim_obj) {
  const auto &adapter = py::cast<PrimitivePyAdapterPtr>(prim_obj);
  MS_EXCEPTION_IF_NULL(adapter);
  auto attached_prim = adapter->attached_primitive();
  if (attached_prim == nullptr) {
    attached_prim = std::make_shared<PrimitivePy>(prim_obj, adapter);
    adapter->set_attached_primitive(attached_prim);
  }
  return attached_prim->cast<PrimitivePtr>();
}

AnfNodePtr CreateSquareNode(const FuncGraphPtr &func_graph, const AnfNodePtr &inp, const ShapeVector &shape_vec,
                            const TypeId &type_id) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto op_obj = python_adapter::GetPyFn(kOpsFunctionName, kSquareOpName)();
  auto op_prim = GetPrimitiveFromPyAdapter(op_obj);
  auto square = func_graph->NewCNode(op_prim, {inp});
  MS_EXCEPTION_IF_NULL(square);
  auto abs = std::make_shared<abstract::AbstractTensor>(TypeIdToType(type_id), shape_vec);
  square->set_abstract(abs);
  return square;
}

ValueNodePtr CreateValueNode(const ValuePtr &value_ptr) {
  MS_EXCEPTION_IF_NULL(value_ptr);
  auto new_node = std::make_shared<ValueNode>(value_ptr);
  MS_EXCEPTION_IF_NULL(new_node);
  auto value_abstract = value_ptr->ToAbstract();
  new_node->set_abstract(value_abstract);
  return new_node;
}

AnfNodePtr CreateReduceSumNode(const FuncGraphPtr &func_graph, const AnfNodePtr &square, const AnfNodePtr &clip_by_norm,
                               const ShapeVector &shape_vec, const TypeId &type_id) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto op_obj = python_adapter::GetPyFn(kOpsFunctionName, kReduceSumOpName)();
  auto op_prim = GetPrimitiveFromPyAdapter(op_obj);
  // Sync the attribute of `ClipByNorm` to `ReduceSum`
  auto clip_by_norm_node = clip_by_norm->cast<CNodePtr>();
  auto clip_by_norm_prim = GetValueNode<PrimitivePtr>(clip_by_norm_node->input(0));
  MS_EXCEPTION_IF_NULL(clip_by_norm_prim);
  auto axis_value = clip_by_norm_prim->GetAttr(kAttrAxis);
  MS_EXCEPTION_IF_NULL(axis_value);
  // Get `axis` vector
  const auto dim = shape_vec.size();
  std::vector<int64_t> axis;
  if (axis_value->isa<ValueSequence>()) {
    axis = GetValue<std::vector<int64_t>>(axis_value);
    if (axis.empty()) {  // reduce_sum for all dimensions
      for (size_t i = 0; i < dim; ++i) {
        (void)axis.emplace_back(i);
      }
    }
  } else if (axis_value->isa<Int64Imm>()) {
    (void)axis.emplace_back(GetValue<int64_t>(axis_value));
  } else {
    MS_EXCEPTION(TypeError) << "For `" << prim::kPrimClipByNorm->name()
                            << "`, the type of attribute `axis` is invalid.";
  }
  // Set abstract to `reduce_sum` op
  int64_t ddim = SizeToLong(dim);
  ShapeVector reduce_sum_output_shape = shape_vec;
  for (const auto &idx : axis) {
    if (idx < -ddim || idx >= ddim) {
      MS_EXCEPTION(ValueError) << "The range of axis value should in [" << -ddim << ", " << ddim
                               << "), but got: " << idx;
    }
    auto positive_idx = idx < 0 ? idx + ddim : idx;
    reduce_sum_output_shape[positive_idx] = 1;
  }
  auto abs = std::make_shared<abstract::AbstractTensor>(TypeIdToType(type_id), reduce_sum_output_shape);
  op_prim->set_attr(kAttrKeepDims, MakeValue<bool>(true));
  auto axis_node = CreateValueNode(MakeValue(axis));
  auto reduce_sum = func_graph->NewCNode(op_prim, {square, axis_node});
  MS_EXCEPTION_IF_NULL(reduce_sum);
  reduce_sum->set_abstract(abs);
  return reduce_sum;
}

AnfNodePtr CreateConstantNode(const FuncGraphPtr &func_graph, const ShapeVector &shape_vec, const TypeId &type_id,
                              const std::string &op_name) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto tensor = std::make_shared<tensor::Tensor>(type_id, shape_vec);
  auto value_node = CreateValueNode(tensor);
  auto op_obj = python_adapter::GetPyFn(kOpsFunctionName, op_name)();
  auto op_prim = GetPrimitiveFromPyAdapter(op_obj);
  auto constant_node = func_graph->NewCNode(op_prim, {value_node});
  MS_EXCEPTION_IF_NULL(constant_node);
  auto abs = std::make_shared<abstract::AbstractTensor>(TypeIdToType(type_id), shape_vec);
  constant_node->set_abstract(abs);
  return constant_node;
}

AnfNodePtr CreateGreaterNode(const FuncGraphPtr &func_graph, const AnfNodePtr &inp_a, const AnfNodePtr &inp_b,
                             const ShapeVector &shape_vec) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto op_obj = python_adapter::GetPyFn(kOpsFunctionName, kGreaterOpName)();
  auto op_prim = GetPrimitiveFromPyAdapter(op_obj);
  auto greater = func_graph->NewCNode(op_prim, {inp_a, inp_b});
  MS_EXCEPTION_IF_NULL(greater);
  auto abs = std::make_shared<abstract::AbstractTensor>(kBool, shape_vec);
  greater->set_abstract(abs);
  return greater;
}

AnfNodePtr CreateSelectNode(const FuncGraphPtr &func_graph, const AnfNodePtr &cond, const AnfNodePtr &inp_a,
                            const AnfNodePtr &inp_b, const ShapeVector &shape_vec, const TypeId &type_id) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto op_obj = python_adapter::GetPyFn(kOpsFunctionName, kSelectOpName)();
  auto op_prim = GetPrimitiveFromPyAdapter(op_obj);
  auto select = func_graph->NewCNode(op_prim, {cond, inp_a, inp_b});
  MS_EXCEPTION_IF_NULL(select);
  auto abs = std::make_shared<abstract::AbstractTensor>(TypeIdToType(type_id), shape_vec);
  select->set_abstract(abs);
  return select;
}

AnfNodePtr CreateSqrtNode(const FuncGraphPtr &func_graph, const AnfNodePtr &reduce_sum, const TypeId &type_id) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto op_obj = python_adapter::GetPyFn(kOpsFunctionName, kSqrtOpName)();
  auto op_prim = GetPrimitiveFromPyAdapter(op_obj);
  auto sqrt = func_graph->NewCNode(op_prim, {reduce_sum});
  MS_EXCEPTION_IF_NULL(sqrt);
  auto abs = std::make_shared<abstract::AbstractTensor>(TypeIdToType(type_id), GetOutputInferShape(reduce_sum));
  sqrt->set_abstract(abs);
  return sqrt;
}

AnfNodePtr CreateMaxNode(const FuncGraphPtr &func_graph, const AnfNodePtr &x, const AnfNodePtr &y,
                         const TypeId &type_id) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto op_obj = python_adapter::GetPyFn(kOpsFunctionName, kMaximumOpName)();
  auto op_prim = GetPrimitiveFromPyAdapter(op_obj);
  auto max = func_graph->NewCNode(op_prim, {x, y});
  MS_EXCEPTION_IF_NULL(max);
  auto x_shape = GetOutputInferShape(x);
  auto y_shape = GetOutputInferShape(y);
  auto output_shape = InferBroadcastShape(x_shape, y_shape, "ClipByNorm", "clip_norm_cast", "l2_norm");
  auto abs = std::make_shared<abstract::AbstractTensor>(TypeIdToType(type_id), output_shape);
  max->set_abstract(abs);
  return max;
}

AnfNodePtr CreateMulNode(const FuncGraphPtr &func_graph, const AnfNodePtr &x, const AnfNodePtr &clip_norm,
                         const ShapeVector &shape_vec, const TypeId &type_id) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto op_obj = python_adapter::GetPyFn(kOpsFunctionName, kMulOpName)();
  auto op_prim = GetPrimitiveFromPyAdapter(op_obj);
  auto mul = func_graph->NewCNode(op_prim, {x, clip_norm});
  MS_EXCEPTION_IF_NULL(mul);
  auto output_shape = shape_vec;
  auto clip_norm_shape = GetOutputInferShape(clip_norm);
  if (clip_norm_shape.size() > output_shape.size()) {
    output_shape = clip_norm_shape;
  }
  auto abs = std::make_shared<abstract::AbstractTensor>(TypeIdToType(type_id), output_shape);
  mul->set_abstract(abs);
  return mul;
}

AnfNodePtr CreateDivNode(const FuncGraphPtr &func_graph, const AnfNodePtr &dividend, const AnfNodePtr &divisor,
                         const ShapeVector &shape_vec, const TypeId &type_id) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto op_obj = python_adapter::GetPyFn(kOpsFunctionName, kDivOpName)();
  auto op_prim = GetPrimitiveFromPyAdapter(op_obj);
  auto div = func_graph->NewCNode(op_prim, {dividend, divisor});
  MS_EXCEPTION_IF_NULL(div);
  auto abs = std::make_shared<abstract::AbstractTensor>(TypeIdToType(type_id), shape_vec);
  div->set_abstract(abs);
  return div;
}

AnfNodePtr CreateCastNode(const FuncGraphPtr &func_graph, const AnfNodePtr &inp, const ShapeVector &shape_vec,
                          const TypeId &src_type_id, const TypeId &dst_type_id) {
  MS_EXCEPTION_IF_NULL(func_graph);
  if (src_type_id == dst_type_id) {
    return inp;
  }
  auto op_obj = python_adapter::GetPyFn(kOpsFunctionName, kCastOpName)();
  auto cast_prim = GetPrimitiveFromPyAdapter(op_obj);
  if (dst_type_id == kNumberTypeFloat16) {
    cast_prim->set_attr(kAttrDstType, kFloat16);
  } else if (dst_type_id == kNumberTypeFloat32) {
    cast_prim->set_attr(kAttrDstType, kFloat32);
  } else {
    MS_EXCEPTION(TypeError) << "For '" << prim::kPrimClipByNorm->name()
                            << "`, the data type of input args only supports float16 or float32.";
  }
  cast_prim->set_attr(kIsBackendCast, MakeValue(true));
  auto abs = std::make_shared<abstract::AbstractTensor>(TypeIdToType(dst_type_id), shape_vec);
  auto cast = func_graph->NewCNode(cast_prim, {inp});
  MS_EXCEPTION_IF_NULL(cast);
  cast->set_abstract(abs);
  return cast;
}

void SetScopeForNewNodes(const std::vector<AnfNodePtr> &nodes, const AnfNodePtr &clip_by_norm_node) {
  auto clip_by_norm_scope = clip_by_norm_node->scope();
  for (auto node : nodes) {
    node->set_scope(clip_by_norm_scope);
  }
}

const AnfNodePtr ProcessClipByNormSplit(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  // Get `ClipByNorm` cnode
  MS_EXCEPTION_IF_NULL(node);
  auto clip_by_norm = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(clip_by_norm);
  constexpr size_t clip_by_norm_inp_num = 2;
  CheckCNodeInputSize(clip_by_norm, clip_by_norm_inp_num);
  // Get input node `x` and `clip_norm`)
  const auto &inp_x = clip_by_norm->input(1);
  constexpr size_t clip_norm_inp_idx = 2;
  const auto &inp_clip_norm = clip_by_norm->input(clip_norm_inp_idx);
  // Get abstract info
  TypeId dst_type_id = kNumberTypeFloat32;
  auto shape_vec = GetOutputInferShape(clip_by_norm);
  auto x_type_id = common::AnfAlgo::GetPrevNodeOutputInferDataType(clip_by_norm, 0);
  // Create `op1 = square(x)` op
  auto square = CreateSquareNode(func_graph, inp_x, shape_vec, x_type_id);
  // Create `op2 = reduce_sum(op1)` op
  auto reduce_sum = CreateReduceSumNode(func_graph, square, clip_by_norm, shape_vec, x_type_id);
  ShapeVector reduce_sum_output_shape = GetOutputInferShape(reduce_sum);
  // Create `op3 = cast(op2)` to float32 data type
  auto reduce_sum_cast = CreateCastNode(func_graph, reduce_sum, reduce_sum_output_shape, x_type_id, dst_type_id);
  // Create `op4 = greater(op3, zeros)` op
  auto zeros_node = CreateConstantNode(func_graph, reduce_sum_output_shape, dst_type_id, prim::kPrimZerosLike->name());
  auto greater = CreateGreaterNode(func_graph, reduce_sum_cast, zeros_node, reduce_sum_output_shape);
  // Create `op5 = select(op4, op3, Ones)` op
  auto ones_node = CreateConstantNode(func_graph, reduce_sum_output_shape, dst_type_id, prim::kPrimOnesLike->name());
  auto safe_reduce_sum_cast =
    CreateSelectNode(func_graph, greater, reduce_sum_cast, ones_node, reduce_sum_output_shape, dst_type_id);
  // Create `op6 = sqrt(op5)` op
  auto sqrt = CreateSqrtNode(func_graph, safe_reduce_sum_cast, dst_type_id);
  // Create `op7 = select(op4, op6, op3)` op
  auto safe_sqrt = CreateSelectNode(func_graph, greater, sqrt, reduce_sum_cast, reduce_sum_output_shape, dst_type_id);
  // Create 'op8 = x * clip_norm' op
  auto inp_x_cast = CreateCastNode(func_graph, inp_x, shape_vec, x_type_id, dst_type_id);
  auto clip_norm_cast = CreateCastNode(func_graph, inp_clip_norm, GetOutputInferShape(inp_clip_norm),
                                       common::AnfAlgo::GetOutputInferDataType(inp_clip_norm, 0), dst_type_id);
  auto mul = CreateMulNode(func_graph, inp_x_cast, clip_norm_cast, shape_vec, dst_type_id);
  // Create `op9 = max(op8, op7)` op
  auto max = CreateMaxNode(func_graph, clip_norm_cast, safe_sqrt, dst_type_id);
  // Create 'op10 = op8 / op9' op
  auto div = CreateDivNode(func_graph, mul, max, shape_vec, dst_type_id);
  std::vector<AnfNodePtr> new_nodes{
    square, reduce_sum, reduce_sum_cast, zeros_node,     greater, ones_node, safe_reduce_sum_cast,
    sqrt,   safe_sqrt,  inp_x_cast,      clip_norm_cast, mul,     max,       div};
  SetScopeForNewNodes(new_nodes, node);
  return div;
}
}  // namespace

AnfNodePtr ClipByNormForGE::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  Reset();
  FuncGraphPtr fg = node->func_graph();
  if (fg != nullptr && IsPrimitiveCNode(node, prim::kPrimClipByNorm)) {
    return ProcessClipByNormSplit(fg, node);
  }
  return nullptr;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
