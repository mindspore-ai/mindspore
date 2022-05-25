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

#include "plugin/device/ascend/optimizer/ir_fission/clip_by_norm_fission.h"
#include <algorithm>
#include "ir/anf.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/optimizer/helper.h"

namespace mindspore {
namespace opt {
namespace {
ShapeVector GetOutputInferShape(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto shape = common::AnfAlgo::GetOutputInferShape(node, 0);
  ShapeVector infer_shape;
  std::transform(shape.begin(), shape.end(), std::back_inserter(infer_shape), SizeToLong);
  return infer_shape;
}

std::vector<int64_t> InferBroadcastShape(std::vector<int64_t> x_shape, std::vector<int64_t> y_shape,
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
    } else if ((x_shape[x_length + i] == abstract::Shape::SHP_ANY) ||
               (y_shape[y_length + i] == abstract::Shape::SHP_ANY)) {
      MS_EXCEPTION(ValueError) << "For '" << op_name << "', input dynamic shape args is not supported.";
    } else {
      MS_EXCEPTION(ValueError) << "For '" << op_name << "', the two input '" << op_x_name << "' and '" << op_y_name
                               << "' can not broadcast";
    }
  }
  return broadcast_shape;
}
}  // namespace

AnfNodePtr ClipByNormSplit::CreateCNodeBase(const FuncGraphPtr &func_graph, const std::vector<AnfNodePtr> &inps,
                                            const std::string &op_name, const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> new_node_inputs = {NewValueNode(std::make_shared<Primitive>(op_name))};
  for (const auto &inp : inps) {
    (void)new_node_inputs.emplace_back(inp);
  }
  auto new_node = NewCNode(new_node_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(new_node);

  new_node->set_scope(node->scope());
  new_node->set_kernel_info(std::make_shared<device::KernelInfo>());
  return new_node;
}

AnfNodePtr ClipByNormSplit::CreateSquareNode(const FuncGraphPtr &func_graph, const AnfNodePtr &inp,
                                             const ShapeVector &shape_vec, const TypeId &type_id) const {
  auto square = CreateCNodeBase(func_graph, {inp}, kSquareOpName, inp);
  auto abs = std::make_shared<abstract::AbstractTensor>(TypeIdToType(type_id), shape_vec);
  square->set_abstract(abs);
  return square;
}

AnfNodePtr ClipByNormSplit::CreateReduceSumNode(const FuncGraphPtr &func_graph, const AnfNodePtr &square,
                                                const AnfNodePtr &clip_by_norm, const ShapeVector &shape_vec,
                                                const TypeId &type_id) const {
  auto reduce_sum = CreateCNodeBase(func_graph, {square}, kReduceSumOpName, square);
  // Sync the attribute of `ClipByNorm` to `ReduceSum`
  auto clip_by_norm_prim = common::AnfAlgo::GetCNodePrimitive(clip_by_norm);
  MS_EXCEPTION_IF_NULL(clip_by_norm_prim);
  auto axis_value = clip_by_norm_prim->GetAttr(kAttrAxis);
  MS_EXCEPTION_IF_NULL(axis_value);
  common::AnfAlgo::SetNodeAttr(kAttrAxis, axis_value, reduce_sum);
  common::AnfAlgo::SetNodeAttr(kAttrKeepDims, MakeValue(true), reduce_sum);
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
    axis.emplace_back(GetValue<int64_t>(axis_value));
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
  reduce_sum->set_abstract(abs);
  return reduce_sum;
}

AnfNodePtr ClipByNormSplit::CreateSqrtNode(const FuncGraphPtr &func_graph, const AnfNodePtr &reduce_sum,
                                           const TypeId &type_id) const {
  auto sqrt = CreateCNodeBase(func_graph, {reduce_sum}, kSqrtOpName, reduce_sum);
  auto abs = std::make_shared<abstract::AbstractTensor>(TypeIdToType(type_id), GetOutputInferShape(reduce_sum));
  sqrt->set_abstract(abs);
  return sqrt;
}

AnfNodePtr ClipByNormSplit::CreateMaxNode(const FuncGraphPtr &func_graph, const AnfNodePtr &x, const AnfNodePtr &y,
                                          const TypeId &type_id) const {
  auto max = CreateCNodeBase(func_graph, {x, y}, kMaximumOpName, y);
  auto x_shape = GetOutputInferShape(x);
  auto y_shape = GetOutputInferShape(y);
  auto output_shape = InferBroadcastShape(x_shape, y_shape, "ClipByNorm", "clip_norm_cast", "l2_norm");
  auto abs = std::make_shared<abstract::AbstractTensor>(TypeIdToType(type_id), output_shape);
  max->set_abstract(abs);
  return max;
}

AnfNodePtr ClipByNormSplit::CreateMulNode(const FuncGraphPtr &func_graph, const AnfNodePtr &x,
                                          const AnfNodePtr &clip_norm, const ShapeVector &shape_vec,
                                          const TypeId &type_id) const {
  auto mul = CreateCNodeBase(func_graph, {x, clip_norm}, kMulOpName, x);
  auto abs = std::make_shared<abstract::AbstractTensor>(TypeIdToType(type_id), shape_vec);
  mul->set_abstract(abs);
  return mul;
}

AnfNodePtr ClipByNormSplit::CreateDivNode(const FuncGraphPtr &func_graph, const AnfNodePtr &dividend,
                                          const AnfNodePtr &divisor, const ShapeVector &shape_vec,
                                          const TypeId &type_id) const {
  auto div = CreateCNodeBase(func_graph, {dividend, divisor}, kDivOpName, divisor);
  auto abs = std::make_shared<abstract::AbstractTensor>(TypeIdToType(type_id), shape_vec);
  div->set_abstract(abs);
  return div;
}

AnfNodePtr ClipByNormSplit::CreateCastNode(const FuncGraphPtr &func_graph, const AnfNodePtr &inp,
                                           const ShapeVector &shape_vec, const TypeId &src_type_id,
                                           const TypeId &dst_type_id) const {
  if (src_type_id == dst_type_id) {
    return inp;
  }

  auto cast = CreateCNodeBase(func_graph, {inp}, kCastOpName, inp);
  if (dst_type_id == kNumberTypeFloat16) {
    common::AnfAlgo::SetNodeAttr(kAttrDstType, kFloat16, cast);
  } else if (dst_type_id == kNumberTypeFloat32) {
    common::AnfAlgo::SetNodeAttr(kAttrDstType, kFloat32, cast);
  } else {
    MS_EXCEPTION(TypeError) << "For '" << prim::kPrimClipByNorm->name()
                            << "`, the data type of input args only supports float16 or float32.";
  }
  common::AnfAlgo::SetNodeAttr(kIsBackendCast, MakeValue(true), cast);
  auto abs = std::make_shared<abstract::AbstractTensor>(TypeIdToType(dst_type_id), shape_vec);
  cast->set_abstract(abs);
  return cast;
}

const BaseRef ClipByNormSplit::DefinePattern() const {
  VarPtr seq_xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimClipByNorm, seq_xs});
}

const AnfNodePtr ClipByNormSplit::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                          const EquivPtr &) const {
  if (!IsPrimitiveCNode(node, prim::kPrimClipByNorm)) {
    return nullptr;
  }
  // Get `ClipByNorm` cnode
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
  // Create `op1 = cast(x)` to float32 data type
  auto x_cast = CreateCastNode(func_graph, inp_x, shape_vec, x_type_id, dst_type_id);
  // Create `op2 = square(op1)` op
  auto square = CreateSquareNode(func_graph, x_cast, shape_vec, dst_type_id);
  // Create 'op3 = reduce_sum(op2)' op
  auto reduce_sum = CreateReduceSumNode(func_graph, square, clip_by_norm, shape_vec, dst_type_id);
  // Create 'op4 = sqrt(op3)' op
  auto sqrt = CreateSqrtNode(func_graph, reduce_sum, dst_type_id);
  // Create 'op5 = cast(clip_norm)' to float32 data type.
  auto clip_norm_cast = CreateCastNode(func_graph, inp_clip_norm, GetOutputInferShape(inp_clip_norm),
                                       common::AnfAlgo::GetOutputInferDataType(inp_clip_norm, 0), dst_type_id);
  // Create 'op6 = x * op5' op
  auto mul = CreateMulNode(func_graph, x_cast, clip_norm_cast, shape_vec, dst_type_id);
  // Create `op7 = max(op5, op4)` op
  auto max = CreateMaxNode(func_graph, clip_norm_cast, sqrt, dst_type_id);
  // Create 'op8 = op6 / op7' op
  auto div = CreateDivNode(func_graph, mul, max, shape_vec, dst_type_id);
  return div;
}
}  // namespace opt
}  // namespace mindspore
