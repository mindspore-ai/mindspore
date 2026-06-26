/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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
#include "backend/common/pass/ir_fission/cdist_grad_fission.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/backend/common/pass_manager/helper.h"
#include "include/utils/anfalgo.h"
#include "utils/shape_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_e.h"
#include "primitive/array_ops.h"
#include "primitive/math_ops.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kCdistGradInputNum = 5;
constexpr size_t kCdistGradInputRank = 3;
constexpr int64_t kCdistGradBatchRank = 1;
constexpr int64_t kGradExpandDim = 0;
constexpr int64_t kInputXExpandDim = -1;
constexpr int64_t kInputYExpandDim = -2;
constexpr int64_t kOutExpandDim = 0;

bool HasStaticShape(const ShapeVector &shape) { return !IsDynamic(shape); }

std::optional<size_t> GetExpandInsertIndex(size_t rank, int64_t expand_dim) {
  const auto insert_index = static_cast<int64_t>(rank) + expand_dim;
  if (insert_index < 0 || insert_index > static_cast<int64_t>(rank)) {
    return std::nullopt;
  }
  return static_cast<size_t>(insert_index);
}

ShapeVector ExpandShapeAt(const ShapeVector &shape, int64_t expand_dim) {
  const auto insert_index = GetExpandInsertIndex(shape.size(), expand_dim);
  if (!insert_index.has_value()) {
    MS_INTERNAL_EXCEPTION(ValueError) << "For CdistGrad fission, invalid expand dim " << expand_dim
                                      << " for shape rank " << shape.size() << ".";
  }
  auto expanded_shape = shape;
  (void)expanded_shape.insert(expanded_shape.cbegin() + static_cast<ShapeVector::difference_type>(insert_index.value()),
                              1);
  return expanded_shape;
}

bool NeedCdistGradFission(const ShapeVector &grad_shape, const ShapeVector &x_shape, const ShapeVector &y_shape,
                          const ShapeVector &out_shape) {
  if (grad_shape.size() != kCdistGradInputRank || x_shape.size() != kCdistGradInputRank ||
      y_shape.size() != kCdistGradInputRank || out_shape.size() != kCdistGradInputRank) {
    return false;
  }
  // Frontend fallback already handles rank-3 non-square rows and batch broadcast. This backend pass only needs
  // to patch the remaining static square-output path that still lowers to CdistGrad on Ascend.
  return grad_shape == out_shape && x_shape[kIndex0] == y_shape[kIndex0] && x_shape[kIndex1] == y_shape[kIndex1] &&
         grad_shape[kIndex0] == x_shape[kIndex0] && grad_shape[kIndex1] == x_shape[kIndex1] &&
         grad_shape[kIndex2] == y_shape[kIndex1];
}

ShapeVector CalcCdistBroadcastShape(ShapeVector x_shape, ShapeVector y_shape) {
  x_shape = ExpandShapeAt(x_shape, kInputXExpandDim);
  y_shape = ExpandShapeAt(y_shape, kInputYExpandDim);
  if (x_shape.size() != y_shape.size()) {
    MS_INTERNAL_EXCEPTION(ValueError) << "For CdistGrad, input_x and input_y should have the same rank.";
  }
  ShapeVector broadcast_shape;
  broadcast_shape.reserve(x_shape.size());
  for (size_t i = 0; i < x_shape.size(); ++i) {
    if (x_shape[i] == 1) {
      broadcast_shape.push_back(y_shape[i]);
    } else if (y_shape[i] == 1 || x_shape[i] == y_shape[i]) {
      broadcast_shape.push_back(x_shape[i]);
    } else {
      MS_INTERNAL_EXCEPTION(ValueError) << "The two input shape can not broadcast, x_shape: " << x_shape
                                        << ", y_shape: " << y_shape;
    }
  }
  return broadcast_shape;
}

AnfNodePtr AddBroadcastToNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node, int64_t expand_dim,
                              const ShapeVector &target_shape, const PatternProcessPass &pass) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(input_node);
  auto kernel_graph = func_graph->cast<std::shared_ptr<session::KernelGraph>>();
  MS_EXCEPTION_IF_NULL(kernel_graph);

  auto axis = kernel_graph->NewValueNode(std::make_shared<Int64Imm>(expand_dim - 1));
  std::vector<AnfNodePtr> expand_dims_inputs = {
    NewValueNode(std::make_shared<Primitive>(prim::kPrimExpandDims->name())), input_node, axis};
  auto expand_dims = pass.NewCNode(expand_dims_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(expand_dims);
  expand_dims->set_scope(input_node->scope());
  auto dtype = common::AnfAlgo::GetOutputInferDataType(input_node, 0);
  auto expand_shape = ExpandShapeAt(common::AnfAlgo::GetOutputInferShape(input_node, 0), expand_dim);
  common::AnfAlgo::SetOutputInferTypeAndShape({dtype}, {expand_shape}, expand_dims.get());
  common::AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(expand_dim - 1), expand_dims);
  common::AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), expand_dims);
  if (expand_shape == target_shape) {
    return expand_dims;
  }

  auto shape_node = opt::CreateValueNodeWithKernelInfo(func_graph, MakeValue(target_shape));
  std::vector<AnfNodePtr> broadcast_to_inputs = {
    NewValueNode(std::make_shared<Primitive>(prim::kPrimBroadcastTo->name())), expand_dims, shape_node};
  auto broadcast_to = pass.NewCNode(broadcast_to_inputs, func_graph);
  common::AnfAlgo::SetOutputInferTypeAndShape({dtype}, {target_shape}, broadcast_to.get());
  common::AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), broadcast_to);
  broadcast_to->set_scope(input_node->scope());
  return broadcast_to;
}
}  // namespace

std::vector<std::string> CdistGradFission::MustExistPrimitiveName() const { return {prim::kPrimCdistGrad->name()}; }

const BaseRef CdistGradFission::DefinePattern() const {
  VarPtr xs = std::make_shared<SeqVar>();
  auto cdist_grad_prim = std::make_shared<Primitive>(prim::kPrimCdistGrad->name());
  return VectorRef(std::vector<BaseRef>{cdist_grad_prim, xs});
}

const AnfNodePtr CdistGradFission::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cdist_grad_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cdist_grad_cnode);
  if (GetBoolAttr(cdist_grad_cnode, kAttrVisited)) {
    return nullptr;
  }
  common::AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), node);
  if (cdist_grad_cnode->size() != kCdistGradInputNum + 1) {
    MS_LOG(INFO) << "The node " << cdist_grad_cnode->DebugString() << " is not equal to " << kCdistGradInputNum
                 << " inputs";
    return nullptr;
  }

  const auto &cdist_grad_inputs = cdist_grad_cnode->inputs();
  auto grad_shape = common::AnfAlgo::GetOutputInferShape(cdist_grad_inputs[kDim1], 0);
  auto x_shape = common::AnfAlgo::GetOutputInferShape(cdist_grad_inputs[kDim2], 0);
  auto y_shape = common::AnfAlgo::GetOutputInferShape(cdist_grad_inputs[kDim3], 0);
  auto out_shape = common::AnfAlgo::GetOutputInferShape(cdist_grad_inputs[kDim4], 0);
  if (!HasStaticShape(grad_shape) || !HasStaticShape(x_shape) || !HasStaticShape(y_shape) ||
      !HasStaticShape(out_shape) || !NeedCdistGradFission(grad_shape, x_shape, y_shape, out_shape)) {
    return nullptr;
  }
  // Ascend CdistGrad still expects dout/x/y/out to be expanded and broadcast to one common shape.
  auto broadcast_shape = CalcCdistBroadcastShape(x_shape, y_shape);
  auto broadcast_grad = AddBroadcastToNode(graph, cdist_grad_inputs[kDim1], kGradExpandDim, broadcast_shape, *this);
  auto broadcast_x = AddBroadcastToNode(graph, cdist_grad_inputs[kDim2], kInputXExpandDim, broadcast_shape, *this);
  auto broadcast_y = AddBroadcastToNode(graph, cdist_grad_inputs[kDim3], kInputYExpandDim, broadcast_shape, *this);
  auto broadcast_out = AddBroadcastToNode(graph, cdist_grad_inputs[kDim4], kOutExpandDim, broadcast_shape, *this);
  auto cdist_grad_prim = std::make_shared<Primitive>(prim::kPrimCdistGrad->name());
  cdist_grad_prim->set_attr(ops::kBatchRank, MakeValue(kCdistGradBatchRank));
  std::vector<AnfNodePtr> new_inputs{
    NewValueNode(cdist_grad_prim), broadcast_grad, broadcast_x, broadcast_y, broadcast_out, cdist_grad_inputs[kDim5]};
  auto new_cnode = NewCNode(new_inputs, graph);
  MS_EXCEPTION_IF_NULL(new_cnode);
  new_cnode->set_abstract(cdist_grad_cnode->abstract());
  new_cnode->set_scope(cdist_grad_cnode->scope());
  new_cnode->set_fullname_with_scope(cdist_grad_cnode->fullname_with_scope());
  common::AnfAlgo::CopyNodeAttrs(cdist_grad_cnode, new_cnode);
  common::AnfAlgo::SetNodeAttr(ops::kBatchRank, MakeValue(kCdistGradBatchRank), new_cnode);
  return new_cnode;
}
}  // namespace opt
}  // namespace mindspore
