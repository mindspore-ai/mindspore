/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "tools/converter/parser/onnx/onnx_deform_conv2d_adjust.h"
#include <vector>
#include <string>
#include <memory>
#include "ops/shape.h"
#include "ops/reshape.h"
#include "ops/nn_ops.h"
#include "include/errorcode.h"
#include "tools/common/tensor_util.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore::lite {
namespace {
CNodePtr NewReshapeOpNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node, const AnfNodePtr &shape_node) {
  MS_ASSERT(func_graph != nullptr && input_node != nullptr && shape_node != nullptr);
  auto reshape_prim = std::make_shared<ops::Reshape>();
  if (reshape_prim == nullptr) {
    MS_LOG(ERROR) << "create reshape failed.";
    return nullptr;
  }
  auto prim_c = reshape_prim->GetPrim();
  MS_CHECK_TRUE_MSG(prim_c != nullptr, nullptr, "get prim return nullptr");
  ValueNodePtr value_node = NewValueNode(prim_c);
  MS_CHECK_TRUE_MSG(value_node != nullptr, nullptr, "create valuenode return nullptr");
  std::vector<AnfNodePtr> op_inputs = {value_node, input_node, shape_node};
  auto reshape = func_graph->NewCNode(op_inputs);
  MS_CHECK_TRUE_MSG(reshape != nullptr, nullptr, "create cnode return nullptr");
  reshape->set_fullname_with_scope(input_node->fullname_with_scope() + "_reshape");
  return reshape;
}

CNodePtr NewShapeOpNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node) {
  auto shape_prim = std::make_shared<ops::Shape>();
  if (shape_prim == nullptr) {
    MS_LOG(ERROR) << "create reshape failed.";
    return nullptr;
  }
  auto prim_c = shape_prim->GetPrim();
  MS_CHECK_TRUE_MSG(prim_c != nullptr, nullptr, "get prim return nullptr");
  ValueNodePtr value_node = NewValueNode(prim_c);
  MS_CHECK_TRUE_MSG(value_node != nullptr, nullptr, "create valuenode return nullptr");
  std::vector<AnfNodePtr> op_inputs = {value_node, input_node};
  auto shape = func_graph->NewCNode(op_inputs);
  MS_CHECK_TRUE_MSG(shape != nullptr, nullptr, "create cnode return nullptr");
  shape->set_fullname_with_scope(input_node->fullname_with_scope() + "_shape");
  return shape;
}
}  // namespace

/*  1. adjust offset from (y1,x1,...,yn,xn) to (x1,...xn,y1,...yn).                offset
 *  2. concat offset and mask as the new offset.                                     |
 *                                                                                 shape
 *                                                                         /                     \
 *                                                                 gather(0) const(-1)  const(2) gather(1,2)
 *                                                                          \        \  /       /
 *     offset    (shape) <--                                                        concat
 *       |         /
 *    [reshape(c/2, 2) + transpose(c: 1,0) + gather(c: 0)/gather(c: 1) + concat(c: 1,0) + reshape(c)]   mask
 *                                                                                     \              /
 *              input0  offset    mask   weight                 input0   weight      concat(offset)
 *                 \       \       /       /                       \       |        /
 *                  MMCVModulatedDeformConv2D         -->          DeformableConv2D
 *                             |                                           |
 *                           output                                     output
 *
 */
bool OnnxDeformConv2dAdjust::Adjust(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto cnodes = func_graph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    if (!opt::CheckPrimitiveType(cnode, prim::kPrimDeformableConv2d)) {
      continue;
    }

    MS_CHECK_TRUE_RET(cnode->size() >= opt::kInputSizeFive, RET_ERROR);
    auto offset_input = cnode->input(opt::kInputIndexTwo);
    MS_CHECK_TRUE_RET(offset_input != nullptr, false);
    auto shape_node = NewShapeOpNode(func_graph, offset_input);
    if (shape_node == nullptr) {
      MS_LOG(ERROR) << "create shape node failed.";
      return false;
    }

    auto gather_0 =
      opt::GenGatherNode(func_graph, shape_node, {0}, shape_node->fullname_with_scope() + "_gather_0", {0});
    MS_CHECK_TRUE_MSG(gather_0 != nullptr, false, "create gather cnode return nullptr");
    auto gather_1 =
      opt::GenGatherNode(func_graph, shape_node, {2, 3}, shape_node->fullname_with_scope() + "_gather_1", {0});
    MS_CHECK_TRUE_MSG(gather_1 != nullptr, false, "create gather cnode return nullptr");
    auto param_0 =
      opt::BuildIntValueParameterNode(func_graph, -1, shape_node->fullname_with_scope() + "_const_0", false);
    MS_CHECK_TRUE_MSG(param_0 != nullptr, false, "create parameter return nullptr");
    auto param_1 =
      opt::BuildIntValueParameterNode(func_graph, 2, shape_node->fullname_with_scope() + "_const_1", false);
    MS_CHECK_TRUE_MSG(param_1 != nullptr, false, "create parameter return nullptr");
    auto concat_shape = opt::GenConcatNode(func_graph, {gather_0, param_0, param_1, gather_1},
                                           shape_node->fullname_with_scope() + "_concat", 0);
    MS_CHECK_TRUE_MSG(concat_shape != nullptr, false, "create concat return nullptr");

    auto reshape = NewReshapeOpNode(func_graph, offset_input, concat_shape);
    if (reshape == nullptr) {
      MS_LOG(ERROR) << "create reshape failed.";
      return false;
    }
    std::vector<int> perm = {0, 2, 1, 3, 4};
    auto transpose = opt::GenTransposeNode(func_graph, reshape, perm, reshape->fullname_with_scope() + "_transpose");
    if (transpose == nullptr) {
      MS_LOG(ERROR) << "create transpose failed.";
      return false;
    }
    auto gather_y = opt::GenGatherNode(func_graph, transpose, {0}, transpose->fullname_with_scope() + "_gather_y", {1});
    if (gather_y == nullptr) {
      MS_LOG(ERROR) << "create gather failed.";
      return false;
    }
    auto gather_x = opt::GenGatherNode(func_graph, transpose, {1}, transpose->fullname_with_scope() + "_gather_x", {1});
    if (gather_x == nullptr) {
      MS_LOG(ERROR) << "create gather failed.";
      return false;
    }

    auto concat = opt::GenConcatNode(func_graph, {gather_x, gather_y}, gather_x->fullname_with_scope() + "_concat", 1);
    if (concat == nullptr) {
      MS_LOG(ERROR) << "create concat failed.";
      return false;
    }
    auto reshape_last = NewReshapeOpNode(func_graph, concat, shape_node);
    if (reshape_last == nullptr) {
      MS_LOG(ERROR) << "create reshape failed.";
      return false;
    }

    auto concat_offset = opt::GenConcatNode(func_graph, {reshape_last, cnode->input(opt::kInputIndexThree)},
                                            offset_input->fullname_with_scope() + "_mask", 1);
    if (concat_offset == nullptr) {
      MS_LOG(ERROR) << "create concat failed.";
      return false;
    }
    // features, filter, offsets, bias(optional)
    std::vector<AnfNodePtr> new_input = {cnode->input(0), cnode->input(1), cnode->input(opt::kInputIndexFour),
                                         concat_offset};
    if (cnode->inputs().size() == opt::kInputIndexSix) {
      new_input.push_back(cnode->input(opt::kInputIndexFive));
    }
    cnode->set_inputs(new_input);
  }
  opt::UpdateManager(func_graph);
  return true;
}
}  // namespace mindspore::lite
