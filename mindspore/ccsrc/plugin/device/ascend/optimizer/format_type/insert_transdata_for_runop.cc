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

#include "plugin/device/ascend/optimizer/format_type/insert_transdata_for_runop.h"
#include <set>
#include "include/common/utils/utils.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kNchwDimNum = 4;
constexpr size_t kDimC = 1;

bool IsDepthwiseCase(const CNodePtr &node, size_t index, const std::string &format, bool is_tuple) {
  if (format != kOpFormat_FRAC_Z) {
    return false;
  }
  abstract::BaseShapePtr base_shape =
    is_tuple ? AnfAlgo::GetPrevNodeOutputDetailShape(node, index) : AnfAlgo::GetOutputDetailShape(node, index);
  MS_EXCEPTION_IF_NULL(base_shape);
  if (base_shape->isa<abstract::Shape>()) {
    auto shape_ptr = base_shape->cast<abstract::ShapePtr>();
    auto shape_vec = shape_ptr->shape();
    return shape_vec.size() == kNchwDimNum && shape_vec[kDimC] == 1;
  }
  return false;
}

bool NeedInsertTransDataForOutput(const CNodePtr &node, size_t index, bool is_tuple) {
  const std::set<std::string> formats_need_transdata = {kOpFormat_ND_RNN_BIAS, kOpFormat_FRACTAL_ZN_RNN,
                                                        kOpFormat_C1HWNCoC0, kOpFormat_FRACTAL_ZN_LSTM};
  auto format = is_tuple ? AnfAlgo::GetPrevNodeOutputFormat(node, index) : AnfAlgo::GetOutputFormat(node, index);
  return formats_need_transdata.count(format) != 0 || IsDepthwiseCase(node, index, format, is_tuple);
}
}  // namespace

bool RunOpInsertTransData::InsertTransdataForOutput(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  bool changed = false;
  bool has_changed = false;
  auto output = graph->output();
  MS_EXCEPTION_IF_NULL(output);
  if (!output->isa<CNode>()) {
    return changed;
  }
  auto cnode = output->cast<CNodePtr>();
  auto inputs_num = common::AnfAlgo::GetInputNum(cnode);
  if (common::AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimMakeTuple)) {
    for (size_t index = 0; index < inputs_num; index++) {
      if (NeedInsertTransDataForOutput(cnode, index, true)) {
        auto cur_cnode_with_index = common::AnfAlgo::GetPrevNodeOutput(cnode, index, false);
        auto trans_node =
          AddTransOpNodeToGraph(graph, cur_cnode_with_index.first, kernel_select_, cur_cnode_with_index.second, false);
        common::AnfAlgo::SetNodeInput(cnode, trans_node, index);
        has_changed = true;
      }
    }
  } else if (AnfAlgo::GetOutputTensorNum(cnode) == 1 && NeedInsertTransDataForOutput(cnode, 0, false)) {
    auto trans_node = AddTransOpNodeToGraph(graph, cnode, kernel_select_, 0, false);
    has_changed = true;
    graph->set_output(trans_node);
  }

  if (has_changed) {
    auto kernel_graph = graph->cast<KernelGraphPtr>();
    MS_EXCEPTION_IF_NULL(kernel_graph);
    auto new_node = kernel_graph->NewCNode(cnode);
    auto manager = kernel_graph->manager();
    MS_EXCEPTION_IF_NULL(manager);
    (void)manager->Replace(cnode, new_node);
    changed = true;
  }
  return changed;
}

bool RunOpInsertTransData::ConvertNodeFormat(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                             const std::string &format, size_t insert_index, size_t input_index,
                                             bool is_insert) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  bool changed = false;
  // convert the format of node to default
  if (NeedInsertTransData(input_shape_, format)) {
    auto input_node = (!is_insert) ? common::AnfAlgo::GetInputNode(cnode, input_index) : node;
    auto trans_node = AddTransOpNodeToGraph(graph, input_node, kernel_select_, insert_index, is_insert);
    common::AnfAlgo::SetNodeInput(cnode, trans_node, input_index);
    changed = true;
  }
  return changed;
}

bool RunOpInsertTransData::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  bool changed = false;
  std::vector<AnfNodePtr> node_list = TopoSort(graph->get_return());
  for (auto &node : node_list) {
    bool has_changed = false;
    MS_EXCEPTION_IF_NULL(node);
    if (!node->cast<CNodePtr>() || !AnfUtils::IsRealKernel(node)) {
      continue;
    }
    size_t input_num = common::AnfAlgo::GetInputTensorNum(node);
    for (size_t index = 0; index < input_num; ++index) {
      auto prev_input_format = AnfAlgo::GetPrevNodeOutputFormat(node, index);
      auto prev_node_out_infer_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(node, index);
      input_shape_ = prev_node_out_infer_shape;
      auto input_format = AnfAlgo::GetInputFormat(node, index);
      // convert the format of node's input or output
      auto input_changed = ConvertNodeFormat(graph, node, prev_input_format, 0, index, false);
      auto output_changed = ConvertNodeFormat(graph, node, input_format, index, index, true);
      has_changed = input_changed || output_changed;
    }
    if (has_changed) {
      auto cnode = node->cast<CNodePtr>();
      auto kernel_graph = graph->cast<KernelGraphPtr>();
      MS_EXCEPTION_IF_NULL(kernel_graph);
      auto new_node = kernel_graph->NewCNode(cnode);
      auto manager = kernel_graph->manager();
      MS_EXCEPTION_IF_NULL(manager);
      (void)manager->Replace(cnode, new_node);
      changed = true;
    }
  }

  changed = InsertTransdataForOutput(graph) || changed;
  return changed;
}
}  // namespace opt
}  // namespace mindspore
