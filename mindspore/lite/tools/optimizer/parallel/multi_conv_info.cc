/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#define USE_DEPRECATED_API
#include "tools/optimizer/parallel/multi_conv_info.h"
#include <string>
#include <algorithm>
#include "tools/optimizer/parallel/spliter.h"
#include "ops/fusion/conv2d_fusion.h"
#include "tools/optimizer/parallel/split_strategy.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"

using mindspore::schema::PrimitiveType_Conv2dTransposeFusion;
namespace mindspore {
namespace opt {
int MultiConvSplit::GenSplitInfo() {
  split_info_.out_num = static_cast<int64_t>(this->strategy_.dev_num);
  for (const auto &dev_type : this->strategy_.dev_types) {
    for (const auto &support_split_device : kSupportSplitedDevices) {
      if (dev_type == support_split_device.first) {
        split_info_.dev_types.push_back(support_split_device.second);
      }
    }
  }
  if (split_info_.dev_types.empty()) {
    MS_LOG(ERROR) << "unsupported DeviceType. ";
    return RET_ERROR;
  }
  // only can get N && H && CIN &&
  std::vector<int64_t> tmp(split_info_.out_num, 0);
  MS_CHECK_FALSE(this->strategy_.strategys.empty(), RET_ERROR);
  for (size_t i = 0; i < this->strategy_.strategys[0].size(); i++) {
    if (this->strategy_.strategys[0][i] == tmp) {
      continue;
    }
    split_info_.axis = static_cast<int64_t>(i);  // NHWC
    split_info_.size_splits.clear();
    split_info_.size_splits = this->strategy_.strategys[0][i];  // cal base on compute_cap
    break;
  }
  split_info_.in_num_conv = num_;
  split_info_.fmk_type = fmk_type_;
  split_info_.extend_bottom = std::vector<int64_t>(split_info_.size_splits.size(), 0);
  split_info_.extend_top = std::vector<int64_t>(split_info_.size_splits.size(), 0);
  split_info_.primitive_type = primitive_type_;
  ori_split_ratios_ = split_info_.size_splits;
  return RET_OK;
}

//  assume we only split to two devices
bool MultiConvSplit::CheckSplitValid() {
  // check conv node prim
  for (const auto &conv_node : conv_nodes_) {
    auto conv_cnode = conv_node->cast<CNodePtr>();
    MS_ASSERT(conv_cnode != nullptr);
    auto conv_prim = ops::GetOperator<ops::Conv2DFusion>(conv_cnode->input(kAnfPrimitiveIndex));
    MS_ASSERT(conv_prim != nullptr);
    MS_CHECK_TRUE_RET(conv_prim->GetAttr(ops::kPadMode) != nullptr, false);
    if (conv_prim->get_pad_mode() != SAME) {
      return false;
    }
  }
  // check final split ratio
  int64_t total_block_count = 0;
  int64_t visited_block = 0;
  auto final_ratios = split_info_.size_splits;
  for (int64_t i = 0; i < split_info_.out_num; ++i) {
    if (i >= static_cast<int64_t>(final_ratios.size()) || final_ratios.at(i) <= 0) {
      return false;
    }
    total_block_count += final_ratios.at(i);
    if (i == 0) {
      visited_block += final_ratios.at(i);
    }
  }
  // check split extend_top
  auto front_conv_node = conv_nodes_.back();
  auto ori_graph_input_shape = Spliter::GetInstance()->graph_node_input_shapes();
  auto ori_front_conv_input_iter = ori_graph_input_shape.find(front_conv_node->fullname_with_scope());
  if (ori_front_conv_input_iter == ori_graph_input_shape.end()) {
    return false;
  }
  MS_CHECK_TRUE_RET(total_block_count != 0, false);
  int64_t split_axis_value_0 = UP_DIV(split_info_.ori_split_axis_value * visited_block, total_block_count);
  if (split_axis_value_0 > split_info_.ori_split_axis_value) {
    return false;
  }
  int64_t split_axis_value_1 = split_info_.ori_split_axis_value - split_axis_value_0;
  split_axis_value_1 += (split_info_.extend_top.back() + split_info_.extend_bottom.back());
  return split_axis_value_1 <= split_info_.ori_split_axis_value;
}

int MultiConvSplit::GetMultiConvNodes(const AnfNodePtr &conv_node) {
  MS_ASSERT(func_graph_ != nullptr && conv_node != nullptr);
  // get nodes to be splited
  // node in graph 1->2->3...
  // node in vector ...->3->2->1
  std::string conv_cnode_name = conv_node->fullname_with_scope();
  auto graph_node_outputs = Spliter::GetInstance()->graph_node_outputs();
  auto it = graph_node_outputs.find(conv_cnode_name);
  if (it == graph_node_outputs.end() || it->second.size() > kDefaultBatch) {
    MS_LOG(ERROR) << "This node may be the last node of graph,it do not has any out-nodes.";
    return RET_ERROR;
  }
  conv_nodes_.push_back(conv_node);
  int32_t index = 0;
  while (index < split_info_.in_num_conv - 1) {
    MS_CHECK_LT(index, static_cast<int32_t>(conv_nodes_.size()), RET_ERROR);
    auto curr_node = conv_nodes_[index];
    MS_ASSERT(curr_node != nullptr);
    auto curr_cnode = curr_node->cast<CNodePtr>();
    MS_CHECK_TRUE_RET(curr_cnode != nullptr, RET_ERROR);
    auto tmp_node = curr_cnode->input(1);
    if (!IsConv2D(tmp_node)) {
      break;
    }
    auto name = tmp_node->fullname_with_scope();
    // check outputs's bigger than two
    it = graph_node_outputs.find(name);
    if (it == graph_node_outputs.end()) {
      return RET_ERROR;
    }
    if (it->second.size() > kDefaultBatch) {
      break;
    }
    conv_nodes_.push_back(tmp_node);
    index++;
  }
  if (conv_nodes_.size() != static_cast<size_t>(split_info_.in_num_conv)) {
    return RET_ERROR;
  }
  return RET_OK;
}

AnfNodePtr MultiConvSplit::MultiConvNHSplit(const AnfNodePtr &node) {
  MS_ASSERT(node != nullptr);
  std::string conv_cnode_name = node->fullname_with_scope();
  // Create Split node and get outputs of Split
  std::vector<AnfNodePtr> split_outputs;
  if (!CreateOutputsOfSplitWithOverlap(func_graph_, conv_nodes_[conv_nodes_.size() - 1], &split_outputs, split_info_,
                                       conv_cnode_name)) {
    MS_LOG(ERROR) << "CreateOutputsOfSplitWithOverlap failed";
    return nullptr;
  }
  // Create Conv node
  int res_conv_numbers = static_cast<int>(conv_nodes_.size() - 1);
  for (int32_t i = res_conv_numbers; i >= 0; i--) {
    std::vector<AnfNodePtr> outputs_node;
    if (!SplitSingleConv(conv_nodes_[i], split_outputs, &outputs_node)) {
      MS_LOG(ERROR) << "SplitSingleConv failed";
      return nullptr;
    }
    split_outputs.clear();
    std::copy(outputs_node.begin(), outputs_node.end(), std::back_inserter(split_outputs));
    outputs_node.clear();
  }
  // Create concate node
  auto concat_node = CreateOutputsOfConcat(func_graph_, node, split_outputs, split_info_, conv_cnode_name);
  MS_CHECK_TRUE_RET(concat_node != nullptr, nullptr);
  split_outputs.clear();
  return concat_node;
}

bool MultiConvSplit::SplitSingleConv(const AnfNodePtr &ori_node, const std::vector<AnfNodePtr> &inputs_node,
                                     std::vector<AnfNodePtr> *outputs_node) {
  MS_ASSERT(ori_node != nullptr && outputs_node != nullptr);
  auto ori_conv_cnode = ori_node->cast<CNodePtr>();
  MS_ASSERT(ori_conv_cnode != nullptr);
  auto ori_attr = ops::GetOperator<ops::Conv2DFusion>(ori_conv_cnode->input(kAnfPrimitiveIndex));
  MS_ASSERT(ori_attr != nullptr);
  for (int output_conv_index = 0; output_conv_index < static_cast<int>(split_info_.out_num); output_conv_index++) {
    // Create Conv node attr
    auto conv_prim = CopyConvPrim(ori_attr);
    auto ori_node_name = ori_node->fullname_with_scope();
    auto graph_node_input_shapes = Spliter::GetInstance()->graph_node_input_shapes();
    auto input_shape_iter = graph_node_input_shapes.find(ori_node_name);
    if (input_shape_iter == graph_node_input_shapes.end()) {
      return true;
    }
    auto input_shapes = input_shape_iter->second;
    auto input_shape = input_shapes.front();
    // adjust primitive
    AdJustConvPrim(conv_prim, input_shape, output_conv_index);
    // node inputs
    std::vector<AnfNodePtr> conv_inputs;
    auto conv_prim_c = conv_prim->GetPrim();
    MS_ASSERT(conv_prim_c != nullptr);
    conv_inputs.push_back(NewValueNode(conv_prim_c));
    AdJustInputs(ori_node, inputs_node, output_conv_index, &conv_inputs);
    // create new conv node
    if (!CreateNewConvNode(ori_node, conv_inputs, output_conv_index, outputs_node)) {
      return false;
    }
  }
  return true;
}

void MultiConvSplit::AdJustInputs(const AnfNodePtr &ori_conv_node, const std::vector<AnfNodePtr> &new_inputs_node,
                                  int output_conv_index, std::vector<AnfNodePtr> *conv_inputs) {
  MS_ASSERT(ori_conv_node != nullptr && conv_inputs != nullptr);
  auto ori_conv_cnode = ori_conv_node->cast<CNodePtr>();
  MS_ASSERT(ori_conv_cnode != nullptr);
  // feature_map
  conv_inputs->push_back(new_inputs_node[output_conv_index]);
  // W+bias
  for (size_t j = kDefaultBatch + 1; j < ori_conv_cnode->size(); j++) {
    conv_inputs->push_back(ori_conv_cnode->input(j));
  }
}

bool MultiConvSplit::CreateNewConvNode(const AnfNodePtr &ori_conv_node, const std::vector<AnfNodePtr> &conv_inputs,
                                       int output_conv_index, std::vector<AnfNodePtr> *outputs_node) {
  MS_ASSERT(ori_conv_node != nullptr && outputs_node != nullptr);
  std::string ori_cnode_name = ori_conv_node->fullname_with_scope();
  // new conv_node
  auto conv_cnode = func_graph_->NewCNode(conv_inputs);
  MS_ASSERT(conv_cnode != nullptr);
  conv_cnode->set_fullname_with_scope(ori_cnode_name + "_" + PARALLEL_NAME_SUFFIX +
                                      std::to_string(output_conv_index + 1));
  conv_cnode->AddAttr(mindspore::ops::kDeviceType,
                      MakeValue(static_cast<int>(split_info_.dev_types[output_conv_index])));
  std::vector<AnfNodePtr> tmp_outputs;
  // conv2d only has one output, set to output_nodes
  if (!GetMultipleOutputsOfAnfNode(func_graph_, conv_cnode, 1, &tmp_outputs)) {
    MS_LOG(ERROR) << "GetMultipleOutputsOfAnfNode failed";
    return false;
  }
  outputs_node->push_back(tmp_outputs[0]->cast<CNodePtr>()->input(1));
  tmp_outputs.clear();
  return true;
}

AnfNodePtr MultiConvSplit::DoSplit(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  MS_ASSERT(unc_graph != nullptr && node != nullptr);
  int ret = GenSplitInfo();
  if (ret != RET_OK) {
    return node;
  }
  func_graph_ = func_graph;
  ret = GetMultiConvNodes(node);
  if (ret != RET_OK) {
    return node;
  }
  return SplitMultiConv(node);
}

AnfNodePtr MultiConvSplitN::SplitMultiConv(const AnfNodePtr &node) {
  MS_ASSERT(node != nullptr);
  if (conv_nodes_.size() == DIMENSION_2D && split_info_.axis == CuttingStragedy::CUT_N) {
    return node;
  }
  return MultiConvNHSplit(node);
}

AnfNodePtr MultiConvSplitH::SplitMultiConv(const AnfNodePtr &node) {
  // update info, N do not need, C do not support
  MS_ASSERT(node != nullptr);
  if (!UpdateSplitInfo(func_graph_, conv_nodes_, &split_info_)) {
    return node;
  }
  if (!CheckSplitValid()) {
    return node;
  }
  return MultiConvNHSplit(node);
}

void MultiConvSplitH::AdJustConvPrim(const api::SharedPtr<ops::Conv2DFusion> &conv_prim, const ShapeVector &input_shape,
                                     int output_conv_index) {
  MS_ASSERT(conv_prim != nullptr);
  MS_ASSERT(input_shape.size() == kInputSizeFour);
  int64_t input_h = input_shape.at(kAxisH);
  int64_t input_w = input_shape.at(kAxisW);
  auto pad_list = GetSplitPadList(conv_prim, input_h, input_w);
  MS_ASSERT(pad_list.size() > 1);
  if (output_conv_index == 0) {
    pad_list[kPadDown] = 0;
  } else if (output_conv_index == static_cast<int>(split_info_.out_num - 1)) {
    pad_list[kPadUp] = 0;
  } else {
    pad_list[kPadUp] = 0;
    pad_list[kPadDown] = 0;
  }
  conv_prim->set_pad_list(pad_list);
}

AnfNodePtr MultiConvSplitCIN::SplitMultiConv(const AnfNodePtr &node) { return nullptr; }

AnfNodePtr MultiConvSplitCOUT::SplitMultiConv(const AnfNodePtr &node) { return nullptr; }

}  // namespace opt
}  // namespace mindspore
