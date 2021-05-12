/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "tools/optimizer/parallel/conv2d_info.h"
#include <memory>
#include <vector>
#include <string>
#include <numeric>
#include "mindspore/core/ops/fusion/conv2d_fusion.h"
#include "mindspore/core/ops/split_with_overlap.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "mindspore/ccsrc/utils/utils.h"
#include "tools/converter/converter_flags.h"
#include "include/errorcode.h"
#include "tools/optimizer/parallel/operator_info_register.h"

using mindspore::schema::PrimitiveType_Conv2DFusion;
namespace mindspore {
namespace opt {

int Conv2DInfo::GetAttrs() { return lite::RET_OK; }

int Conv2DInfo::CheckStrategy(const SplitStrategy &strategy) {
  int split_count = 0;
  Strategys strategys = strategy.strategys;

  // if split N
  if (is_any_not_none(strategys[0][kAxisN])) {
    split_count++;
    split_mode_ = SplitN;
  }
  // if split C_in
  if (is_any_not_none(strategys[0][kAxisCIn])) {
    split_count++;
    split_mode_ = SplitCIN;
    if (strategys[0][kAxisCIn] != strategys[1][kAxisCIn]) {
      MS_LOG(ERROR) << "Strategy ERROR, split C_in, input and kernel must use same strategy.";
      return lite::RET_ERROR;
    }
  }
  // if split C_out
  if (is_any_not_none(strategys[1][kAxisCOut])) {
    split_count++;
    split_mode_ = SplitCOUT;
  }
  // if split H
  if (is_any_not_none(strategys[0][kAxisH])) {
    split_count++;
    split_mode_ = SplitH;
  }
  if (is_any_not_none(strategys[0][kAxisW])) {
    MS_LOG(ERROR) << "Strategy ERROR, doesn't support split W.";
    return lite::RET_ERROR;
  }
  if (is_any_not_none(strategys[1][kAxisH])) {
    MS_LOG(ERROR) << "Strategy ERROR, doesn't support split kernel H.";
    return lite::RET_ERROR;
  }
  if (is_any_not_none(strategys[1][kAxisW])) {
    MS_LOG(ERROR) << "Strategy ERROR, doesn't support split kernel W.";
    return lite::RET_ERROR;
  }
  if (split_count > 1) {
    MS_LOG(ERROR) << "Strategy ERROR, only support split one dimension.";
    return lite::RET_ERROR;
  }

  return lite::RET_OK;
}

AnfNodePtr Conv2DInfo::CreateOutputsOfSplit(const CNodePtr &orig_node, size_t input_index,
                                            std::vector<AnfNodePtr> *split_outputs, size_t split_dim, size_t split_num,
                                            const std::vector<int64_t> &splits, bool trans_format) {
  if (orig_node == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }
  auto conv_prim = GetValueNode<std::shared_ptr<ops::Conv2DFusion>>(cnode_->input(kAnfPrimitiveIndex));
  // prim of split
  auto split_prim = std::make_shared<ops::SplitWithOverlap>();
  split_prim->set_split_dim(split_dim);
  split_prim->set_number_split(split_num);
  split_prim->set_ratio(splits);
  split_prim->set_trans_format(trans_format);
  if (split_mode_ == SplitH) {
    split_prim->set_extend_top(std::vector<int64_t>(split_num, 0));
    auto extend_bottom = conv_prim->get_kernel_size().at(kIndexH) - conv_prim->get_stride().at(kIndexH);
    auto bottom_vector = std::vector<int64_t>(split_num, extend_bottom);
    bottom_vector[split_num - 1] = 0;
    split_prim->set_extend_bottom(bottom_vector);
    split_prim->set_stride(conv_prim->get_stride().at(kIndexH));
    split_prim->set_pad_top(conv_prim->get_pad_list().at(kPadUp));
  } else {
    split_prim->set_extend_top(std::vector<int64_t>(split_num, 0));
    split_prim->set_extend_bottom(std::vector<int64_t>(split_num, 0));
    split_prim->set_stride(0);
    split_prim->set_pad_top(0);
  }
  std::vector<AnfNodePtr> split_inputs = {NewValueNode(split_prim)};
  split_inputs.push_back(orig_node->input(input_index + 1));
  auto split_cnode = func_graph_->NewCNode(split_inputs);
  if (split_cnode == nullptr) {
    MS_LOG(ERROR) << name_ << " : Failed to create split node.";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }

  split_cnode->set_fullname_with_scope("Split_" + name_);
  CreateMultipleOutputsOfAnfNode(split_cnode, split_num, split_outputs);

  return split_cnode;
}

int Conv2DInfo::CheckConv2DPrimitiveType() {
  if (CheckIfFuncGraphIsNull(func_graph_) != lite::RET_OK) {
    return lite::RET_ERROR;
  }
  if (CheckIfAnfNodeIsNull(cnode_) != lite::RET_OK) {
    return lite::RET_ERROR;
  }
  if (!CheckPrimitiveType(cnode_, prim::kPrimConv2D) && !CheckPrimitiveType(cnode_, prim::kPrimConv2DFusion)) {
    return RET_ERROR;
  }
  return RET_OK;
}

int Conv2DInfo::InferParallelCNodes() {
  if (CheckConv2DPrimitiveType() != lite::RET_OK) {
    return RET_ERROR;
  }
  Strategys strategys = strategy_.strategys;
  size_t dev_num = strategy_.dev_num;
  std::vector<AnfNodePtr> feature_split_outputs;
  std::vector<AnfNodePtr> kernel_split_outputs;
  std::vector<AnfNodePtr> bias_split_outputs;
  std::string orig_name = name_;

  parallel_output_nodes_.clear();
  auto conv_prim = GetValueNode<std::shared_ptr<ops::Conv2DFusion>>(cnode_->input(kAnfPrimitiveIndex));
  // split feature and kernel
  switch (split_mode_) {
    case SplitH: {
      name_ = orig_name + "_input";
      auto feature_split_cnode =
        CreateOutputsOfSplit(cnode_, 0, &feature_split_outputs, kAxisH, dev_num, strategys[0][kAxisH], true);
      if ((feature_split_cnode == nullptr) || (feature_split_outputs.size() != IntToSize(dev_num))) {
        MS_LOG(ERROR) << name_ << " : Make split cnode failed.";
        return lite::RET_ERROR;
      }
    } break;
    case SplitN: {
      name_ = orig_name + "_input";
      auto feature_split_cnode =
        CreateOutputsOfSplit(cnode_, 0, &feature_split_outputs, kAxisN, dev_num, strategys[0][kAxisN], true);
      if ((feature_split_cnode == nullptr) || (feature_split_outputs.size() != IntToSize(dev_num))) {
        MS_LOG(ERROR) << name_ << " : Make split cnode failed.";
        return lite::RET_ERROR;
      }
    } break;
    case SplitCOUT: {
      name_ = orig_name + "_kernel";
      auto kernel_split_cnode =
        CreateOutputsOfSplit(cnode_, 1, &kernel_split_outputs, kAxisCOut, dev_num, strategys[1][kAxisCOut], false);
      if ((kernel_split_cnode == nullptr) || (kernel_split_outputs.size() != IntToSize(dev_num))) {
        MS_LOG(ERROR) << name_ << " : Make split cnode failed.";
        return lite::RET_ERROR;
      }
      if (cnode_->size() >= 4) {
        name_ = orig_name + "_bias";
        auto bias_split_cnode =
          CreateOutputsOfSplit(cnode_, 2, &bias_split_outputs, 0, dev_num, strategys[1][kAxisCOut], false);
        if ((bias_split_cnode == nullptr) || (bias_split_outputs.size() != IntToSize(dev_num))) {
          MS_LOG(ERROR) << name_ << " : Make split cnode failed.";
          return lite::RET_ERROR;
        }
      }
    } break;
    case SplitCIN: {
      name_ = orig_name + "_input";
      auto feature_split_cnode =
        CreateOutputsOfSplit(cnode_, 0, &feature_split_outputs, kAxisCIn, dev_num, strategys[0][kAxisCIn], true);
      if ((feature_split_cnode == nullptr) || (feature_split_outputs.size() != IntToSize(dev_num))) {
        MS_LOG(ERROR) << name_ << " : Make split cnode failed.";
        return lite::RET_ERROR;
      }
      name_ = orig_name + "_kernel";
      auto kernel_split_cnode =
        CreateOutputsOfSplit(cnode_, 1, &kernel_split_outputs, kAxisCIn, dev_num, strategys[1][kAxisCIn], false);
      if ((kernel_split_cnode == nullptr) || (kernel_split_outputs.size() != IntToSize(dev_num))) {
        MS_LOG(ERROR) << name_ << " : Make split cnode failed.";
        return lite::RET_ERROR;
      }
    } break;
    default:
      MS_LOG(DEBUG) << "No Split mode chosen";
  }
  name_ = orig_name;

  return ConstructOutputCNodes(conv_prim, feature_split_outputs, kernel_split_outputs, bias_split_outputs);
}

int Conv2DInfo::ConstructOutputCNodes(const std::shared_ptr<ops::Conv2DFusion> &conv_prim,
                                      const std::vector<AnfNodePtr> &feature_split_outputs,
                                      const std::vector<AnfNodePtr> &kernel_split_outputs,
                                      const std::vector<AnfNodePtr> &bias_split_outputs) {
  Strategys strategys = strategy_.strategys;
  size_t dev_num = strategy_.dev_num;

  int cin_strategy_sum = std::accumulate(strategys[0][kAxisCIn].begin(), strategys[0][kAxisCIn].end(), 0);
  int cout_strategy_sum = std::accumulate(strategys[1][kAxisCOut].begin(), strategys[1][kAxisCOut].end(), 0);
  std::string conv_cnode_name = cnode_->fullname_with_scope();
  // construct parallel Conv2D nodes
  for (size_t i = 0; i < dev_num; ++i) {
    std::vector<AnfNodePtr> tmp_outputs;
    bool has_bias = cnode_->size() >= 4;
    // if split cin, only one parallel operator has bias
    if ((i != 0) && split_mode_ == SplitCIN) {
      has_bias = false;
    }
    // copy attr
    auto prim = std::make_shared<ops::Conv2DFusion>();
    prim->set_pad(conv_prim->get_pad());
    prim->set_in_channel(conv_prim->get_in_channel());
    prim->set_out_channel(conv_prim->get_out_channel());
    prim->set_dilation(conv_prim->get_dilation());
    prim->set_format(conv_prim->get_format());
    prim->set_group(conv_prim->get_group());
    prim->set_kernel_size(conv_prim->get_kernel_size());
    prim->set_pad_mode(conv_prim->get_pad_mode());
    prim->set_pad_list(conv_prim->get_pad_list());
    prim->set_stride(conv_prim->get_stride());
    prim->set_activation_type(conv_prim->get_activation_type());

    switch (split_mode_) {
      case SplitH: {
        if (i != 0) {
          auto pad = prim->get_pad_list();
          pad.at(kPadUp) = 0;
          prim->set_pad_list(pad);
        }
        if (i != (dev_num - 1)) {
          auto pad = prim->get_pad_list();
          pad.at(kPadDown) = 0;
          prim->set_pad_list(pad);
        }
      } break;
      case SplitCIN: {
        auto in_channel = prim->get_in_channel();
        if (i == 0) {
          prim->set_in_channel(in_channel * strategys[0][kAxisCIn][0] / cin_strategy_sum);
        } else {
          prim->set_in_channel(in_channel - (in_channel * strategys[0][kAxisCIn][0] / cin_strategy_sum));
        }
      } break;
      case SplitCOUT: {
        auto out_channel = prim->get_out_channel();
        if (i == 0) {
          prim->set_out_channel(out_channel * strategys[1][kAxisCOut][0] / cout_strategy_sum);
        } else {
          prim->set_out_channel(out_channel - (out_channel * strategys[1][kAxisCOut][0] / cout_strategy_sum));
        }
      } break;
      default:
        break;
    }
    std::vector<AnfNodePtr> conv_inputs = {NewValueNode(prim)};
    // if split Cout, feature will not be splited
    if (split_mode_ == SplitCOUT) {
      conv_inputs.push_back(cnode_->input(1));
    } else {
      conv_inputs.push_back(feature_split_outputs[i]);
    }
    // kernel splited only when split Cin and Cout
    if (split_mode_ == SplitCIN || split_mode_ == SplitCOUT) {
      conv_inputs.push_back(kernel_split_outputs[i]);
    } else {
      conv_inputs.push_back(cnode_->input(2));
    }
    if (has_bias) {
      if (split_mode_ == SplitCOUT) {
        conv_inputs.push_back(bias_split_outputs[i]);
      } else {
        conv_inputs.push_back(cnode_->input(3));
      }
    }
    auto conv_cnode = func_graph_->NewCNode(conv_inputs);
    if (conv_cnode == nullptr) {
      MS_LOG(ERROR) << name_ << " : Failed to create parallel Conv2D node " << i;
      return lite::RET_ERROR;
    }
    conv_cnode->set_fullname_with_scope(conv_cnode_name + std::to_string(i));
    CreateMultipleOutputsOfAnfNode(conv_cnode, 1, &tmp_outputs);
    parallel_output_nodes_.push_back(tmp_outputs[0]);
  }
  return lite::RET_OK;
}

int Conv2DInfo::InferReplaceOp() {
  size_t dev_num = strategy_.dev_num;
  if (split_mode_ == SplitCIN) {
    MS_LOG(DEBUG) << name_ << " : Split Cin, infer Forward op.";
    replace_op_ = CreateReduceNode(cnode_, parallel_output_nodes_, kAxisCIn, dev_num, true);
  } else {
    int32_t concat_dim;
    if (split_mode_ == SplitN) {
      concat_dim = kAxisN;
    } else if (split_mode_ == SplitCOUT) {
      // output format is same as feature map
      concat_dim = kAxisCIn;
    } else {
      concat_dim = kAxisH;
    }
    replace_op_ = CreateConcateNode(cnode_, parallel_output_nodes_, concat_dim, dev_num, true);
  }

  if (replace_op_ == nullptr) {
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

AnfNodePtr DepthwiseConv2DInfo::CreateOutputsOfSplit(const CNodePtr &orig_node, size_t input_index,
                                                     std::vector<AnfNodePtr> *split_outputs, size_t split_dim,
                                                     size_t split_num, const std::vector<int64_t> &splits,
                                                     bool trans_format) {
  return nullptr;
}

int DepthwiseConv2DInfo::InferParallelCNodes() {
  if (CheckIfFuncGraphIsNull(func_graph_) != lite::RET_OK) {
    return lite::RET_ERROR;
  }
  if (CheckIfAnfNodeIsNull(cnode_) != lite::RET_OK) {
    return lite::RET_ERROR;
  }
  Strategys strategys = strategy_.strategys;
  size_t dev_num = strategy_.dev_num;
  std::vector<AnfNodePtr> feature_split_outputs;
  std::string orig_name = name_;

  switch (split_mode_) {
    case SplitCIN: {
      MS_LOG(ERROR) << "DepthwiseConv2DInfo doesn't support split Cin.";
      return lite::RET_ERROR;
    } break;
    case SplitCOUT: {
      MS_LOG(ERROR) << "DepthwiseConv2DInfo doesn't support split Cout.";
      return lite::RET_ERROR;
    } break;
    case SplitH: {
      MS_LOG(ERROR) << "DepthwiseConv2DInfo doesn't support split H.";
      return lite::RET_ERROR;
    } break;
    default:
      break;
  }

  parallel_output_nodes_.clear();
  std::string conv_cnode_name = cnode_->fullname_with_scope();
  auto conv_prim = GetValueNode<std::shared_ptr<ops::Conv2DFusion>>(cnode_->input(kAnfPrimitiveIndex));
  if (!IsDwConvNode(conv_prim)) {
    return lite::RET_ERROR;
  }
  // split feature and kernel
  if (split_mode_ == SplitN) {
    name_ = orig_name + "_input";
    auto feature_split_cnode =
      CreateOutputsOfSplit(cnode_, 0, &feature_split_outputs, kAxisN, dev_num, strategys[0][kAxisN], true);
    if ((feature_split_cnode == nullptr) || (feature_split_outputs.size() != IntToSize(dev_num))) {
      MS_LOG(ERROR) << name_ << " : Make split cnode failed.";
      return lite::RET_ERROR;
    }
  }
  name_ = orig_name;

  // construct parallel Conv2D nodes
  for (size_t i = 0; i < dev_num; ++i) {
    std::vector<AnfNodePtr> tmp_outputs;
    // copy attr
    auto prim = std::make_shared<ops::Conv2DFusion>();
    prim->AddAttr(ops::kIsDepthWise, MakeValue<bool>(true));
    prim->set_pad(conv_prim->get_pad());
    prim->set_in_channel(conv_prim->get_in_channel());
    prim->set_out_channel(conv_prim->get_out_channel());
    prim->set_dilation(conv_prim->get_dilation());
    prim->set_format(conv_prim->get_format());
    prim->set_group(conv_prim->get_group());
    prim->set_kernel_size(conv_prim->get_kernel_size());
    prim->set_pad_mode(conv_prim->get_pad_mode());
    prim->set_pad_list(conv_prim->get_pad_list());
    prim->set_stride(conv_prim->get_stride());
    prim->set_activation_type(conv_prim->get_activation_type());

    std::vector<AnfNodePtr> conv_inputs = {NewValueNode(prim)};
    conv_inputs.push_back(feature_split_outputs[i]);
    conv_inputs.push_back(cnode_->input(2));
    if (cnode_->size() >= 4) {
      conv_inputs.push_back(cnode_->input(3));
    }
    auto conv_cnode = func_graph_->NewCNode(conv_inputs);
    if (conv_cnode == nullptr) {
      MS_LOG(ERROR) << name_ << " : Failed to create parallel DepthwiseConv2D node " << i;
      return lite::RET_ERROR;
    }
    conv_cnode->set_fullname_with_scope(conv_cnode_name + std::to_string(i));
    CreateMultipleOutputsOfAnfNode(conv_cnode, 1, &tmp_outputs);
    parallel_output_nodes_.push_back(tmp_outputs[0]);
  }

  return lite::RET_OK;
}

int DepthwiseConv2DInfo::InferReplaceOp() {
  size_t dev_num = strategy_.dev_num;
  replace_op_ = CreateConcateNode(cnode_, parallel_output_nodes_, kAxisN, dev_num, true);
  if (replace_op_ == nullptr) {
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

OPERATOR_INFO_REGISTER(PrimitiveType_Conv2DFusion, kNumberTypeFloat32, OperatorInfoCreator<Conv2DInfo>)
OPERATOR_INFO_REGISTER(PrimitiveType_Conv2DFusion, kNumberTypeInt8, OperatorInfoCreator<Conv2DInfo>)
}  // namespace opt
}  // namespace mindspore
