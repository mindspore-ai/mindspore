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
#include "ops/fusion/conv2d_fusion.h"
#include "ops/split_with_overlap.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "utils/utils.h"
#include "tools/converter/converter_flags.h"
#include "include/errorcode.h"
#include "tools/optimizer/parallel/operator_info_register.h"
#include "tools/optimizer/parallel/spliter.h"
#include "tools/optimizer/fisson/fisson_util.h"
#include "nnacl/op_base.h"

using mindspore::schema::PrimitiveType_Conv2DFusion;
namespace mindspore {
namespace opt {
constexpr auto kConvWithBias = 4;
constexpr auto kAnfConvInput = 1;
constexpr auto kAnfConvWeight = 2;
constexpr auto kAnfConvBias = 3;
int Conv2DInfo::CheckStrategy(const SplitStrategy &strategy) {
  int split_count = 0;
  Strategys strategys = strategy.strategys;
  MS_CHECK_GE(strategys.size(), kInputSizeTwo, RET_ERROR);
  MS_CHECK_GE(strategys[0].size(), kInputSizeFour, RET_ERROR);
  MS_CHECK_GE(strategys[1].size(), kInputSizeFour, RET_ERROR);
  // if split N
  if (is_any_not_none(strategys[0][kAxisN])) {
    split_count++;
    split_mode_ = SplitN;
    splits_ = strategys[0][kAxisN];
  }
  // if split C_in
  if (is_any_not_none(strategys[0][kAxisCIn])) {
    split_count++;
    split_mode_ = SplitCIN;
    splits_ = strategys[0][kAxisCIn];
    if (strategys[0][kAxisCIn] != strategys[1][kAxisCIn]) {
      MS_LOG(ERROR) << "Strategy ERROR, split C_in, input and kernel must use same strategy.";
      return RET_ERROR;
    }
  }
  // if split C_out
  if (is_any_not_none(strategys[1][kAxisCOut])) {
    split_count++;
    split_mode_ = SplitCOUT;
    splits_ = strategys[1][kAxisCOut];
  }
  // if split H
  if (is_any_not_none(strategys[0][kAxisH])) {
    split_count++;
    split_mode_ = SplitH;
    splits_ = strategys[0][kAxisH];
  }
  if (is_any_not_none(strategys[0][kAxisW])) {
    MS_LOG(ERROR) << "Strategy ERROR, doesn't support split W.";
    return RET_ERROR;
  }
  if (is_any_not_none(strategys[1][kAxisH])) {
    MS_LOG(ERROR) << "Strategy ERROR, doesn't support split kernel H.";
    return RET_ERROR;
  }
  if (is_any_not_none(strategys[1][kAxisW])) {
    MS_LOG(ERROR) << "Strategy ERROR, doesn't support split kernel W.";
    return RET_ERROR;
  }
  if (split_count > 1) {
    MS_LOG(ERROR) << "Strategy ERROR, only support split one dimension.";
    return RET_ERROR;
  }
  return RET_OK;
}

int Conv2DInfo::CheckIfSplit() {
  auto conv_prim = GetValueNode<std::shared_ptr<ops::Conv2DFusion>>(cnode_->input(kAnfPrimitiveIndex));
  MS_ASSERT(conv_prim != nullptr);
  auto strides = conv_prim->get_stride();
  std::vector<int64_t> weight_shape;
  std::vector<int64_t> input_shape;

  // for n, h, cin, we should checkout it's input whether bigger than split total ratio
  if (split_mode_ != SplitCOUT) {
    auto input_node_abstract = GetCNodeInputAbstract(cnode_, 1);
    MS_CHECK_TRUE_RET(input_node_abstract != nullptr, RET_ERROR);
    auto weight_node_abstract = GetCNodeInputAbstract(cnode_, 2);
    MS_CHECK_TRUE_RET(weight_node_abstract != nullptr, RET_ERROR);
    if (!utils::isa<abstract::AbstractTensorPtr>(input_node_abstract)) {
      MS_LOG(ERROR) << "conv_input_abstract of should be abstract tensor";
      return RET_ERROR;
    }
    if (!utils::isa<abstract::AbstractTensorPtr>(weight_node_abstract)) {
      MS_LOG(ERROR) << "conv_weight_abstract of should be abstract tensor";
      return RET_ERROR;
    }
    auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(input_node_abstract);
    MS_ASSERT(abstract_tensor != nullptr);
    MS_CHECK_TRUE_RET(abstract_tensor->shape() != nullptr, RET_ERROR);
    input_shape = abstract_tensor->shape()->shape();
    abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(weight_node_abstract);
    MS_ASSERT(abstract_tensor != nullptr);
    MS_CHECK_TRUE_RET(abstract_tensor->shape() != nullptr, RET_ERROR);
    weight_shape = abstract_tensor->shape()->shape();
    int total_ratio = 0;
    total_ratio = std::accumulate(splits_.begin(), splits_.end(), total_ratio);
    if (input_shape.size() != kInputSizeFour || weight_shape.size() != kInputSizeFour) {
      return RET_ERROR;
    }
    auto shape_h = input_shape.at(1);
    auto shape_n = input_shape.at(0);
    if (split_mode_ == SplitH && shape_h < total_ratio) {
      return RET_ERROR;
    }
    if (split_mode_ == SplitN && shape_n < total_ratio) {
      return RET_ERROR;
    }
    // too tiny FLOPs no need to be splited, need to add check for split ratio
    auto current_flops = ApproximateFLOPs(strides, input_shape, weight_shape);
    if (current_flops <= kUserFLOPs) {
      return RET_ERROR;
    }
  }
  return RET_OK;
}

AnfNodePtr Conv2DInfo::CreateOutputsOfSplit(const CNodePtr &orig_node, size_t input_index,
                                            std::vector<AnfNodePtr> *split_outputs, size_t split_dim, size_t split_num,
                                            const std::vector<int64_t> &splits) {
  MS_ASSERT(orig_node != nullptr && split_outputs != nullptr);
  auto graph_node_input_shapes = Spliter::GetInstance()->graph_node_input_shapes();
  auto ori_node_name = orig_node->fullname_with_scope();
  auto input_shape_iter = graph_node_input_shapes.find(ori_node_name);
  if (input_shape_iter == graph_node_input_shapes.end()) {
    return nullptr;
  }
  auto input_shapes = input_shape_iter->second;
  auto input_shape = input_shapes.front();

  auto conv_prim = GetValueNode<std::shared_ptr<ops::Conv2DFusion>>(cnode_->input(kAnfPrimitiveIndex));
  MS_ASSERT(conv_prim != nullptr);
  // prim of split
  auto split_prim = std::make_shared<ops::SplitWithOverlap>();
  MS_CHECK_TRUE_RET(split_prim != nullptr, nullptr);
  std::vector<int64_t> new_splits = splits;
  if (split_mode_ == SplitH) {
    split_prim->set_extend_top(std::vector<int64_t>(split_num, 0));
    MS_CHECK_GE(conv_prim->get_kernel_size().size(), 1, nullptr);
    MS_CHECK_GE(conv_prim->get_stride().size(), 1, nullptr);
    auto extend_bottom = conv_prim->get_kernel_size().at(kIndexH) - conv_prim->get_stride().at(kIndexH);
    auto bottom_vector = std::vector<int64_t>(split_num, extend_bottom);
    MS_CHECK_GE(split_num, 1, nullptr);
    bottom_vector[split_num - 1] = 0;
    split_prim->set_extend_bottom(bottom_vector);
    MS_CHECK_GE(conv_prim->get_pad_list().size(), 1, nullptr);
    MS_CHECK_TRUE_RET(input_shape.size() == DIMENSION_4D, nullptr);
    if (!UpdateRatioWithPadStride(new_splits.data(), new_splits.size(), split_num, input_shape[split_dim],
                                  conv_prim->get_pad_list().at(kPadUp), conv_prim->get_stride().at(kIndexH))) {
      MS_LOG(ERROR) << "UpdateRatioWithPadStride failed";
      return nullptr;
    }
  } else {
    split_prim->set_extend_top(std::vector<int64_t>(split_num, 0));
    split_prim->set_extend_bottom(std::vector<int64_t>(split_num, 0));
  }
  split_prim->set_split_dim(split_dim);
  split_prim->set_number_split(split_num);
  split_prim->set_ratio(new_splits);

  auto split_primitive = NewValueNode(split_prim);
  MS_CHECK_TRUE_MSG(split_primitive != nullptr, nullptr, "create SplitWithOverlap return nullptr");
  std::vector<AnfNodePtr> split_inputs = {split_primitive};
  // ori_conv_node must only have one input
  split_inputs.push_back(orig_node->input(input_index + 1));
  auto split_cnode = func_graph_->NewCNode(split_inputs);
  if (split_cnode == nullptr) {
    MS_LOG(ERROR) << name_ << " : Failed to create split node.";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }
  split_cnode->set_fullname_with_scope("Split_" + name_);
  if (CreateMultipleOutputsOfAnfNode(split_cnode, split_num, split_outputs) != RET_OK) {
    return nullptr;
  }
  return split_cnode;
}

int Conv2DInfo::CheckConv2DPrimitiveType() {
  if (func_graph_ == nullptr) {
    return lite::RET_ERROR;
  }
  if (cnode_ == nullptr) {
    return lite::RET_ERROR;
  }
  if (!CheckPrimitiveType(cnode_, prim::kPrimConv2D) && !CheckPrimitiveType(cnode_, prim::kPrimConv2DFusion)) {
    return RET_ERROR;
  }
  auto prim = GetValueNode<PrimitivePtr>(cnode_->input(kAnfPrimitiveIndex));
  MS_ASSERT(prim != nullptr);
  MS_CHECK_TRUE_RET(prim->GetAttr(ops::kPad) != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(prim->GetAttr(ops::kInChannel) != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(prim->GetAttr(ops::kOutChannel) != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(prim->GetAttr(ops::kDilation) != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(prim->GetAttr(ops::kFormat) != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(prim->GetAttr(ops::kGroup) != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(prim->GetAttr(ops::kKernelSize) != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(prim->GetAttr(ops::kStride) != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(prim->GetAttr(ops::kActivationType) != nullptr, RET_ERROR);
  return RET_OK;
}

int Conv2DInfo::InferParallelCNodes() {
  if (CheckConv2DPrimitiveType() != RET_OK) {
    return RET_ERROR;
  }
  if (CheckIfSplit() != RET_OK) {
    return RET_ERROR;
  }
  size_t dev_num = strategy_.dev_num;
  std::vector<AnfNodePtr> feature_split_outputs;
  std::vector<AnfNodePtr> kernel_split_outputs;
  std::vector<AnfNodePtr> bias_split_outputs;
  std::string orig_name = name_;
  // split feature and kernel
  switch (split_mode_) {
    case SplitN:
    case SplitH: {
      name_ = orig_name + "_input";
      auto feature_split_cnode = CreateOutputsOfSplit(cnode_, 0, &feature_split_outputs, kAxisH, dev_num, splits_);
      if (CheckSplitResult(feature_split_cnode, feature_split_outputs, dev_num) != RET_OK) {
        return RET_ERROR;
      }
      break;
    }
    case SplitCIN:
    case SplitCOUT: {
      MS_LOG(ERROR) << "we do not split mode COUT or CIN";
      break;
    }
    default:
      MS_LOG(DEBUG) << "No Split mode chosen";
  }
  name_ = orig_name;
  parallel_output_nodes_.clear();
  auto conv_prim = GetValueNode<std::shared_ptr<ops::Conv2DFusion>>(cnode_->input(kAnfPrimitiveIndex));
  MS_ASSERT(conv_prim != nullptr);
  return ConstructOutputCNodes(conv_prim, feature_split_outputs, kernel_split_outputs, bias_split_outputs);
}

int Conv2DInfo::ConstructOutputCNodes(const std::shared_ptr<ops::Conv2DFusion> &conv_prim,
                                      const std::vector<AnfNodePtr> &feature_split_outputs,
                                      const std::vector<AnfNodePtr> &kernel_split_outputs,
                                      const std::vector<AnfNodePtr> &bias_split_outputs) {
  MS_ASSERT(conv_prim != nullptr);
  Strategys strategys = strategy_.strategys;
  size_t dev_num = strategy_.dev_num;
  int cin_strategy_sum = std::accumulate(strategys[0][kAxisCIn].begin(), strategys[0][kAxisCIn].end(), 0);
  int cout_strategy_sum = std::accumulate(strategys[1][kAxisCOut].begin(), strategys[1][kAxisCOut].end(), 0);
  std::string conv_cnode_name = cnode_->fullname_with_scope();
  // construct parallel Conv2D nodes
  for (size_t i = 0; i < dev_num; ++i) {
    std::vector<AnfNodePtr> tmp_outputs;
    bool has_bias = cnode_->size() >= kConvWithBias;
    // if split cin, only one parallel operator has bias
    if ((i != 0) && split_mode_ == SplitCIN) {
      has_bias = false;
    }
    // copy attr
    auto prim = std::make_shared<ops::Conv2DFusion>();
    MS_CHECK_TRUE_RET(prim != nullptr, RET_ERROR);
    prim->set_pad(conv_prim->get_pad());
    prim->set_pad_mode(PAD);
    prim->set_in_channel(conv_prim->get_in_channel());
    prim->set_out_channel(conv_prim->get_out_channel());
    prim->set_dilation(conv_prim->get_dilation());
    prim->set_format(conv_prim->get_format());
    prim->set_group(conv_prim->get_group());
    prim->set_kernel_size(conv_prim->get_kernel_size());
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
        MS_CHECK_TRUE_RET(cin_strategy_sum != 0, RET_ERROR);
        MS_CHECK_INT_MUL_NOT_OVERFLOW(in_channel, strategys[0][kAxisCIn][0], RET_ERROR);
        if (i == 0) {
          prim->set_in_channel(in_channel * strategys[0][kAxisCIn][0] / cin_strategy_sum);
        } else {
          prim->set_in_channel(in_channel - (in_channel * strategys[0][kAxisCIn][0] / cin_strategy_sum));
        }
      } break;
      case SplitCOUT: {
        auto out_channel = prim->get_out_channel();
        MS_CHECK_TRUE_RET(cout_strategy_sum != 0, RET_ERROR);
        MS_CHECK_INT_MUL_NOT_OVERFLOW(out_channel, strategys[1][kAxisCOut][0], RET_ERROR);
        if (i == 0) {
          prim->set_out_channel(out_channel * strategys[1][kAxisCOut][0] / cout_strategy_sum);
        } else {
          prim->set_out_channel(out_channel - (out_channel * strategys[1][kAxisCOut][0] / cout_strategy_sum));
        }
      } break;
      default:
        break;
    }
    std::vector<AnfNodePtr> conv_inputs;
    // if split Cout, feature will not be splited
    if (split_mode_ == SplitCOUT) {
      conv_inputs.push_back(cnode_->input(kAnfConvInput));
    } else {
      conv_inputs.push_back(feature_split_outputs[i]);
    }
    // kernel splited only when split Cin and Cout
    if (split_mode_ == SplitCIN || split_mode_ == SplitCOUT) {
      conv_inputs.push_back(kernel_split_outputs[i]);
    } else {
      conv_inputs.push_back(cnode_->input(kAnfConvWeight));
    }
    if (has_bias) {
      if (split_mode_ == SplitCOUT) {
        conv_inputs.push_back(bias_split_outputs[i]);
      } else {
        conv_inputs.push_back(cnode_->input(kAnfConvBias));
      }
    }
    auto conv_cnode = func_graph_->NewCNode(prim, conv_inputs);
    if (conv_cnode == nullptr) {
      MS_LOG(ERROR) << name_ << " : Failed to create parallel Conv2D node " << i;
      return lite::RET_ERROR;
    }
    conv_cnode->set_fullname_with_scope(conv_cnode_name + std::to_string(i));
    (void)CreateMultipleOutputsOfAnfNode(conv_cnode, 1, &tmp_outputs);
    parallel_output_nodes_.push_back(tmp_outputs[0]);
  }
  return lite::RET_OK;
}

int Conv2DInfo::InferReplaceOp() {
  size_t dev_num = strategy_.dev_num;
  if (split_mode_ == SplitCIN) {
    MS_LOG(DEBUG) << name_ << " : Split Cin, infer Forward op.";
    replace_op_ = CreateReduceNode(cnode_, parallel_output_nodes_, dev_num);
  } else {
    int32_t concat_dim;
    if (split_mode_ == SplitN) {
      concat_dim = kAxisN;
    } else if (split_mode_ == SplitCOUT) {
      // output format is same as feature map
      concat_dim = kAxisCOut;
    } else {
      concat_dim = kAxisH;
    }
    replace_op_ = CreateConcateNode(cnode_, parallel_output_nodes_, concat_dim, dev_num);
  }

  if (replace_op_ == nullptr) {
    return RET_ERROR;
  }
  return RET_OK;
}

OPERATOR_INFO_REGISTER(PrimitiveType_Conv2DFusion, kNumberTypeFloat32, false, OperatorInfoCreator<Conv2DInfo>)
OPERATOR_INFO_REGISTER(PrimitiveType_Conv2DFusion, kNumberTypeInt8, false, OperatorInfoCreator<Conv2DInfo>)
}  // namespace opt
}  // namespace mindspore
