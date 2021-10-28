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

#include "tools/optimizer/parallel/depthwise_conv2d_info.h"
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include "nnacl/op_base.h"
#include "ops/fusion/conv2d_fusion.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "utils/utils.h"
#include "tools/converter/converter_flags.h"
#include "include/errorcode.h"
#include "tools/optimizer/parallel/operator_info_register.h"
#include "tools/optimizer/fisson/fisson_util.h"
#include "ops/split_with_overlap.h"
#include "src/tensor.h"
#include "tools/optimizer/parallel/spliter.h"

using mindspore::schema::PrimitiveType_Conv2DFusion;
namespace mindspore {
namespace opt {
namespace {
void SplitConstantData(char *in_data, char **out_data, int64_t num_split, int64_t split_dim_size, int64_t element_bytes,
                       int64_t outer_total_dim, int64_t inner_stride, const int64_t *start_indices,
                       const int64_t *end_indices) {
  MS_ASSERT(in_data != nullptr && out_data != nullptr && start_indices != nullptr && end_indices != nullptr);
  int64_t input_stride = split_dim_size * inner_stride * element_bytes;
  for (int64_t slice_idx = 0; slice_idx < num_split; slice_idx++) {
    int out_stride = (end_indices[slice_idx] - start_indices[slice_idx]) * inner_stride * element_bytes;
    char *src_ptr = in_data + start_indices[slice_idx] * inner_stride * element_bytes;
    for (int64_t out_idx = 0; out_idx < outer_total_dim; out_idx++) {
      (void)(memcpy(out_data[slice_idx] + out_idx * out_stride, src_ptr, out_stride));
      src_ptr += input_stride;
    }
  }
}

void CreateSplitConstantTensors(const tensor::TensorPtr &constant_tensor, const std::vector<int64_t> &splits,
                                int64_t split_dim, std::vector<tensor::TensorPtr> *split_constant_tensors) {
  MS_ASSERT(constant_tensor != nullptr && split_constant_tensors != nullptr);
  auto constant_shape = constant_tensor->shape();
  auto weight_type_id = constant_tensor->data_type();
  int64_t total_block_count = 0;
  int64_t split_num = static_cast<int64_t>(splits.size());
  for (int64_t i = 0; i < split_num; i++) {
    total_block_count += splits.at(i);
  }
  MS_ASSERT(split_dim < static_cast<int64_t>(constant_shape.size()));
  int64_t split_dim_size = constant_shape[split_dim];
  int64_t visited_block = 0;
  std::vector<ShapeVector> split_constant_shapes(split_num, ShapeVector(constant_shape));
  for (int64_t i = 0; i < split_num; i++) {
    // init shape for [split_dim]
    visited_block += splits[i];
    if (total_block_count == 0) {
      MS_LOG(ERROR) << "divisor is zero";
      split_constant_tensors->clear();
      return;
    }
    auto cur_shape = UP_DIV(split_dim_size * visited_block, total_block_count);
    split_constant_shapes.at(i).at(split_dim) = cur_shape;
    auto tensor = std::make_shared<tensor::Tensor>(weight_type_id, split_constant_shapes.at(i));
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "make shared failed.";
      split_constant_tensors->clear();
      return;
    }
    split_constant_tensors->push_back(std::move(tensor));
  }

  std::vector<int64_t> borders;
  borders.emplace_back(0);
  visited_block = 0;
  for (int64_t i = 0; i < split_num - 1; i++) {
    visited_block += splits[i];
    if (total_block_count == 0) {
      MS_LOG(ERROR) << "divisor is zero.";
      split_constant_tensors->clear();
      return;
    }
    auto cur_border = UP_DIV(split_dim_size * visited_block, total_block_count);
    borders.emplace_back(cur_border);
  }
  borders.emplace_back(split_dim_size);
  std::vector<int64_t> start_indices;
  std::vector<int64_t> end_indices;
  for (int64_t i = 0; i < split_num; i++) {
    start_indices.emplace_back(borders[i]);
    end_indices.emplace_back(borders[i + 1]);
  }
  int64_t element_bytes = static_cast<int64_t>(lite::DataTypeSize(constant_tensor->data_type()));
  std::vector<char *> split_constant_tensors_ptr;
  std::transform(
    split_constant_tensors->begin(), split_constant_tensors->end(), split_constant_tensors_ptr.end(),
    [&](const tensor::TensorPtr &constant_tensor) { return (reinterpret_cast<char *>(constant_tensor->data_c())); });
  int64_t outer_total_dim = 1;
  for (int64_t i = 0; i < split_dim; i++) {
    outer_total_dim *= static_cast<size_t>(constant_shape[i]);
  }
  int64_t inner_stride = 1;
  for (int64_t i = static_cast<int64_t>(constant_shape.size()) - 1; i > split_dim; i--) {
    inner_stride *= static_cast<size_t>(constant_shape[i]);
  }
  auto constant_tensor_ptr = reinterpret_cast<char *>(constant_tensor->data_c());
  // init split_constant_tensor_data
  SplitConstantData(constant_tensor_ptr, split_constant_tensors_ptr.data(), split_num, split_dim_size, element_bytes,
                    outer_total_dim, inner_stride, start_indices.data(), end_indices.data());
}

}  // namespace

int DepthwiseConv2DInfo::CheckStrategy(const SplitStrategy &strategy) {
  MS_LOG(INFO) << "DepthwiseConv2DInfo check strategy start";
  // for depthwise conv2d, we only split channel && include split feature map, weight && bias
  // so just get the ratio from strategy
  int split_count = 0;
  Strategys strategys = strategy.strategys;
  MS_CHECK_GE(strategys.size(), kInputSizeTwo, RET_ERROR);
  MS_CHECK_GE(strategys[0].size(), kInputSizeFour, RET_ERROR);
  MS_CHECK_GE(strategys[1].size(), kInputSizeFour, RET_ERROR);
  if (is_any_not_none(strategys[0][kAxisN])) {
    split_count++;
    splits_ = strategys[0][kAxisN];
    split_mode_ = SplitN;
    split_dim_ = kAxisN;
  }
  // if split CIN
  if (is_any_not_none(strategys[0][kAxisCIn])) {
    split_count++;
    splits_ = strategys[0][kAxisCIn];
    split_mode_ = SplitCIN;
    split_dim_ = kAxisCIn;
  }
  // if split COUT
  if (is_any_not_none(strategys[1][kAxisCOut])) {
    split_count++;
    splits_ = strategys[1][kAxisCOut];
    split_mode_ = SplitCOUT;
    split_dim_ = kAxisCOut;
  }
  // if splitH
  if (is_any_not_none(strategys[0][kAxisH])) {
    split_count++;
    splits_ = strategys[0][kAxisH];
    split_mode_ = SplitH;
    split_dim_ = kAxisH;
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
  MS_LOG(INFO) << "DepthwiseConv2DInfo check strategy end";
  return RET_OK;
}

bool DepthwiseConv2DInfo::CheckSplitOutputs(const std::vector<AnfNodePtr> &feature_split_outputs,
                                            const std::vector<AnfNodePtr> &kernel_split_outputs,
                                            const std::vector<AnfNodePtr> &bias_split_outputs) {
  size_t dev_num = splits_.size();
  if (feature_split_outputs.size() != dev_num) {
    return false;
  }
  if (kernel_split_outputs.size() != dev_num) {
    return false;
  }

  bool has_bias = cnode_->size() > kBiasIndex + 1;
  if (has_bias) {
    if (bias_split_outputs.size() != dev_num) {
      return false;
    }
  } else {
    if (!bias_split_outputs.empty()) {
      return false;
    }
  }
  return true;
}

void DepthwiseConv2DInfo::AdJustConvPrim(const std::shared_ptr<ops::Conv2DFusion> &conv_prim,
                                         int64_t *visited_in_channel, int64_t *visited_out_channel,
                                         int64_t *visited_group, int output_conv_index) {
  MS_ASSERT(conv_prim != nullptr && visited_in_channel != nullptr);
  MS_ASSERT(visited_out_channel != nullptr && visited_group != nullptr);
  int64_t dev_num = static_cast<int64_t>(splits_.size());
  int64_t total_ratio = std::accumulate(splits_.begin(), splits_.end(), 0);
  int64_t in_channel = conv_prim->get_in_channel();
  int64_t out_channel = conv_prim->get_out_channel();
  int64_t group = conv_prim->get_group();
  switch (split_mode_) {
    case SplitN: {
      break;
    }
    case SplitH: {
      if (output_conv_index != 0) {
        auto pad = conv_prim->get_pad_list();
        pad.at(kPadUp) = 0;
        conv_prim->set_pad_list(pad);
      }
      if (output_conv_index != (dev_num - 1)) {
        auto pad = conv_prim->get_pad_list();
        pad.at(kPadDown) = 0;
        conv_prim->set_pad_list(pad);
      }
      break;
    }
    case SplitCOUT:
    case SplitCIN: {
      if (output_conv_index != dev_num - 1) {
        NNACL_CHECK_ZERO_RETURN(total_ratio);
        auto curr_channel = in_channel * splits_.at(output_conv_index) / total_ratio;
        conv_prim->set_in_channel(curr_channel);
        (*visited_in_channel) += curr_channel;
      } else {
        conv_prim->set_in_channel(in_channel - *visited_in_channel);
      }

      if (output_conv_index != dev_num - 1) {
        NNACL_CHECK_ZERO_RETURN(total_ratio);
        auto curr_channel = out_channel * splits_.at(output_conv_index) / total_ratio;
        conv_prim->set_out_channel(curr_channel);
        (*visited_out_channel) += curr_channel;
      } else {
        conv_prim->set_out_channel(out_channel - *visited_out_channel);
      }

      if (output_conv_index != dev_num - 1) {
        NNACL_CHECK_ZERO_RETURN(total_ratio);
        auto curr_group = group * splits_.at(output_conv_index) / total_ratio;
        conv_prim->set_group(curr_group);
        (*visited_group) += curr_group;
      } else {
        conv_prim->set_group(group - *visited_group);
      }
      break;
    }
    default: {
      break;
    }
  }
}

void DepthwiseConv2DInfo::AdJustInputs(const std::shared_ptr<ops::Conv2DFusion> &conv_prim,
                                       const std::vector<AnfNodePtr> &feature_split_outputs,
                                       const std::vector<AnfNodePtr> &kernel_split_outputs,
                                       const std::vector<AnfNodePtr> &bias_split_outputs, int output_conv_index) {
  MS_ASSERT(conv_prim != nullptr);
  std::vector<AnfNodePtr> tmp_outputs;
  std::string conv_cnode_name = cnode_->fullname_with_scope();
  bool has_bias = cnode_->size() > kBiasIndex + 1;
  std::vector<AnfNodePtr> conv_inputs = {NewValueNode(conv_prim)};
  if (split_mode_ == SplitN || split_mode_ == SplitH) {
    conv_inputs.push_back(feature_split_outputs.at(output_conv_index));
    conv_inputs.push_back(cnode_->input(kWeightIndex + 1));
    if (has_bias) {
      conv_inputs.push_back(cnode_->input(kBiasIndex + 1));
    }
  } else if (split_mode_ == SplitCIN) {
    conv_inputs.push_back(feature_split_outputs.at(output_conv_index));
    conv_inputs.push_back(kernel_split_outputs[output_conv_index]);
    if (has_bias) {
      conv_inputs.push_back(bias_split_outputs[output_conv_index]);
    }
  } else {
    conv_inputs.push_back(cnode_->input(1));
    conv_inputs.push_back(kernel_split_outputs[output_conv_index]);
    conv_inputs.push_back(bias_split_outputs[output_conv_index]);
  }
  // create new depthwise_conv node
  auto conv_cnode = func_graph_->NewCNode(conv_inputs);
  if (conv_cnode == nullptr) {
    MS_LOG(ERROR) << "new a cnode failed.";
    return;
  }
  conv_cnode->set_fullname_with_scope(conv_cnode_name + std::to_string(output_conv_index));
  (void)CreateMultipleOutputsOfAnfNode(conv_cnode, 1, &tmp_outputs);
  // remember depthwise conv to create concat
  parallel_output_nodes_.push_back(tmp_outputs[0]);
}

int DepthwiseConv2DInfo::ConstructOutputCNodes(const std::shared_ptr<ops::Conv2DFusion> &conv_prim,
                                               const std::vector<AnfNodePtr> &feature_split_outputs,
                                               const std::vector<AnfNodePtr> &kernel_split_outputs,
                                               const std::vector<AnfNodePtr> &bias_split_outputs) {
  MS_ASSERT(conv_prim != nullptr);
  if (!CheckSplitOutputs(feature_split_outputs, kernel_split_outputs, bias_split_outputs)) {
    return RET_ERROR;
  }
  int64_t visited_in_channel = 0;
  int64_t visited_out_channel = 0;
  int64_t visited_group = 0;
  // construct parallel Conv2D nodes
  int dev_num = static_cast<int>(splits_.size());
  for (int i = 0; i < dev_num; ++i) {
    auto new_depth_conv_prim = CopyConvPrim(conv_prim);
    MS_CHECK_TRUE_RET(new_depth_conv_prim != nullptr, RET_ERROR);
    new_depth_conv_prim->set_pad_mode(PAD);
    AdJustConvPrim(new_depth_conv_prim, &visited_in_channel, &visited_out_channel, &visited_group, i);
    AdJustInputs(new_depth_conv_prim, feature_split_outputs, kernel_split_outputs, bias_split_outputs, i);
  }
  return RET_OK;
}

AnfNodePtr DepthwiseConv2DInfo::CreateOutputsOfSplit(const CNodePtr &ori_node, size_t input_index,
                                                     std::vector<AnfNodePtr> *split_outputs, size_t split_num,
                                                     const std::vector<int64_t> &splits) {
  MS_ASSERT(orig_node != nullptr && split_outputs != nullptr);
  auto depth_wise_conv_prim = GetValueNode<std::shared_ptr<ops::Conv2DFusion>>(cnode_->input(kAnfPrimitiveIndex));
  MS_ASSERT(depth_wise_conv_prim != nullptr);
  auto ori_node_name = ori_node->fullname_with_scope();
  auto graph_node_input_shapes = Spliter::GetInstance()->graph_node_input_shapes();
  auto input_shape_iter = graph_node_input_shapes.find(ori_node_name);
  if (input_shape_iter == graph_node_input_shapes.end()) {
    return nullptr;
  }
  auto input_shapes = input_shape_iter->second;
  auto input_shape = input_shapes.front();
  MS_CHECK_TRUE_RET(input_shape.size() > kInputSizeTwo, nullptr);
  int64_t input_h = input_shape.at(kAxisH);
  int64_t input_w = input_shape.at(kAxisW);
  auto pad_list = GetSplitPadList(depth_wise_conv_prim, input_h, input_w);
  depth_wise_conv_prim->set_pad_list(pad_list);
  depth_wise_conv_prim->set_pad_mode(PAD);

  // prim of split
  auto split_prim = std::make_shared<ops::SplitWithOverlap>();
  MS_CHECK_TRUE_RET(split_prim != nullptr, nullptr);
  std::vector<int64_t> new_splits = splits;
  MS_CHECK_TRUE_RET(input_shape.size() > static_cast<size_t>(split_dim_), nullptr);
  if (split_mode_ == SplitH) {
    split_prim->set_extend_top(std::vector<int64_t>(split_num, 0));
    MS_CHECK_GE(depth_wise_conv_prim->get_kernel_size().size(), 1, nullptr);
    MS_CHECK_GE(depth_wise_conv_prim->get_stride().size(), 1, nullptr);
    auto extend_bottom =
      depth_wise_conv_prim->get_kernel_size().at(kIndexH) - depth_wise_conv_prim->get_stride().at(kIndexH);
    auto bottom_vector = std::vector<int64_t>(split_num, extend_bottom);
    bottom_vector[split_num - 1] = 0;
    split_prim->set_extend_bottom(bottom_vector);
    if (!UpdateRatioWithPadStride(new_splits.data(), new_splits.size(), split_num, input_shape[split_dim_],
                                  depth_wise_conv_prim->get_pad_list().at(kPadUp),
                                  depth_wise_conv_prim->get_stride().at(kIndexH))) {
      MS_LOG(ERROR) << "UpdateRatioWithPadStride failed";
      return nullptr;
    }
  } else {
    split_prim->set_extend_top(std::vector<int64_t>(split_num, 0));
    split_prim->set_extend_bottom(std::vector<int64_t>(split_num, 0));
  }
  split_prim->set_split_dim(split_dim_);
  split_prim->set_number_split(split_num);
  split_prim->set_ratio(new_splits);

  std::vector<AnfNodePtr> split_inputs;
  // ori_conv_node must only have one feature input
  split_inputs.push_back(ori_node->input(input_index + 1));
  auto split_cnode = func_graph_->NewCNode(split_prim, split_inputs);
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

int DepthwiseConv2DInfo::CheckDepthWiseConv2DPrimitiveType() {
  auto prim = GetValueNode<PrimitivePtr>(cnode_->input(kAnfPrimitiveIndex));
  MS_CHECK_TRUE_RET(prim != nullptr, RET_ERROR);
  // depth_wise can not be splited in conv_info, we deal with in depthwise_conv_info
  bool is_depth_wise = prim->GetAttr(ops::kIsDepthWise) != nullptr && GetValue<bool>(prim->GetAttr(ops::kIsDepthWise));
  if (!is_depth_wise) {
    return RET_ERROR;
  }
  if (!CheckPrimitiveType(cnode_, prim::kPrimConv2D) && !CheckPrimitiveType(cnode_, prim::kPrimConv2DFusion)) {
    return RET_ERROR;
  }
  MS_CHECK_TRUE_RET(prim->GetAttr(ops::kInChannel) != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(prim->GetAttr(ops::kOutChannel) != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(prim->GetAttr(ops::kGroup) != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(prim->GetAttr(ops::kKernelSize) != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(prim->GetAttr(ops::kStride) != nullptr, RET_ERROR);
  return RET_OK;
}

int DepthwiseConv2DInfo::CreateConstantOutputsOfSplit(std::vector<AnfNodePtr> *split_outputs, int input_index) {
  // split depthwise_conv weight && bias offline
  MS_ASSERT(split_outputs != nullptr);
  int64_t split_dim = kAxisCOut;
  if (input_index == kBiasIndex) {
    split_dim = 0;
  }
  auto constant_node = cnode_->input(input_index + 1);
  auto constant_tensor = GetTensorInfo(constant_node);
  MS_CHECK_TRUE_RET(constant_tensor != nullptr, RET_ERROR);
  std::vector<tensor::TensorPtr> split_constant_tensors;
  CreateSplitConstantTensors(constant_tensor, splits_, split_dim, &split_constant_tensors);
  if (split_constant_tensors.empty()) {
    return RET_ERROR;
  }
  for (const auto &split_constant_tensor : split_constant_tensors) {
    auto parameter_node = func_graph_->add_parameter();
    MS_EXCEPTION_IF_NULL(parameter_node);
    auto type_id_ptr = TypeIdToType(split_constant_tensor->data_type());
    parameter_node->set_abstract(
      std::make_shared<abstract::AbstractTensor>(type_id_ptr, split_constant_tensor->shape()));
    MS_CHECK_TRUE_RET(parameter_node->abstract() != nullptr, RET_ERROR);
    parameter_node->set_default_param(split_constant_tensor);
    parameter_node->set_name(name_);
    split_outputs->push_back(parameter_node);
  }
  if (split_outputs->empty()) {
    return RET_ERROR;
  }
  return RET_OK;
}

int DepthwiseConv2DInfo::InferParallelCNodes() {
  if (CheckDepthWiseConv2DPrimitiveType() != RET_OK) {
    return RET_ERROR;
  }
  size_t dev_num = strategy_.dev_num;
  std::string input_op_name = name_;
  std::vector<AnfNodePtr> feature_split_outputs;
  std::vector<AnfNodePtr> kernel_split_outputs;
  std::vector<AnfNodePtr> bias_split_outputs;
  switch (split_mode_) {
    case SplitN:
    case SplitH: {
      name_ = input_op_name + ("_input");
      auto feature_split_cnode = CreateOutputsOfSplit(cnode_, 0, &feature_split_outputs, dev_num, splits_);
      MS_CHECK_TRUE_RET(feature_split_cnode != nullptr, RET_ERROR);
      if (CheckSplitResult(feature_split_cnode, feature_split_outputs, dev_num) != RET_OK) {
        return RET_ERROR;
      }
      break;
    }
    case SplitCIN:
    case SplitCOUT: {
      name_ = input_op_name + ("_input");
      auto feature_split_cnode = CreateOutputsOfSplit(cnode_, 0, &feature_split_outputs, dev_num, splits_);
      MS_CHECK_TRUE_RET(feature_split_cnode != nullptr, RET_ERROR);
      if (CheckSplitResult(feature_split_cnode, feature_split_outputs, dev_num) != RET_OK) {
        return RET_ERROR;
      }
      name_ = input_op_name + ("_kernel");
      if (CreateConstantOutputsOfSplit(&kernel_split_outputs, 1) != RET_OK) {
        return RET_ERROR;
      }
      if (cnode_->size() > kBiasIndex + 1) {
        name_ = input_op_name + ("_bias");
        if (CreateConstantOutputsOfSplit(&bias_split_outputs, 2) != RET_OK) {
          return RET_ERROR;
        }
      }
      break;
    }
    default: {
      break;
    }
  }
  name_ = input_op_name;
  auto depth_wise_conv_prim = GetValueNode<std::shared_ptr<ops::Conv2DFusion>>(cnode_->input(kAnfPrimitiveIndex));
  MS_ASSERT(depth_wise_conv_prim != nullptr);
  return ConstructOutputCNodes(depth_wise_conv_prim, feature_split_outputs, kernel_split_outputs, bias_split_outputs);
}

int DepthwiseConv2DInfo::InferReplaceOp() {
  size_t dev_num = strategy_.dev_num;
  replace_op_ = CreateConcateNode(cnode_, parallel_output_nodes_, split_dim_, dev_num);
  if (replace_op_ == nullptr) {
    return RET_ERROR;
  }
  return RET_OK;
}

OPERATOR_INFO_REGISTER(PrimitiveType_Conv2DFusion, kNumberTypeFloat32, true, OperatorInfoCreator<DepthwiseConv2DInfo>)
OPERATOR_INFO_REGISTER(PrimitiveType_Conv2DFusion, kNumberTypeInt8, true, OperatorInfoCreator<DepthwiseConv2DInfo>)
}  // namespace opt
}  // namespace mindspore
