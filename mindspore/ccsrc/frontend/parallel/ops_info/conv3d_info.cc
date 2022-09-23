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

#include "frontend/parallel/ops_info/conv3d_info.h"

#include <algorithm>
#include <functional>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "include/common/utils/parallel_context.h"
#include "pipeline/jit/resource.h"

namespace mindspore {
namespace parallel {
std::vector<int64_t> Conv3DInfo::GetStrideAttr() { return GetTupleIntAttr(STRIDES); }

std::vector<int64_t> Conv3DInfo::GetDilationAttr() { return GetTupleIntAttr(DILATIONS); }

Status Conv3DInfo::CheckAttrsBase() {
  if (format_ != NCDHW) {
    MS_LOG(ERROR) << name_ << ": The format must be 'NCDHW', but got " << format_;
    return FAILED;
  }

  if (kernel_size_.size() != 3) {
    MS_LOG(ERROR) << name_ << ": The size of kernel_size'tuple must be 3, but got " << kernel_size_.size();
    return FAILED;
  }

  if (pad_list_.size() != 6) {
    MS_LOG(ERROR) << name_ << ": The size of pad_list must be 6, but got " << pad_list_.size();
    return FAILED;
  }

  if (stride_.size() != 5) {
    MS_LOG(ERROR) << name_ << ": The size of stride must be 5, but got " << stride_.size();
    return FAILED;
  }

  if (stride_[0] != 1 || stride_[1] != 1) {
    MS_LOG(ERROR) << name_ << ": The first two elements of stride must be 1, but the stride is " << stride_;
    return FAILED;
  }

  if (dilation_.size() != 5) {
    MS_LOG(ERROR) << name_ << ": The size of dilation must be 5, but got " << dilation_.size();
    return FAILED;
  }
  MS_LOG(INFO) << name_ << ": The out channel is " << out_channel_ << ", kernel size is " << kernel_size_
               << ", mode is " << mode_ << ", pad mode is " << pad_mode_ << ", pad list is " << pad_list_
               << ", stride is " << stride_ << ", dilation is " << dilation_ << ", group is " << group_
               << ", format is " << format_ << ", the kernel size use dilation is " << kernel_size_use_dilation_;
  return SUCCESS;
}

Status Conv3DInfo::CheckStrategy(const StrategyPtr &strategy) {
  h_dim_need_exchange_overlap_ = false;
  w_dim_need_exchange_overlap_ = false;
  if (CheckStrategyBase(strategy) != SUCCESS) {
    return FAILED;
  }

  std::vector<Dimensions> stra = strategy->GetInputDim();
  Dimensions input_strategy = stra[0];
  Dimensions weight_strategy = stra[1];
  if (input_strategy.size() != 5 || weight_strategy.size() != 5) {
    MS_LOG(ERROR) << name_
                  << ": The size of input strategy or weight strategy must be 5, but the size of input strategy is "
                  << input_strategy.size() << ", the size of weight strategy is " << weight_strategy.size();
    return FAILED;
  }

  if (input_strategy[1] != weight_strategy[1]) {
    MS_LOG(ERROR) << name_ << ": The shard num of c-in for input strategy is " << input_strategy[1]
                  << ", but the shard num of c-in for weight strategy is " << weight_strategy[1];
    return FAILED;
  }

  if (weight_strategy[2] != 1 || weight_strategy[3] != 1 || weight_strategy[4] != 1) {
    MS_LOG(ERROR) << name_ << ": The kernel size can not be split, but the strategy for kernel size is ("
                  << weight_strategy[2] << ", " << weight_strategy[3] << ", " << weight_strategy[4] << ")";
    return FAILED;
  }

  if (input_strategy[4] != 1) {
    MS_LOG(ERROR) << name_
                  << ": Do not support to split the last dimension of input, but the strategy for this dimension is ("
                  << input_strategy[4];
    return FAILED;
  }

  if (input_strategy[2] != 1 || input_strategy[3] != 1) {
    if (CheckHWStrategy(input_strategy[2], input_strategy[3]) != SUCCESS) {
      return FAILED;
    }
  }

  // if the h/w dimension is split, and the pad mode is not "valid", need to exchange overlap
  if (input_strategy[2] > 1 && pad_mode_ != 2) {
    h_dim_need_exchange_overlap_ = true;
  }

  if (input_strategy[3] > 1 && pad_mode_ != 2) {
    w_dim_need_exchange_overlap_ = true;
  }
  return SUCCESS;
}

Status Conv3DInfo::InferTensorMap() {
  // input_strategy: ((n, i, a, b, 1), (o, i, 1, 1, 1))
  // output_strategy: ((n, o, a, b, 1),)
  // dev_matrix: (n, i, a, b, o)
  TensorMap input_tensor_map = {4, 3, 2, 1, -1};
  TensorMap weight_tensor_map = {0, 3, -1, -1, -1};
  TensorMap output_tensor_map = {4, 0, 2, 1, -1};

  (void)inputs_tensor_map_.emplace_back(std::move(input_tensor_map));
  (void)inputs_tensor_map_.emplace_back(std::move(weight_tensor_map));
  (void)outputs_tensor_map_.emplace_back(std::move(output_tensor_map));
  return SUCCESS;
}

std::string Conv3DInfo::ReplaceNodeName() const {
  if (name_.find(CONV3D_INFO) != std::string::npos) {
    return CONV3D;
  }

  MS_LOG(EXCEPTION) << "Invalid name: " << name_;
}

OperatorAttrs Conv3DInfo::CreateConv3DAttrs() {
  auto node_stride = stride_;
  (void)node_stride.erase(node_stride.begin(), node_stride.begin() + 2);
  auto node_dilition = dilation_;
  (void)node_dilition.erase(node_dilition.begin(), node_dilition.begin() + 2);

  Attr out_channel = {OUT_CHANNEL, MakeValue(new_out_channel_)};
  Attr kernel_size = {KERNEL_SIZE, MakeValue(kernel_size_)};
  Attr mode = {MODE, MakeValue(mode_)};
  Attr pad_mode = {PAD_MODE, MakeValue("pad")};
  Attr pad = {PAD, MakeValue(new_pad_list_)};
  Attr stride = {STRIDE, MakeValue(node_stride)};
  Attr dilation = {DILATION, MakeValue(node_dilition)};
  Attr group = {GROUP, MakeValue(group_)};
  Attr data_format = {DATA_FORMAT, MakeValue(format_)};

  OperatorAttrs attrs;
  attrs = {out_channel, kernel_size, mode, pad_mode, pad, stride, dilation, group, data_format};
  return attrs;
}

AnfNodePtr Conv3DInfo::GenerateConv3DNode(const AnfNodePtr &new_input, const CNodePtr &cnode) {
  auto conv3d_attrs = CreateConv3DAttrs();
  auto node_name = ReplaceNodeName();

  if (cnode->size() < 3) {
    MS_LOG(EXCEPTION) << name_ << ": The size of cnode is invalid: " << cnode->size();
  }
  return gen_g_.PushBack({gen_g_.NewOpInst(node_name, conv3d_attrs), new_input, cnode->input(2)});
}

void Conv3DInfo::ComputeReplaceGraph(const CNodePtr &cnode) {
  // Because the NeighborExchangeV2 only support the 4-dim input, and it only exchange the last 2-dim of input, but the
  // input of conv3d is 5-dim, and need to exchange 3/4th-dim of input, so here use some operators to build the graph:
  // slice input (ncdhw) -> transpose(in, (4, 0, 1, 2, 3)) -> reshape(in, (w*n, c, d, h)) -> neighborexchangev2(in)
  // -> reshape(in, (w, n, c, d', h')) -> transpose(in, (1, 2, 3, 4, 0)) -> conv3d
  auto graph = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(graph);

  if (gen_g_.Init(cnode) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": GenerateGraph Init failed";
  }

  // transpose-1
  std::vector<int64_t> t1 = {4, 0, 1, 2, 3};
  auto transpose_1 = gen_g_.PushBack({gen_g_.NewOpInst(TRANSPOSE), gen_g_.virtual_input_node(), CreateTuple(t1)});

  // reshape-1
  auto s = input_slice_shape_;
  if (s.size() != 5) {
    MS_LOG(EXCEPTION) << name_ << ": The size of input slice shape must be 5, but got " << s.size();
  }
  Shape s1 = {s[4] * s[0], s[1], s[2], s[3]};
  auto reshape_1 = gen_g_.PushBack({gen_g_.NewOpInst(RESHAPE), transpose_1, CreateTuple(s1)});

  // neighborexchangev2
  auto neighbor_exchange_v2_attrs = CreateNeighborExchangeV2Attrs();
  auto neighbor_exchange_v2 =
    gen_g_.PushBack({gen_g_.NewOpInst(NEIGHBOREXCHANGEV2, neighbor_exchange_v2_attrs), reshape_1});

  // reshape-2
  Shape s2 = {s[4], s[0], s[1], s[2] + recv_lens_[0] + recv_lens_[1], s[3] + recv_lens_[2] + recv_lens_[3]};
  auto reshape_2 = gen_g_.PushBack({gen_g_.NewOpInst(RESHAPE), neighbor_exchange_v2, CreateTuple(s2)});

  // transopse-2
  std::vector<int64_t> t2 = {1, 2, 3, 4, 0};
  auto transpose_2 = gen_g_.PushBack({gen_g_.NewOpInst(TRANSPOSE), reshape_2, CreateTuple(t2)});

  // conv3d
  auto conv3d = GenerateConv3DNode(transpose_2, cnode);

  std::vector<std::pair<AnfNodePtr, int64_t>> input_nodes = {std::make_pair(transpose_1, 1)};
  replace_graph_ = std::make_shared<std::pair<std::vector<std::pair<AnfNodePtr, int64_t>>, AnfNodePtr>>(
    std::make_pair(input_nodes, conv3d));
}

REGISTER(Conv3DInfo);
}  // namespace parallel
}  // namespace mindspore
