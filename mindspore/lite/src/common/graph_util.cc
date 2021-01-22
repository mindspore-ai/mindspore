/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include <fstream>
#include <sstream>
#include <utility>
#include "src/common/graph_util.h"
#include "src/common/utils.h"
#include "src/common/log_adapter.h"
#include "include/errorcode.h"

namespace mindspore {
namespace lite {
std::vector<size_t> GetGraphInputNodes(const lite::Model *model) {
  MS_ASSERT(model != nullptr);
  MS_ASSERT(!(model->sub_graphs_.empty()));
  std::vector<size_t> ret;
  for (auto graph_in_index : model->sub_graphs_.front()->input_indices_) {
    auto node_size = model->all_nodes_.size();
    for (size_t j = 0; j < node_size; ++j) {
      auto node = model->all_nodes_[j];
      MS_ASSERT(node != nullptr);
      if (std::any_of(node->input_indices_.begin(), node->input_indices_.end(),
                      [&](const uint32_t &node_in_index) { return node_in_index == graph_in_index; })) {
        if (!IsContain<size_t>(ret, j)) {
          ret.emplace_back(j);
        }
      }
    }
  }
  return ret;
}

std::vector<size_t> GetGraphOutputNodes(const lite::Model *model) {
  MS_ASSERT(model != nullptr);
  std::vector<size_t> ret;
  for (auto graph_out_index : model->sub_graphs_.front()->output_indices_) {
    auto node_size = model->all_nodes_.size();
    for (size_t j = 0; j < node_size; ++j) {
      auto node = model->all_nodes_[j];
      MS_ASSERT(node != nullptr);
      if (std::any_of(node->output_indices_.begin(), node->output_indices_.end(),
                      [&](const uint32_t &node_out_index) { return node_out_index == graph_out_index; })) {
        if (!IsContain<size_t>(ret, j)) {
          ret.emplace_back(j);
        }
      }
    }
  }
  return ret;
}

std::vector<size_t> GetLinkedPostNodeIdx(const lite::Model *model, const size_t tensor_idx) {
  MS_ASSERT(model != nullptr);
  std::vector<size_t> post_node_idxes;
  auto nodes_size = model->all_nodes_.size();
  for (size_t i = 0; i < nodes_size; ++i) {
    auto node = model->all_nodes_[i];
    if (node == nullptr) {
      continue;
    }

    auto is_contain = std::any_of(node->input_indices_.begin(), node->input_indices_.end(),
                                  [&](const uint32_t &node_input_idx) { return node_input_idx == tensor_idx; });
    if (is_contain) {
      post_node_idxes.emplace_back(i);
    }
  }
  return post_node_idxes;
}

bool IsPackedOp(schema::PrimitiveType op_type) {
  static std::vector<schema::PrimitiveType> packed_ops = {
    schema::PrimitiveType_Conv2D,          schema::PrimitiveType_DeConv2D,
    schema::PrimitiveType_DepthwiseConv2D, schema::PrimitiveType_DeDepthwiseConv2D,
    schema::PrimitiveType_MatMul,          schema::PrimitiveType_Lstm};
  return IsContain(packed_ops, op_type);
}
}  // namespace lite
}  // namespace mindspore
