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

#include "src/litert/pass/online_fusion/online_fusion_pass.h"
#include <vector>
#include "src/common/ops/populate/populate_register.h"
#include "nnacl/split_parameter.h"
#include "nnacl/reduce_parameter.h"
#include "nnacl/concat_parameter.h"
#include "include/model.h"

namespace mindspore::lite {
OnlineFusionPass::OnlineFusionPass(SearchSubGraph *search_subgrap) : search_subgrap_(search_subgrap) {}

OnlineFusionPass::~OnlineFusionPass() {
  context_ = nullptr;
  model_ = nullptr;
  search_subgrap_ = nullptr;
}

void OnlineFusionPass::DoOnlineFusionPass() {
  if (InitOnlineFusion() == RET_OK) {
    DoOnlineFusion();
  }
}

int OnlineFusionPass::InitOnlineFusion() {
  if (search_subgrap_ != nullptr && search_subgrap_->context_ != nullptr && search_subgrap_->model_ != nullptr) {
    tensors_ = &search_subgrap_->tensors_;
    context_ = search_subgrap_->context_;
    model_ = search_subgrap_->model_;
    src_tensors_ = search_subgrap_->src_tensors_;
    return RET_OK;
  }
  return RET_ERROR;
}

std::vector<std::vector<uint32_t>> OnlineFusionPass::GetFrontNodeIndex(LiteGraph::Node *cur_node) {
  std::vector<std::vector<uint32_t>> front_node;
  auto &cur_node_output_tensor_indices = cur_node->input_indices_;
  for (auto &cur_node_output_tensor_indice : cur_node_output_tensor_indices) {
    auto &cur_node_output_tensor = tensors_->at(cur_node_output_tensor_indice);
    front_node.emplace_back(cur_node_output_tensor.in_nodes_);
  }
  return front_node;
}

std::vector<std::vector<uint32_t>> OnlineFusionPass::GetNextNodeIndex(LiteGraph::Node *cur_node) {
  // get cur_node
  std::vector<std::vector<uint32_t>> next_node;
  auto &cur_node_output_tensor_indices = cur_node->output_indices_;
  for (auto &cur_node_output_tensor_indice : cur_node_output_tensor_indices) {
    auto &cur_node_output_tensor = tensors_->at(cur_node_output_tensor_indice);
    next_node.emplace_back(cur_node_output_tensor.in_nodes_);
  }
  return next_node;
}

OpParameter *OnlineFusionPass::GetNodeOpParameter(LiteGraph::Node *node) {
  MS_ASSERT(node != nullptr);
  auto primitive = node->primitive_;
  auto op_param_func = PopulateRegistry::GetInstance()->GetParameterCreator(
    GetPrimitiveType(primitive, SCHEMA_VERSION::SCHEMA_CUR), SCHEMA_VERSION::SCHEMA_CUR);
  if (op_param_func == nullptr) {
    return nullptr;
  }
  return op_param_func(primitive);
}

flatbuffers::Offset<mindspore::schema::Attribute> OnlineFusionPass::SetDataToUint8Vector(
  void *src, size_t len, flatbuffers::FlatBufferBuilder *fbb, const char *attr_name) {
  std::vector<uint8_t> data(len, 0);
  (void)memcpy(data.data(), src, len);
  flatbuffers::Offset<mindspore::schema::Attribute> attr =
    mindspore::schema::CreateAttributeDirect(*fbb, attr_name, &data);
  return attr;
}
}  // namespace mindspore::lite
