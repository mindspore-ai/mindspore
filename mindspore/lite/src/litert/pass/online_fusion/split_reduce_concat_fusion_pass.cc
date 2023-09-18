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

#include "src/litert/pass/online_fusion/split_reduce_concat_fusion_pass.h"
#include <vector>
#include "src/common/ops/populate/populate_register.h"
#include "nnacl/split_parameter.h"
#include "nnacl/concat_parameter.h"
#include "nnacl/reduce_parameter.h"
#include "include/model.h"

namespace {
constexpr size_t kInitialSize = 1024;
}  // namespace

namespace mindspore::lite {
void SplitReduceConcatOnlineFusionPass::DoOnlineFusion() {
  DoSplitReduceConcatFusionPass();  // split + reduce + concat op fusion
}

void SplitReduceConcatOnlineFusionPass::DoSplitReduceConcatFusionPass() {
  auto &device_list = context_->device_list_;
  if (device_list.size() != 1 || device_list.front().device_type_ != DT_CPU) {
    return;
  }
  node_list_ = model_->graph_.all_nodes_;
  // loop all node
  for (uint32_t i = 0; i < node_list_.size(); i++) {
    // the first node is split op
    DoSplitReduceConcatFusion(i);
  }
  return;
}

bool SplitReduceConcatOnlineFusionPass::SatifyReduceConcatParse(SearchSubGraph::Subgraph *subgraph, uint32_t in_node,
                                                                int split_concat_axis,
                                                                std::vector<uint32_t> *positions) {
  // reduce op
  uint32_t reduce_node1_index = in_node;
  auto &reduce_node = node_list_.at(reduce_node1_index);
  if (GetPrimitiveType(reduce_node->primitive_, SCHEMA_VERSION::SCHEMA_CUR) != schema::PrimitiveType_ReduceFusion) {
    return false;
  }
  auto *reduce_param = reinterpret_cast<ReduceParameter *>(GetNodeOpParameter(reduce_node));
  if (reduce_param == nullptr) {
    return false;
  }
  if (reduce_param->mode_ != mindspore::schema::ReduceMode_ReduceSum ||
      reduce_param->keep_dims_ == false) {  // only support reducesum
    free(reduce_param);
    reduce_param = nullptr;
    return false;
  }
  free(reduce_param);
  reduce_param = nullptr;
  subgraph->nodes_.emplace_back(reduce_node1_index);

  // concat op
  auto next_node = GetNextNodeIndex(reduce_node);
  if (next_node.size() != 1 && next_node.front().size() != 1) {
    return false;
  }
  uint32_t concat_node1_index = next_node.front().front();
  auto &concat_node1 = node_list_.at(concat_node1_index);
  if (GetPrimitiveType(concat_node1->primitive_, SCHEMA_VERSION::SCHEMA_CUR) != schema::PrimitiveType_Concat) {
    return false;
  }
  auto *concat_param = reinterpret_cast<ConcatParameter *>(GetNodeOpParameter(concat_node1));
  if (concat_param == nullptr) {
    return false;
  }
  if (concat_param->axis_ != split_concat_axis) {
    free(concat_param);
    concat_param = nullptr;
    return false;
  }
  free(concat_param);
  concat_param = nullptr;

  if (subgraph->ends_.size() == 0) {
    subgraph->ends_.emplace_back(concat_node1_index);
  } else if (subgraph->ends_.at(0) != concat_node1_index) {
    return false;
  }
  auto reduce_out = reduce_node->output_indices_.front();
  const auto &concat_in = concat_node1->input_indices_;
  for (size_t i = 0; i < concat_in.size(); ++i) {
    if (concat_in[i] == reduce_out) {
      positions->push_back(i);
    }
  }
  return true;
}

void SplitReduceConcatOnlineFusionPass::DeleteOriginNode(SearchSubGraph::Subgraph *subgraph,
                                                         const std::vector<uint32_t> &positions) {
  auto sub_graph = model_->graph_.sub_graphs_.at(0);
  auto &subgraph_node_indices = sub_graph->node_indices_;
  for (auto &node_index : subgraph->nodes_) {
    // delete tensors_ tensor info
    auto &input_indices = node_list_.at(node_index)->input_indices_;
    for (auto input_indice : input_indices) {
      tensors_->at(input_indice).in_nodes_.clear();
      tensors_->at(input_indice).out_nodes_.clear();
    }
    node_list_.at(node_index)->input_indices_.clear();
    node_list_.at(node_index)->output_indices_.clear();

    auto indice_itr = std::find(subgraph_node_indices.begin(), subgraph_node_indices.end(), node_index);
    subgraph_node_indices.erase(indice_itr);
  }
  for (auto &node_index : subgraph->ends_) {
    // delete tensors_ tensor info
    auto &input_indices = node_list_.at(node_index)->input_indices_;
    bool reserve_concat = input_indices.size() != positions.size();
    for (size_t i = (reserve_concat ? 1 : 0); i < positions.size(); ++i) {
      tensors_->at(input_indices[positions[i]]).in_nodes_.clear();
      tensors_->at(input_indices[positions[i]]).out_nodes_.clear();
    }
    auto &output_indices = node_list_.at(subgraph->heads_.front())->output_indices_;
    for (auto output_indice : output_indices) {
      tensors_->at(output_indice).out_nodes_.clear();
      tensors_->at(output_indice).out_nodes_.emplace_back(subgraph->heads_.front());
    }
    if (reserve_concat) {
      for (size_t i = positions.size() - 1; i > 0; --i) {
        input_indices.erase(input_indices.begin() + positions[i]);
      }
    } else {
      node_list_.at(node_index)->input_indices_.clear();
      node_list_.at(node_index)->output_indices_.clear();
      auto indice_itr = std::find(subgraph_node_indices.begin(), subgraph_node_indices.end(), node_index);
      subgraph_node_indices.erase(indice_itr);
    }
  }
}

void SplitReduceConcatOnlineFusionPass::DoSplitReduceConcatFusion(uint32_t node_id) {
  node_list_ = model_->graph_.all_nodes_;
  auto &node = node_list_[node_id];
  // the first node is split op
  if (GetPrimitiveType(node->primitive_, SCHEMA_VERSION::SCHEMA_CUR) != schema::PrimitiveType_Split) {
    return;
  }
  auto *split_param_base = GetNodeOpParameter(node);
  auto *split_param = reinterpret_cast<SplitParameter *>(split_param_base);
  if (split_param == nullptr) {
    return;
  }
  auto ReleaseSplitParameter = [split_param]() {
    if (split_param->op_parameter_.destroy_func_ != nullptr) {
      split_param->op_parameter_.destroy_func_(&split_param->op_parameter_);
    }
    free(split_param);
  };
  auto split_concat_axis = split_param->split_dim_;
  if (split_concat_axis < 0) {
    ReleaseSplitParameter();
    return;
  }
  auto output_indices = node->output_indices_;
  SearchSubGraph::Subgraph subgraph;
  subgraph.heads_.emplace_back(node_id);
  std::vector<uint32_t> positions;
  for (auto &output_indice : output_indices) {
    auto &tensor = tensors_->at(output_indice);
    auto &in_nodes = tensor.in_nodes_;
    for (auto &in_node : in_nodes) {
      // satisfy reduce + concat struct
      if (!SatifyReduceConcatParse(&subgraph, in_node, split_concat_axis, &positions)) {
        ReleaseSplitParameter();
        return;
      }
    }
  }
  if (positions.size() != output_indices.size() || positions.empty()) {
    ReleaseSplitParameter();
    return;
  }
  for (size_t i = 1; i < positions.size(); ++i) {
    if (positions[i] - positions[i - 1] != 1) {
      ReleaseSplitParameter();
      return;
    }
  }
  // do fusion
  auto ret = CreateCustomNode(node, &subgraph, split_param, positions);
  ReleaseSplitParameter();
  if (ret != RET_OK) {
    return;
  }

  DeleteOriginNode(&subgraph, positions);

  MS_LOG(INFO) << "split + reduce + concat op fusion to custom op success.";
  return;
}

int SplitReduceConcatOnlineFusionPass::CreateCustomNode(LiteGraph::Node *node, SearchSubGraph::Subgraph *subgraph,
                                                        SplitParameter *split_param,
                                                        const std::vector<uint32_t> &positions) {
  MS_ASSERT(node != nullptr);
  flatbuffers::FlatBufferBuilder fbb(kInitialSize);

  std::vector<flatbuffers::Offset<mindspore::schema::Attribute>> attrs;
  attrs.emplace_back(SetDataToUint8Vector(split_param, sizeof(SplitParameter), &fbb, "split_primitive"));
  attrs.emplace_back(
    SetDataToUint8Vector(split_param->split_sizes_, sizeof(int) * split_param->num_split_, &fbb, "split_sizes"));

  auto val_offset = schema::CreateCustomDirect(fbb, "SplitReduceConcatFusion", &attrs);
  auto prim_offset =
    schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(PrimType::PrimType_Custom), val_offset.o);
  fbb.Finish(prim_offset);

  void *prim = malloc(fbb.GetSize());
  if (prim == nullptr) {
    MS_LOG(ERROR) << "malloc SplitReduceConcatFusion primitive failed.";
    return RET_ERROR;
  }
  (void)memcpy(prim, fbb.GetBufferPointer(), fbb.GetSize());
  auto online_fusion_prim = flatbuffers::GetRoot<schema::Primitive>(prim);
  if (online_fusion_prim == nullptr) {
    MS_LOG(ERROR) << "GetRoot SplitReduceConcatFusion primitive failed.";
    free(prim);
    return RET_ERROR;
  }
  fbb.Clear();
  model_->node_bufs_.push_back(prim);

  static uint64_t index = 0;
  node->name_ = "SplitReduceConcatFusion" + std::to_string(index++);
  node->primitive_ = online_fusion_prim;
  node->node_type_ = PrimType::PrimType_Inner_SplitReduceConcatFusion;
  node->input_indices_ = model_->graph_.all_nodes_.at(subgraph->heads_.front())->input_indices_;
  if (positions.size() == model_->graph_.all_nodes_.at(subgraph->ends_.front())->input_indices_.size()) {
    node->output_indices_ = model_->graph_.all_nodes_.at(subgraph->ends_.front())->output_indices_;
  } else {
    node->output_indices_ = {
      model_->graph_.all_nodes_.at(subgraph->ends_.front())->input_indices_.at(positions.front())};
  }
  return RET_OK;
}

int DoSplitReduceConcatFusionPass(SearchSubGraph *search_subgrap) {
  SplitReduceConcatOnlineFusionPass split_reduce_concat_fusion(search_subgrap);
  split_reduce_concat_fusion.DoOnlineFusionPass();
  return RET_OK;
}

REG_ONLINE_FUSION_PASS(DoSplitReduceConcatFusionPass);
}  // namespace mindspore::lite
