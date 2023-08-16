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

#include "src/litert/pass/online_fusion/cast_gather_reduce_fusion_pass.h"
#include <vector>
#include "src/litert/pass/online_fusion/online_fusion_utils.h"
#include "src/common/ops/populate/populate_register.h"
#include "nnacl/reduce_parameter.h"
#include "include/model.h"

namespace {
constexpr size_t kInitialSize = 1024;
}  // namespace

namespace mindspore::lite {
void CastGatherReduceOnlineFusionPass::DoOnlineFusion() {
  DoCastGatherReduceFusionPass();  // cast + gather + reduce op fusion
}

void CastGatherReduceOnlineFusionPass::DoCastGatherReduceFusionPass() {
  auto &device_list = context_->device_list_;
  if (device_list.size() != 1 || device_list.front().device_type_ != DT_CPU) {
    return;
  }
  node_list_ = model_->graph_.all_nodes_;
  // loop all node
  for (uint32_t i = 0; i < node_list_.size(); i++) {
    // the first node is split op
    if (DoCastGatherReduceFusion(i)) {
      MS_LOG(INFO) << "cast + gather + reduce op fusion to custom op success.";
    }
  }
  return;
}

bool CastGatherReduceOnlineFusionPass::DoCastGatherReduceFusion(uint32_t node_id) {
  node_list_ = model_->graph_.all_nodes_;
  auto &node = node_list_[node_id];

  SearchSubGraph::Subgraph subgraph;
  std::vector<uint32_t> new_input_indices;
  if (GetPrimitiveType(node->primitive_, SCHEMA_VERSION::SCHEMA_CUR) == schema::PrimitiveType_Cast) {
    auto cast_in_indices = node->input_indices_;
    if (cast_in_indices.size() != C2NUM) {
      return false;
    }
    auto &cast_in1_tensor = tensors_->at(cast_in_indices.at(1));
    if (cast_in1_tensor.type_ != SearchSubGraph::TensorType::CONSTANT) {
      return false;
    }

    auto input_data = src_tensors_->at(cast_in_indices.at(0));
    if (input_data->shape().size() != C2NUM) {
      return false;
    }

    subgraph.heads_.emplace_back(node_id);

    auto next_node = GetNextNodeIndex(node);
    if (next_node.size() != 1 || next_node.front().size() != 1) {
      return false;
    }

    uint32_t next_node1_index = next_node.front().front();
    if (!SatifyGatherReduceParse(&subgraph, next_node1_index, &new_input_indices)) {
      return false;
    }
    new_input_indices.at(1) = cast_in_indices[0];
  } else if (GetPrimitiveType(node->primitive_, SCHEMA_VERSION::SCHEMA_CUR) == schema::PrimitiveType_Gather) {
    auto front_node = GetFrontNodeIndex(node);
    if (front_node.size() != C3NUM || front_node.at(1).size() != 1) {
      return false;
    }
    uint32_t cast_node1_index = front_node.at(1).front();
    auto &cast_node1 = node_list_.at(cast_node1_index);
    if (GetPrimitiveType(cast_node1->primitive_, SCHEMA_VERSION::SCHEMA_CUR) == schema::PrimitiveType_Cast) {
      return false;
    }

    if (!SatifyGatherReduceParse(&subgraph, node_id, &new_input_indices)) {
      return false;
    }
  } else {
    return false;
  }

  // do fusion
  auto ret = CreateCastGatherReduceCustomNode(node, &subgraph, &new_input_indices);
  if (ret != RET_OK) {
    return false;
  }

  DeleteCastGatherReduceOriginNode(&subgraph);
  return true;
}

int CastGatherReduceOnlineFusionPass::CreateCastGatherReduceCustomNode(LiteGraph::Node *node,
                                                                       SearchSubGraph::Subgraph *subgraph,
                                                                       std::vector<uint32_t> *new_input_indices) {
  MS_ASSERT(node != nullptr);
  flatbuffers::FlatBufferBuilder fbb(kInitialSize);

  std::vector<flatbuffers::Offset<mindspore::schema::Attribute>> attrs;
  auto val_offset = schema::CreateCustomDirect(fbb, "CastGatherReduceFusion", &attrs);
  auto prim_offset =
    schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(PrimType::PrimType_Custom), val_offset.o);
  fbb.Finish(prim_offset);

  void *prim = malloc(fbb.GetSize());
  if (prim == nullptr) {
    MS_LOG(ERROR) << "malloc CastGatherReduceFusion primitive failed.";
    return RET_ERROR;
  }
  (void)memcpy(prim, fbb.GetBufferPointer(), fbb.GetSize());
  auto online_fusion_prim = flatbuffers::GetRoot<schema::Primitive>(prim);
  if (online_fusion_prim == nullptr) {
    MS_LOG(ERROR) << "GetRoot CastGatherReduceFusion primitive failed.";
    free(prim);
    return RET_ERROR;
  }
  fbb.Clear();
  model_->node_bufs_.push_back(prim);

  static uint64_t index = 0;
  node->name_ = "CastGatherReduceFusion" + std::to_string(index++);
  node->primitive_ = online_fusion_prim;
  node->node_type_ = PrimType::PrimType_Inner_SplitReduceConcatFusion;
  node->input_indices_ = *new_input_indices;
  node->output_indices_ = model_->graph_.all_nodes_.at(subgraph->ends_.front())->output_indices_;
  return RET_OK;
}

bool CastGatherReduceOnlineFusionPass::SatifyGatherReduceParse(SearchSubGraph::Subgraph *subgraph, uint32_t in_node,
                                                               std::vector<uint32_t> *new_input_indices) {
  // gather  op
  uint32_t gather_node1_index = in_node;
  auto &gather_node1 = node_list_.at(gather_node1_index);
  if (GetPrimitiveType(gather_node1->primitive_, SCHEMA_VERSION::SCHEMA_CUR) != schema::PrimitiveType_Gather) {
    return false;
  }

  auto gather_in_indices = gather_node1->input_indices_;
  if (gather_in_indices.size() != C3NUM) {
    return false;
  }
  auto &gather_in0_tensor = tensors_->at(gather_in_indices.at(0));
  if (gather_in0_tensor.type_ != SearchSubGraph::TensorType::CONSTANT) {
    return false;
  }
  auto &gather_in2_tensor = tensors_->at(gather_in_indices.at(C2NUM));
  if (gather_in2_tensor.type_ != SearchSubGraph::TensorType::CONSTANT ||
      !IsIntScalarValue(src_tensors_->at(gather_in_indices.at(C2NUM)), 0)) {
    return false;
  }
  if (subgraph->heads_.size() == 0) {
    auto input_data = src_tensors_->at(gather_in_indices.at(1));
    if (input_data->shape().size() != C2NUM) {
      return false;
    }
    subgraph->heads_.emplace_back(gather_node1_index);
  } else {
    subgraph->nodes_.emplace_back(gather_node1_index);
  }
  new_input_indices->emplace_back(gather_in_indices[C0NUM]);
  new_input_indices->emplace_back(gather_in_indices[C1NUM]);
  new_input_indices->emplace_back(gather_in_indices[C2NUM]);

  // reduce op
  auto next_node = GetNextNodeIndex(gather_node1);
  if (next_node.size() != 1 || next_node.front().size() != 1) {
    return false;
  }
  uint32_t reduce_node1_index = next_node.front().front();
  auto &reduce_node1 = node_list_.at(reduce_node1_index);

  auto reduce_in_indices = reduce_node1->input_indices_;
  if (reduce_in_indices.size() != C2NUM) {
    return false;
  }
  auto &reduce_in1_tensor = tensors_->at(reduce_in_indices.at(1));
  if (reduce_in1_tensor.type_ != SearchSubGraph::TensorType::CONSTANT ||
      !IsIntScalarValue(src_tensors_->at(reduce_in_indices.at(C1NUM)), 1)) {
    return false;
  }

  if (GetPrimitiveType(reduce_node1->primitive_, SCHEMA_VERSION::SCHEMA_CUR) != schema::PrimitiveType_ReduceFusion) {
    return false;
  }
  auto *reduce_param = reinterpret_cast<ReduceParameter *>(GetNodeOpParameter(reduce_node1));
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
  subgraph->ends_.emplace_back(reduce_node1_index);

  return true;
}

void CastGatherReduceOnlineFusionPass::DeleteCastGatherReduceOriginNode(SearchSubGraph::Subgraph *subgraph) {
  auto sub_graph = model_->graph_.sub_graphs_.at(0);
  auto &subgraph_node_indices = sub_graph->node_indices_;
  for (auto &node_index : subgraph->nodes_) {
    // delete tensors_ tensor info
    auto &input_indices = node_list_.at(node_index)->input_indices_;
    for (auto input_indice : input_indices) {
      tensors_->at(input_indice).in_nodes_.clear();
      tensors_->at(input_indice).out_nodes_.clear();
    }
    node_list_.at(node_index)->output_indices_.clear();
    node_list_.at(node_index)->input_indices_.clear();

    auto indice_itr = std::find(subgraph_node_indices.begin(), subgraph_node_indices.end(), node_index);
    subgraph_node_indices.erase(indice_itr);
  }
  for (auto &node_index : subgraph->ends_) {
    // delete tensors_ tensor info
    auto &input_indices = node_list_.at(node_index)->input_indices_;
    for (auto input_indice : input_indices) {
      tensors_->at(input_indice).out_nodes_.clear();
      tensors_->at(input_indice).in_nodes_.clear();
    }
    auto &output_indices = node_list_.at(node_index)->output_indices_;
    for (auto output_indice : output_indices) {
      tensors_->at(output_indice).out_nodes_.clear();
      tensors_->at(output_indice).out_nodes_.emplace_back(subgraph->heads_.front());
    }

    node_list_.at(node_index)->input_indices_.clear();
    node_list_.at(node_index)->output_indices_.clear();

    auto indice_itr = std::find(subgraph_node_indices.begin(), subgraph_node_indices.end(), node_index);
    subgraph_node_indices.erase(indice_itr);
  }
}

int DoCastGatherReduceFusionPass(SearchSubGraph *search_subgraph) {
  CastGatherReduceOnlineFusionPass cast_gather_reduce_online_fusion(search_subgraph);
  cast_gather_reduce_online_fusion.DoOnlineFusionPass();
  return RET_OK;
}

REG_ONLINE_FUSION_PASS(DoCastGatherReduceFusionPass);
}  // namespace mindspore::lite
