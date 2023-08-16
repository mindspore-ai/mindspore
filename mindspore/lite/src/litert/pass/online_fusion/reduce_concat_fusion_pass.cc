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

#include "src/litert/pass/online_fusion/reduce_concat_fusion_pass.h"
#include <vector>
#include "src/litert/pass/online_fusion/online_fusion_utils.h"
#include "src/common/ops/populate/populate_register.h"
#include "nnacl/reduce_parameter.h"
#include "nnacl/concat_parameter.h"
#include "include/model.h"

namespace {
constexpr size_t kInitialSize = 1024;
const std::set<int64_t> kLastAxisSizeSet = {16, 32, 64, 128};
}  // namespace

namespace mindspore::lite {
void ReduceConcatOnlineFusionPass::DoOnlineFusion() {
  DoReduceConcatFusionPass();  // reduce + concat op fusion
}

bool ReduceConcatOnlineFusionPass::DoReduceConcatFusion(uint32_t node_id) {
  node_list_ = model_->graph_.all_nodes_;
  auto &node = node_list_[node_id];
  // the first node is split op
  if (GetPrimitiveType(node->primitive_, SCHEMA_VERSION::SCHEMA_CUR) != schema::PrimitiveType_Concat) {
    return false;
  }

  auto *concat_param = reinterpret_cast<ConcatParameter *>(GetNodeOpParameter(node));
  if (concat_param == nullptr) {
    return false;
  }
  if (concat_param->axis_ != C2NUM && concat_param->axis_ != -1) {
    free(concat_param);
    concat_param = nullptr;
    return false;
  }
  free(concat_param);
  concat_param = nullptr;

  auto input_indices = node->input_indices_;
  SearchSubGraph::Subgraph subgraph;
  subgraph.ends_.emplace_back(node_id);
  std::vector<uint32_t> new_input_indices;
  std::vector<uint32_t> positions;
  int reduce_count = 0;

  int lastAxisSize = 0;
  for (size_t i = 0; i < input_indices.size(); i++) {
    auto &input_indice = input_indices[i];
    auto &tensor = tensors_->at(input_indice);
    auto &in_nodes = tensor.out_nodes_;

    if (in_nodes.size()) {
      auto in_node = in_nodes.at(0);
      uint32_t reduce_node1_index = in_node;
      auto &reduce_node1 = node_list_.at(reduce_node1_index);

      if (GetPrimitiveType(reduce_node1->primitive_, SCHEMA_VERSION::SCHEMA_CUR) ==
          schema::PrimitiveType_ReduceFusion) {
        if (SatifyReduceConcatParse(reduce_node1_index, &lastAxisSize)) {
          subgraph.heads_.emplace_back(reduce_node1_index);
          new_input_indices.emplace_back(reduce_node1->input_indices_.at(0));
          positions.emplace_back(1);
          reduce_count++;
          continue;
        }
      }
    }

    auto concat_in_tensor_shape = src_tensors_->at(input_indice)->shape();
    if (concat_in_tensor_shape.size() != C3NUM) {
      if (i != input_indices.size() - 1) {
        return false;
      }
    } else if (kLastAxisSizeSet.find(concat_in_tensor_shape[C2NUM]) != kLastAxisSizeSet.end()) {
      if (lastAxisSize == 0) {
        lastAxisSize = concat_in_tensor_shape[C2NUM];
      } else if (concat_in_tensor_shape[C2NUM] != lastAxisSize) {
        return false;
      }
    } else {
      return false;
    }

    new_input_indices.emplace_back(input_indice);
    positions.emplace_back(0);
  }
  if (subgraph.heads_.size() < C20NUM || reduce_count < C10NUM) {
    return false;
  }

  // do fusion
  auto ret = CreateReduceConcatCustomNode(node, &subgraph, &new_input_indices, &positions);
  if (ret != RET_OK) {
    return false;
  }

  DeleteReduceConcatOriginNode(&subgraph, positions);
  return true;
}

int ReduceConcatOnlineFusionPass::CreateReduceConcatCustomNode(LiteGraph::Node *node,
                                                               SearchSubGraph::Subgraph *subgraph,
                                                               std::vector<uint32_t> *new_input_indices,
                                                               std::vector<uint32_t> *positions) {
  MS_ASSERT(node != nullptr);
  flatbuffers::FlatBufferBuilder fbb(kInitialSize);

  std::vector<flatbuffers::Offset<mindspore::schema::Attribute>> attrs;
  auto val_offset = schema::CreateCustomDirect(fbb, "ReduceConcatFusion", &attrs);
  auto prim_offset =
    schema::CreatePrimitive(fbb, static_cast<schema::PrimitiveType>(PrimType::PrimType_Custom), val_offset.o);
  fbb.Finish(prim_offset);

  void *prim = malloc(fbb.GetSize());
  if (prim == nullptr) {
    MS_LOG(ERROR) << "malloc ReduceConcatFusion primitive failed.";
    return RET_ERROR;
  }
  (void)memcpy(prim, fbb.GetBufferPointer(), fbb.GetSize());
  auto online_fusion_prim = flatbuffers::GetRoot<schema::Primitive>(prim);
  if (online_fusion_prim == nullptr) {
    MS_LOG(ERROR) << "GetRoot ReduceConcatFusion primitive failed.";
    free(prim);
    return RET_ERROR;
  }
  fbb.Clear();
  model_->node_bufs_.push_back(prim);

  static uint64_t index = 0;
  node->name_ = "ReduceConcatFusion" + std::to_string(index++);
  node->primitive_ = online_fusion_prim;
  node->node_type_ = PrimType::PrimType_Inner_ReduceConcatFusion;
  node->input_indices_ = *new_input_indices;
  node->output_indices_ = model_->graph_.all_nodes_.at(subgraph->ends_.front())->output_indices_;
  return RET_OK;
}

void ReduceConcatOnlineFusionPass::DeleteReduceConcatOriginNode(SearchSubGraph::Subgraph *subgraph,
                                                                const std::vector<uint32_t> &positions) {
  auto sub_graph = model_->graph_.sub_graphs_.at(0);
  auto &subgraph_node_indices = sub_graph->node_indices_;

  for (auto &node_index : subgraph->heads_) {
    // delete tensors_ tensor info
    auto &out_indices = node_list_.at(node_index)->output_indices_;
    for (auto out_indice : out_indices) {  // clear out node relate
      tensors_->at(out_indice).out_nodes_.clear();
      tensors_->at(out_indice).out_nodes_.emplace_back(subgraph->heads_.front());
    }

    // clear input and out indices
    node_list_.at(node_index)->input_indices_.clear();
    node_list_.at(node_index)->output_indices_.clear();

    auto indice_itr = std::find(subgraph_node_indices.begin(), subgraph_node_indices.end(), node_index);
    subgraph_node_indices.erase(indice_itr);
  }
}

void ReduceConcatOnlineFusionPass::DoReduceConcatFusionPass() {
  auto &device_list = context_->device_list_;
  if (device_list.size() != 1 || device_list.front().device_type_ != DT_CPU ||
      device_list.front().device_info_.cpu_device_info_.enable_float16_) {
    return;
  }
  node_list_ = model_->graph_.all_nodes_;
  // loop all node
  for (uint32_t i = 0; i < node_list_.size(); i++) {
    // the first node is split op
    if (DoReduceConcatFusion(i)) {
      MS_LOG(INFO) << "reduce + concat op fusion to custom op success.";
    }
  }
  return;
}

bool ReduceConcatOnlineFusionPass::SatifyReduceConcatParse(uint32_t in_node, int *lastAxisSize) {
  auto &reduce_node1 = node_list_.at(in_node);
  auto reduce_in_indices = reduce_node1->input_indices_;
  if (reduce_in_indices.size() != C2NUM) {
    return false;
  }
  auto reduce_in0_tensor_shape = src_tensors_->at(reduce_in_indices.at(C0NUM))->shape();
  if (reduce_in0_tensor_shape.size() != C3NUM ||
      kLastAxisSizeSet.find(reduce_in0_tensor_shape[C2NUM]) == kLastAxisSizeSet.end()) {
    return false;
  } else {
    if (*lastAxisSize == 0) {
      *lastAxisSize = reduce_in0_tensor_shape[C2NUM];
    } else if (*lastAxisSize != reduce_in0_tensor_shape[C2NUM]) {
      return false;
    }
  }
  auto &reduce_in1_tensor = tensors_->at(reduce_in_indices.at(C1NUM));
  if (reduce_in1_tensor.type_ != SearchSubGraph::TensorType::CONSTANT ||
      !IsIntScalarValue(src_tensors_->at(reduce_in_indices.at(C1NUM)), 1)) {
    return false;
  }

  auto *reduce_param = reinterpret_cast<ReduceParameter *>(GetNodeOpParameter(reduce_node1));
  if (reduce_param == nullptr) {
    return false;
  }
  if (reduce_param->mode_ != mindspore::schema::ReduceMode_ReduceSum ||
      reduce_param->keep_dims_ == false) {  // only support reducesum model and keep_dim is true
    free(reduce_param);
    reduce_param = nullptr;
    return false;
  }
  free(reduce_param);  // free reduce param data
  reduce_param = nullptr;
  return true;
}

int DoReduceConcatFusionPass(SearchSubGraph *search_subgrap) {
  ReduceConcatOnlineFusionPass reduce_concat_fusion(search_subgrap);
  reduce_concat_fusion.DoOnlineFusionPass();
  return RET_OK;
}

REG_ONLINE_FUSION_PASS(DoReduceConcatFusionPass);
}  // namespace mindspore::lite
