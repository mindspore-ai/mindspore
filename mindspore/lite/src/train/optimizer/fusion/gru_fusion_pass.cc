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

#include "src/train/optimizer/fusion/gru_fusion_pass.h"
#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "src/common/log_adapter.h"
#include "include/errorcode.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
namespace {
constexpr size_t kSplitOutSize = 3;
constexpr uint32_t kAdd0 = 0;
constexpr uint32_t kAdd1 = 1;
constexpr uint32_t kAdd2 = 2;
constexpr uint32_t kAdd3 = 3;
constexpr uint32_t kAdd4 = 4;
constexpr uint32_t kAdd5 = 5;
constexpr uint32_t kSub = 6;
constexpr uint32_t kMul0 = 7;
constexpr uint32_t kMul1 = 8;
constexpr uint32_t kTanh = 9;
constexpr uint32_t kSigmoid0 = 10;
constexpr uint32_t kSigmoid1 = 11;
constexpr uint32_t kSplit0 = 12;
constexpr uint32_t kSplit1 = 13;
constexpr uint32_t kMatmul0 = 14;
constexpr uint32_t kMatmul1 = 15;
constexpr uint32_t kInputH = 16;
constexpr uint32_t kInputI = 17;
constexpr auto kCustomGRU = "CustomGRU";

bool CheckCommon(schema::MetaGraphT *graph, uint32_t node_index, schema::PrimitiveType type, size_t in_nums,
                 size_t out_nums) {
  if (graph->nodes.size() <= node_index) {
    return false;
  }
  const auto &node = graph->nodes[node_index];
  if (node == nullptr || node->primitive == nullptr) {
    return false;
  }
  const auto &value = node->primitive->value;
  if (value.type != type) {
    return false;
  }
  if (value.value == nullptr) {
    return false;
  }
  if ((in_nums > 0 && node->inputIndex.size() != in_nums) || node->outputIndex.size() != out_nums) {
    return false;
  }
  return std::all_of(node->inputIndex.begin(), node->inputIndex.end(),
                     [&graph](uint32_t tensor_index) { return graph->allTensors.size() > tensor_index; }) &&
         std::all_of(node->outputIndex.begin(), node->outputIndex.end(),
                     [&graph](uint32_t tensor_index) { return graph->allTensors.size() > tensor_index; });
}

template <schema::PrimitiveType T, typename P>
bool CheckArithmetic(schema::MetaGraphT *graph, uint32_t node_index) {
  if (!CheckCommon(graph, node_index, T, kInputSize1, 1)) {
    return false;
  }
  const auto &node = graph->nodes[node_index];
  const auto &value = node->primitive->value;
  const auto add_attr = static_cast<const P *>(value.value);
  if (add_attr->activation_type != schema::ActivationType_NO_ACTIVATION) {
    return false;
  }
  auto tensor_indexes = node->inputIndex;
  (void)tensor_indexes.insert(tensor_indexes.end(), node->outputIndex.begin(), node->outputIndex.end());
  std::vector<int> shape;
  for (size_t i = 0; i < tensor_indexes.size(); ++i) {
    if (i == 0) {
      shape = graph->allTensors[tensor_indexes[i]]->dims;
      continue;
    }
    if (graph->allTensors[tensor_indexes[i]]->dims != shape) {
      return false;
    }
  }
  return true;
}

template <schema::ActivationType T>
bool CheckActivation(schema::MetaGraphT *graph, uint32_t node_index) {
  if (!CheckCommon(graph, node_index, schema::PrimitiveType_Activation, 1, 1)) {
    return false;
  }
  const auto &value = graph->nodes[node_index]->primitive->value;
  const auto add_attr = static_cast<const schema::ActivationT *>(value.value);
  if (add_attr->activation_type != T) {
    return false;
  }
  return true;
}

bool CheckBiasAdd(schema::MetaGraphT *graph, uint32_t node_index) {
  if (!CheckCommon(graph, node_index, schema::PrimitiveType_AddFusion, kInputSize1, 1) &&
      !CheckCommon(graph, node_index, schema::PrimitiveType_BiasAdd, kInputSize1, 1)) {
    return false;
  }
  const auto &node = graph->nodes[node_index];
  const auto &value = node->primitive->value;
  if (value.type == schema::PrimitiveType_AddFusion) {
    const auto add_attr = static_cast<const schema::AddFusionT *>(value.value);
    if (add_attr->activation_type != schema::ActivationType_NO_ACTIVATION) {
      return false;
    }
  }
  auto in_shape0 = graph->allTensors[node->inputIndex[0]]->dims;
  auto in_shape1 = graph->allTensors[node->inputIndex[1]]->dims;
  if (in_shape1.size() != 1 || in_shape0.empty() || in_shape0.back() != in_shape1.back()) {
    return false;
  }
  return true;
}

bool CheckMatmul(schema::MetaGraphT *graph, uint32_t node_index) {
  if (!CheckCommon(graph, node_index, schema::PrimitiveType_MatMulFusion, kInputSize1, 1)) {
    return false;
  }
  const auto &node = graph->nodes[node_index];
  const auto &value = node->primitive->value;
  const auto matmul_attr = static_cast<const schema::MatMulFusionT *>(value.value);
  if (matmul_attr->activation_type != schema::ActivationType_NO_ACTIVATION) {
    return false;
  }
  auto out_shape = graph->allTensors[node->outputIndex.front()]->dims;
  return out_shape.size() == kInputSize1;
}

bool CheckSplit(schema::MetaGraphT *graph, uint32_t node_index) {
  if (!CheckCommon(graph, node_index, schema::PrimitiveType_Split, 1, kSplitOutSize)) {
    return false;
  }
  const auto &node = graph->nodes[node_index];
  if (node->inputIndex.size() != 1 || node->outputIndex.size() != kSplitOutSize) {
    return false;
  }
  auto in_shape0 = graph->allTensors[node->inputIndex[0]]->dims;
  auto out_shape0 = graph->allTensors[node->outputIndex[0]]->dims;
  auto out_shape1 = graph->allTensors[node->outputIndex[1]]->dims;
  auto out_shape2 = graph->allTensors[node->outputIndex[kInputSize1]]->dims;
  if (out_shape0 != out_shape1 || out_shape0 != out_shape2) {
    return false;
  }
  if (in_shape0.empty() || out_shape0.empty()) {
    return false;
  }
  if (in_shape0.back() != (out_shape0.back() + out_shape1.back() + out_shape2.back())) {
    return false;
  }
  return true;
}

bool CheckStack(schema::MetaGraphT *graph, uint32_t node_index) {
  if (!CheckCommon(graph, node_index, schema::PrimitiveType_Stack, 0, 1)) {
    return false;
  }
  const auto &node = graph->nodes[node_index];
  const auto &value = node->primitive->value;
  const auto stack_attr = static_cast<const schema::StackT *>(value.value);
  auto out_shape = graph->allTensors[node->outputIndex.front()]->dims;
  if (out_shape.empty()) {
    return false;
  }
  auto axis = stack_attr->axis;
  if (axis < 0) {
    axis += static_cast<int64_t>(out_shape.size());
  }
  return axis == 0;
}

bool CheckSqueeze(schema::MetaGraphT *graph, uint32_t node_index) {
  if (!CheckCommon(graph, node_index, schema::PrimitiveType_Squeeze, 0, 1)) {
    return false;
  }
  const auto &node = graph->nodes[node_index];
  if (node->inputIndex.size() != 1 && node->inputIndex.size() != kInputSize1) {
    return false;
  }
  int axis = 0;
  if (node->inputIndex.size() == kInputSize1) {
    const auto &data = graph->allTensors[node->inputIndex[1]]->data;
    if (data.size() != sizeof(int)) {
      return false;
    }
    axis = *(reinterpret_cast<const int *>(data.data()));
  } else {
    const auto &value = node->primitive->value;
    const auto squeeze_attr = static_cast<const schema::SqueezeT *>(value.value);
    if (squeeze_attr->axis.size() != 1) {
      return false;
    }
    axis = squeeze_attr->axis.front();
  }
  auto in_shape = graph->allTensors[node->inputIndex[0]]->dims;
  if (in_shape.empty()) {
    return false;
  }
  if (axis < 0) {
    axis += static_cast<int>(in_shape.size());
  }
  return axis == 0;
}

std::vector<int> GetStridedSlicePoints(const schema::TensorT *tensor, int64_t mask) {
  if (tensor->data.empty()) {
    return {};
  }
  auto origin_data = reinterpret_cast<const int *>(tensor->data.data());
  size_t num = tensor->data.size() / sizeof(int);
  std::vector<int> data;
  for (size_t i = 0; i < num; ++i) {
    bool ineffective = (mask & (1 << i));
    int cur_point = ineffective ? 0 : origin_data[i];
    data.push_back(cur_point);
  }
  return data;
}

bool CheckStridedSlice(schema::MetaGraphT *graph, uint32_t node_index, int batch_position) {
  if (!CheckCommon(graph, node_index, schema::PrimitiveType_StridedSlice, C4NUM, 1)) {
    return false;
  }
  const auto &node = graph->nodes[node_index];
  const auto &step_tensor = graph->allTensors[node->inputIndex.back()];
  if (!step_tensor->data.empty()) {
    const auto data = reinterpret_cast<int *>(step_tensor->data.data());
    auto size = step_tensor->data.size() / sizeof(int);
    if (std::any_of(data, data + size, [](int val) { return val != 1; })) {
      return false;
    }
  }
  auto in_shape = graph->allTensors[node->inputIndex.front()]->dims;
  auto out_shape = graph->allTensors[node->outputIndex.back()]->dims;
  if (in_shape.size() != out_shape.size() || in_shape.empty()) {
    return false;
  }
  for (size_t i = 1; i < in_shape.size(); ++i) {
    if (in_shape[i] != out_shape[i]) {
      return false;
    }
  }
  const auto &value = node->primitive->value;
  const auto strided_slice_attr = static_cast<const schema::StridedSliceT *>(value.value);
  if (strided_slice_attr->ellipsis_mask != 0 || strided_slice_attr->new_axis_mask != 0 ||
      strided_slice_attr->shrink_axis_mask != 0) {
    return false;
  }
  auto begin = GetStridedSlicePoints(graph->allTensors[node->inputIndex[1]].get(), strided_slice_attr->begin_mask);
  if (begin.empty()) {
    return false;
  }
  return begin.front() == batch_position;
}

bool CheckGruCell(schema::MetaGraphT *graph, uint32_t node_index) {
  if (!CheckCommon(graph, node_index, schema::PrimitiveType_Custom, C6NUM, 1)) {
    return false;
  }
  const auto &node = graph->nodes[node_index];
  const auto &value = node->primitive->value;
  const auto gru_attr = static_cast<const schema::CustomT *>(value.value);
  return gru_attr->type == kCustomGRU;
}

std::unique_ptr<schema::CustomT> CreateCustom() {
  auto ConvertToAttr = [](const std::string &key, const std::vector<uint8_t> &value) {
    auto attr = std::make_unique<schema::AttributeT>();
    attr->name = key;
    attr->data = value;
    return attr;
  };
  auto attrs = std::make_unique<schema::CustomT>();
  MS_CHECK_TRUE_MSG(attrs != nullptr, nullptr, "Create CustomT failed.");
  attrs->type = kCustomGRU;
  std::vector<uint8_t> transpose_a{false};
  std::vector<uint8_t> transpose_b{true};
  std::vector<uint8_t> built_in{true};

  attrs->attr.push_back(ConvertToAttr("transpose_a", transpose_a));
  attrs->attr.push_back(ConvertToAttr("transpose_b", transpose_b));
  attrs->attr.push_back(ConvertToAttr("builtin", built_in));
  return attrs;
}

struct InNodeInfo {
  int node_index;
  std::vector<uint32_t> in_indexes;
};

struct OutNodeInfo {
  int node_index;
  uint32_t out_index;
};

struct camp {
  bool operator()(uint32_t left, uint32_t right) const { return left > right; }
};
}  // namespace

class LinkInfoManager {
 public:
  explicit LinkInfoManager(schema::MetaGraphT *graph) : graph_{graph} {
    auto &all_nodes = graph->nodes;
    for (int node_index = 0; node_index < static_cast<int>(all_nodes.size()); ++node_index) {
      auto in_indexes = all_nodes[node_index]->inputIndex;
      for (uint32_t index = 0; index < static_cast<uint32_t>(in_indexes.size()); ++index) {
        if (link_info_manager_.find(in_indexes[index]) == link_info_manager_.end()) {
          link_info_manager_[in_indexes[index]] = std::make_pair(std::vector<InNodeInfo>{}, OutNodeInfo{-1, 0});
        }
        auto &in_infos = link_info_manager_[in_indexes[index]].first;
        auto iter = in_infos.begin();
        for (; iter != in_infos.end(); ++iter) {
          if (iter->node_index == node_index) {
            break;
          }
        }
        if (iter != in_infos.end()) {
          iter->in_indexes.push_back(index);
        } else {
          in_infos.push_back({node_index, {index}});
        }
      }

      auto out_indexes = all_nodes[node_index]->outputIndex;
      for (uint32_t index = 0; index < static_cast<uint32_t>(out_indexes.size()); ++index) {
        link_info_manager_[out_indexes[index]].second = OutNodeInfo{node_index, index};
      }
    }
  }

  const auto &GetLinkInfos() const { return link_info_manager_; }

  void Replace(uint32_t node_index, std::unique_ptr<CNodeT> node) { graph_->nodes[node_index].swap(node); }

  void AddDeleteNodes(const std::set<uint32_t> &node_indexes) {
    delete_nodes_.insert(node_indexes.begin(), node_indexes.end());
  }

  void UpdateMetaGraph() {
    auto &main_graph = graph_->subGraph.front();
    for (auto node_index : delete_nodes_) {
      graph_->nodes.erase(graph_->nodes.begin() + node_index);
    }
    main_graph->nodeIndices.clear();
    for (uint32_t index = 0; index < static_cast<uint32_t>(graph_->nodes.size()); ++index) {
      main_graph->nodeIndices.push_back(index);
    }
    std::map<uint32_t, uint32_t> tensor_maps;
    BuildTensorMap(&tensor_maps);
    auto UpdateTensorIndex = [&tensor_maps](std::vector<uint32_t> *origin) {
      auto origin_indexes = *origin;
      origin->clear();
      (void)std::transform(origin_indexes.begin(), origin_indexes.end(), std::back_inserter(*origin),
                           [&tensor_maps](uint32_t origin_index) { return tensor_maps[origin_index]; });
    };
    UpdateTensorIndex(&graph_->inputIndex);
    for (auto &node : graph_->nodes) {
      UpdateTensorIndex(&node->inputIndex);
      UpdateTensorIndex(&node->outputIndex);
    }
    UpdateTensorIndex(&graph_->outputIndex);
    main_graph->inputIndices = graph_->inputIndex;
    main_graph->outputIndices = graph_->outputIndex;
    main_graph->tensorIndices.clear();
    for (uint32_t index = 0; index < static_cast<uint32_t>(tensor_maps.size()); ++index) {
      main_graph->tensorIndices.push_back(index);
    }
    std::vector<std::unique_ptr<TensorT>> tensors;
    graph_->allTensors.swap(tensors);
    graph_->allTensors.resize(tensor_maps.size());
    for (auto &tensor_map : tensor_maps) {
      graph_->allTensors[tensor_map.second].swap(tensors[tensor_map.first]);
    }
  }

 private:
  void BuildTensorMap(std::map<uint32_t, uint32_t> *tensor_maps) {
    uint32_t new_index = 0;
    auto InsertElements = [tensor_maps, &new_index](const std::vector<uint32_t> &indexes) mutable {
      for (auto index : indexes) {
        if (tensor_maps->find(index) != tensor_maps->end()) {
          continue;
        }
        (*tensor_maps)[index] = new_index++;
      }
    };
    InsertElements(graph_->inputIndex);
    for (auto &node : graph_->nodes) {
      InsertElements(node->inputIndex);
      InsertElements(node->outputIndex);
    }
    InsertElements(graph_->outputIndex);
  }

  schema::MetaGraphT *graph_{nullptr};
  std::set<uint32_t, camp> delete_nodes_;
  // tensor_index, <in_node_infos, out_node_info>
  std::map<uint32_t, std::pair<std::vector<InNodeInfo>, OutNodeInfo>> link_info_manager_;
};

class GruCellFusion {
 public:
  GruCellFusion() = default;
  ~GruCellFusion() = default;
  STATUS Run(schema::MetaGraphT *graph) {
    MS_ASSERT(graph != nullptr);
    MS_ASSERT(graph->subGraph.size() == 1);
    link_info_manager_ = std::make_shared<LinkInfoManager>(graph);
    graph_ = graph;
    DefinePattern();
    for (uint32_t node_index = 0; node_index < static_cast<uint32_t>(graph->nodes.size()); ++node_index) {
      if (!MatchPattern(node_index)) {
        continue;
      }
      if (CreateCustomGruCell() != RET_OK) {
        MS_LOG(ERROR) << "Create Custom-Gru failed.";
        return RET_ERROR;
      }
    }
    link_info_manager_->UpdateMetaGraph();
    return RET_OK;
  }

 private:
  struct NodeInfo {
    struct InTensorInfo {
      bool is_const{false};
      uint32_t node_index_{0};
      uint32_t tensor_index_{0};
    };
    struct OutTensorInfo {
      uint32_t node_index_{0};
      uint32_t tensor_index_{0};
    };
    bool (*checker)(schema::MetaGraphT *graph, uint32_t node_index);
    std::vector<InTensorInfo> in_infos;
    std::vector<OutTensorInfo> out_infos;
  };

  void DefinePattern() {
    int match_order = 0;
    pattern_[{match_order++, kAdd0}] = {
      CheckArithmetic<schema::PrimitiveType_AddFusion, schema::AddFusionT>, {{false, kTanh, 0}, {false, kMul0, 0}}, {}};
    pattern_[{match_order++, kTanh}] = {
      CheckActivation<schema::ActivationType_TANH>, {{false, kAdd1, 0}}, {{kSub, 1}, {kAdd0, 0}}};
    pattern_[{match_order++, kMul0}] = {CheckArithmetic<schema::PrimitiveType_MulFusion, schema::MulFusionT>,
                                        {{false, kSigmoid0, 0}, {false, kSub, 0}},
                                        {{kAdd0, 1}}};
    pattern_[{match_order++, kAdd1}] = {CheckArithmetic<schema::PrimitiveType_AddFusion, schema::AddFusionT>,
                                        {{false, kSplit0, 2}, {false, kMul1, 0}},
                                        {{kTanh, 0}}};
    pattern_[{match_order++, kSub}] = {CheckArithmetic<schema::PrimitiveType_SubFusion, schema::SubFusionT>,
                                       {{false, kInputH, 0}, {false, kTanh, 0}},
                                       {{kMul0, 1}}};
    pattern_[{match_order++, kSigmoid0}] = {
      CheckActivation<schema::ActivationType_SIGMOID>, {{false, kAdd2, 0}}, {{kMul0, 0}}};
    pattern_[{match_order++, kSplit0}] = {CheckSplit, {{false, kAdd3, 0}}, {{kAdd4, 0}, {kAdd2, 0}, {kAdd1, 0}}};
    pattern_[{match_order++, kMul1}] = {CheckArithmetic<schema::PrimitiveType_MulFusion, schema::MulFusionT>,
                                        {{false, kSigmoid1, 0}, {false, kSplit1, 2}},
                                        {{kAdd1, 1}}};
    pattern_[{match_order++, kAdd2}] = {CheckArithmetic<schema::PrimitiveType_AddFusion, schema::AddFusionT>,
                                        {{false, kSplit0, 1}, {false, kSplit1, 1}},
                                        {{kSigmoid0, 0}}};
    pattern_[{match_order++, kSigmoid1}] = {
      CheckActivation<schema::ActivationType_SIGMOID>, {{false, kAdd4, 0}}, {{kMul1, 0}}};
    pattern_[{match_order++, kAdd3}] = {CheckBiasAdd, {{false, kMatmul0, 0}, {true}}, {{kSplit0, 0}}};
    pattern_[{match_order++, kSplit1}] = {CheckSplit, {{false, kAdd5, 0}}, {{kAdd4, 1}, {kAdd2, 1}, {kMul1, 1}}};
    pattern_[{match_order++, kAdd4}] = {CheckArithmetic<schema::PrimitiveType_AddFusion, schema::AddFusionT>,
                                        {{false, kSplit0, 0}, {false, kSplit1, 0}},
                                        {{kSigmoid1, 0}}};
    pattern_[{match_order++, kAdd5}] = {CheckBiasAdd, {{false, kMatmul1, 0}, {true}}, {{kSplit1, 0}}};
    pattern_[{match_order++, kMatmul0}] = {CheckMatmul, {{false, kInputI, 0}, {true}}, {{kAdd3, 0}}};
    pattern_[{match_order++, kMatmul1}] = {CheckMatmul, {{false, kInputH, 0}, {true}}, {{kAdd5, 0}}};
  }

  bool FillRealPattern(uint32_t node_index, std::map<uint32_t, NodeInfo> *real_pattern) {
    const auto &link_infos = link_info_manager_->GetLinkInfos();
    if (real_pattern->find(node_index) != real_pattern->end()) {
      return false;
    }
    real_pattern->insert({node_index, {nullptr}});
    auto in_tensor_indexes = graph_->nodes[node_index]->inputIndex;
    for (auto tensor_index : in_tensor_indexes) {
      if (link_infos.find(tensor_index) == link_infos.end()) {
        return false;
      }
      const auto &tensor_out_info = link_infos.at(tensor_index).second;
      if (tensor_out_info.node_index < 0) {
        real_pattern->at(node_index).in_infos.push_back({true});
      } else {
        real_pattern->at(node_index)
          .in_infos.push_back({false, static_cast<uint32_t>(tensor_out_info.node_index), tensor_out_info.out_index});
      }
    }
    auto out_tensor_indexes = graph_->nodes[node_index]->outputIndex;
    for (auto tensor_index : out_tensor_indexes) {
      if (link_infos.find(tensor_index) == link_infos.end()) {
        return false;
      }
      const auto &in_tensor_out_info = link_infos.at(tensor_index).first;
      for (const auto &in_node_info : in_tensor_out_info) {
        for (auto index : in_node_info.in_indexes) {
          real_pattern->at(node_index).out_infos.push_back({static_cast<uint32_t>(in_node_info.node_index), index});
        }
      }
    }
    return true;
  }

  bool CheckPattern(const std::map<uint32_t, NodeInfo> &real_pattern,
                    const std::pair<int, uint32_t> &pattern_node_index) {
    const auto &real_in_infos = real_pattern.at(real_node_map_.at(pattern_node_index.second)).in_infos;
    const auto &virtual_in_infos = pattern_.at(pattern_node_index).in_infos;
    if (real_in_infos.size() != virtual_in_infos.size()) {
      return false;
    }
    for (size_t i = 0; i < virtual_in_infos.size(); ++i) {
      if (virtual_in_infos[i].is_const) {
        if (!real_in_infos[i].is_const) {
          return false;
        }
        continue;
      }
      if (virtual_in_infos[i].tensor_index_ != real_in_infos[i].tensor_index_) {
        return false;
      }
      if (real_node_map_.find(virtual_in_infos[i].node_index_) == real_node_map_.end()) {
        real_node_map_.insert({virtual_in_infos[i].node_index_, real_in_infos[i].node_index_});
      } else if (real_node_map_.at(virtual_in_infos[i].node_index_) != real_in_infos[i].node_index_) {
        return false;
      }
    }
    const auto &real_out_infos = real_pattern.at(real_node_map_.at(pattern_node_index.second)).out_infos;
    const auto &virtual_out_infos = pattern_.at(pattern_node_index).out_infos;
    if (virtual_out_infos.empty()) {
      return true;
    }
    if (real_out_infos.size() != virtual_out_infos.size()) {
      return false;
    }
    for (size_t i = 0; i < virtual_out_infos.size(); ++i) {
      if (virtual_out_infos[i].tensor_index_ != real_out_infos[i].tensor_index_) {
        return false;
      }
      if (real_node_map_.find(virtual_out_infos[i].node_index_) == real_node_map_.end()) {
        real_node_map_.insert({virtual_out_infos[i].node_index_, real_out_infos[i].node_index_});
      } else if (real_node_map_.at(virtual_out_infos[i].node_index_) != real_out_infos[i].node_index_) {
        return false;
      }
    }
    return true;
  }

  bool CheckClosure(const std::map<uint32_t, uint32_t> &node_map) {
    std::set<uint32_t> real_nodes;
    (void)std::for_each(node_map.begin(), node_map.end(),
                        [&real_nodes](std::pair<uint32_t, uint32_t> pair) { real_nodes.insert(pair.second); });
    if (real_nodes.size() != node_map.size()) {
      return false;
    }
    const auto &link_infos = link_info_manager_->GetLinkInfos();
    for (uint32_t start = kAdd1; start <= kMatmul1; ++start) {
      if (node_map.find(start) == node_map.end()) {
        return false;
      }
      const auto &node = graph_->nodes[node_map.at(start)];
      auto out_tensor_indexes = node->outputIndex;
      for (auto out_index : out_tensor_indexes) {
        if (link_infos.find(out_index) == link_infos.end()) {
          return false;
        }
        for (const auto &in_node_info : link_infos.at(out_index).first) {
          if (real_nodes.find(in_node_info.node_index) == real_nodes.end()) {
            return false;
          }
        }
      }
    }
    return true;
  }

  bool MatchPattern(uint32_t add_index) {
    real_node_map_.clear();
    real_node_map_[kAdd0] = add_index;
    std::map<uint32_t, NodeInfo> real_pattern;
    for (const auto &pair : pattern_) {
      if (real_node_map_.find(pair.first.second) == real_node_map_.end()) {
        return false;
      }
      auto node_index = real_node_map_[pair.first.second];
      if (!pair.second.checker(graph_, node_index)) {
        return false;
      }
      if (!FillRealPattern(node_index, &real_pattern)) {
        return false;
      }
      if (!CheckPattern(real_pattern, pair.first)) {
        return false;
      }
    }
    auto weight_hidden_index = graph_->nodes[real_node_map_[kMatmul1]]->inputIndex[1];
    auto weight_hidden_shape = graph_->allTensors[weight_hidden_index]->dims;
    if (weight_hidden_shape.size() != C2NUM || weight_hidden_shape[0] != weight_hidden_shape[1] * C3NUM) {
      return false;
    }
    return CheckClosure(real_node_map_);
  }

  STATUS CreateCustomGruCell() {
    std::vector<uint32_t> inputs;
    inputs.push_back(graph_->nodes[real_node_map_[kMatmul0]]->inputIndex[0]);  // x
    inputs.push_back(graph_->nodes[real_node_map_[kMatmul0]]->inputIndex[1]);  // weight_input
    inputs.push_back(graph_->nodes[real_node_map_[kMatmul1]]->inputIndex[1]);  // weight_hidden
    inputs.push_back(graph_->nodes[real_node_map_[kAdd3]]->inputIndex[1]);     // bias_input
    inputs.push_back(graph_->nodes[real_node_map_[kAdd5]]->inputIndex[1]);     // bias_hidden
    inputs.push_back(graph_->nodes[real_node_map_[kMatmul1]]->inputIndex[0]);  // init_h
    auto outputs = graph_->nodes[real_node_map_[kAdd0]]->outputIndex;
    auto attrs = CreateCustom();
    MS_CHECK_TRUE_RET(attrs != nullptr, RET_NULL_PTR);
    auto prim_t = std::make_unique<schema::PrimitiveT>();
    MS_CHECK_TRUE_MSG(prim_t != nullptr, RET_ERROR, "Create PrimitiveT failed.");
    prim_t->value.type = schema::PrimitiveType_Custom;
    prim_t->value.value = attrs.release();
    auto custom_gru = std::make_unique<schema::CNodeT>();
    MS_CHECK_TRUE_MSG(custom_gru != nullptr, RET_ERROR, "Create Custom-Gru failed.");
    custom_gru->name = graph_->nodes[real_node_map_[kAdd0]]->name;
    custom_gru->inputIndex = inputs;
    custom_gru->outputIndex = outputs;
    custom_gru->primitive = std::move(prim_t);
    link_info_manager_->Replace(real_node_map_[kAdd0], std::move(custom_gru));
    std::set<uint32_t> delete_nodes;
    for (uint32_t i = kAdd1; i <= kMatmul1; ++i) {
      delete_nodes.insert(real_node_map_[i]);
    }
    link_info_manager_->AddDeleteNodes(delete_nodes);
    return RET_OK;
  }

  std::map<std::pair<int, uint32_t>, NodeInfo> pattern_;
  std::map<uint32_t, uint32_t> real_node_map_;
  schema::MetaGraphT *graph_{nullptr};
  std::shared_ptr<LinkInfoManager> link_info_manager_{nullptr};
};

STATUS GruFusionPass::Run(schema::MetaGraphT *graph) {
#ifndef ENABLE_ARM64
  return RET_OK;
#endif
  if (graph == nullptr) {
    MS_LOG(ERROR) << "graph is a nullptr.";
    return RET_NULL_PTR;
  }
  if (graph->subGraph.size() != 1) {
    return RET_OK;
  }
  if (FuseToGruCell(graph) != RET_OK) {
    return RET_ERROR;
  }
  return FuseGruCell(graph);
}

STATUS GruFusionPass::FuseToGruCell(schema::MetaGraphT *graph) {
  GruCellFusion gru_cell_fusion{};
  if (gru_cell_fusion.Run(graph) != RET_OK) {
    MS_LOG(ERROR) << "Fuse GruCell failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS GruFusionPass::FuseGruCell(schema::MetaGraphT *graph) {
  link_info_manager_ = std::make_shared<LinkInfoManager>(graph);
  for (uint32_t i = 0; i < static_cast<uint32_t>(graph->nodes.size()); ++i) {
    if (!CheckStack(graph, i)) {
      continue;
    }
    std::vector<uint32_t> strided_slices;
    std::vector<uint32_t> squeezes;
    std::vector<uint32_t> gru_cells;
    if (!MatchPatten(graph, i, &strided_slices, &squeezes, &gru_cells)) {
      continue;
    }
    if (CreateGru(graph, i, strided_slices, squeezes, gru_cells) != RET_OK) {
      MS_LOG(ERROR) << "Fuse GruCell failed.";
      return RET_ERROR;
    }
  }
  link_info_manager_->UpdateMetaGraph();
  link_info_manager_ = nullptr;
  return RET_OK;
}

bool GruFusionPass::MatchPatten(schema::MetaGraphT *graph, uint32_t stack_index, std::vector<uint32_t> *strided_slices,
                                std::vector<uint32_t> *squeezes, std::vector<uint32_t> *gru_cells) {
  auto &link_infos = link_info_manager_->GetLinkInfos();
  auto &stack_node = graph->nodes[stack_index];
  int batch_point = 0;
  auto CommonCheck = [&link_infos](uint32_t tensor_index) {
    if (link_infos.find(tensor_index) == link_infos.end()) {
      return std::make_pair(false, 0);
    }
    const auto &in_node_info = link_infos.at(tensor_index).first;
    if (in_node_info.size() != 1 && in_node_info.front().in_indexes.size() != 1) {
      return std::make_pair(false, 0);
    }
    auto node_index = link_infos.at(tensor_index).second.node_index;
    if (node_index < 0) {
      return std::make_pair(false, 0);
    }
    return std::make_pair(true, node_index);
  };
  for (auto tensor_index : stack_node->inputIndex) {
    auto check_info = CommonCheck(tensor_index);
    if (!check_info.first || !CheckGruCell(graph, check_info.second)) {
      return false;
    }
    gru_cells->push_back(check_info.second);
    auto &gru_cell_node = graph->nodes[check_info.second];
    check_info = CommonCheck(gru_cell_node->inputIndex.front());
    if (!check_info.first || !CheckSqueeze(graph, check_info.second)) {
      return false;
    }
    squeezes->push_back(check_info.second);
    auto &squeeze_node = graph->nodes[check_info.second];
    check_info = CommonCheck(squeeze_node->inputIndex.front());
    if (!check_info.first || !CheckStridedSlice(graph, check_info.second, batch_point)) {
      return false;
    }
    strided_slices->push_back(check_info.second);
    ++batch_point;
  }
  if (strided_slices->empty()) {
    return false;
  }
  uint32_t input_index = graph->nodes[strided_slices->front()]->inputIndex.front();
  if (std::any_of(strided_slices->begin(), strided_slices->end(), [input_index, graph](uint32_t strided_slice) {
        return graph->nodes[strided_slice]->inputIndex.front() != input_index;
      })) {
    return false;
  }
  auto in_shape = graph->allTensors[input_index]->dims;
  if (in_shape.empty() || in_shape.front() != batch_point) {
    return false;
  }
  return CheckGruCellConnection(graph, *gru_cells);
}

bool GruFusionPass::CheckGruCellConnection(schema::MetaGraphT *graph, const std::vector<uint32_t> &gru_cells) {
  auto &first_node = graph->nodes[gru_cells.front()];
  if (first_node->inputIndex.size() != C6NUM) {
    return false;
  }
  auto init_h = first_node->outputIndex.front();
  for (size_t i = 1; i < gru_cells.size(); ++i) {
    auto &node = graph->nodes[gru_cells[i]];
    if (node->inputIndex.size() != first_node->inputIndex.size()) {
      return false;
    }
    for (size_t j = 1; j < C5NUM; ++j) {
      if (node->inputIndex[j] != first_node->inputIndex[j]) {
        return false;
      }
    }
    if (node->inputIndex[C5NUM] != init_h) {
      return false;
    }
    init_h = node->outputIndex.front();
  }
  return true;
}

STATUS GruFusionPass::CreateGru(schema::MetaGraphT *graph, uint32_t stack_index,
                                const std::vector<uint32_t> &strided_slices, const std::vector<uint32_t> &squeezes,
                                const std::vector<uint32_t> &gru_cells) {
  auto &gru_cell_node = graph->nodes[gru_cells.front()];
  gru_cell_node->inputIndex[0] = graph->nodes[strided_slices.front()]->inputIndex[0];
  gru_cell_node->outputIndex[0] = graph->nodes[stack_index]->outputIndex[0];
  std::set<uint32_t> delete_node{stack_index};
  (void)delete_node.insert(strided_slices.begin(), strided_slices.end());
  (void)delete_node.insert(squeezes.begin(), squeezes.end());
  (void)delete_node.insert(gru_cells.begin() + 1, gru_cells.end());
  link_info_manager_->AddDeleteNodes(delete_node);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
