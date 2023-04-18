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
#include "plugin/device/ascend/optimizer/ir_fission/add_status_input_for_random_operator.h"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <set>
#include <utility>
#include "proto/random_status.pb.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/anf_runtime_algorithm.h"

namespace mindspore::opt {
namespace {
struct RandomNode {
  std::string name;
  std::string code;
  size_t status0;
  size_t status1;
};

// key: code. value: ordered in index
std::map<std::string, std::vector<RandomNode>> Deserialization(const std::string &proto_str, uint32_t graph_id) {
  std::map<std::string, std::vector<RandomNode>> snap_map;
  mindspore::RandomNodeList proto_random_node_list;
  proto_random_node_list.ParseFromString(proto_str);
  for (auto &proto_random_node : proto_random_node_list.nodes()) {
    if (graph_id != proto_random_node.graph_id()) {
      continue;
    }
    snap_map[proto_random_node.code()].emplace_back(RandomNode{
      proto_random_node.name(), proto_random_node.code(), proto_random_node.status0(), proto_random_node.status1()});
  }
  return snap_map;
}

bool CheckMatch(const std::map<std::string, std::vector<AnfNodePtr>> &filter_map,
                const std::map<std::string, std::vector<RandomNode>> &snap_map) {
  if (filter_map.size() != snap_map.size()) {
    MS_LOG(WARNING) << "filter_map size " << filter_map.size() << " and snap_map size " << snap_map.size()
                    << " not match.";
    return false;
  }
  for (const auto &[key, filter_list] : filter_map) {
    auto iter = snap_map.find(key);
    if (iter == snap_map.end()) {
      MS_LOG(WARNING) << "filter_map has key " << key << " but snap_map has not.";
      return false;
    }
    const auto &snap_list = iter->second;
    if (filter_list.size() != snap_list.size()) {
      MS_LOG(WARNING) << "Key " << key << " in filter_map size " << filter_list.size() << " and snap_map one size "
                      << snap_list.size() << " not match.";
      return false;
    }
    for (size_t i = 0; i < filter_list.size(); ++i) {
      MS_EXCEPTION_IF_NULL(filter_list[i]);
      auto cnode_name = common::AnfAlgo::GetCNodeName(filter_list[i]);
      if (snap_list[i].name.find(cnode_name) == std::string::npos) {
        MS_LOG(WARNING) << "Key " << key << " index " << i << " filter_map node name " << cnode_name
                        << " and snap_map node name " << snap_list[i].name << " not match.";
        return false;
      }
    }
  }
  return true;
}

std::map<std::string, std::vector<AnfNodePtr>> FilterRandomNodeFromToposortList(
  const std::vector<AnfNodePtr> &toposort_list) {
  std::map<std::string, std::vector<AnfNodePtr>> random_node_filter_map;
  for (const auto &node : toposort_list) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode_name = common::AnfAlgo::GetCNodeName(node);
    if (kRandomNodeWhiteList.find(cnode_name) == kRandomNodeWhiteList.end()) {
      continue;
    }
    std::string key = "";
    auto debug_info = trace::GetSourceCodeDebugInfo(node->debug_info());
    if (debug_info != nullptr) {
      auto location = debug_info->location();
      if (location != nullptr) {
        key = location->file_name() + ":" + std::to_string(location->line());
      }
    }
    random_node_filter_map[key].push_back(node);
  }

  return random_node_filter_map;
}

std::pair<size_t, size_t> GetSnapStatus(const std::map<std::string, std::vector<RandomNode>> &snap_map,
                                        const std::string &key, size_t index) {
  if (snap_map.empty()) {
    return {0, 0};
  }
  auto iter = snap_map.find(key);
  if (iter == snap_map.end()) {
    return {0, 0};
  }
  const auto &list = iter->second;
  if (list.size() <= index) {
    return {0, 0};
  }
  return {list[index].status0, list[index].status1};
}

ValueNodePtr CreateInput(const KernelGraphPtr &kg, size_t value) {
  std::vector<int64_t> shape = {1};
  TensorTypePtr tensor_type = std::make_shared<TensorType>(kUInt64);
  MS_EXCEPTION_IF_NULL(tensor_type);
  tensor::DeviceInfo device_info{kOpFormat_DEFAULT, tensor_type};
  tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(TypeId::kNumberTypeUInt64, shape);
  MS_EXCEPTION_IF_NULL(tensor);
  tensor->set_device_info(device_info);
  auto data_ptr = tensor->data_c();
  MS_EXCEPTION_IF_NULL(data_ptr);
  auto ptr = static_cast<size_t *>(data_ptr);
  *ptr = value;
  auto value_node = std::make_shared<ValueNode>(tensor);
  MS_EXCEPTION_IF_NULL(value_node);
  auto abstract = tensor->ToAbstract();
  value_node->set_abstract(abstract);
  auto indices_kernel_info = std::make_shared<device::KernelInfo>();
  MS_EXCEPTION_IF_NULL(indices_kernel_info);
  value_node->set_kernel_info(indices_kernel_info);
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
  builder.SetOutputsFormat({kOpFormat_DEFAULT});
  builder.SetOutputsDeviceType({kNumberTypeUInt64});
  builder.SetOutputsKernelObjectType({kernel::KernelObjectType::TENSOR});
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), value_node.get());
  kg->AddValueNodeToGraph(value_node);
  return value_node;
}
}  // namespace
bool AddStatusInputForRandomOperator::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  bool changed = false;
  // get random snap from attr
  std::map<std::string, std::vector<RandomNode>> snap_map = {};
  if (graph->has_attr(kAttrRandomOpSnapShot)) {
    auto value = graph->get_attr(kAttrRandomOpSnapShot);
    MS_EXCEPTION_IF_NULL(value);
    snap_map = Deserialization(GetValue<std::string>(value), kernel_graph->graph_id());
  }

  std::vector<AnfNodePtr> node_list = TopoSort(graph->get_return());
  auto filter_map = FilterRandomNodeFromToposortList(node_list);
  if (!snap_map.empty() && !CheckMatch(filter_map, snap_map)) {
    MS_LOG(WARNING) << "Graph " << graph->ToString() << " attr " << kAttrRandomOpSnapShot
                    << " and actual nodes is not matched, this attr will be ignored.";
    snap_map = {};
  }
  for (const auto &[k, v] : filter_map) {
    for (size_t i = 0; i < v.size(); ++i) {
      MS_EXCEPTION_IF_NULL(v[i]);
      auto cnode = v[i]->cast<CNodePtr>();
      std::string node_fullname = cnode->fullname_with_scope();
      auto [s0, s1] = GetSnapStatus(snap_map, k, i);
      cnode->add_input(CreateInput(kernel_graph, s0));
      cnode->add_input(CreateInput(kernel_graph, s1));
    }
  }
  return changed;
}
}  // namespace mindspore::opt
