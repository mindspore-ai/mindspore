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
  std::map<std::string, int64_t> seed_attr;
};

// key: code. value: ordered in index
std::map<std::string, std::vector<RandomNode>> Deserialization(const std::string &proto_str) {
  std::map<std::string, std::vector<RandomNode>> snap_map;
  mindspore::RandomNodeList proto_random_node_list;
  auto ret = proto_random_node_list.ParseFromString(proto_str);
  if (!ret) {
    MS_LOG(WARNING) << "Parse proto str " << proto_str << " failed, random status will be ignored.";
    return snap_map;
  }
  for (auto &proto_random_node : proto_random_node_list.nodes()) {
    snap_map[proto_random_node.code()].emplace_back(
      RandomNode{proto_random_node.name(), proto_random_node.code(), proto_random_node.status0(),
                 proto_random_node.status1(), [](const auto &protobuf_map) -> std::map<std::string, int64_t> {
                   std::map<std::string, int64_t> ret;
                   for (const auto &[key, value] : protobuf_map) {
                     ret.emplace(key, value);
                   }
                   return ret;
                 }(proto_random_node.seed_attr())});
  }
  return snap_map;
}

bool CheckMatch(const std::map<std::string, std::vector<AnfNodePtr>> &random_node_in_cur_graph,
                const std::map<std::string, std::vector<RandomNode>> &snap_map) {
  if (random_node_in_cur_graph.size() > snap_map.size()) {
    MS_LOG(WARNING) << "filter_map size " << random_node_in_cur_graph.size() << " and snap_map size " << snap_map.size()
                    << " not match.";
    return false;
  }
  for (const auto &[key, filter_list] : random_node_in_cur_graph) {
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
    std::string key = {};
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

std::tuple<size_t, size_t, std::map<std::string, int64_t>> GetSnapStatus(
  const std::map<std::string, std::vector<RandomNode>> &snap_map, const std::string &key, size_t index) {
  if (snap_map.empty()) {
    return {0, 0, {}};
  }
  auto iter = snap_map.find(key);
  if (iter == snap_map.end()) {
    return {0, 0, {}};
  }
  const auto &list = iter->second;
  if (list.size() <= index) {
    return {0, 0, {}};
  }
  return {list[index].status0, list[index].status1, list[index].seed_attr};
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
    snap_map = Deserialization(GetValue<std::string>(value));
  }

  std::vector<AnfNodePtr> node_list = TopoSort(graph->get_return());
  auto random_node_in_cur_graph = FilterRandomNodeFromToposortList(node_list);
  if (!snap_map.empty() && !CheckMatch(random_node_in_cur_graph, snap_map)) {
    MS_LOG(WARNING) << "Graph " << graph->ToString() << " attr " << kAttrRandomOpSnapShot
                    << " and actual nodes is not matched, this attr will be ignored.";
    snap_map = {};
  }
  for (const auto &[k, v] : random_node_in_cur_graph) {
    for (size_t i = 0; i < v.size(); ++i) {
      MS_EXCEPTION_IF_NULL(v[i]);
      auto cnode = v[i]->cast<CNodePtr>();
      std::string node_fullname = cnode->fullname_with_scope();
      auto [s0, s1, seed_attrs] = GetSnapStatus(snap_map, k, i);
      for (const auto &[attr_name, attr_value] : seed_attrs) {
        if (!common::AnfAlgo::HasNodeAttr(attr_name, cnode)) {
          MS_LOG(EXCEPTION) << "Node " << cnode->fullname_with_scope() << " does not have attr " << attr_name
                            << ", but read " << attr_value << " from checkpoint.";
        }
        int64_t value = common::AnfAlgo::GetNodeAttr<int64_t>(cnode, attr_name);
        if (value == 0) {
          MS_LOG(EXCEPTION) << "Node " << cnode->fullname_with_scope() << " have attr " << attr_name << " value is "
                            << value << ", in this case the randomness cannot be fixed.";
        }
        if (value != attr_value) {
          MS_LOG(EXCEPTION)
            << "Node " << cnode->fullname_with_scope() << " have attr " << attr_name << " value is " << value
            << ", but read " << attr_value
            << " from checkpoint. When loading the operators’ random state, please do not modify the network script.";
        }
      }
      cnode->add_input(CreateInput(kernel_graph, s0));
      cnode->add_input(CreateInput(kernel_graph, s1));
    }
  }
  return changed;
}
}  // namespace mindspore::opt
