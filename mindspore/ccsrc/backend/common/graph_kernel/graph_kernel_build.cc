/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#ifdef ENABLE_AKG
#include "backend/common/graph_kernel/graph_kernel_build.h"

#include <fstream>
#include <utility>
#include <string>
#include <map>
#include <unordered_set>
#include "include/common/utils/anfalgo.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "kernel/akg/akg_kernel_json_generator.h"
#include "backend/common/graph_kernel/graph_kernel_helper.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"
#include "kernel/akg/akg_kernel_build_manager.h"

namespace mindspore::graphkernel {
namespace {
void GetTopoValidNodes(const FuncGraphPtr &func_graph, CNodePtrList *topo_valid_nodes) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(topo_valid_nodes);
  auto nodes = TopoSort(func_graph->get_return());
  for (auto &node : nodes) {
    if (node == nullptr || !node->isa<CNode>() || !AnfUtils::IsRealKernel(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    topo_valid_nodes->push_back(cnode);
  }
}

bool IsAkgOp(const AnfNodePtr &node) {
  if (node == nullptr || !node->isa<CNode>()) {
    return false;
  }
  static std::unordered_set<std::string> ops{"UnPadAkg", "PadAkg", "ElemAny"};
  auto name = AnfUtils::GetCNodeName(node);
  return ops.find(name) != ops.end();
}
}  // namespace

bool SafeSplitSchemer::Split(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  Run(func_graph);
  return !split_plan_.empty();
}

void SafeSplitSchemer::Run(const FuncGraphPtr &func_graph) {
  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, true);
    func_graph->set_manager(mng);
  }
  SplitNodes(func_graph);
  if (split_plan_.size() != need_inline_.size() || split_plan_.empty() || (split_plan_.size() == 1 && !NeedInline(0))) {
    split_plan_.clear();
    need_inline_.clear();
    return;
  }
  GroupReturnNode(func_graph);
}

void SafeSplitSchemer::SplitNodes(const FuncGraphPtr &func_graph) {
  CNodePtrList topo_valid_nodes;
  GetTopoValidNodes(func_graph, &topo_valid_nodes);
  for (size_t i = 0; i < topo_valid_nodes.size(); ++i) {
    const auto &node = topo_valid_nodes[i];
    node_group_[node] = i;
    if (IsPrimitiveCNode(node, prim::kPrimInplaceAssign)) {
      // Let InplaceAssign and its second input in a same group.
      auto input2 = node->input(kIndex2);
      node_group_[input2] = i;
    }
  }

  std::map<size_t, AnfNodePtrList> group_nodes;
  // Nodes with same group id will stay in the same group.
  for (const auto &node : topo_valid_nodes) {
    auto group_id = node_group_[node];
    group_nodes[group_id].push_back(node);
  }

  node_group_.clear();
  for (const auto &it : group_nodes) {
    for (const auto &node : it.second) {
      node_group_[node] = split_plan_.size();
    }
    split_plan_.push_back(it.second);
    // If a group has >= 2 nodes or AKG specific node, then this group will stay in a sub graph(need_inline = 0).
    if (it.second.size() > 1 || (it.second.size() == 1 && IsAkgOp(it.second.back()))) {
      need_inline_.push_back(0);
    } else {
      need_inline_.push_back(1);
    }
  }
}

void GraphKernelBuild::Init() {
  // Init KernelMeta.
  if (bin_map_ == nullptr) {
    bin_map_ = kernel::KernelMeta::GetInstance();
    if (!bin_map_->initialized()) {
      bin_map_->Initialize();
    }
  }

  // Init AkgKernelBuilder.
  if (Callback::Instance()->GetTargetFromContext() == kGPUDevice) {
    kernel_builder_ = kernel::AkgKernelBuildManager::Instance().GetAkgKernelBuilder(kGPUDevice);
  } else if (Callback::Instance()->GetTargetFromContext() == kAscendDevice) {
    kernel_builder_ = kernel::AkgKernelBuildManager::Instance().GetAkgKernelBuilder(kAscendDevice);
  } else {
    const auto kernel_generator = GraphKernelFlags::GetInstance().kernel_generator;
    if (kernel_generator == "AKG") {
      kernel_builder_ = kernel::AkgKernelBuildManager::Instance().GetAkgKernelBuilder(kCPUDevice);
    } else if (kernel_generator == "BISHENG") {
      kernel_builder_ = kernel::AkgKernelBuildManager::Instance().GetAkgKernelBuilder(kernel_generator);
    } else {
      MS_EXCEPTION(UnknownError) << "Kernel generator only supports AKG and BISHENG but got " << kernel_generator
                                 << ".";
    }
  }
}

bool GraphKernelBuild::Process(const FuncGraphPtr &func_graph, int iter) {
  bool changed = false;
  std::vector<kernel::JsonNodePair> nodes;
  CollectNodes(func_graph, &nodes);
  // No nodes need to be compiled.
  if (nodes.empty()) {
    MS_LOG(DEBUG) << "There are no Akg kernel to be compiled.";
    return changed;
  }
  // Update cache before compiling. Some nodes may already have compiled cache(e.g. compiled from previous network
  // running), these nodes do not need to be compiled again.
  auto need_compile_nodes = CollectNotCachedNodes(nodes);
  MS_LOG(INFO) << "Iter " << iter << ": Total Akg kernel number is " << nodes.size() << ", "
               << need_compile_nodes.size() << " of them need to be compiled, and "
               << (nodes.size() - need_compile_nodes.size()) << " of them use the compilation cache.";
  // Parallel compile.
  ParallelBuild(need_compile_nodes);
  // Update cache after compiling. Nodes that still not have compile cache means they compiled failed.
  auto remaining_nodes = CollectNotCachedNodes(need_compile_nodes);
  // Split nodes that compile failed.
  changed = SplitNodes(remaining_nodes);

  return changed;
}

kernel::JsonNodePair GraphKernelBuild::CollectNode(const AnfNodePtr &node) const {
  FuncGraphPtr sub_func_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(node);
  MS_EXCEPTION_IF_NULL(sub_func_graph);
  auto mng = sub_func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(sub_func_graph, true);
    sub_func_graph->set_manager(mng);
  }
  AnfNodePtrList node_list;
  AnfNodePtrList input_list;
  AnfNodePtrList output_list;
  kernel::GetValidKernelNodes(sub_func_graph, &node_list, &input_list, &output_list);
  DumpOption option;
  option.get_target_info = true;
  AkgKernelJsonGenerator akg_kernel_json_generator(option);
  if (!akg_kernel_json_generator.CollectFusedJson(node_list, input_list, output_list)) {
    MS_EXCEPTION(UnknownError) << "Collect op info file failed. op[" << node->fullname_with_scope() << "].";
  }
  return std::make_pair(akg_kernel_json_generator, node);
}

void GraphKernelBuild::CollectNodes(const FuncGraphPtr &func_graph, std::vector<kernel::JsonNodePair> *nodes) const {
  if (func_graph == nullptr) {
    return;
  }
  MS_EXCEPTION_IF_NULL(nodes);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto todo = TopoSort(func_graph->get_return());
  for (auto iter = todo.crbegin(); iter != todo.crend(); ++iter) {
    auto node = *iter;
    // Only processes graph kernel node
    if (node == nullptr || !common::AnfAlgo::IsGraphKernel(node) || AnfAlgo::GetKernelMod(node) != nullptr) {
      continue;
    }
    auto json_node = CollectNode(node);
    nodes->push_back(json_node);
  }
}

std::vector<kernel::JsonNodePair> GraphKernelBuild::CollectNotCachedNodes(
  const std::vector<kernel::JsonNodePair> &nodes) {
  MS_EXCEPTION_IF_NULL(bin_map_);
  MS_EXCEPTION_IF_NULL(kernel_builder_);
  std::vector<kernel::JsonNodePair> res;
  for (const auto &[json_generator, node] : nodes) {
    if (node == nullptr) {
      continue;
    }
    // Skip node that already set kernel mod(created from compile cache).
    if (AnfAlgo::GetKernelMod(node) != nullptr) {
      continue;
    }
    const auto &kernel_name = json_generator.kernel_name();
    // Skip node that already has cache.
    if (kernel_pack_.find(kernel_name) != kernel_pack_.end()) {
      kernel_builder_->AkgSetKernelMod(kernel_pack_[kernel_name], json_generator, node);
      MS_LOG(DEBUG) << "Set cached kernel for node [" << node->fullname_with_scope() << "] with kernel name ["
                    << kernel_name << "]";
      continue;
    }
    std::string json_path = bin_map_->kernel_meta_path() + kernel_name + kernel::kJsonSuffix;
    std::ifstream kernel_json(json_path);
    // Json file not exits, which means the node does not have cache.
    if (!kernel_json.is_open()) {
      (void)res.emplace_back(json_generator, node);
      continue;
    }
    kernel_json.close();
    // For GPU and CPU, we need to insert json path to bin_map_(KernelMeta) first, otherwise AkgSearchCache will fail.
    (void)bin_map_->Insert(kernel_name, json_path);
    auto cached_kernel_pack = kernel_builder_->AkgSearchCache(kernel_name);
    // Node cache found.
    if (cached_kernel_pack != nullptr) {
      kernel_pack_[kernel_name] = cached_kernel_pack;
      kernel_builder_->AkgSetKernelMod(cached_kernel_pack, json_generator, node);
      MS_LOG(DEBUG) << "Set cached kernel for node [" << node->fullname_with_scope() << "] with kernel name ["
                    << kernel_name << "]";
      continue;
    }
    // Node cache not found.
    (void)res.emplace_back(json_generator, node);
  }
  return res;
}

void GraphKernelBuild::ParallelBuild(const std::vector<kernel::JsonNodePair> &nodes) {
  std::vector<kernel::JsonNodePair> uniq_nodes;
  std::unordered_set<std::string> kernel_names;
  // AkgKernelBuilder::ParallelBuild can not process duplicate nodes, so we need to filter these nodes first.
  for (const auto &[json_generator, node] : nodes) {
    const auto &kernel_name = json_generator.kernel_name();
    if (kernel_names.find(kernel_name) == kernel_names.end()) {
      (void)kernel_names.insert(kernel_name);
      (void)uniq_nodes.emplace_back(json_generator, node);
    }
  }
  if (!uniq_nodes.empty()) {
    MS_EXCEPTION_IF_NULL(kernel_builder_);
    (void)kernel_builder_->ParallelBuild(uniq_nodes);
  }
}

bool GraphKernelBuild::SplitNodes(const std::vector<kernel::JsonNodePair> &nodes) {
  bool result = false;
  std::unordered_set<std::string> kernel_names;
  for (const auto &[json_generator, node] : nodes) {
    const auto &kernel_name = json_generator.kernel_name();
    // Print kernel name of nodes that compile failed.
    if (kernel_names.find(kernel_name) == kernel_names.end()) {
      (void)kernel_names.insert(kernel_name);
      MS_LOG(WARNING) << "Nodes that with kernel name [" << kernel_name
                      << "] do not have compile cache after compiling and will be split.";
    }
    MS_EXCEPTION_IF_NULL(node);
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (!splitter_.TrySplit(cnode)) {
      // This means the compiled failed node also can not be split.
      MS_LOG(EXCEPTION) << "Node [" << node->fullname_with_scope() << "] with kernel name [" << kernel_name
                        << "] compiled failed and can not be split.";
    }
    result = true;
  }
  return result;
}

bool GraphKernelBuild::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, true);
    func_graph->set_manager(mng);
  }

  Init();

  bool changed = false;
  bool need_traverse = true;
  int iter = 1;
  while (need_traverse) {
    need_traverse = Process(func_graph, iter);
    iter++;
    changed = need_traverse || changed;
    if (need_traverse) {
      mng->RemoveRoots();
      mng->KeepRoots({func_graph});
    }
  }

  return changed;
}
}  // namespace mindspore::graphkernel
#endif
