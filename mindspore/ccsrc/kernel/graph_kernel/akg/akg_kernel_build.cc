/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "kernel/graph_kernel/akg/akg_kernel_build.h"

#include <chrono>
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <iostream>
#include "mindspore/core/ops/framework_ops.h"
#include "ir/dtype.h"
#include "ir/func_graph.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "kernel/framework_utils.h"
#include "kernel/graph_kernel/graph_kernel_json_generator.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/debug/profiler/profiling.h"

namespace mindspore {
namespace kernel {
constexpr int32_t MAX_ERROR_LEN = 1024;
constexpr int32_t PROCESS_NUM = 16;
constexpr int32_t TIME_OUT = 300;
constexpr auto kLogLevel = "log_level";

bool AkgKernelBuilder::ParallelBuild(const std::vector<JsonNodePair> &build_args) {
  struct timeval start_time;
  struct timeval end_time;
  (void)gettimeofday(&start_time, nullptr);
  MS_LOG(INFO) << "Akg start parallel build. kernel count: " << build_args.size();

  KernelPool kp;
  auto ret = kp.Init(build_args);
  if (ret != 0) {
    MS_LOG(ERROR) << "KernelPool init failed.";
    return false;
  }

  std::set<size_t> fetched_ids;
  ret = kp.FetchKernels(&fetched_ids);
  if (ret != 0) {
    MS_LOG(ERROR) << "KernelPool FetchKernels failed.";
    return false;
  }

  if (!fetched_ids.empty()) {
    auto jsons = GetKernelJsonsByHashId(build_args, fetched_ids);

    auto client = GetClient();
    MS_EXCEPTION_IF_NULL(client);
    if (!client->CompilerStart(PROCESS_NUM, TIME_OUT)) {
      MS_LOG(ERROR) << "Akg start failed.";
      return false;
    }
    auto attrs = CollectBuildAttrs();
    if (!attrs.empty() && !client->CompilerSendAttr(attrs)) {
      MS_LOG(ERROR) << "Akg send attr failed.";
      return false;
    }
    if (!client->CompilerSendData(jsons)) {
      MS_LOG(ERROR) << "Akg send data failed.";
      return false;
    }
    if (!client->CompilerWait()) {
      MS_LOG(ERROR) << "Akg compile failed.";
      return false;
    }
  }

  ret = kp.UpdateAndWait(fetched_ids);
  if (ret != 0) {
    MS_LOG(ERROR) << "KernelPool UpdateAndWait failed.";
    return false;
  }

  if (kp.Release() != 0) {
    MS_LOG(ERROR) << "KernelPool release failed.";
    return false;
  }

  (void)gettimeofday(&end_time, nullptr);
  const uint64_t kUSecondInSecond = 1000000;
  uint64_t cost = kUSecondInSecond * static_cast<uint64_t>(end_time.tv_sec - start_time.tv_sec);
  cost += static_cast<uint64_t>(end_time.tv_usec - start_time.tv_usec);
  MS_LOG(INFO) << "Akg kernel build time: " << cost << " us.";

  return true;
}

bool AkgKernelBuilder::AkgOpParallelBuild(const std::vector<JsonNodePair> &build_args) {
  repeat_nodes_.clear();
  auto new_build_args = GetNotCachedKernels(build_args);
  if (new_build_args.empty()) {
    return true;
  }

  build_attrs_[kLogLevel] = "ERROR";
  if (!ParallelBuild(new_build_args)) {
    return false;
  }

  // All unique done here, cache them and set kernel.
  if (!InsertToCache(build_args)) {
    MS_LOG(ERROR) << "Insert cache failed.";
    return false;
  }

  if (!HandleRepeatNodes()) {
    MS_LOG(ERROR) << "Handle repeat nodes failed.";
    return false;
  }

  return true;
}

bool AkgKernelBuilder::SingleOpParallelBuild(const std::vector<AnfNodePtr> &anf_nodes) {
  std::vector<JsonNodePair> json_and_node;
  for (const auto &anf_node : anf_nodes) {
    MS_EXCEPTION_IF_NULL(anf_node);
    // Node already has kernel mod, no need to process it.
    if (AnfAlgo::GetKernelMod(anf_node) != nullptr) {
      continue;
    }
    graphkernel::DumpOption option;
    option.get_target_info = true;
    GraphKernelJsonGenerator graph_kernel_json_generator(option);
    auto cnode = anf_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    bool is_custom_node = IsPrimitiveCNode(cnode, prim::kPrimCustom) || IsAKGSparseOP(cnode);
    // Graph kernel node and Custom node need to generate composite json
    if (common::AnfAlgo::IsGraphKernel(cnode) || is_custom_node) {
      FuncGraphPtr func_graph = is_custom_node ? cnode->func_graph() : common::AnfAlgo::GetCNodeFuncGraphPtr(cnode);
      MS_EXCEPTION_IF_NULL(func_graph);
      auto mng = func_graph->manager();
      if (mng == nullptr) {
        mng = Manage(func_graph, true);
        func_graph->set_manager(mng);
      }
      if (is_custom_node) {
        // in this case, the cnode is a CustomOp (no matter whether graph kernel mode is enabled or not)
        // generate the fused json for the single kernel cnode
        if (!graph_kernel_json_generator.CollectFusedJsonWithSingleKernel(cnode)) {
          MS_EXCEPTION(UnknownError) << "Collect op info failed. op[" << anf_node->fullname_with_scope() << "].";
        }
      } else {
        // in this case, the cnode is a IsGraphKernel when graph kernel mode is enabled
        // generate the fused json for the graph kernel subgraph
        std::vector<AnfNodePtr> node_list;
        std::vector<AnfNodePtr> input_list;
        std::vector<AnfNodePtr> output_list;
        GetValidKernelNodes(func_graph, &node_list, &input_list, &output_list);
        if (!graph_kernel_json_generator.CollectFusedJson(node_list, input_list, output_list)) {
          MS_EXCEPTION(UnknownError) << "Collect op info failed. op[" << anf_node->fullname_with_scope() << "].";
        }
      }
    } else {
      if (!graph_kernel_json_generator.CollectJson(anf_node)) {
        MS_EXCEPTION(UnknownError) << "Collect op info failed. op[" << anf_node->fullname_with_scope() << "].";
      }
    }
    (void)json_and_node.emplace_back(std::move(graph_kernel_json_generator), anf_node);
  }

  if (json_and_node.empty()) {
    MS_LOG(INFO) << "There is no akg kernel to be compiled.";
    return true;
  }

  bool res = AkgOpParallelBuild(json_and_node);
  if (!res) {
    MS_LOG(ERROR) << "Akg build kernel failed.";
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
