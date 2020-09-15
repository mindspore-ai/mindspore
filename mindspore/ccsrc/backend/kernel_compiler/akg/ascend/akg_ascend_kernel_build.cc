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

#include "backend/kernel_compiler/akg/ascend/akg_ascend_kernel_build.h"

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include "ir/dtype.h"
#include "ir/func_graph.h"
#include "backend/kernel_compiler/kernel.h"
#include "backend/kernel_compiler/common_utils.h"
#include "backend/kernel_compiler/tbe/tbe_utils.h"
#include "backend/kernel_compiler/akg/ascend/akg_ascend_kernel_mod.h"
#include "backend/kernel_compiler/akg/akg_kernel_json_generator.h"
#include "backend/kernel_compiler/akg/akg_kernel_attrs_process.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/session/kernel_build_client.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr int32_t PROCESS_NUM = 16;
constexpr int32_t TIME_OUT = 300;

void SetKernelMod(const KernelPackPtr &kernel_pack, const AkgKernelJsonGenerator &json_generator,
                  const AnfNodePtr &anf_node) {
  auto kernel_mod_ptr = std::make_shared<AkgKernelMod>(kernel_pack);
  kernel_mod_ptr->SetInputSizeList(json_generator.input_size_list());
  kernel_mod_ptr->SetOutputSizeList(json_generator.output_size_list());
  AnfAlgo::SetKernelMod(kernel_mod_ptr, anf_node.get());
}
}  // namespace

std::vector<std::string> AkgAscendKernelBuilder::GetNotCachedKernelJsons(const std::vector<JsonNodePair> &build_args) {
  // Remove cached nodes, gether unique nodes, and collect repeated nodes which need postprecess.
  std::vector<std::string> jsons;
  std::unordered_set<std::string> kernel_name_set;
  for (const auto &[json_generator, anf_node] : build_args) {
    MS_EXCEPTION_IF_NULL(anf_node);
    auto kernel_name = json_generator.kernel_name();
    MS_LOG(DEBUG) << "Akg start compile op: " << kernel_name;
    auto cached_kernel_pack = tbe::TbeUtils::SearchCache(kernel_name, GetProcessorStr(anf_node));
    if (cached_kernel_pack != nullptr) {
      MS_LOG(DEBUG) << "Use cached kernel, kernel_name[" << kernel_name << "], fullname_with_scope["
                    << anf_node->fullname_with_scope() << "].";
      SetKernelMod(cached_kernel_pack, json_generator, anf_node);
      continue;
    }

    if (kernel_name_set.count(kernel_name) != 0) {
      repeat_nodes_.push_back({json_generator, anf_node});
      continue;
    }
    kernel_name_set.insert(kernel_name);
    auto kernel_json = json_generator.kernel_json_str();
    kernel::SaveJsonInfo(kernel_name, kernel_json);
    jsons.push_back(kernel_json);
  }
  return jsons;
}

bool AkgAscendKernelBuilder::InsertToCache(const std::vector<JsonNodePair> &build_args) {
  for (const auto &[json_generator, anf_node] : build_args) {
    auto kernel_name = json_generator.kernel_name();
    auto new_kernel_pack = tbe::TbeUtils::InsertCache(kernel_name, GetProcessorStr(anf_node));
    if (new_kernel_pack == nullptr) {
      MS_LOG(ERROR) << "Insert to cache failed, kernel_name[" << kernel_name << "], fullname_with_scope["
                    << anf_node->fullname_with_scope() << "].";
      return false;
    }
    SetKernelMod(new_kernel_pack, json_generator, anf_node);
    MS_LOG(DEBUG) << "Akg compile " << kernel_name << " kernel and insert cache successfully!";
  }
  return true;
}

bool AkgAscendKernelBuilder::HandleRepeatNodes() {
  for (const auto &[json_generator, anf_node] : repeat_nodes_) {
    auto kernel_name = json_generator.kernel_name();
    auto cached_kernel_pack = tbe::TbeUtils::SearchCache(kernel_name, GetProcessorStr(anf_node));
    if (cached_kernel_pack == nullptr) {
      MS_LOG(ERROR) << "Use cached kernel failed, kernel_name[" << kernel_name << "], fullname_with_scope["
                    << anf_node->fullname_with_scope() << "].";
      return false;
    }
    MS_LOG(INFO) << "Use just compiled kernel, kernel_name[" << kernel_name << "], fullname_with_scope["
                 << anf_node->fullname_with_scope() << "].";
    SetKernelMod(cached_kernel_pack, json_generator, anf_node);
  }
  return true;
}

bool AkgAscendKernelBuilder::AkgOpParallelBuild(const std::vector<JsonNodePair> &build_args) {
  repeat_nodes_.clear();
  auto jsons = GetNotCachedKernelJsons(build_args);
  if (jsons.empty()) {
    return true;
  }

  // Start building in AKG
  if (!AscendKernelBuildClient::Instance().AkgStart(PROCESS_NUM, TIME_OUT)) {
    MS_LOG(ERROR) << "Akg start failed.";
    return false;
  }
  if (!AscendKernelBuildClient::Instance().AkgSendData(jsons)) {
    MS_LOG(ERROR) << "Akg send data failed.";
    return false;
  }
  if (!AscendKernelBuildClient::Instance().AkgWait()) {
    MS_LOG(ERROR) << "Akg compile failed.";
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

bool AkgAscendKernelParallelBuild(const std::vector<AnfNodePtr> &anf_nodes) {
  std::vector<JsonNodePair> json_and_node;
  for (const auto &anf_node : anf_nodes) {
    MS_EXCEPTION_IF_NULL(anf_node);
    AkgKernelJsonGenerator akg_kernel_json_generator;
    auto cnode = anf_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (AnfAlgo::IsGraphKernel(cnode)) {
      auto func_graph = AnfAlgo::GetCNodeFuncGraphPtr(cnode);
      MS_EXCEPTION_IF_NULL(func_graph);
      auto mng = func_graph->manager();
      if (mng == nullptr) {
        mng = Manage(func_graph, true);
        func_graph->set_manager(mng);
      }
      std::vector<AnfNodePtr> node_list, input_list, output_list;
      MS_LOG(INFO) << "Akg start compile composite op[" << anf_node->fullname_with_scope() << "]";
      GetValidKernelNodes(func_graph, &node_list, &input_list, &output_list);
      if (!akg_kernel_json_generator.CollectFusedJson(node_list, input_list, output_list)) {
        MS_EXCEPTION(UnknownError) << "Akg build failed composite op[" << anf_node->fullname_with_scope() << "].";
      }
    } else {
      if (!akg_kernel_json_generator.CollectJson(anf_node)) {
        MS_EXCEPTION(UnknownError) << "Akg build failed basic op[" << anf_node->fullname_with_scope() << "].";
      }
    }
    json_and_node.push_back({akg_kernel_json_generator, anf_node});
  }

  if (json_and_node.empty()) {
    MS_LOG(DEBUG) << "There is no kernel needed to be compiled.";
    return true;
  }

  AkgAscendKernelBuilder akg_ascend_kernel_builder;
  return akg_ascend_kernel_builder.AkgOpParallelBuild(json_and_node);
}
}  // namespace kernel
}  // namespace mindspore
