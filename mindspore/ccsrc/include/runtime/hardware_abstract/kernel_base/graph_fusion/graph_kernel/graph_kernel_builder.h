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

#ifndef MINDSPORE_CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_KERNEL_BASE_GRAPH_FUSION_GRAPH_KERNEL_GRAPH_KERNEL_BUILD_H_
#define MINDSPORE_CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_KERNEL_BASE_GRAPH_FUSION_GRAPH_KERNEL_GRAPH_KERNEL_BUILD_H_

#include <string>
#include <utility>
#include <tuple>
#include <vector>
#include <map>
#include <set>
#include "nlohmann/json.hpp"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/kash/kernel_pack.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_build_client.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/graph_kernel_json_generator.h"

namespace mindspore {
namespace kernel {
using graphkernel::GraphKernelJsonGenerator;
using JsonNodePair = std::pair<GraphKernelJsonGenerator, AnfNodePtr>;

class RUNTIME_HARDWARE_EXPORT GraphKernelBuilder {
 public:
  GraphKernelBuilder() = default;
  virtual ~GraphKernelBuilder() = default;

  virtual KernelPackPtr SearchKernelCache(const std::string &kernel_name);
  virtual KernelPackPtr InsertKernelCache(const std::string &kernel_name);
  virtual void LoadCache();

  virtual KernelBuildClient *GetClient() = 0;
  virtual void SetKernelMod(const KernelPackPtr &kernel_pack, const GraphKernelJsonGenerator &json_generator,
                            const AnfNodePtr &anf_node) = 0;
  virtual void SaveJsonInfo(const string &kernel_name, const string &kernel_json) = 0;
  virtual bool SingleOpParallelBuild(const std::vector<AnfNodePtr> &anf_nodes) = 0;
  virtual bool ParallelBuild(const std::vector<JsonNodePair> &build_args) = 0;

  static std::function<std::tuple<FuncGraphPtr, AnfNodePtrList, AnfNodePtrList>(const AnfNodePtrList &)> build_func_;

 protected:
  std::vector<std::string> GetKernelJsonsByHashId(const std::vector<JsonNodePair> &build_args,
                                                  const std::set<size_t> &fetched_ids);
  std::vector<JsonNodePair> GetNotCachedKernels(const std::vector<JsonNodePair> &build_args);

  bool InsertToCache(const std::vector<JsonNodePair> &build_args);
  bool HandleRepeatNodes();

  std::vector<JsonNodePair> repeat_nodes_;
  nlohmann::json build_attrs_;
  std::string CollectBuildAttrs();
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_KERNEL_BASE_GRAPH_FUSION_GRAPH_KERNEL_GRAPH_KERNEL_BUILD_H_
