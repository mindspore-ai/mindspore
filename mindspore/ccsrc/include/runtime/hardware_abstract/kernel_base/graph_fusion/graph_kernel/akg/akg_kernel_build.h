/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#ifndef CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_KERNEL_BASE_GRAPH_FUSION_GRAPH_KERNEL_AKG_AKG_KERNEL_BUILD_H_
#define CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_KERNEL_BASE_GRAPH_FUSION_GRAPH_KERNEL_AKG_AKG_KERNEL_BUILD_H_

#include <vector>
#include <string>
#include "nlohmann/json.hpp"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_build_client.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/graph_kernel_json_generator.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/graph_kernel_builder.h"

namespace mindspore {
namespace kernel {
using graphkernel::GraphKernelJsonGenerator;

class RUNTIME_HARDWARE_EXPORT AkgKernelBuilder : public GraphKernelBuilder {
 public:
  AkgKernelBuilder() = default;
  virtual ~AkgKernelBuilder() = default;

  bool SingleOpParallelBuild(const std::vector<AnfNodePtr> &anf_nodes) override;
  bool ParallelBuild(const std::vector<JsonNodePair> &build_args) override;

 private:
  virtual std::string GetPlatform() const { return "default"; }
  bool AkgOpParallelBuild(const std::vector<JsonNodePair> &build_args);
};
}  // namespace kernel
}  // namespace mindspore

#endif  // CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_KERNEL_BASE_GRAPH_FUSION_GRAPH_KERNEL_AKG_AKG_KERNEL_BUILD_H_
