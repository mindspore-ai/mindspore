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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_BISHENG_CPU_BISHENG_CPU_KERNEL_BUILD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_BISHENG_CPU_BISHENG_CPU_KERNEL_BUILD_H_
#include <string>
#include <utility>
#include <vector>
#include "kernel/akg/akg_kernel_build.h"
#include "kernel/akg/akg_kernel_build_manager.h"
#include "base/base.h"

namespace mindspore {
namespace kernel {
const char kBISHENG[] = "BISHENG";
using JsonNodePair = std::pair<AkgKernelJsonGenerator, AnfNodePtr>;

class BishengCpuKernelBuilder : public AkgKernelBuilder {
 public:
  BishengCpuKernelBuilder() = default;
  ~BishengCpuKernelBuilder() = default;

  kernel::KernelBuildClient *GetClient() override { return 0; }
  void AkgSetKernelMod(const KernelPackPtr &kernel_pack, const AkgKernelJsonGenerator &json_generator,
                       const AnfNodePtr &anf_node) override {
    return;
  };
  void BishengSetKernelMod(const string &kernel_name, const AkgKernelJsonGenerator &json_generator,
                           const AnfNodePtr &anf_node);
  void AkgSaveJsonInfo(const string &kernel_name, const string &kernel_json) override;
  bool ParallelBuild(const std::vector<JsonNodePair> &json_and_node) override;
};

REG_AKG_KERNEL_BUILDER(kBISHENG, BishengCpuKernelBuilder);
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_BISHENG_CPU_BISHENG_CPU_KERNEL_BUILD_H_
