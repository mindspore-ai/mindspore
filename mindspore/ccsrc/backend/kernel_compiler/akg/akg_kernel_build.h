/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_AKG_KERNEL_BUILD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_AKG_KERNEL_BUILD_H_

#include <string>
#include <utility>
#include <vector>
#include <map>
#include "ir/anf.h"
#include "backend/kernel_compiler/kernel.h"
#include "backend/session/kernel_build_client.h"
#include "backend/kernel_compiler/akg/akg_kernel_json_generator.h"

namespace mindspore {
namespace kernel {
using JsonNodePair = std::pair<AkgKernelJsonGenerator, AnfNodePtr>;

class AkgKernelBuilder {
 public:
  AkgKernelBuilder() = default;
  ~AkgKernelBuilder() = default;

  virtual KernelBuildClient *GetClient() = 0;
  virtual KernelPackPtr AkgSearchCache(const std::string &kernel_name, const std::string &processor) = 0;
  virtual KernelPackPtr AkgInsertCache(const std::string &kernel_name, const std::string &processor) = 0;
  virtual void AkgSetKernelMod(const KernelPackPtr &kernel_pack, const AkgKernelJsonGenerator &json_generator,
                               const AnfNodePtr &anf_node) = 0;
  virtual void AkgSaveJsonInfo(const string &kernel_name, const string &kernel_json) = 0;
  bool AkgKernelParallelBuild(const std::vector<AnfNodePtr> &anf_nodes);

 private:
  std::vector<std::string> GetNotCachedKernelJsons(const std::vector<JsonNodePair> &build_args);
  bool InsertToCache(const std::vector<JsonNodePair> &build_args);
  bool HandleRepeatNodes();
  bool AkgOpParallelBuild(const std::vector<JsonNodePair> &build_args);
  std::vector<JsonNodePair> repeat_nodes_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_AKG_KERNEL_BUILD_H_
