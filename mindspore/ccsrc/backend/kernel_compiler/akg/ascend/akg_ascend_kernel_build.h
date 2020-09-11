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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_ASCEND_AKG_ASCEND_KERNEL_BUILD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_ASCEND_AKG_ASCEND_KERNEL_BUILD_H_

#include <string>
#include <utility>
#include <vector>
#include <map>
#include "ir/anf.h"
#include "backend/kernel_compiler/kernel.h"
#include "backend/kernel_compiler/akg/akg_kernel_json_generator.h"

namespace mindspore {
namespace kernel {
using JsonNodePair = std::pair<AkgKernelJsonGenerator, AnfNodePtr>;

class AkgAscendKernelBuilder {
 public:
  AkgAscendKernelBuilder() = default;
  ~AkgAscendKernelBuilder() = default;
  bool AkgOpParallelBuild(const std::vector<JsonNodePair> &build_args);

 private:
  std::vector<std::string> GetNotCachedKernelJsons(const std::vector<JsonNodePair> &build_args);
  bool InsertToCache(const std::vector<JsonNodePair> &build_args);
  bool HandleRepeatNodes();

  std::vector<JsonNodePair> repeat_nodes_;
};

bool AkgAscendKernelParallelBuild(const std::vector<AnfNodePtr> &anf_nodes);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_ASCEND_AKG_ASCEND_KERNEL_BUILD_H_
