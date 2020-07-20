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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_REPLACE_NODE_BY_PROXY_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_REPLACE_NODE_BY_PROXY_H_
#include <utility>
#include <vector>
#include <string>

#include "backend/optimizer/common/pass.h"
#include "ir/func_graph.h"
#include "ir/anf.h"
#include "utils/utils.h"
#include "backend/kernel_compiler/kernel_build_info.h"

namespace mindspore {
namespace opt {
class ReplaceNodeByProxy : public Pass {
 public:
  explicit ReplaceNodeByProxy(const std::string &name) : Pass(name) {}
  ~ReplaceNodeByProxy() override = default;
  bool Run(const FuncGraphPtr &graph) override;

 private:
  kernel::KernelBuildInfoPtr GenerateKernelBuildInfo(const CNodePtr &cnode);
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_REPLACE_NODE_BY_PROXY_H_
