/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_OVERLAP_1B1F_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_OVERLAP_1B1F_H_
#include <string>
#include <unordered_map>
#include "include/backend/visible.h"
#include "include/backend/common/pass_manager/pass.h"
#include "include/backend/common/kernel_graph/kernel_graph.h"

namespace mindspore {
namespace opt {
class BACKEND_COMMON_EXPORT Overlap1b1f : public Pass {
 public:
  explicit Overlap1b1f(const std::string &name = "overlap_1b1f") : Pass(name) {}
  ~Overlap1b1f() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  bool DoOverlap1b1f(const KernelGraphPtr &kernel_graph);
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_OVERLAP_1B1F_H_
