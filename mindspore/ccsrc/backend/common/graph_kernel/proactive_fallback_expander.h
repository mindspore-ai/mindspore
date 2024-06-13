/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_PROACTIVE_FALLBACK_EXPANDER_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_PROACTIVE_FALLBACK_EXPANDER_H_

#include <string>
#include <unordered_set>

#include "include/backend/optimizer/pass.h"
#include "include/backend/optimizer/optimizer.h"

namespace mindspore::graphkernel {
/**
 * @brief this pass fallbacks some highlevel operations to its low level representations.
 * e.g.: AddExt(x, y, alpha) -> Add(x, Mul(alpha, y)). This transformation makes further optimizations easy.
 */
class ProactiveFallbackExpander : public opt::Pass {
 public:
  ProactiveFallbackExpander() : Pass("proactive_fallback_expander") {}
  ~ProactiveFallbackExpander() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 protected:
  const std::unordered_set<std::string> &GetFallbackOps();
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_PROACTIVE_FALLBACK_EXPANDER_H_
