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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_GE_EXPANDER_FALLBACK_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_GE_EXPANDER_FALLBACK_H_

#include "include/backend/optimizer/pass.h"
#include "ir/func_graph.h"

namespace mindspore {
namespace opt {
class ExpanderFallback : public Pass {
 public:
  ExpanderFallback() : Pass("expander_fallback") {}
  ~ExpanderFallback() override = default;
  bool Run(const FuncGraphPtr &graph) override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_GE_EXPANDER_FALLBACK_H_
