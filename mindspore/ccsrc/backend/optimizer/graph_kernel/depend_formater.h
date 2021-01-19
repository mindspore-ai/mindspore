
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_DEPEND_FORMATER_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_DEPEND_FORMATER_H_

#include <map>
#include <memory>

#include "backend/optimizer/common/pass.h"
#include "ir/func_graph.h"

namespace mindspore {
namespace opt {
class DependFormater : public Pass {
 public:
  DependFormater() : Pass("depend_formater") {}
  ~DependFormater() override = default;
  bool Run(const FuncGraphPtr &graph) override;
};
using DependFormaterPtr = std::shared_ptr<DependFormater>;
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_DEPEND_FORMATER_H_
