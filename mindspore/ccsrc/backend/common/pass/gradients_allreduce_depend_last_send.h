/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_GRADIENTS_ALLREDUCE_DEPEBD_LAST_SEND_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_GRADIENTS_ALLREDUCE_DEPEBD_LAST_SEND_H_
#include <vector>
#include <string>
#include <memory>

#include "include/backend/optimizer/pass.h"
#include "ir/func_graph.h"
#include "ir/anf.h"
#include "include/backend/optimizer/helper.h"
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class BACKEND_EXPORT GradientsAllReduceDependLastSend : public Pass {
 public:
  GradientsAllReduceDependLastSend() : Pass("adjust_depend_for_parallel_optimizer_recompute_all_gather") {}
  ~GradientsAllReduceDependLastSend() override = default;
  bool Run(const FuncGraphPtr &graph) override;

 private:
  bool InsertDependBetweenAllReduceAndSend(const FuncGraphPtr &graph, const std::vector<CNodePtr> &addn_list,
                                           const CNodePtr &last_send) const;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_GRADIENTS_ALLREDUCE_DEPEBD_LAST_SEND_H_
