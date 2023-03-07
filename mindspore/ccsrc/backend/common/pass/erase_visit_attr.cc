/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "backend/common/pass/erase_visit_attr.h"
#include <memory>
#include "kernel/common_utils.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/optimizer/helper.h"

namespace mindspore {
namespace opt {
const BaseRef EraseVisitAttr::DefinePattern() const {
  std::shared_ptr<Var> V = std::make_shared<CondVar>(Visited);
  std::shared_ptr<Var> Xs = std::make_shared<SeqVar>();
  return VectorRef({V, Xs});
}

const AnfNodePtr EraseVisitAttr::Process(const FuncGraphPtr &, const AnfNodePtr &node, const EquivPtr &) const {
  common::AnfAlgo::EraseNodeAttr(kAttrVisited, node);
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
