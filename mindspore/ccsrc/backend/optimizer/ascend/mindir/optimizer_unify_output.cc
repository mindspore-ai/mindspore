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

#include "backend/optimizer/ascend/mindir/optimizer_unify_output.h"

#include <vector>
#include <memory>

#include "abstract/abstract_value.h"
#include "backend/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kFtrlOutputNum = 3;
constexpr size_t kMomentumOutputNum = 2;
constexpr size_t kRMSPropOutputNum = 3;
constexpr size_t kCenteredRMSPropOutputNum = 4;

CNodePtr ProcessOutput(const FuncGraphPtr &graph, const AnfNodePtr &node, const size_t output_size) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  auto cnode_ptr = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode_ptr);

  auto abstract = cnode_ptr->abstract();
  MS_EXCEPTION_IF_NULL(abstract);

  if (AnfAlgo::HasNodeAttr("optim_output_passed", cnode_ptr) && abstract->isa<abstract::AbstractTuple>()) {
    return nullptr;
  }
  AnfAlgo::SetNodeAttr("optim_output_passed", MakeValue(true), cnode_ptr);

  std::vector<AbstractBasePtr> abstract_list;
  for (size_t i = 0; i < output_size; i++) {
    abstract_list.push_back(abstract->Clone());
  }
  auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(abstract_list);
  cnode_ptr->set_abstract(abstract_tuple);

  auto index = NewValueNode(static_cast<int64_t>(0));
  auto get_item = graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), cnode_ptr, index});
  MS_EXCEPTION_IF_NULL(get_item);

  get_item->set_abstract(abstract->Clone());
  return get_item;
}
}  // namespace

const BaseRef FtrlUnifyOutput::DefinePattern() const {
  VarPtr var = std::make_shared<Var>();
  VarPtr accum = std::make_shared<Var>();
  VarPtr linear = std::make_shared<Var>();
  VarPtr grad = std::make_shared<Var>();
  VarPtr lr = std::make_shared<Var>();
  VarPtr l1 = std::make_shared<Var>();
  VarPtr l2 = std::make_shared<Var>();
  VarPtr lr_power = std::make_shared<Var>();
  VarPtr u = std::make_shared<SeqVar>();
  VectorRef pattern({prim::kPrimApplyFtrl, var, accum, linear, grad, lr, l1, l2, lr_power, u});
  return pattern;
}

const AnfNodePtr FtrlUnifyOutput::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  return ProcessOutput(graph, node, kFtrlOutputNum);
}

const BaseRef MomentumUnifyOutput::DefinePattern() const {
  VarPtr var = std::make_shared<Var>();
  VarPtr accum = std::make_shared<Var>();
  VarPtr lr = std::make_shared<Var>();
  VarPtr grad = std::make_shared<Var>();
  VarPtr momentum = std::make_shared<Var>();
  VarPtr u = std::make_shared<SeqVar>();
  VectorRef pattern({prim::kPrimApplyMomentum, var, accum, lr, grad, momentum, u});
  return pattern;
}

const AnfNodePtr MomentumUnifyOutput::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                              const EquivPtr &) const {
  return ProcessOutput(graph, node, kMomentumOutputNum);
}

const BaseRef RMSPropUnifyOutput::DefinePattern() const {
  VarPtr inputs = std::make_shared<SeqVar>();
  VectorRef pattern({prim::kPrimApplyRMSProp, inputs});
  return pattern;
}

const AnfNodePtr RMSPropUnifyOutput::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                             const EquivPtr &) const {
  return ProcessOutput(graph, node, kRMSPropOutputNum);
}

const BaseRef CenteredRMSPropUnifyOutput::DefinePattern() const {
  VarPtr var = std::make_shared<Var>();
  VarPtr mg = std::make_shared<Var>();
  VarPtr ms = std::make_shared<Var>();
  VarPtr mom = std::make_shared<Var>();
  VarPtr grad = std::make_shared<Var>();
  VarPtr lr = std::make_shared<Var>();
  VarPtr rho = std::make_shared<Var>();
  VarPtr momentum = std::make_shared<Var>();
  VarPtr epsilon = std::make_shared<Var>();
  VarPtr u = std::make_shared<SeqVar>();
  VectorRef pattern({prim::kPrimApplyCenteredRMSProp, var, mg, ms, mom, grad, lr, rho, momentum, epsilon, u});
  return pattern;
}

const AnfNodePtr CenteredRMSPropUnifyOutput::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                     const EquivPtr &) const {
  return ProcessOutput(graph, node, kCenteredRMSPropOutputNum);
}
}  // namespace opt
}  // namespace mindspore
