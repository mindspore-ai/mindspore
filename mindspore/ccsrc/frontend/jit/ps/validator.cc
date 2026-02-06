/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#include "frontend/jit/ps/validator.h"

#include <memory>
#include <mutex>
#include <string>

#include "primitive/structure_ops.h"
#include "primitive/sequence_ops.h"
#include "primitive/other_ops.h"
#include "primitive/nn_optimizer_ops.h"
#include "primitive/framework_ops.h"
#include "ir/manager.h"
#include "ir/dtype.h"
#include "ir/graph_utils.h"
#include "utils/anf_utils.h"
#include "utils/compile_config.h"
#include "utils/trace_info.h"
#include "frontend/jit/ps/static_analysis/prim.h"
#include "frontend/jit/ps/parse/resolve.h"
#include "frontend/jit/ps/debug/trace.h"
#include "abstract/abstract_function.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_h.h"
#include "primitive/auto_generate/gen_ops_primitive_i.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_p.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "primitive/auto_generate/gen_ops_primitive_u.h"

namespace mindspore {
namespace validator {
using mindspore::abstract::AbstractBase;
using mindspore::abstract::AbstractFunction;
using mindspore::abstract::AbstractJTagged;
using mindspore::abstract::AbstractList;
using mindspore::abstract::AbstractMapTensor;
using mindspore::abstract::AbstractProblem;
using mindspore::abstract::AbstractRefTensor;
using mindspore::abstract::AbstractRowTensor;
using mindspore::abstract::AbstractScalar;
using mindspore::abstract::AbstractSequence;
using mindspore::abstract::AbstractTensor;
using mindspore::abstract::AbstractTuple;
using mindspore::abstract::AbstractType;
void ValidateOperation(const AnfNodePtr &node) {
  if (!IsValueNode<Primitive>(node)) {
    return;
  }

  // Primitive must in whitelist
  auto prim = GetValueNode<PrimitivePtr>(node);
  MS_EXCEPTION_IF_NULL(prim);
  if (prim->isa<prim::DoSignaturePrimitive>()) {
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, node)
      << "Illegal DoSignaturePrimitive '" << prim->name() << "' in the graph."
      << "node:" << node->DebugString() << ", location:" << trace::GetDebugInfoStr(node->debug_info());
  }
  if (abstract::IsInWhiteList(prim)) {
    return;
  }
  if (prim->HasAttr("is_load")) {
    return;
  }
  if (prim->name() == "PyExecute") {
    return;
  }
  if (prim->name() == "TensorMove") {
    return;
  }

  if (prim->isa<PrimitivePy>()) {
    MS_LOG(DEBUG) << "Primitive " << prim->name() << " has python evaluator.";
    return;
  }
  if (prim->name() == "fake_bprop") {
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, node)
      << "Illegal primitive: " << GetValue<std::string>(prim->GetAttr("info")) << "node:" << node->DebugString()
      << ", location:" << trace::GetDebugInfoStr(node->debug_info());
  }

  MS_LOG_WITH_NODE(EXCEPTION, node) << "Illegal primitive: " << prim->name()
                                    << ". Please check whether to use unsupported primitive:" << node->DebugString()
                                    << ", location:" << trace::GetDebugInfoStr(node->debug_info());
}

bool CheckAbstractScalar(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  AbstractBasePtr abstract = node->abstract();
  if (abstract->isa<AbstractScalar>()) {
    TypePtr type = abstract->GetTypeTrack();
    MS_EXCEPTION_IF_NULL(type);
    if (type->isa<EnvType>() || type->isa<MsClassType>()) {
      MS_LOG_WITH_NODE(EXCEPTION, node) << "Illegal type in the graph: " << abstract->ToString()
                                        << ", node: " << node->DebugString()
                                        << ", location:" << trace::GetDebugInfoStr(node->debug_info());
    }
    auto real_node = node;
    if (IsPrimitiveCNode(node, prim::kPrimReturn) || IsPrimitiveCNode(node, prim::kPrimDepend)) {
      real_node = real_node->cast<CNodePtr>()->input(1);
    }
    // Only allow string/number type from external.
    if (type->isa<External>() && !IsValueNode<StringImm>(real_node) && !IsValueNode<FP32Imm>(real_node) &&
        !IsValueNode<FP64Imm>(real_node)) {
      MS_LOG_WITH_NODE(EXCEPTION, node) << "Illegal type in the graph: " << abstract->ToString()
                                        << ", node: " << real_node->DebugString()
                                        << "\nPlease check your code:" << trace::GetDebugInfoStr(node->debug_info());
    }
    // When a DeadNode is renormalized before, its abstract may be changed to
    // AbstractScalar(std:: make_shared<Int32Imm>(0), std:: make_shared<Problem>()).
    if (type->isa<Problem>()) {
      auto value = abstract->GetValueTrack();
      MS_EXCEPTION_IF_NULL(value);
      node->set_abstract(value->ToAbstract());
    }
    // Backend not support string, need convert to tensor.
    // JoinedStr: AbstractScalar(Type: String) --> AbstractTensor.
    if (IsPrimitiveCNode(node, prim::kPrimJoinedStr)) {
      node->set_abstract(std::make_shared<AbstractTensor>(kInt64, ShapeVector({1})));
    }
    return true;
  }
  return false;
}

void ValidateAbstract(const AnfNodePtr &node, const FuncGraphPtr &func_graph) {
  if (node == nullptr) {
    MS_LOG(DEBUG) << "Node to validate is invalid";
    return;
  }
  AbstractBasePtr abstract = node->abstract();
  if (abstract == nullptr) {
    MS_LOG(DEBUG) << "Abstract is null in node: " << node->DebugString();
    return;
  }
  if (CheckAbstractScalar(node)) {
    return;
  }
  if (abstract->isa<AbstractProblem>()) {
    // NOTICE: validate dead code?
    MS_LOG(DEBUG) << "AbstractProblem in the graph: " << abstract->ToString();
    return;
  }

  if (abstract->isa<abstract::AbstractEvent>()) {
    auto scalar_abs = std::make_shared<AbstractScalar>(0);
    node->set_abstract(scalar_abs);
    MS_LOG(DEBUG) << "Update event abstract: " << scalar_abs->ToString();

    if (node->isa<ValueNode>()) {
      ValuePtr val = scalar_abs->BuildValue();
      MS_EXCEPTION_IF_NULL(val);
      AnfNodePtr value_node = NewValueNode(val);
      value_node->set_abstract(scalar_abs);
      auto mgr = func_graph->manager();
      MS_EXCEPTION_IF_NULL(mgr);
      mgr->Replace(node, value_node);
    }
    return;
  }

  bool is_legal_abstract = abstract->isa<AbstractType>() || abstract->isa<AbstractFunction>() ||
                           abstract->isa<AbstractTuple>() || abstract->isa<AbstractList>() ||
                           abstract->isa<AbstractTensor>() || abstract->isa<AbstractRowTensor>() ||
                           abstract->isa<AbstractRefTensor>() || abstract->isa<AbstractMapTensor>() ||
                           abstract->isa<abstract::AbstractNone>() || abstract->isa<abstract::AbstractMonad>() ||
                           abstract->isa<abstract::AbstractScript>();
  if (is_legal_abstract && !abstract->isa<abstract::FunctionalAbstractClosure>()) {
    return;
  }

  // Other types show exception
  MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, node)
    << "Illegal type in the graph: " << abstract->ToString() << ", node: " << node->DebugString()
    << "\nPlease check your code:" << trace::GetDebugInfoStr(node->debug_info());
}

void CheckValueTuple(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto value_node = node->cast_ptr<ValueNode>();
  MS_EXCEPTION_IF_NULL(value_node);
  const auto &value = value_node->value();
  MS_EXCEPTION_IF_NULL(value);
  auto value_tuple = value->cast_ptr<ValueTuple>();
  MS_EXCEPTION_IF_NULL(value_tuple);
  const auto &tuple_values = value_tuple->value();
  for (const auto &tuple_value : tuple_values) {
    auto input_node = NewValueNode(tuple_value);
    ValidateOperation(input_node);
  }
}

void CheckAssignReturnValue(const AnfNodePtr &node) {
  static const PrimitiveSet assign_prims = {prim::kPrimAssign, prim::kPrimAssignAdd, prim::kPrimAssignSub};
  if (IsPrimitiveCNode(node, prim::kPrimDepend)) {
    auto real_input = node->cast_ptr<CNode>()->input(1);
    while (IsPrimitiveCNode(real_input, prim::kPrimDepend)) {
      real_input = real_input->cast_ptr<CNode>()->input(1);
    }
    if (!IsOneOfPrimitiveCNode(real_input, assign_prims)) {
      return;
    }
  } else if (!IsOneOfPrimitiveCNode(node, assign_prims)) {
    return;
  }
  auto fg = node->func_graph();
  MS_EXCEPTION_IF_NULL(fg);
  auto mgr = fg->manager();
  MS_EXCEPTION_IF_NULL(mgr);
  auto &node_users = mgr->node_users();
  auto iter = node_users.find(node);
  if (iter == node_users.end()) {
    return;
  }
  static const PrimitiveSet virtual_prims = {
    prim::kPrimImageSummary, prim::kPrimScalarSummary, prim::kPrimTensorSummary, prim::kPrimHistogramSummary,
    prim::kPrimMakeTuple,    prim::kPrimStateSetItem,  prim::kPrimTupleGetItem,  prim::kPrimLoad,
    prim::kPrimPartial,      prim::kPrimDepend,        prim::kPrimUpdateState,   prim::kPrimDynamicLossScale};
  auto users = iter->second;
  for (const auto &user : users) {
    auto user_node = user.first;
    if (!IsOneOfPrimitiveCNode(user_node, virtual_prims)) {
      MS_LOG(WARNING) << "Deprecated: the return value of Assign/AssignAdd/AssignSub operator will be removed "
                      << "in subsequent releases.\n"
                      << "You can modify the code from:\na = P.Assign()(param, value)\nb = a * 2\nto: \n"
                      << "P.Assign()(param, value)\nb = param * 2\n"
                      << "Please check your code:" << trace::GetDebugInfoStr(node->debug_info());
    }
  }
}

void CheckDeadNodeInOutputRecursively(const AnfNodePtr &node, const AbstractBasePtr &abstract) {
  if (abstract == nullptr) {
    return;
  }
  TypePtr type = abstract->BuildType();
  MS_EXCEPTION_IF_NULL(type);
  if (type->isa<Problem>() || type->isa<Function>()) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Function in output is not supported. Please check your code. "
                                      << trace::GetDebugInfoStr(node->debug_info());
  }
  if (abstract->isa<AbstractSequence>()) {
    auto abs_seq = abstract->cast_ptr<AbstractSequence>();
    for (const auto &elem : abs_seq->elements()) {
      CheckDeadNodeInOutputRecursively(node, elem);
    }
  }
}

void ValidateTopGraphOutput(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto abstract = node->abstract();
  CheckDeadNodeInOutputRecursively(node, abstract);
}

void ValidateScope(const AnfNodePtr &node, const std::string &pass_name) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<ValueNode>() || node->isa<Parameter>()) {
    return;
  }
  if (!AnfUtils::IsRealKernel(node)) {
    return;
  }
  if (node->abstract() != nullptr && node->abstract()->isa<abstract::AbstractFunction>()) {
    return;
  }
  if (node->scope() == nullptr || node->scope() == kDefaultScope) {
    MS_LOG(ERROR) << "In " << pass_name << ", failed to find scope for node "
                  << node->DebugString(AnfNode::DebugStringLevel::kLevel2);
  }
  if (node->scope() == kDefaultScopeUnderGuard) {
    MS_LOG(INFO) << "In " << pass_name << ", encounter kDefaultScopeUnderGuard for node: "
                 << node->DebugString(AnfNode::DebugStringLevel::kLevel2);
  }
}

void Validate(const FuncGraphPtr &func_graph) {
  ValidateTopGraphOutput(func_graph->output());
  const auto &all_nodes = TopoSort(func_graph->return_node(), SuccDeeperSimple);
  for (auto node : all_nodes) {
    TraceGuard guard(MakeTraceInfo<TraceCopy>(node->debug_info()));
    if (common::GetCompileConfig("CHECK_PASS_NODE_SCOPE") == "1") {
      ValidateScope(node, "Validate");
    }
    CheckAssignReturnValue(node);
    while (IsPrimitiveCNode(node, prim::kPrimReturn) || IsPrimitiveCNode(node, prim::kPrimDepend)) {
      node = node->cast_ptr<CNode>()->input(1);
    }
    if (IsValueNode<ValueTuple>(node)) {
      CheckValueTuple(node);
      continue;
    }
    ValidateOperation(node);
  }
  for (const auto &node : all_nodes) {
    ValidateAbstract(node, func_graph);
  }
}
}  // namespace validator
}  // namespace mindspore
