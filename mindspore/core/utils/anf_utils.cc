/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "utils/anf_utils.h"
#include <memory>
#include <string>
#include <list>
#include <algorithm>
#include "mindspore/core/ops/core_ops.h"
#include "utils/trace_base.h"
#include "utils/hash_map.h"
#include "utils/os.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace {
class AbstractMutexManager {
 public:
  static AbstractMutexManager &GetInstance() {
    static AbstractMutexManager instance;
    return instance;
  }

  std::recursive_mutex *GetAbstractLock(const AnfNode *node) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    if (is_valid_) {
      return &mu_for_nodes_[node];
    } else {
      return nullptr;
    }
  }

  void Close() {
    is_valid_ = false;
    mu_for_nodes_.clear();
  }
  void Open() { is_valid_ = true; }

 private:
  mindspore::HashMap<const AnfNode *, std::recursive_mutex> mu_for_nodes_;
  std::recursive_mutex mu_;
  bool is_valid_ = false;
};

struct CustomActorInfo {
  CustomActorInfo(const AnfUtils::CustomActorCallback &func, const std::string &type_name, const CNodePtr &cnode)
      : actor_func(func), type_name(type_name), base_cnode_ptr(cnode) {}
  ~CustomActorInfo() = default;

  // Key for user data.
  constexpr static char key[] = "CustomActor";
  AnfUtils::CustomActorCallback actor_func = {};
  std::string type_name;
  CNodeWeakPtr base_cnode_ptr;
};
using CustomActorInfoPtr = std::shared_ptr<CustomActorInfo>;

struct CNodeCustomInfo {
  CNodeCustomInfo(const AnfNodePtr &inferop, const AnfNodePtr &initop) : infer_node(inferop), init_node(initop) {}
  ~CNodeCustomInfo() = default;
  // Key for user data.
  constexpr static char key[] = "CustomNodeInfo";
  AnfNodeWeakPtr infer_node;
  AnfNodeWeakPtr init_node;
};
using CNodeCustomInfoPtr = std::shared_ptr<CNodeCustomInfo>;
struct RealInputInfo {
  explicit RealInputInfo(const CNodePtr &cnode) : base_cnode_ptr(cnode), real_input_nodes() {}
  ~RealInputInfo() = default;
  // Key for user data.
  constexpr static char key[] = "RealInputInfo";
  CNodeWeakPtr base_cnode_ptr;
  // HashMap <input_index, pair<pre_node, pre_node_output_index>> is used to record the real input node to infer the
  // dynamic shape information of the nodes located at the boundary of the graph partition, such as heterogeneous
  // scenario and so on.
  mindspore::HashMap<size_t, std::pair<AnfNodeWeakPtr, size_t>> real_input_nodes;
};

AnfNodePtr NewCustomActorNode(const CustomActorInfoPtr &actor_info, const FuncGraphPtr &g) {
  MS_EXCEPTION_IF_NULL(g);
  auto custom_actor_node = std::make_shared<AnfNode>(g);
  custom_actor_node->set_user_data<CustomActorInfo>(actor_info);
  return custom_actor_node;
}
}  // namespace

AbstractScope::AbstractScope(std::recursive_mutex *mu) : mu_(mu) {
  if (mu_ != nullptr) {
    mu_->lock();
  }
}

AbstractScope::AbstractScope(AbstractScope &&other) {
  mu_ = other.mu_;
  other.mu_ = nullptr;
}

AbstractScope &AbstractScope::operator=(AbstractScope &&other) {
  mu_ = other.mu_;
  other.mu_ = nullptr;
  return *this;
}

AbstractScope::~AbstractScope() {
  if (mu_ != nullptr) {
    mu_->unlock();
  }
}

AbstractScope AnfUtils::GetAbstractLock(const AnfNode *node) {
  return AbstractScope(AbstractMutexManager::GetInstance().GetAbstractLock(node));
}

void AnfUtils::OpenAbstractLock() { AbstractMutexManager::GetInstance().Open(); }

void AnfUtils::CloseAbstractLock() { AbstractMutexManager::GetInstance().Close(); }

// If the node's shape is dynamic shape or dynamic rank, return true.
bool AnfUtils::IsNodeOutputShapeDynamic(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto base_shape = node->Shape();
  if (base_shape == nullptr) {
    MS_LOG(INFO) << "Invalid base shape, node: " << node->fullname_with_scope();
    return false;
  }
  return base_shape->IsDynamic();
}

bool AnfUtils::IsRealKernel(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
#ifndef ENABLE_SECURITY
  static const PrimitiveSet virtual_prims = {
    prim::kPrimImageSummary,    prim::kPrimScalarSummary, prim::kPrimTensorSummary, prim::kPrimHistogramSummary,
    prim::kPrimMakeTuple,       prim::kPrimStateSetItem,  prim::kPrimTupleGetItem,  prim::kPrimReturn,
    prim::kPrimPartial,         prim::kPrimDepend,        prim::kPrimUpdateState,   prim::kPrimLoad,
    prim::kPrimDynamicLossScale};
#else
  static const PrimitiveSet virtual_prims = {
    prim::kPrimMakeTuple,   prim::kPrimStateSetItem, prim::kPrimTupleGetItem,
    prim::kPrimReturn,      prim::kPrimPartial,      prim::kPrimDepend,
    prim::kPrimUpdateState, prim::kPrimLoad,         prim::kPrimDynamicLossScale};
#endif
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr) {
    // parameter and value node is a real kernel too
    return true;
  }
  if (cnode->size() == 0) {
    MS_LOG(EXCEPTION) << "Illegal null input of cnode(%s)" << node->DebugString() << trace::DumpSourceLines(node);
  }

  auto kernel_info = cnode->kernel_info();
  if (kernel_info) {
    auto runtime_cache = kernel_info->runtime_cache();
    if (runtime_cache.runtime_cache().is_real_kernel() != Uncached) {
      return (runtime_cache.runtime_cache().is_real_kernel() == True);
    }
  }
  bool res = !IsOneOfPrimitive(cnode->input(kAnfPrimitiveIndex), virtual_prims);

  if (kernel_info) {
    auto runtime_cache = kernel_info->runtime_cache();
    if (res) {
      runtime_cache.runtime_cache().set_real_kernel(True);
    } else {
      runtime_cache.runtime_cache().set_real_kernel(False);
    }
  }

  return res;
}

bool AnfUtils::IsRealCNodeKernel(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  if (IsPrimitiveCNode(node, prim::kPrimReturn)) {
    return true;
  }
  return AnfUtils::IsRealKernel(node);
}

std::string AnfUtils::GetCNodeName(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<CNode>()) {
    auto primitive = GetCNodePrimitive(node);
    if (primitive != nullptr) {
      if (primitive->name() == "Custom") {
        auto uniq_name = primitive->GetAttr("uniq_name");
        if (uniq_name) {
          return GetValue<std::string>(uniq_name);
        }
      }
      return primitive->name();
    }

    // Check whether call node's input is not a value node which contains FuncGraph.
    auto cnode = dyn_cast<CNode>(node);
    MS_EXCEPTION_IF_NULL(cnode);
    if (cnode->size() == 0 || !IsValueNode<FuncGraph>(cnode->input(0))) {
      return "";
    }

    auto func_graph = GetCNodeFuncGraph(node);
    MS_EXCEPTION_IF_NULL(func_graph);
    if (func_graph->has_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL)) {
      std::string fg_name = "GraphKernel_";
      fg_name += GetValue<std::string>(func_graph->get_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL));
      return fg_name;
    }
    return func_graph->ToString();
  }
  MS_LOG(EXCEPTION) << "Unknown anf node type " << node->DebugString() << trace::DumpSourceLines(node);
}

size_t AnfUtils::GetInputTensorNum(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr) {
    MS_LOG(EXCEPTION) << "Only cnode has real input, but this anf is " << node->DebugString()
                      << trace::DumpSourceLines(node);
  }
  {
    // cppcheck-suppress unreadVariable
    auto lock = AnfUtils::GetAbstractLock(cnode.get());
    ssize_t input_tensor_num = cnode->input_tensor_num();
    if (input_tensor_num >= 0) {
      return static_cast<size_t>(input_tensor_num);
    }
  }

  size_t input_num = cnode->inputs().size();
  if (input_num == 0) {
    MS_LOG(EXCEPTION) << "Cnode inputs size can't be zero" << trace::DumpSourceLines(node);
  }
  // Exclude inputs[0].
  --input_num;

  // Exclude monad inputs for real cnodes.
  if (input_num > 0 && AnfUtils::IsRealKernel(cnode)) {
    auto &inputs = cnode->inputs();
    // Search monad inputs, backward.
    for (auto iter = inputs.rbegin(); iter != inputs.rend(); ++iter) {
      // cppcheck-suppress unreadVariable
      auto lock = AnfUtils::GetAbstractLock((*iter).get());
      if (!HasAbstractMonad(*iter)) {
        // Stop count if we encounter a non-monad input.
        break;
      }
      --input_num;
    }
  }
  // cppcheck-suppress unreadVariable
  auto lock = AnfUtils::GetAbstractLock(cnode.get());
  cnode->set_input_tensor_num(static_cast<ssize_t>(input_num));
  return input_num;
}

size_t AnfUtils::GetOutputTensorNum(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = node->kernel_info();
  if (kernel_info) {
    auto runtime_cache = kernel_info->runtime_cache();
    if (runtime_cache.runtime_cache().is_valid()) {
      ssize_t output_tensor_num = runtime_cache.runtime_cache().output_tensor_num();
      if (output_tensor_num >= 0) {
        return static_cast<size_t>(output_tensor_num);
      }
    }
  }

  size_t res;
  TypePtr type = node->Type();
  if (type == nullptr) {
    res = 0;
  } else if (type->isa<Tuple>()) {
    auto tuple_type = type->cast<TuplePtr>();
    MS_EXCEPTION_IF_NULL(tuple_type);
    res = tuple_type->size();
    // Some nodes could have monad outputs like RpcRecv. We need to jump these outputs.
    if (NeedJumpMonadOutput(node) && tuple_type->elements()[res - 1]->isa<MonadType>()) {
      for (size_t i = 0; i < tuple_type->elements().size(); i++) {
        if (tuple_type->elements()[i]->isa<MonadType>()) {
          res = i;
          break;
        }
      }
    }
  } else if (type->isa<List>()) {
    auto list_type = type->cast<ListPtr>();
    MS_EXCEPTION_IF_NULL(list_type);
    res = list_type->size();
  } else if (type->isa<TypeNone>()) {
    res = 0;
  } else if (type->isa<CSRTensorType>()) {
    // Currently, CSRTensor only supports 2-D matrix (shape has 2 values). 5 outputs = 3 Tensors + 2 shape values.
    constexpr size_t kCSRTensorOutputNum = 5;
    res = kCSRTensorOutputNum;
  } else if (type->isa<COOTensorType>()) {
    // Currently, COOTensor only supports 2-D matrix (shape has 2 values). 4 outputs = 2 Tensors + 2 shape values.
    constexpr size_t kCOOTensorOutputNum = 4;
    res = kCOOTensorOutputNum;
  } else if (NeedJumpMonadOutput(node) && type->isa<MonadType>()) {
    // Some nodes could have monad outputs like RpcRecv. We need to jump these outputs.
    res = 0;
  } else {
    res = 1;
  }

  if (kernel_info) {
    auto runtime_cache = kernel_info->runtime_cache();
    if (runtime_cache.runtime_cache().is_valid()) {
      runtime_cache.runtime_cache().set_output_tensor_num(static_cast<ssize_t>(res));
    }
  }
  return res;
}

void AnfUtils::SetNodeAttr(const std::string &key, const ValuePtr &value, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "Only cnode has attr, but this anf is " << node->DebugString() << trace::DumpSourceLines(node);
  }
  // single op cnode.
  auto primitive = GetCNodePrimitive(node);
  if (primitive != nullptr) {
    primitive->set_attr(key, value);
    return;
  }
  // graph kernel cnode.
  auto fg = GetCNodeFuncGraph(node);
  MS_EXCEPTION_IF_NULL(fg);
  fg->set_attr(key, value);
}

int64_t AnfUtils::GetIntValue(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto value_node = anf_node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value();
  return GetIntValue(value);
}

int64_t AnfUtils::GetIntValue(const ValuePtr &value) {
  if (value->isa<Int64Imm>()) {
    return GetValue<int64_t>(value);
  } else if (value->isa<Int32Imm>()) {
    return IntToLong(GetValue<int>(value));
  } else {
    MS_LOG(EXCEPTION) << "The value should be Int32Imm or Int64Imm, but got " << value->ToString();
  }
  return 0;
}

std::pair<AnfNodePtr, size_t> AnfUtils::VisitKernel(const AnfNodePtr &anf_node, size_t index) {
  MS_EXCEPTION_IF_NULL(anf_node);
  const PrimitiveSet follow_first_input_prims = {prim::kPrimDepend, prim::kPrimLoad};
  if (anf_node->isa<ValueNode>()) {
    return std::make_pair(anf_node, 0);
  } else if (anf_node->isa<Parameter>()) {
    return std::make_pair(anf_node, 0);
  } else if (IsCustomActorNode(anf_node)) {
    return std::make_pair(anf_node, 0);
  } else if (anf_node->isa<CNode>()) {
    auto cnode = anf_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto input0 = cnode->input(0);
    MS_EXCEPTION_IF_NULL(input0);
    if (IsPrimitive(input0, prim::kPrimMakeTuple)) {
      if (GetInputTensorNum(cnode) == 0) {
        return std::make_pair(nullptr, 0);
      }
      auto node = cnode->input(index + IntToSize(1));
      MS_EXCEPTION_IF_NULL(node);
      return VisitKernel(node, 0);
    } else if (IsPrimitive(input0, prim::kPrimTupleGetItem)) {
      if (cnode->inputs().size() != kTupleGetItemInputSize) {
        MS_LOG(EXCEPTION) << "The node tuple_get_item must have 2 inputs!";
      }
      auto input2 = cnode->input(kInputNodeOutputIndexInTupleGetItem);
      auto item_idx = AnfUtils::GetIntValue(input2);
      return VisitKernel(cnode->input(kRealInputNodeIndexInTupleGetItem), LongToSize(item_idx));
    } else if (IsPrimitiveCNode(cnode, prim::kPrimUpdateState)) {
      return VisitKernel(cnode->input(kUpdateStateRealInput), 0);
    } else if (IsOneOfPrimitive(input0, follow_first_input_prims)) {
      return VisitKernel(cnode->input(kRealInputIndexInDepend), 0);
    } else {
      return std::make_pair(anf_node, index);
    }
  } else {
    MS_LOG(EXCEPTION) << "The input is invalid";
  }
}

bool AnfUtils::IsGraphKernel(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto func_graph = GetCNodeFuncGraph(node);
  return func_graph != nullptr && func_graph->has_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL);
}

bool AnfUtils::IsNodeInGraphKernel(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  return node->func_graph() != nullptr && node->func_graph()->has_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL);
}

void AnfUtils::SetDumpFlag(const AnfNodePtr &node) {
  if (node == nullptr || !node->isa<CNode>()) {
    return;
  }
  auto prim = GetCNodePrimitive(node);
  if (prim != nullptr) {
    prim->set_attr(kAttrDump, MakeValue(kValueTrue));
  }
}

bool AnfUtils::GetDumpFlag(const AnfNodePtr &node) {
  if (node == nullptr || !node->isa<CNode>()) {
    return false;
  }
  auto prim = GetCNodePrimitive(node);
  if (prim != nullptr) {
    auto attr = prim->GetAttr(kAttrDump);
    if (attr != nullptr && attr->isa<StringImm>() && attr->cast<StringImmPtr>()->value() == kValueTrue) {
      return true;
    }
  }
  return false;
}

bool AnfUtils::HasDumpFlag(const AnfNodePtr &node) {
  if (node == nullptr || !node->isa<CNode>()) {
    return false;
  }
  auto prim = GetCNodePrimitive(node);
  if (prim != nullptr) {
    return prim->HasAttr(kAttrDump);
  }
  return false;
}

bool AnfUtils::IsCustomActorNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  return node->has_user_data<CustomActorInfo>();
}

bool AnfUtils::IsCutomActorNodeSame(const AnfNodePtr &node1, const AnfNodePtr &node2) {
  MS_EXCEPTION_IF_NULL(node1);
  MS_EXCEPTION_IF_NULL(node2);
  if (!IsCustomActorNode(node1) || !IsCustomActorNode(node2)) {
    MS_LOG(EXCEPTION) << "Two node are not all Custom Actor Node!";
  }

  auto actor_info1 = node1->user_data<CustomActorInfo>();
  MS_EXCEPTION_IF_NULL(actor_info1);
  std::string actor_type1 = actor_info1->type_name;

  auto actor_info2 = node2->user_data<CustomActorInfo>();
  MS_EXCEPTION_IF_NULL(actor_info2);
  std::string actor_type2 = actor_info2->type_name;

  return (actor_type1 == actor_type2);
}

std::string AnfUtils::GetCustomActorType(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!IsCustomActorNode(node)) {
    MS_LOG(EXCEPTION) << node->fullname_with_scope() << " is not a custom actor node!";
  }

  auto actor_info = node->user_data<CustomActorInfo>();
  MS_EXCEPTION_IF_NULL(actor_info);
  return actor_info->type_name;
}

std::string AnfUtils::GetCustomActorName(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!IsCustomActorNode(node)) {
    MS_LOG(EXCEPTION) << node->fullname_with_scope() << " is not a custom actor node!";
  }

  auto actor_info = node->user_data<CustomActorInfo>();
  MS_EXCEPTION_IF_NULL(actor_info);
  auto base_node = actor_info->base_cnode_ptr.lock();
  MS_EXCEPTION_IF_NULL(base_node);
  std::string actor_name = actor_info->type_name + "_of_" + base_node->fullname_with_scope();
  return actor_name;
}

CNodePtr AnfUtils::GetCustomActorBaseNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!IsCustomActorNode(node)) {
    MS_LOG(EXCEPTION) << node->fullname_with_scope() << " is not a custom actor node!";
  }

  auto actor_info = node->user_data<CustomActorInfo>();
  MS_EXCEPTION_IF_NULL(actor_info);
  return actor_info->base_cnode_ptr.lock();
}

AnfUtils::CustomActorCallback AnfUtils::GetCustomFunc(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!IsCustomActorNode(node)) {
    MS_LOG(EXCEPTION) << node->fullname_with_scope() << " is not a custom actor node!";
  }

  auto actor_info = node->user_data<CustomActorInfo>();
  MS_EXCEPTION_IF_NULL(actor_info);
  return actor_info->actor_func;
}

AnfNodePtr AnfUtils::NewInitActorNode(AnfUtils::CustomActorCallback f, const CNodePtr &base_cnode) {
  MS_EXCEPTION_IF_NULL(base_cnode);
  auto actor_info = std::make_shared<CustomActorInfo>(f, kInit, base_cnode);
  return NewCustomActorNode(actor_info, base_cnode->func_graph());
}

AnfNodePtr AnfUtils::NewInferActorNode(AnfUtils::CustomActorCallback f, const CNodePtr &base_cnode) {
  MS_EXCEPTION_IF_NULL(base_cnode);
  auto actor_info = std::make_shared<CustomActorInfo>(f, kInfer, base_cnode);
  return NewCustomActorNode(actor_info, base_cnode->func_graph());
}

void AnfUtils::SetCustomInfoToBaseNode(const AnfNodePtr &base_cnode, const AnfNodePtr &inferop,
                                       const AnfNodePtr &initop) {
  MS_EXCEPTION_IF_NULL(base_cnode);
  MS_EXCEPTION_IF_NULL(inferop);
  MS_EXCEPTION_IF_NULL(initop);

  auto actor_info = std::make_shared<CNodeCustomInfo>(inferop, initop);
  base_cnode->set_user_data<CNodeCustomInfo>(actor_info);
}

AnfNodePtr AnfUtils::GetCustomInferopNode(const AnfNodePtr &base_cnode) {
  MS_EXCEPTION_IF_NULL(base_cnode);
  auto actor_info = base_cnode->user_data<CNodeCustomInfo>();
  if (actor_info == nullptr) {
    return nullptr;
  }
  return actor_info->infer_node.lock();
}

mindspore::HashMap<size_t, std::pair<AnfNodeWeakPtr, size_t>> &AnfUtils::GetRealInputNodes(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto real_input_info = cnode->user_data<RealInputInfo>();
  if (real_input_info == nullptr) {
    real_input_info = std::make_shared<RealInputInfo>(cnode);
    cnode->set_user_data(real_input_info);
  }
  return real_input_info->real_input_nodes;
}

bool AnfUtils::NeedJumpMonadOutput(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr) {
    return false;
  }

  std::vector<std::string> jump_monad_output_nodes = {kRpcRecvOpName};
  if (std::find(jump_monad_output_nodes.begin(), jump_monad_output_nodes.end(), GetCNodeName(cnode)) !=
      jump_monad_output_nodes.end()) {
    return true;
  }
  return false;
}

void FlatParameterFinder::AddParameter(const ParameterPtr &param) {
  auto tensor = dyn_cast<tensor::Tensor>(param->default_param());
  if (tensor == nullptr) {
    return;
  }
  auto [chunk, offset] = tensor->GetChunkOffset();
  if (chunk != nullptr) {
    (void)param_to_flat_param_.emplace(param, FlatParamInfo{nullptr, chunk, offset});
    return;
  }
  if (tensor->shape_c().size() == 1) {
    (void)candidate_flat_params_.emplace(tensor->data_c(), param);
  }
}

void FlatParameterFinder::AddNodes(const std::vector<AnfNodePtr> &nodes) {
  for (auto &node : nodes) {
    auto param = dyn_cast<Parameter>(node);
    if (param != nullptr) {
      AddParameter(param);
    }
  }
}

void FlatParameterFinder::UpdateFlatParameters() {
  if (candidate_flat_params_.empty()) {
    return;
  }
  for (auto &entry : param_to_flat_param_) {
    auto &info = entry.second;
    if (info.flat_param == nullptr) {
      auto iter = candidate_flat_params_.find(info.chunk);
      if (iter != candidate_flat_params_.end()) {
        (void)flat_params_.emplace(iter->second);
        info.flat_param = iter->second;
      }
    }
  }
  candidate_flat_params_.clear();
}

std::pair<ParameterPtr, size_t> FlatParameterFinder::FindFlatParameter(const ParameterPtr &param) {
  UpdateFlatParameters();
  auto iter = param_to_flat_param_.find(param);
  if (iter == param_to_flat_param_.end()) {
    return {nullptr, 0};
  }
  auto &flat_param = iter->second.flat_param;
  if (flat_param == nullptr) {
    MS_LOG(WARNING) << "Find flat Parameter for " << param->ToString() << " failed";
    return {nullptr, 0};
  }
  return {flat_param, iter->second.offset};
}

const std::set<ParameterPtr> &FlatParameterFinder::GetFlatParameters() {
  UpdateFlatParameters();
  return flat_params_;
}
}  // namespace mindspore
