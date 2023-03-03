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
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"
#include <map>
#include <memory>
#include <sstream>
#include <utility>
#include <unordered_map>
#include <algorithm>
#include "mindspore/core/ops/core_ops.h"
#include "utils/anf_utils.h"
#include "utils/ms_context.h"
#include "backend/common/graph_kernel/model/op_node.h"
#include "backend/common/graph_kernel/model/graph_builder.h"
#include "runtime/hardware/device_context_manager.h"

namespace mindspore::graphkernel {
std::string GkUtils::ExtractGraphKernelName(const AnfNodePtrList &nodes, const std::string &prefix,
                                            const std::string &postfix) {
  std::stringstream name;
  if (!prefix.empty()) {
    name << prefix << "_";
  }
  for (const auto &node : nodes) {
    if (AnfUtils::IsGraphKernel(node)) {
      auto fg_flag_val = GetCNodeFuncGraph(node)->get_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL);
      name << GetValue<std::string>(fg_flag_val) << "_";
    } else if (node->isa<CNode>() && AnfUtils::IsRealKernel(node)) {
      name << GetCNodePrimitive(node)->name() << "_";
    }
  }
  if (!postfix.empty()) {
    name << postfix;
  }
  return name.str();
}

AnfNodePtrList GkUtils::SpreadTuples(const AnfNodePtrList &nodes, size_t begin_index) {
  AnfNodePtrList result;
  for (size_t i = begin_index; i < nodes.size(); i++) {
    if (IsPrimitiveCNode(nodes[i], prim::kPrimMakeTuple)) {
      auto mt = nodes[i]->cast<CNodePtr>();
      // recursively spread all inner tuples.
      auto mt_inputs = SpreadTuples(mt->inputs(), 1);
      (void)result.insert(result.cend(), mt_inputs.cbegin(), mt_inputs.cend());
    } else {
      result.push_back(nodes[i]);
    }
  }
  return result;
}

std::vector<PrimitivePtr> GkUtils::GetValidOps(const std::vector<OpWithLevel> &ops_with_level, unsigned int level,
                                               const std::vector<std::string> &enable_ops_only,
                                               const std::vector<std::string> &enable_ops,
                                               const std::vector<std::string> &disable_ops) {
  std::vector<PrimitivePtr> ops;
  auto new_prim = [](const std::string &name) { return std::make_shared<Primitive>(name); };
  if (!enable_ops_only.empty()) {
    (void)std::transform(enable_ops_only.begin(), enable_ops_only.end(), std::back_inserter(ops), new_prim);
    return ops;
  }
  auto target = Callback::Instance()->GetTargetFromContext();
  for (const auto &[op_target, op_level, op] : ops_with_level) {
    if (op_target == kAllTarget || op_target == target) {
      if (level >= op_level) {
        (void)ops.emplace_back(op);
      }
    }
  }
  if (!enable_ops.empty()) {
    (void)std::transform(enable_ops.begin(), enable_ops.end(), std::back_inserter(ops), new_prim);
  }
  if (!disable_ops.empty()) {
    auto iter = std::remove_if(ops.begin(), ops.end(), [&disable_ops](const PrimitivePtr &p) {
      return std::find(disable_ops.begin(), disable_ops.end(), p->name()) != disable_ops.end();
    });
    (void)ops.erase(iter, ops.cend());
  }
  return ops;
}

std::vector<PrimitivePtr> GkUtils::FilterExcludedOps(const std::vector<PrimitivePtr> &ops) {
#ifndef MSLITE_ENABLE_GRAPH_KERNEL
  if (Callback::Instance()->GetTargetFromContext() != kGPUDevice) {
    return ops;
  }
  std::vector<PrimitivePtr> dst_ops;
  const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {kGPUDevice, MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  MS_EXCEPTION_IF_NULL(device_context);
  auto deprecated_ptr = device_context->GetDeprecatedInterface();
  MS_EXCEPTION_IF_NULL(deprecated_ptr);
  auto major_compute_capability = deprecated_ptr->GetGPUCapabilityMajor();
  std::unordered_map<std::string, int> limited_capacity_ops = {
    {prim::kPrimConv2D->name(), 7}, {prim::kPrimMatMul->name(), 7}, {prim::kPrimBatchMatMul->name(), 7}};
  std::vector<std::string> final_filter_ops;
  for (auto op : ops) {
    if (limited_capacity_ops.find(op->name()) != limited_capacity_ops.end() &&
        limited_capacity_ops[op->name()] != major_compute_capability) {
      final_filter_ops.emplace_back(op->name());
    } else {
      dst_ops.emplace_back(op);
    }
  }
  // Give hint for excluded src_ops.
  static bool give_hint = false;
  if (!give_hint && final_filter_ops.size() > 0) {
    give_hint = true;
    for (size_t i = 0; i < final_filter_ops.size(); ++i) {
      MS_LOG(WARNING) << "For op : " << final_filter_ops[i]
                      << " can not be enabled in GraphKernel because the current device's computing capacity is "
                      << major_compute_capability << ", which is != " << limited_capacity_ops[final_filter_ops[i]];
    }
  }
  return dst_ops;
#else
  return ops;
#endif
}

bool GkUtils::IsKeepBasicNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto prim = GetCNodePrimitive(node);
  auto target = Callback::Instance()->GetTargetFromContext();
  if (prim == nullptr) {
    return false;
  }
  // Heterogeneous computing is not support yet
  // so if node's primitive_target is inconsistent with target from context
  // the node cannot be added to the cluster list.
  if (prim->HasAttr("primitive_target") && GetValue<std::string>(prim->GetAttr("primitive_target")) != target) {
    return true;
  }
  // dynamic shape nodes is not supported yet.
  // the "skip" is used by inplace node.
  // the kAttrIsInternalOutputNopNode is used by internal output of KernelGraph.
  const std::vector<std::string> exclude_bool_attrs = {kAttrInputIsDynamicShape, kAttrOutputIsDynamicShape, "skip",
                                                       kAttrIsInternalOutputNopNode};
  if (std::any_of(exclude_bool_attrs.cbegin(), exclude_bool_attrs.cend(), [&prim](const std::string &attr_name) {
        return prim->HasAttr(attr_name) && GetValue<bool>(prim->GetAttr(attr_name));
      })) {
    return true;
  }

  // If node contain attribute in contagious_attrs, it have to keep basic no matter what the value is.
  const std::vector<std::string> contagious_attrs = {"inplace_group", "inplace_algo", "inplace_output_index",
                                                     "aggregate", "aggregate_input_index"};
  if (std::any_of(contagious_attrs.cbegin(), contagious_attrs.cend(),
                  [&prim](const std::string &attr_name) -> bool { return prim->HasAttr(attr_name); })) {
    return true;
  }
  return false;
}

CNodePtr GkUtils::NewRealCNode(const std::vector<AnfNodePtr> &inputs, const FuncGraphPtr &func_graph,
                               const std::vector<inner::NodeBase> &out_info_list, const CallbackPtr &cb) {
  auto cnode = func_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(cnode);

  if (out_info_list.size() == 0) {
    MS_LOG(EXCEPTION) << "CNode must have output!";
  }

  // Setup abstract.
  AbstractBasePtrList abs_list;
  (void)std::transform(
    out_info_list.begin(), out_info_list.end(), std::back_inserter(abs_list), [](const inner::NodeBase &out_info) {
      auto abs_tensor = std::make_shared<abstract::AbstractTensor>(TypeIdToType(out_info.type), out_info.shape);
      return abs_tensor;
    });
  if (abs_list.size() == 1) {
    cnode->set_abstract(abs_list[0]);
  } else {
    cnode->set_abstract(std::make_shared<abstract::AbstractTuple>(abs_list));
  }

  // Setup kernel build info.
  cb->SetBasicNodeKernelInfo(cnode, out_info_list);
  func_graph->AddNode(cnode);
  return cnode;
}

FuncGraphPtr GkUtils::LiteGraph2AnfGraph(const inner::LiteGraphPtr &lite_graph, const CallbackPtr &cb) {
  auto func_graph = std::make_shared<FuncGraph>();
  std::map<inner::NodePtr, AnfNodePtr> node_map;
  for (const auto &inp : lite_graph->inputs()) {
    auto param = func_graph->add_parameter();
    node_map[inp] = param;
    param->set_abstract(std::make_shared<abstract::AbstractTensor>(TypeIdToType(inp->type), inp->shape));
    cb->SetBasicNodeKernelInfo(param, {{inp->shape, inp->type, inp->format}});
  }
  // Create CNodes.
  for (const auto &op_node : lite_graph->GetOrderedNodes()) {
    if (op_node->NodeType() != inner::NType::Primitive) {
      MS_LOG(EXCEPTION) << "Node " << op_node->debug_name() << " should be a Primitive node";
    }
    auto op = std::static_pointer_cast<inner::PrimOp>(op_node);
    auto primitive = std::make_shared<Primitive>(op->op(), op->attrs());
    auto prim = GetOpsPrim(primitive->name());
    if (prim != nullptr) {
      (void)primitive->AddAttr(kAttrInputNames, prim->GetAttr(kAttrInputNames));
      (void)primitive->AddAttr(kAttrOutputNames, prim->GetAttr(kAttrOutputNames));
    }
    AnfNodePtrList inputs = {NewValueNode(primitive)};
    (void)std::transform(op->inputs().begin(), op->inputs().end(), std::back_inserter(inputs),
                         [&node_map, &cb](const inner::NodePtr &inp) -> AnfNodePtr {
                           const auto iter = node_map.find(inp);
                           if (iter != node_map.end()) {
                             return iter->second;
                           } else {
                             if (inp->NodeType() != inner::NType::Value) {
                               MS_LOG(EXCEPTION) << "Node " << inp->debug_name() << " should be a Value node";
                             }
                             auto inp_value = inp->As<inner::ConstTensorNode>()->data();
                             auto value_node = NewValueNode(inp_value);
                             value_node->set_abstract(inp_value->ToAbstract());
                             cb->SetBasicNodeKernelInfo(value_node, {{inp->shape, inp->type, inp->format}});
                             return value_node;
                           }
                         });
    auto output_info_list = op->outputs();
    if (output_info_list.empty()) {
      (void)output_info_list.emplace_back(static_cast<inner::NodeBase>(*op));
    }
    auto cnode = NewRealCNode(inputs, func_graph, output_info_list, cb);
    MS_EXCEPTION_IF_NULL(cnode);
    node_map[op_node] = cnode;
  }
  if (lite_graph->GetOutputs().empty()) {
    MS_LOG(EXCEPTION) << "The output of LiteGraph " << lite_graph->name() << " is empty.";
  } else if (lite_graph->GetOutputs().size() == 1) {
    func_graph->set_output(node_map[lite_graph->GetOutputs()[0]]);
  } else {
    AnfNodePtrList mt_inputs;
    AbstractBasePtrList out_abs_list;
    (void)std::transform(lite_graph->GetOutputs().begin(), lite_graph->GetOutputs().end(),
                         std::back_inserter(mt_inputs), [&node_map, &out_abs_list](const inner::NodePtr &out) {
                           auto out_node = node_map[out];
                           MS_EXCEPTION_IF_NULL(out_node);
                           (void)out_abs_list.emplace_back(out_node->abstract());
                           return out_node;
                         });
    auto mt = func_graph->NewCNode(prim::kPrimMakeTuple, mt_inputs);
    mt->set_abstract(std::make_shared<abstract::AbstractTuple>(out_abs_list));
    cb->SetEmptyKernelInfo(mt);
    func_graph->AddNode(mt);
    func_graph->set_output(mt);
  }
  return func_graph;
}

inner::LiteGraphPtr GkUtils::AnfGraph2LiteGraph(const FuncGraphPtr &func_graph,
                                                HashMap<inner::NodePtr, AnfNodePtr> *op_node_map) {
  std::string name = "Default";
  if (func_graph->has_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL)) {
    name = GetValue<std::string>(func_graph->get_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL));
  }
  inner::GraphBuilder gb(name);
  std::map<AnfNodePtr, inner::NodePtr> node_map;
  auto todos = TopoSort(func_graph->output());
  const auto &params = func_graph->parameters();
  auto cb = Callback::Instance();
  auto ExtractBuildInfo = [&cb](const AnfNodePtr &node) {
    auto shape = cb->GetOutputShape(node, 0);
    auto type = cb->GetOutputType(node, 0);
    auto format = cb->GetOutputFormat(node, 0);
    return inner::NodeBase({shape, type, format});
  };
  // set inputs
  for (auto &p : params) {
    node_map[p] = gb.Parameter(ExtractBuildInfo(p));
  }
  // set ops
  for (auto node : todos) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      continue;
    }
    if (IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
      break;
    }
    auto prim = GetCNodePrimitive(cnode);
    MS_EXCEPTION_IF_NULL(prim);
    inner::NodePtrList inputs;
    (void)std::transform(cnode->inputs().begin() + 1, cnode->inputs().end(), std::back_inserter(inputs),
                         [&node_map, &gb](const AnfNodePtr &no) {
                           const auto iter = node_map.find(no);
                           if (iter != node_map.end()) {
                             return iter->second;
                           } else {
                             auto tensor = GetValueNode<tensor::TensorPtr>(no);
                             MS_EXCEPTION_IF_NULL(tensor);
                             return gb.Value(tensor);
                           }
                         });
    auto op = gb.Op(AnfUtils::GetCNodeName(node), ExtractBuildInfo(node), inputs, prim->attrs());
    node_map[node] = op;
    if (op_node_map != nullptr) {
      (*op_node_map)[op] = node;
    }
  }
  // set outputs
  auto output_node = func_graph->output();
  if (IsPrimitiveCNode(output_node, prim::kPrimMakeTuple)) {
    inner::NodePtrList outputs;
    auto mt = output_node->cast<CNodePtr>();
    (void)std::transform(mt->inputs().begin() + 1, mt->inputs().end(), std::back_inserter(outputs),
                         [&node_map](const AnfNodePtr &no) { return node_map[no]; });
    gb.SetOutputs(std::move(outputs));
  } else {
    gb.SetOutputs({node_map[output_node]});
  }
  return gb.Get();
}

FuncGraphManagerPtr GkUtils::GetFuncGraphManager(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  if (manager == nullptr) {
    manager = Manage(func_graph, true);
    func_graph->set_manager(manager);
  }
  return manager;
}

void GkUtils::UpdateFuncGraphManager(const FuncGraphManagerPtr &mng, const FuncGraphPtr &func_graph) {
  mng->RemoveRoots();
  mng->KeepRoots({func_graph});
}

PrimitivePtr GkUtils::GetOpsPrim(const std::string &name) {
  const auto &op_primc_fns = ops::OpPrimCRegister::GetInstance().GetPrimCMap();
  auto const iter = op_primc_fns.find(name);
  if (iter == op_primc_fns.end()) {
    return nullptr;
  }
  return iter->second();
}

void GkUtils::GetValidKernelNodes(const FuncGraphPtr &func_graph, AnfNodePtrList *node_list, AnfNodePtrList *input_list,
                                  AnfNodePtrList *output_list) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node_list);
  AnfNodePtrList todos = TopoSort(func_graph->output());
  (void)std::copy_if(todos.cbegin(), todos.cend(), std::back_inserter(*node_list), AnfUtils::IsRealCNodeKernel);

  if (input_list != nullptr) {
    const auto &parameters = func_graph->parameters();
    (void)input_list->insert(input_list->cend(), parameters.cbegin(), parameters.cend());
  }
  if (output_list != nullptr) {
    if (IsPrimitiveCNode(todos.back(), prim::kPrimMakeTuple)) {
      auto fg_output = todos.back()->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(fg_output);
      (void)output_list->insert(output_list->cend(), fg_output->inputs().cbegin() + 1, fg_output->inputs().cend());
    } else {
      (void)output_list->emplace_back(func_graph->output());
    }
  }
}

int64_t GkUtils::GetChannelInConvFormat(const std::string &format_string) {
  constexpr size_t nchwc_len = 5;
  if (format_string.size() <= nchwc_len || format_string.find("NCHW") != 0) {
    MS_LOG(EXCEPTION) << "Format must be NCHWnc, but got [" << format_string << "]";
  }
  constexpr size_t n_pos = 4;
  auto channel = format_string.substr(n_pos, format_string.size() - nchwc_len);
  return std::stol(channel);
}
}  // namespace mindspore::graphkernel
