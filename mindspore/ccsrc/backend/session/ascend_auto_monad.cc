/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "backend/session/ascend_auto_monad.h"
#include <set>
#include <map>
#include <stack>
#include <vector>
#include <string>
#include <tuple>
#include <queue>
#include <utility>
#include <memory>
#include <algorithm>
#include "utils/ms_context.h"
#include "utils/ordered_map.h"
#include "base/core_ops.h"
#include "debug/anf_ir_dump.h"
#include "pipeline/jit/base.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "runtime/device/ascend/kernel_select_ascend.h"

namespace mindspore {
namespace session {
namespace {

// Pair of graph and its actual arguments.
using GraphArgPair = std::pair<KernelGraphPtr, std::vector<AnfNodePtr>>;

// We start label id from 0, and use 0xFFFFFFFF to indicate label not set.
constexpr uint32_t kNoLabel = 0xFFFFFFFF;

// Primitive attribute for argument link assign.
const char LINK[] = "link";

bool IsSaveGraph() {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  return context_ptr->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
}

void DumpAllGraphs(NotNull<KernelGraphPtr> kg, std::set<KernelGraphPtr> *memo) {
  if (memo->find(kg) != memo->end()) {
    return;
  }
  memo->insert(kg);
  std::string file_name = "ascend_auto_monad_" + std::to_string(kg->graph_id()) + ".ir";
  DumpIR(file_name, kg.get());
  for (auto &child : kg->child_graph_order()) {
    auto cg = child.lock();
    if (cg) {
      DumpAllGraphs(NOT_NULL(cg), memo);
    }
  }
}

void DumpGraphForDebug(NotNull<KernelGraphPtr> kg) {
  if (IsSaveGraph()) {
    std::set<KernelGraphPtr> memo;
    DumpAllGraphs(kg, &memo);
  }
}

void DumpExecuteOrder(NotNull<KernelGraphPtr> kg) {
  if (!IsSaveGraph()) {
    return;
  }
  std::string filename = "ascend_execute_order_" + std::to_string(kg->graph_id()) + ".dat";
  auto filepath = pipeline::GetSaveGraphsPathName(filename);
  char real_path[PATH_MAX] = {0};
#if defined(_WIN32) || defined(_WIN64)
  if (_fullpath(filepath, filename.c_str(), PATH_MAX) == nullptr) {
    MS_LOG(DEBUG) << "dir " << filename << " does not exit.";
  }
#else
  if (realpath(filepath.c_str(), real_path) == nullptr) {
    MS_LOG(DEBUG) << "Dir " << filepath << " does not exit.";
  }
#endif

  std::ofstream fout(real_path);
  if (!fout.is_open()) {
    MS_LOG(ERROR) << "Open file '" << real_path << "' failed!";
    return;
  }

  fout << "Execute order:\n";
  int index = 0;
  for (auto &cnode : kg->execution_order()) {
    MS_EXCEPTION_IF_NULL(cnode);
    if (IsPrimitiveCNode(cnode, prim::kPrimLabelSet)) {
      fout << "L" << AnfAlgo::GetNodeAttr<uint32_t>(cnode, kAttrLabelIndex) << ":\n";
    }
    fout << "  [" << index << "], " << cnode->DebugString();
    if (AnfAlgo::HasNodeAttr(kAttrLabelIndex, cnode)) {
      fout << " : L" << AnfAlgo::GetNodeAttr<uint32_t>(cnode, kAttrLabelIndex);
    }
    if (AnfAlgo::HasNodeAttr(kAttrLabelSwitchList, cnode)) {
      auto labels = AnfAlgo::GetNodeAttr<std::vector<uint32_t>>(cnode, kAttrLabelSwitchList);
      fout << " : ";
      for (size_t i = 0; i < labels.size(); ++i) {
        fout << ((i > 0) ? ", L" : "L") << labels[i];
      }
    }
    fout << '\n';
    index++;
  }
  fout.close();
}

// Return kNoLabel when label id attribute not set for the graph.
uint32_t GetGraphLabel(const KernelGraphPtr &kg) {
  auto value = kg->get_attr(kAttrLabelIndex);
  if (value == nullptr) {
    return kNoLabel;
  }
  return GetValue<uint32_t>(value);
}

struct CallBranch {
  KernelGraphPtr graph;
  std::vector<AnfNodePtr> args;
};

struct CallSite {
  // Call/Switch/SwitchLayer
  CNodePtr cnode;

  // The last monad before call.
  AnfNodePtr last_monad = nullptr;

  // Branch graph called.
  std::vector<CallBranch> callees;

  // Parameter for return value.
  AnfNodePtr out_param = nullptr;

  // Label id for return.
  uint32_t return_label = kNoLabel;

  // Label param to index map.
  std::map<AnfNodePtr, uint32_t> label_indexes;

  // True if this is a tail call.
  bool tail = false;
};

struct ReturnPoint {
  CallSite *call_site = nullptr;
};

struct CallInfo {
  std::vector<CallSite> call_sites;
  std::vector<ReturnPoint> return_points;
  AnfNodePtr label_param = nullptr;
};

//
// ParameterPool cache parameters by its abstract, so that we can reuse
// parameter with same abstract to store return values.
//
class ParameterPool {
 public:
  explicit ParameterPool(const KernelGraphPtr &top_graph) : top_graph_(top_graph) {}
  ~ParameterPool() = default;

  // Create or get a parameter from pool with the given abstract.
  AnfNodePtr GetParameter(const abstract::AbstractBasePtr &abs) {
    // Find parameter in pool by the given abstract.
    auto iter = std::find_if(paras_.begin(), paras_.end(), [&abs](auto &para) {
      auto para_abs = para->abstract();
      // Reuse output parameter with compatible abstract.
      return IsCompatible(abs, para_abs);
    });
    // Return the parameter if found.
    if (iter != paras_.end()) {
      return *iter;
    }
    // If parameter not found with the given abstract, create a new one.
    auto para = top_graph_->NewParameter(abs);
    auto out_para = top_graph_->TransTupleToMakeTuple(para);
    // This is required, so that device memory can be allocated for it.
    top_graph_->AddChildGraphResult(out_para);
    // Save new para to pool.
    paras_.push_back(out_para);
    return out_para;
  }

 protected:
  // Check if one abstract is compatible with another abstract.
  static bool IsCompatible(const abstract::AbstractBasePtr &a1, const abstract::AbstractBasePtr &a2) {
    if (a1 == nullptr || a2 == nullptr) {
      return false;
    }
    if (a1->isa<abstract::AbstractTensor>() && a2->isa<abstract::AbstractTensor>()) {
      // This make AbstractRef compatible with AbstractTensor.
      auto &t1 = static_cast<abstract::AbstractTensor &>(*a1);
      auto &t2 = static_cast<abstract::AbstractTensor &>(*a2);
      return t1 == t2;
    }
    return *a1 == *a2;
  }

 private:
  // The top graph.
  const KernelGraphPtr &top_graph_;

  // Cached parameters.
  std::vector<AnfNodePtr> paras_;
};

//
// Base class for context.
//
class BaseContext {
 public:
  void MarkVisited(const KernelGraphPtr &kg) { visited_graphs_.insert(kg); }

  bool IsVisited(const KernelGraphPtr &kg) const { return visited_graphs_.find(kg) != visited_graphs_.end(); }

  const std::set<KernelGraphPtr> &visited_graphs() const { return visited_graphs_; }

  void ClearVisited() { visited_graphs_.clear(); }

 private:
  std::set<KernelGraphPtr> visited_graphs_;
};

//
// AscendAutoMonadContext holds some shared states during auto-monad.
//
class AscendAutoMonadContext : public BaseContext {
 public:
  explicit AscendAutoMonadContext(const KernelGraphPtr &kg) : top_graph_(kg), param_pool_(kg) {}
  ~AscendAutoMonadContext() = default;

  // Label id start from 1, and increased by 1 for each new id.
  uint32_t NewLabel() { return label_id_++; }

  // Current label id, also the number of label ids we currently used.
  uint32_t CurrentLabel() const { return label_id_; }

  // Create a new parameter.
  // Output parameters are all created on top graph.
  AnfNodePtr CreateParameter(const AbstractBasePtr &abs) {
    auto para = top_graph_->NewParameter(abs);
    auto out_para = top_graph_->TransTupleToMakeTuple(para);
    // This is required, so that device memory can be allocated for it.
    top_graph_->AddChildGraphResult(out_para);
    return out_para;
  }

  // Get or create a temporary parameter for the given abstract.
  AnfNodePtr GetTempParameter(const AbstractBasePtr &abs) { return param_pool_.GetParameter(abs); }

  const KernelGraphPtr &TopGraph() const { return top_graph_; }

  // Map kernel_graph to its call info.
  OrderedMap<KernelGraphPtr, CallInfo> call_info_map;

 private:
  // The top graph.
  const KernelGraphPtr &top_graph_;

  // The parameter pool that cache parameters for return value.
  ParameterPool param_pool_;

  // Current label id.
  uint32_t label_id_ = 0;
};

//
// Call info finder finds graph call information.
//
class CallInfoFinder {
 public:
  static void Run(AscendAutoMonadContext *context) {
    CallInfoFinder finder(context->TopGraph(), context);
    finder.Run();
  }

 private:
  CallInfoFinder(const KernelGraphPtr &kg, AscendAutoMonadContext *context) : kernel_graph_(kg), context_(*context) {}
  ~CallInfoFinder() = default;

  void Run() {
    FindCallSites();
    FindCallReturns();
  }

  // Find all call sites.
  void FindCallSites() {
    auto call_info = CreateCallInfo();
    if (call_info == nullptr) {
      // Skip if call_info for this graph already existed.
      return;
    }
    // Update directly called sub-graphs.
    kernel_graph_->UpdateChildGraphOrder();
    // Find Call/Switch/SwitchLayer nodes, and make CallSites for them.
    AnfNodePtr last_monad = nullptr;
    auto nodes = TopoSort(kernel_graph_->output());
    for (auto &node : nodes) {
      MS_EXCEPTION_IF_NULL(node);
      if (HasAbstractUMonad(node)) {
        // Found a node with UMonad abstract, set it as the last monad.
        last_monad = node;
      } else if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimCall)) {
        MakeCallSite(node->cast<CNodePtr>(), last_monad, call_info);
      } else if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimSwitch) ||
                 AnfAlgo::CheckPrimitiveType(node, prim::kPrimSwitchLayer)) {
        MakeSwitchCallSite(node->cast<CNodePtr>(), last_monad, call_info);
      }
    }
    // Set the last call as tail call if it is the output node.
    // We don't set tail call for top graph because return is always required.
    if (kernel_graph_ != context_.TopGraph() && !call_info->call_sites.empty()) {
      auto real_output = GetRealNode(kernel_graph_->output());
      if (real_output == call_info->call_sites.back().cnode) {
        call_info->call_sites.back().tail = true;
      }
    }
    // Recursively find CallSites from sub-graphs.
    for (auto &call_site : call_info->call_sites) {
      for (auto &callee : call_site.callees) {
        CallInfoFinder finder(callee.graph, &context_);
        finder.FindCallSites();
      }
    }
  }

  // Find call-return pairs.
  void FindCallReturns() {
    for (auto &entry : context_.call_info_map) {
      auto &caller = entry.first;
      auto &call_info = entry.second;
      for (auto &call_site : call_info.call_sites) {
        for (auto &callee : call_site.callees) {
          MakeGraphLabel(callee.graph);
        }
        if (!call_site.tail) {
          SearchCallReturns(caller, &call_site);
        }
      }
    }
  }

  // Create entry label for the given graph if not set.
  void MakeGraphLabel(const KernelGraphPtr &kg) {
    auto label = GetGraphLabel(kg);
    if (label == kNoLabel) {
      // Allocate a new label id and save it to the graph.
      label = context_.NewLabel();
      kg->set_attr(kAttrLabelIndex, MakeValue(label));
    }
  }

  // Search return points for all non-tail calls.
  void SearchCallReturns(const KernelGraphPtr &caller, CallSite *call_site) {
    std::set<KernelGraphPtr> visited = {caller};
    std::queue<CallSite *> call_sites;
    call_sites.push(call_site);
    while (!call_sites.empty()) {
      auto site = call_sites.front();
      call_sites.pop();
      for (auto &callee : site->callees) {
        auto &kg = callee.graph;
        if (visited.find(kg) != visited.end()) {
          // Skip visited graphs.
          continue;
        }
        // Mark visited.
        visited.emplace(kg);
        // Check callee.
        auto &call_info = context_.call_info_map[kg];
        auto &sites = call_info.call_sites;
        if (!sites.empty() && sites.back().tail) {
          // Follow tail call.
          call_sites.push(&sites.back());
        } else {
          // Find a call-return relation.
          HandleCallReturn(caller, call_site, kg);
        }
      }
    }
  }

  // Handle a call-return relation.
  void HandleCallReturn(const KernelGraphPtr &caller, CallSite *call_site, const KernelGraphPtr &callee) {
    // Create a label for the return point.
    if (call_site->return_label == kNoLabel) {
      call_site->return_label = context_.NewLabel();
    }
    // Create a parameter for the return value.
    if (call_site->out_param == nullptr) {
      call_site->out_param = context_.CreateParameter(call_site->cnode->abstract());
    }
    // Add a return point for the callee graph.
    auto &call_info = context_.call_info_map[callee];
    auto &return_point = call_info.return_points.emplace_back();
    return_point.call_site = call_site;

    // Setup label index if there are multi return points.
    const auto n_return_points = call_info.return_points.size();
    if (n_return_points > 1) {
      if (n_return_points == 2) {
        // Create a parameter to store label index.
        const ShapeVector shape = {1};
        auto abs = std::make_shared<abstract::AbstractTensor>(kInt32, shape);
        call_info.label_param = context_.CreateParameter(abs);
        // Add label index for the first call site.
        call_info.return_points.front().call_site->label_indexes.emplace(call_info.label_param, 0);
      }
      // Add label index for the current call site.
      auto label_index = static_cast<uint32_t>(call_info.return_points.size() - 1);
      call_site->label_indexes.emplace(call_info.label_param, label_index);
    }
  }

  // Create a CallInfo for current kernel graph, return null if it is already existed.
  CallInfo *CreateCallInfo() {
    auto [iter, ok] = context_.call_info_map.add(kernel_graph_);
    if (!ok) {
      // CallInfo already existed.
      return nullptr;
    }
    return &(iter->second);
  }

  // Create CallSite for Call node.
  void MakeCallSite(const CNodePtr &cnode, const AnfNodePtr &last_monad, CallInfo *call_info) {
    auto &call_site = call_info->call_sites.emplace_back();
    call_site.cnode = cnode;
    call_site.last_monad = last_monad;
    call_site.callees.emplace_back(GetCallBranch(cnode));
  }

  // Create CallSite for Switch/SwitchLayer node.
  void MakeSwitchCallSite(const CNodePtr &cnode, const AnfNodePtr &last_monad, CallInfo *call_info) {
    auto &call_site = call_info->call_sites.emplace_back();
    call_site.cnode = cnode;
    call_site.last_monad = last_monad;
    call_site.callees = GetSwitchBranches(cnode);
  }

  CallBranch GetCallBranch(const CNodePtr &cnode) {
    auto input_graph = cnode->input(kCallKernelGraphIndex);
    MS_EXCEPTION_IF_NULL(input_graph);
    auto kg = GetValueNode<KernelGraphPtr>(input_graph);
    MS_EXCEPTION_IF_NULL(kg);
    constexpr size_t call_arg_index = 2;
    auto &inputs = cnode->inputs();
    std::vector<AnfNodePtr> args{inputs.begin() + call_arg_index, inputs.end()};
    return {.graph = kg, .args = std::move(args)};
  }

  std::vector<CallBranch> GetSwitchBranches(const CNodePtr &cnode) {
    constexpr size_t cond_start_index = 2;
    std::vector<CallBranch> branches;
    for (size_t index = cond_start_index; index < cnode->inputs().size(); ++index) {
      branches.emplace_back(GetSwitchBranch(cnode, index));
    }
    return branches;
  }

  CallBranch GetSwitchBranch(const CNodePtr &cnode, size_t index) {
    auto partial_cnode = dyn_cast<CNode>(cnode->input(index));
    if (partial_cnode == nullptr) {
      return {nullptr, {}};
    }
    auto &inputs = partial_cnode->inputs();
    if (!IsPrimitive(inputs.at(0), prim::kPrimPartial)) {
      MS_LOG(EXCEPTION) << "Invalid switch node: " << cnode->DebugString();
    }
    auto graph = GetValueNode<KernelGraphPtr>(inputs.at(1));
    constexpr size_t arg_index = 2;
    std::vector<AnfNodePtr> args{inputs.begin() + arg_index, inputs.end()};
    return {.graph = graph, .args = std::move(args)};
  }

  static AnfNodePtr GetRealNode(const AnfNodePtr &node) {
    if (!IsPrimitiveCNode(node, prim::kPrimDepend)) {
      return node;
    }
    return GetRealNode(node->cast<CNodePtr>()->input(1));
  }

 private:
  const KernelGraphPtr &kernel_graph_;
  AscendAutoMonadContext &context_;
};

//
// AscendAutoMonadConverter convert control flow to monad form
// for a kernel graph and its children graphs recursively.
//
class AscendAutoMonadConverter {
 public:
  static void Run(AscendAutoMonadContext *context) {
    for (auto &entry : context->call_info_map) {
      AscendAutoMonadConverter converter(entry.first, context, &entry.second);
      converter.Run();
    }
  }

 private:
  AscendAutoMonadConverter(const KernelGraphPtr &kg, AscendAutoMonadContext *context, CallInfo *call_info)
      : kernel_graph_(kg), context_(*context), call_info_(*call_info) {}
  ~AscendAutoMonadConverter() = default;

  void Run() {
    // Setup entry label if found.
    SetupEntryLabel();

    // Handle call sites.
    for (auto &call_site : call_info_.call_sites) {
      HandleCallSite(call_site);
    }
    // Handle return points.
    HandleReturnPoints();
    // Let output depend on monad.
    if (monad_) {
      MakeMonadDepend();
    }
  }

  void HandleCallSite(const CallSite &call_site) {
    // Update last_monad_.
    last_monad_ = call_site.last_monad;

    // The call/switch/switch_layer cnode.
    auto &cnode = call_site.cnode;

    // Get branches of the call_site.
    // for call, there is one branch;
    // for switch, the first one is true branch;
    // for switch_layer, the first one is 0 branch.
    auto &branches = call_site.callees;

    // Link arguments and find labels for branches.
    std::vector<KernelGraphPtr> graphes;
    std::vector<uint32_t> labels;
    graphes.reserve(branches.size());
    labels.reserve(branches.size());
    for (auto &[graph, args] : branches) {
      MS_EXCEPTION_IF_NULL(graph);
      auto linked_args = LinkArguments(args, graph);
      if (linked_args != nullptr) {
        monad_ = UpdateState(GetMonad(), linked_args);
      }
      graphes.push_back(graph);
      labels.push_back(GetGraphLabel(graph));
    }

    // Assign label indexes if required.
    AssignLabelIndexes(call_site);

    // For Switch, we reverse the graphes and labels, so that the false branch
    // is the first one, since for kernel LabelSwitch, false is the first branch.
    if (AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimSwitch)) {
      std::reverse(graphes.begin(), graphes.end());
      std::reverse(labels.begin(), labels.end());
    }

    // Create LabelGoto or LabelSwitch node.
    auto label_goto_switch = MakeLabelGotoSwitch(cnode, graphes, labels);

    // Setup return label and output if required.
    if (call_site.return_label != kNoLabel) {
      auto label_node = LabelSet(call_site.return_label);
      AnfNodePtr output = call_site.out_param;
      MS_EXCEPTION_IF_NULL(output);
      const bool is_single_call = call_site.label_indexes.empty();
      if (is_single_call) {
        // For single call, let output depend on the label node,
        // this ensures the return label is set before output is used.
        output = MakeDepend(output, label_node);
      } else {
        // For multi-return call, assign result from temp parameter to
        // output parameter, this prevent result be overwritten by next call.
        auto tmp_param = context_.GetTempParameter(output->abstract());
        output = AssignAll(output, tmp_param);
        monad_ = UpdateState(GetMonad(), output);
      }
      // Replace the the call/switch node with the output.
      ReplaceNode(cnode, output);
      return;
    }

    // If no return label required, it should be a tail call.
    if (!call_site.tail) {
      MS_LOG(EXCEPTION) << "Return label not set for non-tail call " << cnode->DebugString();
    }
    // For tail calls, replace origin call node with label_goto/label_switch.
    ReplaceNode(cnode, label_goto_switch);
    kernel_graph_->set_end_goto(label_goto_switch);
  }

  // Assign label indexes to label parameters for a call site.
  void AssignLabelIndexes(const CallSite &call_site) {
    for (auto &[label_param, label_index] : call_site.label_indexes) {
      auto index_value = GetIndexValueNode(label_index);
      auto assign = Assign(label_param, index_value);
      monad_ = UpdateState(GetMonad(), assign);
    }
  }

  // Create or reuse ValueNode for the index.
  ValueNodePtr GetIndexValueNode(uint32_t index) {
    auto iter = index_nodes_.find(index);
    if (iter != index_nodes_.end()) {
      // Reuse ValueNode for same index.
      return iter->second;
    }
    // Create a new ValueNode on top graph for the index.
    auto &top_graph = context_.TopGraph();
    std::vector<int64_t> data = {static_cast<int64_t>(index)};
    auto tensor = std::make_shared<tensor::Tensor>(data, kInt32);
    auto value_node = top_graph->NewValueNode(tensor->ToAbstract(), tensor);
    top_graph->AddValueNodeToGraph(value_node);
    index_nodes_.emplace(index, value_node);
    return value_node;
  }

  // Replace a node with new node in current kernel graph.
  // We also replace the arguments used for sub-graph calls.
  void ReplaceNode(const AnfNodePtr &old_node, const AnfNodePtr &new_node) {
    kernel_graph_->ReplaceNode(NOT_NULL(old_node), NOT_NULL(new_node));
    for (auto &call_site : call_info_.call_sites) {
      for (auto &callee : call_site.callees) {
        std::replace(callee.args.begin(), callee.args.end(), old_node, new_node);
      }
    }
  }

  // Make a label_goto or label_switch for a Call/Switch/SwitchLayer node.
  CNodePtr MakeLabelGotoSwitch(const CNodePtr &cnode, const std::vector<KernelGraphPtr> &graphes,
                               const std::vector<uint32_t> &labels) {
    // Create LabelGoto or LabelSwitch according the cnode type.
    const bool is_call = AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimCall);
    auto label_goto_switch = (is_call ? LabelGoto(labels.front()) : LabelSwitch(cnode->input(1), labels));

    // Set child graph attribute for the LabelGoto or LabelSwitch node.
    SetChildGrapAttr(label_goto_switch, graphes);

    // Mark the label_switch node is for 'switch_layer' if it is.
    if (AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimSwitchLayer)) {
      AnfAlgo::SetNodeAttr(kAttrSwitchLayer, prim::kValueOne, label_goto_switch);
    }
    return label_goto_switch;
  }

  //
  // Handle return points.
  // use label_goto for single return point;
  // use label_switch for multi return points.
  //
  void HandleReturnPoints() {
    auto &return_points = call_info_.return_points;
    // No return points.
    if (return_points.empty()) {
      return;
    }
    // Assign output according the return points.
    AssignOutput(return_points);
    // Single return point.
    if (return_points.size() == 1) {
      // Insert label_goto for return.
      auto &return_point = return_points.front();
      auto return_goto = LabelGoto(return_point.call_site->return_label);
      AnfAlgo::SetNodeAttr(kAttrReturn, prim::kValueOne, return_goto);
      kernel_graph_->set_end_goto(return_goto);
      return;
    }
    // Multi return points.
    std::vector<uint32_t> return_labels;
    return_labels.reserve(return_points.size());
    // Get return labels from return points.
    std::transform(return_points.begin(), return_points.end(), std::back_inserter(return_labels),
                   [](const ReturnPoint &return_point) { return return_point.call_site->return_label; });
    // Insert label_switch for multi return points.
    auto &label_param = call_info_.label_param;
    MS_EXCEPTION_IF_NULL(label_param);
    auto return_switch = LabelSwitch(label_param, return_labels);
    AnfAlgo::SetNodeAttr(kAttrReturn, prim::kValueOne, return_switch);
    kernel_graph_->set_end_goto(return_switch);
  }

  // Assign graph output to the output parameter.
  void AssignOutput(const std::vector<ReturnPoint> &return_points) {
    // For single call: we directly assign output to the output parameter of the call site;
    // For multi call: we assign output to a temp parameter, and let caller assign the
    // temp parameter to a output parameter after returned.
    auto call_site = return_points.front().call_site;
    MS_EXCEPTION_IF_NULL(call_site);
    const bool is_single_call = (return_points.size() == 1 && call_site->label_indexes.empty());
    AnfNodePtr out_param =
      (is_single_call ? call_site->out_param : context_.GetTempParameter(kernel_graph_->output()->abstract()));
    MS_EXCEPTION_IF_NULL(out_param);
    auto assign_output = AssignAll(out_param, kernel_graph_->output());
    monad_ = UpdateState(GetMonad(), assign_output);
  }

  //
  // Link actual arguments to graph's formal arguments.
  // for multi-args:
  //   r = Call(fg, arg1, arg2, u)
  // linked arguments:
  //   r1 = Assign(para1, arg1, c)
  //   r2 = Assign(para2, arg2, c)
  //   tuple = MakeTuple(r1, r2, u)
  //
  // for single-arg:
  //   r = Call(fg, arg)
  // linked arguments:
  //   r = Assign(para1, arg1, c)
  //
  // for empty-arg:
  //   r = Call(fg)
  // linked arguments return null.
  //
  AnfNodePtr LinkArguments(const std::vector<AnfNodePtr> &args, const KernelGraphPtr &graph) {
    auto &paras = graph->inputs();
    if (args.size() != paras.size()) {
      MS_LOG(EXCEPTION) << "Wrong arg number! " << graph->ToString() << " " << args.size() << " != " << paras.size();
    }
    // If no argument, return null.
    if (args.empty()) {
      return nullptr;
    }
    // Single argument.
    if (args.size() == 1) {
      auto &value = args.front();
      if (HasAbstractMonad(value) || paras.front() == value) {
        // No assign for single monad argument, return it.
        return value;
      }
      return AssignAll(paras.front(), value, true);
    }
    // Multi arguments.
    AnfNodePtrList tuple_inputs;
    tuple_inputs.reserve(args.size() + 1);
    tuple_inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    for (size_t i = 0; i < args.size(); ++i) {
      auto &value = args.at(i);
      if (HasAbstractMonad(value)) {
        // No assign for monad arguments.
        tuple_inputs.emplace_back(value);
        continue;
      }
      // Assign general arguments.
      auto &target = paras.at(i);
      if (target == value) {
        continue;
      }
      tuple_inputs.emplace_back(AssignAll(target, value, true));
    }
    return kernel_graph_->NewCNode(tuple_inputs);
  }

  // For some cnode, attributes may set to primitive instance, so we create a new prim instance for each cnode.
  AnfNodePtr NewPrimitive(const PrimitivePtr &prim) { return NewValueNode(std::make_shared<Primitive>(prim->name())); }

  AnfNodePtr GetLinkMonad() {
    if (last_monad_ != nullptr) {
      return last_monad_;
    }
    return GetMonad();
  }

  // Make a assign cnode.
  CNodePtr Assign(const AnfNodePtr &target, const AnfNodePtr &source, bool is_link = false) {
    auto monad = (is_link ? GetLinkMonad() : GetMonad());
    auto assign_prim = std::make_shared<Primitive>(prim::kPrimAssign->name());
    if (is_link) {
      // Mark this assign is to link real argument to formal argument.
      assign_prim->set_attr(LINK, prim::kValueOne);
    }
    auto assign = NewValueNode(assign_prim);
    auto cnode = kernel_graph_->NewCNode({assign, target, source, monad});
    cnode->set_abstract(target->abstract());
    return cnode;
  }

  // AissgnAll support tuple to tuple assign.
  AnfNodePtr AssignAll(const AnfNodePtr &target, const AnfNodePtr &source, bool is_link = false) {
    if (!AnfAlgo::CheckPrimitiveType(target, prim::kPrimMakeTuple)) {
      // Assign single value.
      return Assign(target, source, is_link);
    }
    // Assign tuple.
    std::vector<AnfNodePtr> targets = AnfAlgo::GetAllOutput(target, {prim::kPrimTupleGetItem});
    std::vector<AnfNodePtr> sources = AnfAlgo::GetAllOutput(source, {prim::kPrimTupleGetItem});
    if (targets.size() != sources.size()) {
      MS_LOG(EXCEPTION) << "Target size " << targets.size() << " != source size " << sources.size();
    }
    AnfNodePtrList tuple_inputs;
    tuple_inputs.reserve(targets.size() + 1);
    tuple_inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    for (size_t i = 0; i < targets.size(); ++i) {
      tuple_inputs.emplace_back(Assign(targets[i], sources[i], is_link));
    }
    return kernel_graph_->NewCNode(tuple_inputs);
  }

  // Insert UpdateState after input node.
  AnfNodePtr UpdateState(const AnfNodePtr &state, const AnfNodePtr &input) {
    auto update_state = NewValueNode(prim::kPrimUpdateState);
    auto update_state_cnode = kernel_graph_->NewCNode({update_state, state, input});
    update_state_cnode->set_abstract(state->abstract());
    return update_state_cnode;
  }

  //
  // Make entry label for current graph.
  // from:
  //   def sub_graph(x, y):
  //     return add(x, y)
  // to:
  //   def sub_graph(x, y, c):
  //     c = LabelSet(c) : entry_label
  //     return add(x, y)
  //
  void SetupEntryLabel() {
    auto entry_label = GetGraphLabel(kernel_graph_);
    if (entry_label != kNoLabel) {
      // Set entry label.
      auto label_node = LabelSet(entry_label);
      // Make start label the first one in execution order.
      kernel_graph_->set_start_label(label_node);
    }
  }

  // Make a Depend cnode.
  CNodePtr MakeDepend(const AnfNodePtr &origin, const AnfNodePtr &input) {
    auto depend = NewValueNode(prim::kPrimDepend);
    auto depend_cnode = kernel_graph_->NewCNode({depend, origin, input});
    depend_cnode->set_abstract(origin->abstract());
    return depend_cnode;
  }

  // Let output depend on monad.
  void MakeMonadDepend() {
    auto monad = GetMonad();
    auto origin_output = kernel_graph_->output();
    MS_EXCEPTION_IF_NULL(origin_output);
    if (origin_output != monad) {
      auto depend_cnode = MakeDepend(origin_output, monad);
      kernel_graph_->set_output(depend_cnode);
    }
  }

  // Gets the last monad node, we use a separated UMonad for control flow.
  AnfNodePtr &GetMonad() {
    if (monad_ == nullptr) {
      monad_ = GetMonadValue();
    }
    return monad_;
  }

  // Gets the monad const value node.
  AnfNodePtr &GetMonadValue() {
    if (monad_value_ == nullptr) {
      // We should create monad value node by kernel graph,
      // so that kernel_info is properly set for it.
      monad_value_ = kernel_graph_->NewValueNode(kUMonad->ToAbstract(), kUMonad);
    }
    return monad_value_;
  }

  // Make a LabelGoto node.
  CNodePtr LabelGoto(uint32_t label_id) {
    auto monad = GetMonad();
    auto label_goto = NewPrimitive(prim::kPrimLabelGoto);
    auto cnode = kernel_graph_->NewCNode({label_goto, monad});
    AnfAlgo::SetNodeAttr(kAttrLabelIndex, MakeValue(label_id), cnode);
    cnode->set_abstract(monad->abstract());
    monad_ = cnode;
    return cnode;
  }

  // Make a LabelSet node.
  CNodePtr LabelSet(uint32_t label_id) {
    auto monad = GetMonad();
    auto label_set = NewPrimitive(prim::kPrimLabelSet);
    auto cnode = kernel_graph_->NewCNode({label_set, monad});
    AnfAlgo::SetNodeAttr(kAttrLabelIndex, MakeValue(label_id), cnode);
    cnode->set_abstract(monad->abstract());
    monad_ = cnode;
    return cnode;
  }

  // Make a LabelSwitch node.
  CNodePtr LabelSwitch(const AnfNodePtr &cond, const std::vector<uint32_t> &labels) {
    auto monad = GetMonad();
    auto label_switch = NewPrimitive(prim::kPrimLabelSwitch);
    auto cnode = kernel_graph_->NewCNode({label_switch, cond, monad});
    auto label_list = MakeValue(labels);
    AnfAlgo::SetNodeAttr(kAttrLabelSwitchList, label_list, cnode);
    cnode->set_abstract(monad->abstract());
    monad_ = cnode;
    return cnode;
  }

  // Set child graph attribute for label_goto/label_switch node.
  void SetChildGrapAttr(const AnfNodePtr &node, const std::vector<KernelGraphPtr> &graphs) {
    AnfAlgo::SetNodeAttr(kAttrChildGraph, MakeValue(graphs), node);
  }

 private:
  const KernelGraphPtr &kernel_graph_;
  AscendAutoMonadContext &context_;

  // Call info for current kernel graph.
  CallInfo &call_info_;

  // The last monad for Call/Switch node.
  AnfNodePtr last_monad_;

  // The current control flow monad.
  AnfNodePtr monad_;

  // The control flow monad const value node.
  AnfNodePtr monad_value_;

  // Index value node cache for reuse.
  std::map<uint32_t, ValueNodePtr> index_nodes_;
};

constexpr size_t kAssignTargetIndex = 1;
constexpr size_t kAssignSourceIndex = 2;

class ExecuteOrderGenerator {
 public:
  class Context : public BaseContext {};
  ExecuteOrderGenerator(Context &context, const KernelGraphPtr &graph) : context_(context), graph_(graph) {}
  ~ExecuteOrderGenerator() = default;

  void Run() {
    GenerateExecuteOrder();
    EraseParameter();
    EraseLabel();
    UnfoldRepeatedLabels();
  }

 private:
  void GenerateGraphOrder(const KernelGraphPtr &graph) {
    ExecuteOrderGenerator generator(context_, graph);
    generator.GenerateExecuteOrder();
  }

  uint32_t FindMaxLabelId(const std::vector<CNodePtr> &nodes) {
    uint32_t max_label = 0;
    for (auto &node : nodes) {
      if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimLabelSet)) {
        auto label_id = AnfAlgo::GetNodeAttr<uint32_t>(node, kAttrLabelIndex);
        max_label = std::max(label_id, max_label);
      }
    }
    return max_label;
  }

  void HandleLabelSwitch(const AnfNodePtr &node, std::vector<uint32_t> *labels, std::vector<uint32_t> *switch_labels,
                         std::multimap<uint32_t, uint32_t> *labels_multimap) {
    bool is_new_labels = false;
    auto label_list = AnfAlgo::GetNodeAttr<std::vector<uint32_t>>(node, kAttrLabelSwitchList);
    std::vector<uint32_t> new_labels;
    new_labels.reserve(label_list.size());
    for (auto label_id : label_list) {
      auto iter = std::find_if(labels->begin(), labels->end(), [label_id](auto id) { return id == label_id; });
      // Use new label if find repeated label.
      if (iter == labels->end()) {
        new_labels.emplace_back(label_id);
        continue;
      }
      new_labels.emplace_back(++max_label_);
      labels_multimap->insert(std::pair<uint32_t, uint32_t>(*iter, max_label_));
      is_new_labels = true;
    }
    labels->insert(labels->end(), new_labels.begin(), new_labels.end());
    switch_labels->insert(switch_labels->end(), new_labels.begin(), new_labels.end());
    if (is_new_labels) {
      AnfAlgo::SetNodeAttr(kAttrLabelSwitchList, MakeValue(new_labels), node);
    }
  }

  void HandleLabelGoto(const AnfNodePtr &node, std::vector<uint32_t> *labels, std::vector<uint32_t> *switch_labels,
                       std::multimap<uint32_t, uint32_t> *labels_multimap) {
    auto label_id = AnfAlgo::GetNodeAttr<uint32_t>(node, kAttrLabelIndex);
    auto iter = std::find(switch_labels->begin(), switch_labels->end(), label_id);
    if (iter == switch_labels->end()) {
      labels->emplace_back(label_id);
      return;
    }
    AnfAlgo::SetNodeAttr(kAttrLabelIndex, MakeValue(++max_label_), node);
    labels_multimap->insert(std::pair<uint32_t, uint32_t>(*iter, max_label_));
    labels->emplace_back(max_label_);
  }

  // Unfold Repeated Labels, avoid same label in labelswitches.
  void UnfoldRepeatedLabels() {
    auto nodes = graph_->execution_order();
    std::vector<uint32_t> labels;
    std::vector<uint32_t> switch_labels;
    std::multimap<uint32_t, uint32_t> labels_multimap;
    max_label_ = FindMaxLabelId(nodes);
    for (auto &node : nodes) {
      if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimLabelSwitch)) {
        HandleLabelSwitch(node, &labels, &switch_labels, &labels_multimap);
        continue;
      }
      if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimLabelGoto)) {
        HandleLabelGoto(node, &labels, &switch_labels, &labels_multimap);
        continue;
      }
    }
    InsertLabelSet(&nodes, labels_multimap);
    graph_->set_label_num(max_label_ + 1);
    graph_->set_execution_order(nodes);
  }

  void InsertLabelSet(std::vector<CNodePtr> *nodes, const std::multimap<uint32_t, uint32_t> &labels_multimap) {
    for (auto labels : labels_multimap) {
      auto old_label = labels.first;
      auto new_label = labels.second;
      auto iter = std::find_if(nodes->begin(), nodes->end(), [old_label](auto node) {
        if (!AnfAlgo::CheckPrimitiveType(node, prim::kPrimLabelSet)) {
          return false;
        }
        auto label_id = AnfAlgo::GetNodeAttr<uint32_t>(node, kAttrLabelIndex);
        return label_id == old_label;
      });
      if (iter == nodes->end()) {
        MS_LOG(EXCEPTION) << "Not found labelset:" << old_label;
      }
      auto label_set = NewValueNode(std::make_shared<Primitive>(prim::kPrimLabelSet->name()));
      auto cnode = graph_->NewCNode({label_set});
      AnfAlgo::CopyNodeAttrs(*iter, cnode);
      AnfAlgo::SetNodeAttr(kAttrLabelIndex, MakeValue(new_label), cnode);
      auto monad = graph_->NewValueNode(kUMonad->ToAbstract(), kUMonad);
      cnode->set_abstract(monad->abstract());
      device::ascend::SelectKernelInfo(cnode);
      nodes->insert(iter, cnode);
    }
  }

  void AppendGraphOrder(std::vector<CNodePtr> *execution_order, const KernelGraphPtr &graph) {
    auto &order = graph->execution_order();
    execution_order->insert(execution_order->end(), order.begin(), order.end());
  }

  bool HasSubGraphs(const CNodePtr &cnode) { return (cnode && AnfAlgo::HasNodeAttr(kAttrChildGraph, cnode)); }

  std::vector<KernelGraphPtr> GetSubGraphs(const CNodePtr &cnode) {
    return AnfAlgo::GetNodeAttr<std::vector<KernelGraphPtr>>(cnode, kAttrChildGraph);
  }

  void EraseNodeFromExecOrder(const AnfNodePtr &node, const NotNull<std::vector<CNodePtr> *> exec_order) {
    MS_EXCEPTION_IF_NULL(node);
    auto exec_iter = std::find(exec_order->begin(), exec_order->end(), node);
    if (exec_iter == exec_order->end()) {
      MS_LOG(EXCEPTION) << "Cannot find " << node->DebugString() << " in exec order.";
    }
    exec_order->erase(exec_iter);
  }

  void GenerateExecuteOrder() {
    // Mark graph is visited.
    context_.MarkVisited(graph_);

    // Generate topo-sorted kernel cnodes list for this graph.
    graph_->SetExecOrderByDefault();

    std::vector<CNodePtr> execution_order;
    const auto &cnodes = graph_->execution_order();
    for (auto &cnode : cnodes) {
      // Push current node to execution order list.
      execution_order.push_back(cnode);
      // For cnode with sub-graphs, such as LabelSwitch, LabelGoto,
      // Generate execute order for these sub-graphs,
      // and then append them to current execution order list.
      if (HasSubGraphs(cnode)) {
        auto sub_graphs = GetSubGraphs(cnode);
        if (!AnfAlgo::HasNodeAttr(kAttrSwitchLayer, cnode)) {
          // For Switch, we use reversed order to generate sub-graph's execution order,
          // because the true branch of LabelSwitch is the second one, but
          // we want to make true branch ahead of false branch in the generated
          // execution order.
          std::reverse(sub_graphs.begin(), sub_graphs.end());
        }
        for (auto &sub_graph : sub_graphs) {
          if (context_.IsVisited(sub_graph)) {
            // Skip visited sub-graphs.
            continue;
          }
          GenerateGraphOrder(sub_graph);
          AppendGraphOrder(&execution_order, sub_graph);
        }
        // Clear ChildGraph attribute after execute order generated.
        AnfAlgo::EraseNodeAttr(kAttrChildGraph, cnode);
      }
    }
    // Save generated execution order into the graph.
    graph_->set_execution_order(std::move(execution_order));
  }

  std::set<CNodePtr> GetAllNodes(std::set<CNodePtr> *search_list) {
    const auto &all_graphs = context_.visited_graphs();
    std::set<CNodePtr> all_nodes;
    for (auto &graph : all_graphs) {
      auto out = graph->get_return();
      MS_EXCEPTION_IF_NULL(out);
      search_list->insert(out->cast<CNodePtr>());
      auto nodes = TopoSort(out);
      for (auto &node : nodes) {
        MS_EXCEPTION_IF_NULL(node);
        auto cnode = node->cast<CNodePtr>();
        if (cnode != nullptr) {
          all_nodes.insert(cnode);
        }
      }
    }
    return all_nodes;
  }

  static const AnfNodePtr &GetRealNode(const AnfNodePtr &input) {
    if (IsPrimitiveCNode(input, prim::kPrimLoad) || IsPrimitiveCNode(input, prim::kPrimDepend)) {
      return input->cast<CNodePtr>()->inputs().at(1);
    }
    return input;
  }

  void RemoveSameInputsAssigns(std::vector<CNodePtr> *exec_order) {
    for (auto iter = exec_order->begin(); iter != exec_order->end();) {
      auto &node = *iter;
      auto &inputs = node->inputs();
      if (IsPrimitiveCNode(node, prim::kPrimAssign) &&
          (inputs.at(kAssignTargetIndex) == GetRealNode(inputs.at(kAssignSourceIndex)))) {
        iter = exec_order->erase(iter);
      } else {
        ++iter;
      }
    }
  }

  // Erase redundant parameters and assign nodes.
  void EraseParameter() {
    // Copy out execution order list.
    auto exec_order = graph_->execution_order();
    std::set<CNodePtr> search_list(exec_order.begin(), exec_order.end());

    // Remove assigns that target and source are same.
    RemoveSameInputsAssigns(&exec_order);

    // Get all nodes and all graphs
    std::set<CNodePtr> all_nodes = GetAllNodes(&search_list);
    auto &all_graphs = context_.visited_graphs();

    // Count parameter write times by check all assign nodes.
    auto param_write_times = CountParameterAssigns(search_list);

    // Erase redundant assigns.
    for (auto iter = exec_order.begin(); iter != exec_order.end();) {
      auto &node = *iter;
      // We only try to erase argument link assign nodes,
      // other assign nodes are skipped.
      if (IsLinkAssign(node)) {
        auto &target = node->inputs().at(kAssignTargetIndex);
        MS_EXCEPTION_IF_NULL(target);
        auto para = param_write_times.find(target);
        if (para != param_write_times.end() && para->second == 1) {
          // If target only write once, replace target with source and erase assign node.
          auto &source = node->inputs().at(kAssignSourceIndex);
          auto kg = target->func_graph()->cast<KernelGraphPtr>();
          MS_EXCEPTION_IF_NULL(kg);
          kg->ReplaceNode(NOT_NULL(target), NOT_NULL(source));

          // replace parameter in graph input
          for (auto &g : all_graphs) {
            auto child_graph_inputs = g->MutableInputs();
            std::replace(child_graph_inputs->begin(), child_graph_inputs->end(), target, source);
            MS_LOG(DEBUG) << "Replace parameter " << target->DebugString() << " by " << source->DebugString()
                          << " in graph " << g->graph_id() << " inputs";
          }

          // replace parameter in node
          for (auto &iter_node : all_nodes) {
            for (size_t i = 0; i < iter_node->size(); ++i) {
              if (iter_node->input(i) == target) {
                MS_LOG(INFO) << "Replace " << iter_node->DebugString() << " input " << i << " by "
                             << source->DebugString();
                iter_node->set_input(i, source);
              }
            }
          }
          iter = exec_order.erase(iter);
          continue;
        }
      }
      // Go next node.
      ++iter;
    }
    // Set new execution order with redundant assign removed.
    graph_->set_execution_order(std::move(exec_order));
  }

  // Count parameter write times by check all assign nodes.
  std::map<AnfNodePtr, int> CountParameterAssigns(const std::set<CNodePtr> &search_list) {
    auto ref_map = graph_->GetRefMap();
    std::multimap<AnfNodePtr, std::tuple<size_t, AnfNodePtr, size_t>> ref_multimap;
    std::set<AnfNodePtr> root_inputs(graph_->inputs().begin(), graph_->inputs().end());
    std::transform(ref_map.begin(), ref_map.end(), std::inserter(ref_multimap, ref_multimap.end()),
                   [](const std::pair<std::pair<AnfNodePtr, size_t>, std::pair<AnfNodePtr, size_t>> &p)
                     -> std::pair<AnfNodePtr, std::tuple<size_t, AnfNodePtr, size_t>> {
                     return {p.first.first, {p.first.second, p.second.first, p.second.second}};
                   });
    auto validate_ref_parameter = [](AnfNodePtr node) -> AnfNodePtr {
      if (node->isa<CNode>() && AnfAlgo::CheckPrimitiveType(node, prim::KPrimTransData)) {
        auto cnode = node->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(cnode);
        auto first_input = cnode->input(kFirstDataInputIndex);
        MS_EXCEPTION_IF_NULL(first_input);
        return first_input;
      }
      return node;
    };

    // Find all graph input parameters.
    std::map<AnfNodePtr, int> param_write_times;
    const auto &all_graphs = context_.visited_graphs();
    for (const auto &graph : all_graphs) {
      for (auto &input : graph->inputs()) {
        if (input->isa<Parameter>()) {
          param_write_times.emplace(input, 0);
        }
      }
    }
    // Search all nodes for parameter write assigns.
    for (auto &node : search_list) {
      std::set<AnfNodePtr> refed_parameters;
      for (auto [iter, end] = ref_multimap.equal_range(node); iter != end; ++iter) {
        refed_parameters.insert(validate_ref_parameter(std::get<1>(iter->second)));
      }
      for (auto &in : node->inputs()) {
        auto visit_node = AnfAlgo::VisitKernelWithReturnType(in, 0).first;
        visit_node = validate_ref_parameter(visit_node);
        if (!visit_node->isa<Parameter>() || root_inputs.find(visit_node) != root_inputs.end()) {
          continue;
        }
        if (refed_parameters.find(visit_node) != refed_parameters.end()) {
          auto iter = param_write_times.find(visit_node);
          if (iter != param_write_times.end()) {
            // Found a parameter writer, count it.
            ++(iter->second);
          }
        }
      }
    }
    return param_write_times;
  }

  // Check if a node is an assign for argument link.
  bool IsLinkAssign(const AnfNodePtr &node) {
    auto cnode = dyn_cast<CNode>(node);
    if (cnode == nullptr) {
      return false;
    }
    auto prim = GetValueNode<PrimitivePtr>(cnode->inputs().at(0));
    if (!IsPrimitiveEquals(prim, prim::kPrimAssign)) {
      return false;
    }
    return prim->GetAttr(LINK) == prim::kValueOne;
  }

  // Erase LabelGoto and LabelSet
  void EraseLabel() {
    // Find used labels (as jump target).
    std::set<uint32_t> label_used;
    auto exec_order = graph_->execution_order();
    for (auto iter = exec_order.begin(); iter != exec_order.end();) {
      auto &node = *iter;
      if (IsPrimitiveCNode(node, prim::kPrimLabelSwitch)) {
        auto labels = AnfAlgo::GetNodeAttr<std::vector<uint32_t>>(node, kAttrLabelSwitchList);
        for (auto label : labels) {
          label_used.insert(label);
        }
      } else if (IsPrimitiveCNode(node, prim::kPrimLabelGoto)) {
        auto label = AnfAlgo::GetNodeAttr<uint32_t>(node, kAttrLabelIndex);
        auto next = std::next(iter);
        if (next != exec_order.end() && IsPrimitiveCNode(*next, prim::kPrimLabelSet)) {
          // The LabelGoto that jump to next node can be removed.
          auto next_label = AnfAlgo::GetNodeAttr<uint32_t>(*next, kAttrLabelIndex);
          if (next_label == label) {
            iter = exec_order.erase(iter);
            continue;
          }
        }
        label_used.insert(label);
      }
      ++iter;
    }
    // Erase unused LabelSet nodes.
    for (auto iter = exec_order.begin(); iter != exec_order.end();) {
      auto &node = *iter;
      if (IsPrimitiveCNode(node, prim::kPrimLabelSet)) {
        auto label = AnfAlgo::GetNodeAttr<uint32_t>(node, kAttrLabelIndex);
        if (label_used.find(label) == label_used.end()) {
          iter = exec_order.erase(iter);
          continue;
        }
      }
      ++iter;
    }
    graph_->set_execution_order(std::move(exec_order));
  }

  Context &context_;
  const KernelGraphPtr graph_;
  uint32_t max_label_ = 0;
};

}  // namespace

void AscendAutoMonad::Run() {
  MS_LOG(DEBUG) << "Ascend auto-monad start.";
  auto kg = kernel_graph_.get();
  AscendAutoMonadContext context(kg);
  CallInfoFinder::Run(&context);
  AscendAutoMonadConverter::Run(&context);
  kernel_graph_->set_label_num(context.CurrentLabel() + 1);
  MS_LOG(DEBUG) << "Ascend auto-monad finish.";
  DumpGraphForDebug(kernel_graph_);
}

void AscendAutoMonad::GenerateExecuteOrder() {
  MS_LOG(DEBUG) << "Ascend generate execute order start.";
  ExecuteOrderGenerator::Context context;
  ExecuteOrderGenerator generator(context, kernel_graph_.get());
  generator.Run();
  MS_LOG(DEBUG) << "Ascend generate execute order finish.";
  DumpExecuteOrder(kernel_graph_);
}
}  // namespace session
}  // namespace mindspore
