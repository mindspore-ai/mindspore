/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include <utility>
#include <memory>
#include <algorithm>
#include "utils/ms_context.h"
#include "base/core_ops.h"
#include "debug/anf_ir_dump.h"
#include "pipeline/jit/base.h"
#include "backend/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace session {
namespace {

// Pair of graph and its actual arguments.
using GraphArgPair = std::pair<KernelGraphPtr, std::vector<AnfNodePtr>>;

// We start label id from 1, and use 0 to indicate label not set.
constexpr uint32_t kNoLabel = 0;

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

using ParameterPoolPtr = std::shared_ptr<ParameterPool>;

class BaseContext {
 public:
  void MarkVisited(const KernelGraphPtr &kg) { visited_graphs_.insert(kg); }

  bool IsVisited(const KernelGraphPtr &kg) const { return visited_graphs_.find(kg) != visited_graphs_.end(); }

  const std::set<KernelGraphPtr> &visited_graphs() const { return visited_graphs_; }

 private:
  std::set<KernelGraphPtr> visited_graphs_;
};

//
// AscendAutoMonadContext holds some shared states during auto-moand.
//
class AscendAutoMonadContext : public BaseContext {
 public:
  explicit AscendAutoMonadContext(const KernelGraphPtr &kg) : top_graph_(kg) {}
  ~AscendAutoMonadContext() = default;

  // Label id start from 1, and increased by 1 for each new id.
  uint32_t NewLabel() { return label_id_++; }

  // Current label id, also the number of label ids we currently used.
  uint32_t CurrentLabel() const { return label_id_; }

  // Create a new parameter pool.
  ParameterPoolPtr NewParameterPool() { return std::make_shared<ParameterPool>(top_graph_); }

 private:
  // The top graph.
  const KernelGraphPtr &top_graph_;

  // Current label id.
  uint32_t label_id_ = 1;
};

//
// AscendAutoMonadConverter convert control flow to monad form
// for a kernel graph and its children graphs recursively.
//
class AscendAutoMonadConverter {
 public:
  AscendAutoMonadConverter(AscendAutoMonadContext *context, const KernelGraphPtr &kg)
      : context_(*context), kernel_graph_(kg) {}

  ~AscendAutoMonadConverter() = default;

  void Run() {
    // Skip if graph already visited.
    if (context_.IsVisited(kernel_graph_)) {
      return;
    }
    context_.MarkVisited(kernel_graph_);

    // Update directly called sub-graphs.
    kernel_graph_->UpdateChildGraphOrder();

    Prepare();

    // Setup entry label if needed.
    auto entry_label = GetGraphLabel(kernel_graph_);
    if (entry_label != kNoLabel) {
      SetupEntryLabel(entry_label);
    }

    // Handle call and switch nodes.
    HandleCallSwitch();

    // Let output depend on monad.
    if (monad_) {
      MakeMonadDepend();
    }
  }

 private:
  //
  // Prepare information for control flow processing.
  //
  void Prepare() {
    AnfNodePtr last_monad = nullptr;
    auto nodes = TopoSort(kernel_graph_->output());
    for (auto &node : nodes) {
      MS_EXCEPTION_IF_NULL(node);
      if (HasAbstractUMonad(node)) {
        // Found a node with UMonad abstract, set it as the last monad.
        last_monad = node;
      }
      auto cnode = node->cast<CNodePtr>();
      if (cnode == nullptr) {
        continue;
      }
      if (cnode->size() < 1) {
        MS_LOG(EXCEPTION) << "Invalid CNode: " << cnode->DebugString() << std::endl;
      }
      if (AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimCall) ||
          AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimSwitch)) {
        // Found call/switch node, set it as the tail call node.
        tail_call_node_ = cnode;
        call_switch_nodes_.emplace_back(cnode);
        monad_map_.emplace(cnode, last_monad);
      } else if (tail_call_node_ != nullptr && AnfAlgo::IsRealKernel(cnode)) {
        // Set no tail call if we found real kernel cnode after call/switch.
        tail_call_node_ = nullptr;
      }
    }
  }

  //
  // Handle call and switch node, return true if tail call found.
  //
  void HandleCallSwitch() {
    // Handle call switch nodes.
    for (auto &cnode : call_switch_nodes_) {
      if (AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimCall)) {
        HandleCall(cnode);
      } else if (AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimSwitch)) {
        HandleSwitch(cnode);
      } else {
        MS_LOG(EXCEPTION) << "Not a call/switch node: " << cnode->DebugString();
      }
    }
    // If no tail call, assign output value to output parameter,
    // and then goto the return label if set.
    if (tail_call_node_ == nullptr) {
      if (output_parameter_) {
        auto assign_output = AssignAll(output_parameter_, kernel_graph_->output());
        monad_ = UpdateState(GetMonad(), assign_output);
      }
      if (return_label_ != kNoLabel) {
        (void)LabelGoto(return_label_);
      }
    }
  }

  //
  // Convert call node:
  //     out = Call(graph, arg)
  // to:
  //     r = link_args(graph.para, arg, c)
  //     c = UpdateState(c, r)
  //     c = LabelGoto(c) : L1
  //
  void HandleCall(const CNodePtr &cnode) {
    // Update last_monad_.
    last_monad_ = monad_map_[cnode];

    // The callee graph.
    auto graph = GetCallGraph(cnode);
    MS_EXCEPTION_IF_NULL(graph);

    // Link arguments for the sub-graph.
    constexpr size_t call_arg_index = 2;
    auto &inputs = cnode->inputs();
    std::vector<AnfNodePtr> args(inputs.begin() + call_arg_index, inputs.end());
    auto linked_args = LinkArguments(args, graph);
    if (linked_args != nullptr) {
      monad_ = UpdateState(GetMonad(), linked_args);
    }

    // Goto sub-graph label.
    uint32_t graph_label = GetOrCreateGraphLabel(graph);
    auto goto_node = LabelGoto(graph_label);

    // Set child graph attribute, so that subsequence steps such
    // as 'select kernel' can handle sub graphs.
    SetChildGrapAttr(goto_node, {graph});

    // Setup return label if this is not a tail call.
    const bool is_tail_call = (cnode == tail_call_node_);
    const bool need_return = !is_tail_call;
    auto [para_pool, output_para, return_label] = MakeReturn(cnode, need_return);

    // Handle sub-graph recursively.
    HandleSubGraph(graph, para_pool, output_para, return_label);
  }

  //
  // Convert switch node:
  //     branch1 = Partial(graph1, arg)
  //     branch2 = Partial(graph2, arg)
  //     out = Switch(cond, branch1, branch2)
  // to:
  //     r = link_args(graph1, arg)
  //     c = UpdateState(c, r)
  //     r = link_args(graph2, arg)
  //     c = UpdateState(c, r)
  //     c = LabelSwitch(cond, c) : L1, L2
  //     c = LabelSet(c) : <return label>
  //
  void HandleSwitch(const CNodePtr &cnode) {
    // Update last_monad_.
    last_monad_ = monad_map_[cnode];

    // Get both branches of the switch, true branch first.
    auto branches = GetSwitchBranches(cnode);

    // Link arguments and generate labels for branches.
    std::vector<KernelGraphPtr> graphes;
    std::vector<uint32_t> labels;
    graphes.reserve(branches.size());
    labels.reserve(graphes.size());
    for (auto &[graph, args] : branches) {
      if (graph == nullptr) {
        MS_LOG(EXCEPTION) << "Invalid switch: " << cnode->DebugString();
      }
      auto linked_args = LinkArguments(args, graph);
      if (linked_args != nullptr) {
        monad_ = UpdateState(GetMonad(), linked_args);
      }
      graphes.push_back(graph);
      labels.push_back(GetOrCreateGraphLabel(graph));
    }

    // Since true/false branches is reversed in kernel LabelSwitch,
    // We reverse graphes and labels to make false branch first.
    std::reverse(graphes.begin(), graphes.end());
    std::reverse(labels.begin(), labels.end());

    // Add LabelSwith node.
    auto switch_node = LabelSwitch(cnode->input(1), labels);

    // Set child graph attribute for switch node.
    SetChildGrapAttr(switch_node, graphes);

    // Setup return label if required.
    const bool is_tail_call = (cnode == tail_call_node_);
    const bool need_return = (return_label_ == kNoLabel || !is_tail_call);
    auto [para_pool, output_para, return_label] = MakeReturn(cnode, need_return);

    // Handle sub-graphs recursively.
    for (auto &graph : graphes) {
      HandleSubGraph(graph, para_pool, output_para, return_label);
    }
  }

  ParameterPoolPtr GetParameterPool(bool is_last_call) {
    if (!is_last_call) {
      // There are multiple calls in this graph, use a new parameter pool
      // for each of them except the last one.
      return context_.NewParameterPool();
    }
    // For last call, try reuse parameter pool from the caller.
    if (para_pool_ == nullptr) {
      para_pool_ = context_.NewParameterPool();
    }
    return para_pool_;
  }

  // Make return part of a call for the LabelGoto/LabelSwitch node.
  std::tuple<ParameterPoolPtr, AnfNodePtr, uint32_t> MakeReturn(const CNodePtr &cnode, bool need_return) {
    // Find a parameter pool for output parameter.
    const bool is_last_call = (cnode == call_switch_nodes_.back());
    auto para_pool = GetParameterPool(is_last_call);

    // Prepare return label and output parameter.
    uint32_t return_label = return_label_;
    auto output_para = para_pool->GetParameter(cnode->abstract());
    auto output = output_para;

    // Setup return label if return is required.
    if (need_return) {
      // Set a new label at return point.
      return_label = context_.NewLabel();
      auto label_node = LabelSet(return_label);
      // Let output depend on the label node, this ensures the
      // return label is set before output is used.
      output = MakeDepend(output, label_node);
    }

    // Replace the the switch node with the output.
    kernel_graph_->ReplaceNode(NOT_NULL(cnode), NOT_NULL(output));
    return {para_pool, output_para, return_label};
  }

  // Handle sub-graphs recursively.
  void HandleSubGraph(const KernelGraphPtr &graph, const ParameterPoolPtr &para_pool, const AnfNodePtr &out_para,
                      uint32_t return_label) {
    AscendAutoMonadConverter converter(&context_, graph);
    converter.para_pool_ = para_pool;
    converter.output_parameter_ = out_para;
    converter.return_label_ = return_label;
    converter.Run();
  }

  KernelGraphPtr GetCallGraph(const CNodePtr &cnode) {
    auto input_graph = cnode->input(kCallKernelGraphIndex);
    MS_EXCEPTION_IF_NULL(input_graph);
    return GetValueNode<KernelGraphPtr>(input_graph);
  }

  GraphArgPair GetSwitchBranch(const CNodePtr &cnode, size_t index) {
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
    return {graph, {inputs.begin() + arg_index, inputs.end()}};
  }

  std::vector<GraphArgPair> GetSwitchBranches(const CNodePtr &cnode) {
    constexpr size_t true_index = 2;
    constexpr size_t false_index = 3;
    // True branch first, then false branch.
    return {GetSwitchBranch(cnode, true_index), GetSwitchBranch(cnode, false_index)};
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
      return Assign(paras.front(), value, true);
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
      tuple_inputs.emplace_back(Assign(target, value, true));
    }
    return kernel_graph_->NewCNode(tuple_inputs);
  }

  // For some cnode, attributes may set to primitive instance, so we create a new prim instance for each cnode.
  AnfNodePtr NewPrimitive(const PrimitivePtr &prim) { return NewValueNode(std::make_shared<Primitive>(prim->name())); }

  AnfNodePtr GetAssignMonad() {
    if (last_monad_ != nullptr) {
      return last_monad_;
    }
    return GetMonadValue();
  }

  // Make a assign cnode.
  CNodePtr Assign(const AnfNodePtr &target, const AnfNodePtr &source, bool is_link = false) {
    auto monad = GetAssignMonad();
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
  AnfNodePtr AssignAll(const AnfNodePtr &target, const AnfNodePtr &source) {
    if (!AnfAlgo::CheckPrimitiveType(target, prim::kPrimMakeTuple)) {
      // Assign single value.
      return Assign(target, source);
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
      tuple_inputs.emplace_back(Assign(targets[i], sources[i]));
    }
    return kernel_graph_->NewCNode(tuple_inputs);
  }

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
  void SetupEntryLabel(uint32_t entry_label) {
    // Set entry label.
    auto label_node = LabelSet(entry_label);
    // Make start label the first one in execution order.
    kernel_graph_->set_start_label(label_node);
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
    auto depend_cnode = MakeDepend(origin_output, monad);
    kernel_graph_->set_output(depend_cnode);
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
    kernel_graph_->set_end_goto(cnode);  // make 'goto' the last one in execute order.
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

  // Return kNoLabel when label id attribute not set for the graph.
  uint32_t GetGraphLabel(const KernelGraphPtr &kg) {
    auto value = kg->get_attr(kAttrLabelIndex);
    if (value == nullptr) {
      return kNoLabel;
    }
    return GetValue<uint32_t>(value);
  }

  // Get or create entry label for the given graph.
  uint32_t GetOrCreateGraphLabel(const KernelGraphPtr &kg) {
    auto label = GetGraphLabel(kg);
    if (label == kNoLabel) {
      // Allocate a new label id and save it to the graph.
      label = context_.NewLabel();
      kg->set_attr(kAttrLabelIndex, MakeValue(label));
    }
    return label;
  }

  void SetChildGrapAttr(const AnfNodePtr &node, const std::vector<KernelGraphPtr> &graphs) {
    AnfAlgo::SetNodeAttr(kAttrChildGraph, MakeValue(graphs), node);
  }

 private:
  AscendAutoMonadContext &context_;
  const KernelGraphPtr kernel_graph_;

  // Tail call node, null if not found.
  CNodePtr tail_call_node_;

  // Call/Switch nodes.
  std::vector<CNodePtr> call_switch_nodes_;

  // Call/Switch node to monad map.
  std::map<CNodePtr, AnfNodePtr> monad_map_;

  // The last monad for Call/Switch node.
  AnfNodePtr last_monad_;

  // The current control flow monad.
  AnfNodePtr monad_;

  // The control flow monad const value node.
  AnfNodePtr monad_value_;

  // Parameter to store the return value.
  AnfNodePtr output_parameter_;

  // Parameter pool for output parameter allocation.
  ParameterPoolPtr para_pool_;

  // The return label id.
  uint32_t return_label_ = kNoLabel;
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
  }

 private:
  void GenerateGraphOrder(const KernelGraphPtr &graph) {
    ExecuteOrderGenerator generator(context_, graph);
    generator.GenerateExecuteOrder();
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
    for (auto cnode : cnodes) {
      // Push current node to execution order list.
      execution_order.push_back(cnode);
      // For cnode with sub-graphs, such as LabelSwitch, LabelGoto,
      // Generate execute order for these sub-graphs,
      // and then append them to current execution order list.
      if (HasSubGraphs(cnode)) {
        // We use reversed order to generate sub-graph's execution order,
        // because the true branch of LabelSwitch is the second one, but
        // we want to make true branch ahead of false branch in the generated
        // execution order.
        auto sub_graphs = GetSubGraphs(cnode);
        for (auto iter = sub_graphs.rbegin(); iter != sub_graphs.rend(); iter++) {
          auto &sub_graph = *iter;
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

  static const AnfNodePtr &GetRealNode(const AnfNodePtr &input) {
    if (IsPrimitiveCNode(input, prim::kPrimLoad) || IsPrimitiveCNode(input, prim::kPrimDepend)) {
      return input->cast<CNodePtr>()->inputs().at(1);
    }
    return input;
  }

  // Erase redundant parameters and assign nodes.
  void EraseParameter() {
    // Copy out execution order list.
    auto exec_order = graph_->execution_order();

    // Remove assigns that target and source are same.
    for (auto iter = exec_order.begin(); iter != exec_order.end();) {
      auto &node = *iter;
      auto &inputs = node->inputs();
      if (IsPrimitiveCNode(node, prim::kPrimAssign) &&
          (inputs.at(kAssignTargetIndex) == GetRealNode(inputs.at(kAssignSourceIndex)))) {
        iter = exec_order.erase(iter);
      } else {
        ++iter;
      }
    }

    // Count parameter write times by check all assign nodes.
    auto param_write_times = CountParameterAssigns(exec_order);

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
  std::map<AnfNodePtr, int> CountParameterAssigns(const std::vector<CNodePtr> &all_nodes) {
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
    for (auto &node : all_nodes) {
      if (!IsPrimitiveCNode(node, prim::kPrimAssign)) {
        continue;
      }
      auto &target = node->inputs().at(kAssignTargetIndex);
      MS_EXCEPTION_IF_NULL(target);
      auto iter = param_write_times.find(target);
      if (iter != param_write_times.end()) {
        // Found a parameter writer, count it.
        ++(iter->second);
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
};

}  // namespace

void AscendAutoMonad::Run() {
  MS_LOG(DEBUG) << "Ascend auto-monad start.";
  AscendAutoMonadContext context(kernel_graph_.get());
  AscendAutoMonadConverter converter(&context, kernel_graph_.get());
  converter.Run();
  kernel_graph_->set_label_num(context.CurrentLabel());
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
