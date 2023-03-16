/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/hardware/ascend_auto_monad.h"
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
#include "mindspore/core/ops/core_ops.h"
#include "include/common/debug/anf_ir_dump.h"
#include "pipeline/jit/base.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/device/ascend/hal/device/kernel_select_ascend.h"

namespace mindspore::session {
namespace {
// We start label id from 0, and use 0xFFFFFFFF to indicate label not set.
constexpr uint32_t kNoLabel = 0xFFFFFFFF;

// We start input index from 2 for AssignOp, as for inputs[2] is input, inputs[1] is output;
constexpr size_t kInputIndex = 2;

// Primitive attribute for argument link assign.
const char LINK[] = "link";

// Attribute to indicate that the node should not be eliminated.
// Used to keep argument Assign nodes for recursive graphs.
const char KEEP[] = "keep";

// Attribute to indicate that this is an assign for output.
const char OUTPUT[] = "output";

// Attribute to indicate that the node is last node in an iteration.
const char ITEREND[] = "PROFILING_ITER_END";

const auto kSingleOutput = 1;
const auto kFirstOutput = 0;
constexpr size_t kFirstIndex = 1;

#ifdef ENABLE_DUMP_IR
bool IsSaveGraph() {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  return context->CanDump(kIntroductory);
}

void DumpAllGraphs(NotNull<KernelGraphPtr> kg, std::set<KernelGraphPtr> *memo) {
  MS_EXCEPTION_IF_NULL(memo);
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

void DumpGraphForDebug(const NotNull<KernelGraphPtr> &kg) {
  if (IsSaveGraph()) {
    std::set<KernelGraphPtr> memo;
    DumpAllGraphs(kg, &memo);
  }
}
#endif

#ifndef ENABLE_SECURITY
void DumpExecuteOrder(const NotNull<KernelGraphPtr> &kg) {
  if (!IsSaveGraph()) {
    return;
  }
  std::string filename = "ascend_execute_order_" + std::to_string(kg->graph_id()) + ".dat";
  auto filepath = GetSaveGraphsPathName(filename);
  if (filepath.size() >= PATH_MAX) {
    MS_LOG(ERROR) << "File path: " << filepath << " is too long.";
    return;
  }
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
      fout << "L" << common::AnfAlgo::GetNodeAttr<uint32_t>(cnode, kAttrLabelIndex) << ":\n";
    }
    fout << "  [" << index << "], " << cnode->DebugString();
    if (common::AnfAlgo::HasNodeAttr(kAttrLabelIndex, cnode)) {
      fout << " : L" << common::AnfAlgo::GetNodeAttr<uint32_t>(cnode, kAttrLabelIndex);
    }
    if (common::AnfAlgo::HasNodeAttr(kAttrLabelSwitchList, cnode)) {
      auto labels = common::AnfAlgo::GetNodeAttr<std::vector<uint32_t>>(cnode, kAttrLabelSwitchList);
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
#endif

KernelGraphPtr GetValueNodeKernelGraph(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto value_node = node->cast<ValueNodePtr>();
  if (value_node == nullptr) {
    return nullptr;
  }
  auto value = value_node->value();
  if (value == nullptr) {
    return nullptr;
  }
  auto kernel_graph = value->cast<KernelGraphPtr>();
  return kernel_graph;
}

// Return kNoLabel when label id attribute not set for the graph.
uint32_t GetGraphLabel(const KernelGraphPtr &kg) {
  auto value = kg->get_attr(kAttrLabelIndex);
  if (value == nullptr) {
    return kNoLabel;
  }
  return GetValue<uint32_t>(value);
}

bool CheckCallInline(const CNodePtr &cnode) {
  if (!common::AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimCall)) {
    return false;
  }
  auto call_graph = cnode->input(kFirstIndex);
  auto sub_kernel_graph = GetValueNodeKernelGraph(call_graph);
  return sub_kernel_graph->need_inline();
}

bool IsCompatible(const abstract::AbstractBasePtr &a1, const abstract::AbstractBasePtr &a2);

bool CheckAbstractTupleIsCompatible(const abstract::AbstractBasePtr &a1, const abstract::AbstractBasePtr &a2) {
  auto &a1_tuple = dynamic_cast<abstract::AbstractTuple &>(*a1);
  auto &a2_tuple = dynamic_cast<abstract::AbstractTuple &>(*a2);
  auto &a1_elements = a1_tuple.elements();
  auto &a2_elements = a2_tuple.elements();
  if (a1_elements.size() != a2_elements.size()) {
    return false;
  }
  for (size_t i = 0; i < a1_elements.size(); i++) {
    MS_EXCEPTION_IF_NULL(a1_elements[i]);
    MS_EXCEPTION_IF_NULL(a2_elements[i]);
    if (!IsCompatible(a1_elements[i], a2_elements[i])) {
      return false;
    }
  }
  return true;
}

bool CheckTensorAndScalar(const abstract::AbstractBasePtr &a1, const abstract::AbstractBasePtr &a2) {
  MS_EXCEPTION_IF_NULL(a1);
  MS_EXCEPTION_IF_NULL(a2);
  if (a1->isa<abstract::AbstractTensor>() && a2->isa<abstract::AbstractScalar>()) {
    auto a1_element = dynamic_cast<abstract::AbstractUndetermined &>(*a1).element();
    if (IsCompatible(a1_element, a2)) {
      return true;
    }
  }
  return false;
}

// Check if one abstract is compatible with another abstract.
bool IsCompatible(const abstract::AbstractBasePtr &a1, const abstract::AbstractBasePtr &a2) {
  if (a1 == nullptr || a2 == nullptr) {
    return false;
  }
  if (a1 == a2) {
    return true;
  }
  // Check AbstractTuple.
  if (a1->isa<abstract::AbstractTuple>() && a2->isa<abstract::AbstractTuple>()) {
    return CheckAbstractTupleIsCompatible(a1, a2);
  }
  // Consider the following two cases as compatible：
  // a1: AbstractScalar(Type: Bool, Value: ValueAny, Shape: NoShape)
  // a2: AbstractTensor(shape: (), element: AbstractScalar(Type: Bool, Value: ValueAny, Shape: NoShape), value:...)
  if (CheckTensorAndScalar(a1, a2) || CheckTensorAndScalar(a2, a1)) {
    return true;
  }
  // Check AbstractTensor and AbstractRefTensor.
  auto type1 = a1->BuildType();
  auto type2 = a2->BuildType();
  if (type1 != type2 && *type1 != *type2) {
    return false;
  }
  auto shape1 = a1->BuildShape();
  auto shape2 = a2->BuildShape();
  if (shape1 == shape2) {
    return true;
  }
  if (shape1->isa<abstract::Shape>() && shape2->isa<abstract::Shape>()) {
    const auto &shape1_vec = shape1->cast<abstract::ShapePtr>()->shape();
    const auto &shape2_vec = shape2->cast<abstract::ShapePtr>()->shape();
    if ((shape1_vec == ShapeVector({1}) && shape2_vec.empty()) ||
        (shape1_vec.empty() && shape2_vec == ShapeVector({1}))) {
      return true;
    }
  }
  return *shape1 == *shape2;
}

struct CallBranch {
  KernelGraphPtr graph;
  std::vector<AnfNodePtr> args;
};

struct CallSite {
  // Call/Switch/SwitchLayer
  CNodePtr cnode;

  // CNode after transferring to LabelGoto/LabelSwitch/LabelSet.
  CNodePtr conversion_cnode;

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

  // True if this is a recursive call.
  bool recursive = false;

  // True if this is a tail call.
  bool tail = false;

  // True if this call is a disable tail-opt call.
  bool disable_tail = false;
};

struct ReturnPoint {
  CallSite *call_site = nullptr;
};

struct CallInfo {
  // Call sites in current graph.
  std::vector<CallSite> call_sites;

  // Return points of current graph.
  std::vector<ReturnPoint> return_points;

  // Parameter to store label index, if there are
  // multi return points, this should be set.
  AnfNodePtr label_param = nullptr;

  // Return monad.
  AnfNodePtr return_monad_ = nullptr;

  // True if current graph is involved with recursive calls.
  bool recursive = false;
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
    MS_EXCEPTION_IF_NULL(abs);
    // Find parameter in pool by the given abstract.
    auto iter = std::find_if(paras_.begin(), paras_.end(), [&abs](auto &para) {
      MS_EXCEPTION_IF_NULL(para);
      auto para_abs = para->abstract();
      // Reuse output parameter with compatible abstract.
      return IsCompatible(abs, para_abs);
    });
    // Return the parameter if found.
    if (iter != paras_.end()) {
      return *iter;
    }
    // If parameter not found with the given abstract, create a new one.
    MS_EXCEPTION_IF_NULL(top_graph_);
    auto para = top_graph_->NewParameter(abs);
    auto out_para = top_graph_->TransTupleToMakeTuple(para);
    // This is required, so that device memory can be allocated for it.
    top_graph_->AddChildGraphResult(out_para);
    // Save new para to pool.
    paras_.push_back(out_para);
    return out_para;
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
  void MarkVisited(const KernelGraphPtr &kg) { (void)visited_graphs_.insert(kg); }

  [[nodiscard]] bool IsVisited(const KernelGraphPtr &kg) const {
    return visited_graphs_.find(kg) != visited_graphs_.end();
  }

  const std::set<KernelGraphPtr> &visited_graphs() { return visited_graphs_; }

  void ClearVisited() { visited_graphs_.clear(); }

  virtual ~BaseContext() = default;

 private:
  std::set<KernelGraphPtr> visited_graphs_;
};

//
// AscendAutoMonadContext holds some shared states during auto-monad.
//
class AscendAutoMonadContext : public BaseContext {
 public:
  explicit AscendAutoMonadContext(const KernelGraphPtr &kg) : top_graph_(kg), param_pool_(kg) {}
  ~AscendAutoMonadContext() override = default;

  // Label id start from 1, and increased by 1 for each new id.
  uint32_t NewLabel() { return label_id_++; }

  // Current label id, also the number of label ids we currently used.
  [[nodiscard]] uint32_t CurrentLabel() const { return label_id_; }

  // Create a new parameter.
  // Output parameters are all created on top graph.
  AnfNodePtr CreateParameter(const AbstractBasePtr &abs) {
    MS_EXCEPTION_IF_NULL(abs);
    auto para = top_graph_->NewParameter(abs);
    auto out_para = top_graph_->TransTupleToMakeTuple(para);
    // This is required, so that device memory can be allocated for it.
    top_graph_->AddChildGraphResult(out_para);
    return out_para;
  }

  // Get or create a temporary parameter for the given abstract.
  AnfNodePtr GetTempParameter(const AbstractBasePtr &abs) { return param_pool_.GetParameter(abs); }

  [[nodiscard]] const KernelGraphPtr &TopGraph() const { return top_graph_; }

  // Has already created an stack.
  [[nodiscard]] bool HasInitedStack() const { return inited_stack_; }

  // Set flag to indicate whether has already created an stack or not.
  void SetInitedStack(bool flag) { inited_stack_ = flag; }

  // The graphs has recursion.
  [[nodiscard]] bool HasRecursiveCall() const { return has_recursive_call_; }
  // The graphs has subgraph multi-call.
  [[nodiscard]] bool HasSubgraphMultiCall() const { return has_subgraph_multicall_; }
  // set flag to indicate whether has recursion.
  void SetRecursiveCall(bool flag) { has_recursive_call_ = flag; }
  // set flag to indicate whether has multi-call.
  void SetSubGraphMultiCall(bool flag) { has_subgraph_multicall_ = flag; }

  // Map kernel_graph to its call info.
  OrderedMap<KernelGraphPtr, CallInfo> call_info_map;

 private:
  // The top graph.
  const KernelGraphPtr &top_graph_;

  // The parameter pool that cache parameters for return value.
  ParameterPool param_pool_;

  // Current label id.
  uint32_t label_id_ = 0;

  // Create an stack for multi-call and non-tail recursion.
  bool inited_stack_ = false;
  // The graphs has recursion or not.
  bool has_recursive_call_ = false;
  // The graphs has subgraph multi-call or not.
  bool has_subgraph_multicall_ = false;
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
    FindRecursiveCalls();
    DisableTailCalls();
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
    MS_EXCEPTION_IF_NULL(kernel_graph_);
    kernel_graph_->UpdateChildGraphOrder();
    // Find Call/Switch/SwitchLayer nodes, and make CallSites for them.
    AnfNodePtr last_monad = nullptr;
    auto nodes = TopoSort(kernel_graph_->output());
    for (auto &node : nodes) {
      MS_EXCEPTION_IF_NULL(node);
      if (HasAbstractUMonad(node)) {
        // Found a node with UMonad abstract, set it as the last monad.
        last_monad = node;
        call_info->return_monad_ = last_monad;
      } else if (common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimCall)) {
        MakeCallSite(node->cast<CNodePtr>(), last_monad, call_info);
        call_info->return_monad_ = nullptr;
      } else if (common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimSwitch) ||
                 common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimSwitchLayer)) {
        MakeSwitchCallSite(node->cast<CNodePtr>(), last_monad, call_info);
        call_info->return_monad_ = nullptr;
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

  // Find recursive non-tail calls.
  void FindRecursiveCalls() {
    for (auto &[caller, call_info] : context_.call_info_map) {
      for (auto &call_site : call_info.call_sites) {
        if (!call_site.tail) {
          SearchRecursiveCall(caller, &call_site);
        }
      }
    }
  }

  // Disable tail call optimization for recursive call graphs.
  void DisableTailCalls() {
    for (auto &entry : context_.call_info_map) {
      auto &call_info = entry.second;
      if (call_info.recursive && !call_info.call_sites.empty()) {
        call_info.call_sites.back().tail = false;
        call_info.call_sites.back().disable_tail = true;
      }
    }
  }

  // Find call-return pairs.
  void FindCallReturns() {
    for (auto &[caller, call_info] : context_.call_info_map) {
      if (caller->need_inline() && (call_info.recursive || call_info.call_sites.size() != 0)) {
        MS_LOG(INFO) << "Do not inline cell reuse because it has sub-graph call, graph id: " << caller->graph_id();
        caller->set_need_inline(false);
      }
    }
    for (auto &[caller, call_info] : context_.call_info_map) {
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
    MS_EXCEPTION_IF_NULL(kg);
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
      MS_EXCEPTION_IF_NULL(site);
      for (auto &callee : site->callees) {
        auto &kg = callee.graph;
        if (visited.find(kg) != visited.end()) {
          // Skip visited graphs.
          continue;
        }
        // Mark visited.
        (void)visited.emplace(kg);
        // Check callee.
        auto &call_info = context_.call_info_map[kg];
        auto &sites = call_info.call_sites;
        if (!sites.empty() && sites.back().tail) {
          // Follow tail call.
          call_sites.push(&sites.back());
        } else {
          // Find a call-return relation.
          HandleCallReturn(call_site, kg);
        }
      }
    }
  }

  struct SearchRecursiveContext {
    const KernelGraphPtr &start_caller;
    CallSite *start_site;
    std::set<KernelGraphPtr> visited;
    std::vector<KernelGraphPtr> call_path;
  };

  // Search recursive call from a call-site.
  void SearchRecursiveCall(const KernelGraphPtr &start_caller, CallSite *start_site) {
    SearchRecursiveContext context{.start_caller = start_caller, .start_site = start_site};
    DoSearchRecursiveCall(start_caller, *start_site, &context);
  }

  void DoSearchRecursiveCall(const KernelGraphPtr &graph, const CallSite &call_site, SearchRecursiveContext *ctx) {
    MS_EXCEPTION_IF_NULL(ctx);
    // Record call path.
    ctx->call_path.push_back(graph);
    // Handle callee graphs.
    for (auto &callee : call_site.callees) {
      auto &sub_graph = callee.graph;
      if (sub_graph == ctx->start_caller) {
        // Find a recursive call path.
        for (auto &g : ctx->call_path) {
          // Mark recursive for all graphs in call path.
          context_.call_info_map[g].recursive = true;
        }
        // Mark recursive for the start call-site.
        MS_EXCEPTION_IF_NULL(ctx->start_site);
        ctx->start_site->recursive = true;
        continue;
      }
      if (ctx->visited.find(sub_graph) != ctx->visited.end()) {
        // Skip visited graphs.
        continue;
      }
      // Mark visited.
      (void)ctx->visited.emplace(sub_graph);
      // Check call sites in the sub-graph.
      auto &call_info = context_.call_info_map[sub_graph];
      auto &sites = call_info.call_sites;
      for (auto &site : sites) {
        if (!site.callees.empty()) {
          DoSearchRecursiveCall(sub_graph, site, ctx);
        }
      }
    }
    // Don't forget this.
    ctx->call_path.pop_back();
  }

  // Handle a call-return relation.
  void HandleCallReturn(CallSite *call_site, const KernelGraphPtr &callee) {
    MS_EXCEPTION_IF_NULL(call_site);
    MS_EXCEPTION_IF_NULL(callee);
    // Create a label for the return point.
    if (call_site->return_label == kNoLabel) {
      call_site->return_label = context_.NewLabel();
    }
    MS_EXCEPTION_IF_NULL(call_site->cnode);
    MS_EXCEPTION_IF_NULL(callee->output());
    if (!IsCompatible(call_site->cnode->abstract(), callee->output()->abstract())) {
      MS_LOG(EXCEPTION) << "call_site node: " << call_site->cnode->DebugString() << " has different abstract() with "
                        << callee->ToString() << " output(), [ " << call_site->cnode->abstract()->ToString()
                        << " != " << callee->output()->abstract()->ToString() << " ],"
                        << "Do not support this situation, pls check if the graghs are correct.";
    }

    // Create a parameter for the return value.
    if (call_site->out_param == nullptr && !CheckCallInline(call_site->cnode)) {
      call_site->out_param = context_.CreateParameter(call_site->cnode->abstract());
    }
    // Add a return point for the callee graph.
    auto &call_info = context_.call_info_map[callee];
    auto &return_point = call_info.return_points.emplace_back();
    return_point.call_site = call_site;

    // Setup label index if there are multi return points.
    const auto n_return_points = call_info.return_points.size();
    const size_t return_point_sizes = 2;
    if (n_return_points > 1 && !CheckCallInline(call_site->cnode)) {
      if (n_return_points == return_point_sizes) {
        // Create a parameter to store label index.
        const ShapeVector shape = {1};
        auto abs = std::make_shared<abstract::AbstractTensor>(kInt32, shape);
        call_info.label_param = context_.CreateParameter(abs);
        // Add label index for the first call site.
        (void)call_info.return_points.front().call_site->label_indexes.emplace(call_info.label_param, 0);
        // Judge the last call_site whether is loop, set recursive attr if yes.
        if (!call_info.call_sites.empty() && call_info.call_sites.back().disable_tail) {
          SearchRecursiveCall(callee, &call_info.call_sites.back());
        }
      }
      // Add label index for the current call site.
      auto label_index = static_cast<uint32_t>(call_info.return_points.size() - 1);
      (void)call_site->label_indexes.emplace(call_info.label_param, label_index);
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
  static void MakeCallSite(const CNodePtr &cnode, const AnfNodePtr &last_monad, CallInfo *call_info) {
    auto &call_site = call_info->call_sites.emplace_back();
    call_site.cnode = cnode;
    call_site.last_monad = last_monad;
    (void)call_site.callees.emplace_back(GetCallBranch(cnode));
  }

  // Create CallSite for Switch/SwitchLayer node.
  void MakeSwitchCallSite(const CNodePtr &cnode, const AnfNodePtr &last_monad, CallInfo *call_info) const {
    auto &call_site = call_info->call_sites.emplace_back();
    call_site.cnode = cnode;
    call_site.last_monad = last_monad;
    call_site.callees = GetSwitchBranches(cnode);
  }

  static CallBranch GetCallBranch(const CNodePtr &cnode) {
    auto input_graph = cnode->input(kPartialGraphIndex);
    MS_EXCEPTION_IF_NULL(input_graph);
    auto kg = GetValueNode<KernelGraphPtr>(input_graph);
    MS_EXCEPTION_IF_NULL(kg);
    constexpr int64_t call_arg_index = 2;
    auto &inputs = cnode->inputs();
    std::vector<AnfNodePtr> args{inputs.begin() + call_arg_index, inputs.end()};
    return {.graph = kg, .args = std::move(args)};
  }

  [[nodiscard]] std::vector<CallBranch> GetSwitchBranches(const CNodePtr &cnode) const {
    constexpr size_t cond_start_index = 2;
    std::vector<CallBranch> branches;
    for (size_t index = cond_start_index; index < cnode->inputs().size(); ++index) {
      (void)branches.emplace_back(GetSwitchBranch(cnode, index));
    }
    return branches;
  }

  static CallBranch GetSwitchBranch(const CNodePtr &cnode, size_t index) {
    auto partial_cnode = dyn_cast<CNode>(cnode->input(index));
    if (partial_cnode == nullptr) {
      return {nullptr, {}};
    }
    auto &inputs = partial_cnode->inputs();
    if (!IsPrimitive(inputs.at(0), prim::kPrimPartial)) {
      MS_LOG(EXCEPTION) << "Invalid switch node: " << cnode->DebugString();
    }
    auto graph = GetValueNode<KernelGraphPtr>(inputs.at(1));
    constexpr int64_t arg_index = 2;
    std::vector<AnfNodePtr> args{inputs.begin() + arg_index, inputs.end()};
    return {.graph = graph, .args = std::move(args)};
  }

  static AnfNodePtr GetRealNode(const AnfNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    if (!IsPrimitiveCNode(node, prim::kPrimDepend)) {
      return node;
    }
    MS_EXCEPTION_IF_NULL(node->cast<CNodePtr>());
    return GetRealNode(node->cast<CNodePtr>()->input(1));
  }

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
    MS_EXCEPTION_IF_NULL(context);
    for (auto &entry : context->call_info_map) {
      AscendAutoMonadConverter converter(entry.first, context, &entry.second);
      converter.Run();
    }
    const auto &top_graph = context->TopGraph();
    SetIterEndAttrForTopGraph(context, top_graph);
  }

 private:
  AscendAutoMonadConverter(const KernelGraphPtr &kg, AscendAutoMonadContext *context, CallInfo *call_info)
      : kernel_graph_(kg),
        context_(*context),
        call_info_(*call_info),
        name_index_(0),
        need_stackops_(call_info->recursive) {}
  ~AscendAutoMonadConverter() = default;

  void Run() {
    // need inline
    if (kernel_graph_->need_inline()) {
      return;
    }
    // Create an stack
    InitStack();
    // Setup entry label if found.
    SetupEntryLabel();

    // Handle call sites.
    for (auto &call_site : call_info_.call_sites) {
      HandleCallSite(&call_site);
    }
    // Handle return points.
    HandleReturnPoints();
    // Let output depend on monad.
    if (monad_) {
      MakeMonadDepend();
    }
    // Handle recursive call.
    MS_EXCEPTION_IF_NULL(kernel_graph_);
    kernel_graph_->SetExecOrderByDefault();
    if (call_info_.recursive) {
      const auto &nodes = kernel_graph_->execution_order();
      common::AnfAlgo::SetNodeAttr(kAttrRecursiveStart, prim::kValueOne, *nodes.begin());
      common::AnfAlgo::SetNodeAttr(kAttrRecursiveEnd, prim::kValueOne, *nodes.rbegin());
    }
    for (auto &call_site : call_info_.call_sites) {
      if (need_stackops_ && call_site.recursive) {
        MS_EXCEPTION_IF_NULL(call_site.cnode);
        MS_LOG(INFO) << "graph:" << kernel_graph_->ToString() << ", loop call_site:" << call_site.cnode->DebugString();
        InsertStackOps(call_site);
      }
    }
  }

  // Set iteration end points for Profiling.
  static void SetIterEndAttrForTopGraph(AscendAutoMonadContext *context, const KernelGraphPtr &kg) {
    MS_EXCEPTION_IF_NULL(kg);
    kg->SetExecOrderByDefault();
    auto &nodes = kg->execution_order();
    auto end_iter = nodes.rend();
    std::set<KernelGraphPtr> memo;
    (void)memo.insert(kg);
    auto call_info = context->call_info_map[kg];
    if (call_info.call_sites.empty()) {
      SetIterEndAttr(context, kg, false);
      return;
    } else {
      const auto &end_node = call_info.call_sites.back().cnode;
      end_iter = std::find(nodes.rbegin(), nodes.rend(), end_node);
    }
    for (auto iter = nodes.rbegin(); iter != end_iter; ++iter) {
      if (!AnfUtils::IsRealCNodeKernel(*iter)) {
        continue;
      }
      if (common::AnfAlgo::CheckPrimitiveType(*iter, prim::kPrimLabelSet)) {
        const auto &last_call_site = context->call_info_map[kg].call_sites.back();
        for (auto &branch : last_call_site.callees) {
          if (memo.find(branch.graph) != memo.end()) {
            continue;
          }
          FindProfilingEndPoints(context, branch.graph, &memo);
        }
        break;
      }
      common::AnfAlgo::SetNodeAttr(ITEREND, prim::kValueOne, *iter);
      MS_EXCEPTION_IF_NULL(*iter);
      MS_LOG(INFO) << "Set profiling iter-end points: " << (*iter)->DebugString();
      return;
    }
  }

  // Set Attr to the iter-end points.
  static void SetIterEndAttr(AscendAutoMonadContext *context, const KernelGraphPtr &kg, bool has_call_site) {
    MS_EXCEPTION_IF_NULL(kg);
    kg->SetExecOrderByDefault();
    auto &nodes = kg->execution_order();
    auto end_iter = nodes.rend();
    if (has_call_site) {
      const auto &end_node = context->call_info_map[kg].call_sites.back().cnode;
      end_iter = std::find(nodes.rbegin(), nodes.rend(), end_node);
    }
    for (auto iter = nodes.rbegin(); iter != end_iter; ++iter) {
      if (!AnfUtils::IsRealCNodeKernel(*iter)) {
        continue;
      }
      if (common::AnfAlgo::CheckPrimitiveType(*iter, prim::kPrimLabelGoto) &&
          common::AnfAlgo::HasNodeAttr(kAttrReturn, *iter)) {
        continue;
      }
      if (common::AnfAlgo::CheckPrimitiveType(*iter, prim::kPrimLabelGoto) ||
          common::AnfAlgo::CheckPrimitiveType(*iter, prim::kPrimLabelSwitch) ||
          common::AnfAlgo::CheckPrimitiveType(*iter, prim::kPrimLabelSet)) {
        MS_LOG(INFO) << "This node is Labelxxxx, do not found iter end.";
        break;
      }
      common::AnfAlgo::SetNodeAttr(ITEREND, prim::kValueOne, *iter);
      MS_EXCEPTION_IF_NULL(*iter);
      MS_LOG(INFO) << "Set profiling iter-end points: " << (*iter)->DebugString();
      return;
    }
    MS_LOG(INFO) << "Do not find iter_end point.";
  }

  // Find all iteration end points recursively.
  static void FindProfilingEndPoints(AscendAutoMonadContext *context, const KernelGraphPtr &kg,
                                     std::set<KernelGraphPtr> *memo) {
    MS_EXCEPTION_IF_NULL(memo);
    MS_EXCEPTION_IF_NULL(context);
    (void)memo->insert(kg);
    auto call_info = context->call_info_map[kg];
    // 1. find the last call site; if no call site, goto step 3.
    // 2. Judge the call site whether is tail call or not.
    // 3. if yes, recursively find call site in subgraph; if no, find the last TBE node and set extra attr.
    if (!call_info.call_sites.empty()) {
      const auto &last_call_site = call_info.call_sites.back();
      if (last_call_site.tail) {
        for (auto &branch : last_call_site.callees) {
          if (memo->find(branch.graph) != memo->end()) {
            continue;
          }
          FindProfilingEndPoints(context, branch.graph, memo);
        }
      } else {
        SetIterEndAttr(context, kg, true);
      }
    } else {
      SetIterEndAttr(context, kg, false);
    }
  }

  // Create a Stack for StackOps if needed.
  void InitStack() {
    if (!context_.HasInitedStack() && need_stackops_) {
      auto top_graph = context_.TopGraph();
      MS_EXCEPTION_IF_NULL(top_graph);
      auto exec_order = top_graph->execution_order();
      auto stack_init = StackInit(top_graph);
      AnfAlgo::KeepOrder(top_graph, stack_init, *exec_order.begin());
      auto stack_destroy = StackDestroy(top_graph);
      AnfAlgo::KeepOrder(top_graph, *exec_order.rbegin(), stack_destroy);
      top_graph->SetExecOrderByDefault();
      context_.SetRecursiveCall(true);
      context_.SetInitedStack(true);
    }
  }

  // Insert StackOps for call_site in the recursive graph.
  void InsertStackOps(const CallSite &call_site) {
    MS_EXCEPTION_IF_NULL(kernel_graph_);
    auto call_point = call_site.conversion_cnode;
    auto exec_order = kernel_graph_->execution_order();
    std::vector<AnfNodePtr> before_nodes;
    std::vector<CNodePtr> stack_pushs;
    bool find_call_point = false;
    for (auto &node : exec_order) {
      auto node_name = common::AnfAlgo::GetCNodeName(node);
      if (node == call_point) {
        find_call_point = true;
        continue;
      }
      if (!find_call_point) {
        if (node_name == kLabelGotoOpName || node_name == kLabelSwitchOpName || node_name == kLabelSetOpName ||
            node_name == prim::kPrimAssign->name()) {
          MS_LOG(DEBUG) << "Ignore goto/switch/set/assign ops";
        } else {
          before_nodes.push_back(node);
          MS_EXCEPTION_IF_NULL(node);
          MS_LOG(DEBUG) << "push back node:" << node->DebugString();
        }
        continue;
      }
      if (node->size() == 0 || node_name == kLabelGotoOpName || node_name == kLabelSetOpName ||
          node_name == prim::kPrimAssign->name()) {
        continue;
      }
      FindInputNode(before_nodes, node, &stack_pushs);
    }
    InsertStackPush(kernel_graph_, call_point, stack_pushs);
  }

  // Find nodes which need StackOps, and insert StackOps for node.
  void FindInputNode(const std::vector<AnfNodePtr> &before_nodes, const CNodePtr &node,
                     std::vector<CNodePtr> *stack_pushs) {
    MS_EXCEPTION_IF_NULL(node);
    uint32_t start_index = 1;
    if (common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimAssign)) {
      start_index = kInputIndex;
    }
    for (uint32_t i = start_index; i < node->inputs().size(); i++) {
      auto node_input = node->input(i);
      // not need to save monad.
      if (HasAbstractMonad(node_input)) {
        continue;
      }
      MS_EXCEPTION_IF_NULL(node_input);
      MS_LOG(DEBUG) << "check node input[" << i << "]: " << node_input->DebugString();
      if (node_input->isa<Parameter>()) {
        MS_LOG(DEBUG) << "node_input:" << node_input->DebugString() << " is a param";
        CNodePtr stack_pop = InsertStackPop(node_input, stack_pushs);
        node->set_input(i, stack_pop);
        KeepOrderForStackPop(kernel_graph_, stack_pop, node);
        continue;
      }
      auto iter = std::find_if(before_nodes.begin(), before_nodes.end(),
                               [node_input](auto before_node) { return before_node == node_input; });
      if (iter != before_nodes.end()) {
        CNodePtr stack_pop = InsertStackPop(*iter, stack_pushs);
        node->set_input(i, stack_pop);
        KeepOrderForStackPop(kernel_graph_, stack_pop, node);
      }
    }
  }

  // Create StackOps for node_input.
  CNodePtr InsertStackPop(const AnfNodePtr &node_input, std::vector<CNodePtr> *stack_pushs) {
    MS_EXCEPTION_IF_NULL(node_input);
    MS_EXCEPTION_IF_NULL(stack_pushs);
    auto stack_push = StackPush(node_input);
    (void)stack_pushs->emplace_back(stack_push);
    auto stack_pop = StackPop();
    MS_EXCEPTION_IF_NULL(stack_pop);
    stack_pop->set_abstract(node_input->abstract());
    return stack_pop;
  }

  // Arrange StackPushs according to the rules of the last pop-up StackPush first,
  // while ensuring that the last StackPush node is next to the jump_node.
  static void InsertStackPush(const KernelGraphPtr &kg, const CNodePtr &jump_node,
                              const std::vector<CNodePtr> &stack_pushs) {
    MS_LOG(DEBUG) << "There are " << stack_pushs.size() << " stack_push ops";
    if (stack_pushs.empty()) {
      return;
    }
    for (uint32_t i = 1; i < stack_pushs.size(); i++) {
      AnfAlgo::KeepOrder(kg, stack_pushs[i], stack_pushs[i - 1]);
    }
    auto nodes = kg->execution_order();
    auto node_iter = std::find(nodes.begin(), nodes.end(), jump_node);
    AnfAlgo::KeepOrder(kg, stack_pushs[0], jump_node);
    if (node_iter != nodes.begin()) {
      AnfAlgo::KeepOrder(kg, *(node_iter - 1), *stack_pushs.rbegin());
    }
  }

  // Ensure StackPop is next to the jump_node.
  static void KeepOrderForStackPop(const KernelGraphPtr &kg, const CNodePtr &pop, const CNodePtr &jump_node) {
    auto nodes = kg->execution_order();
    auto node_iter = std::find(nodes.cbegin(), nodes.cend(), jump_node);
    if (node_iter == nodes.cend()) {
      MS_LOG(EXCEPTION) << "Cannot find node: " << jump_node->DebugString();
    }
    // Insert between jump_node-1 and jump_node.
    if (node_iter != nodes.begin()) {
      CNodePtr node = *(node_iter - 1);
      AnfAlgo::KeepOrder(kg, node, pop);
    }
    AnfAlgo::KeepOrder(kg, pop, jump_node);
  }

  static void RemoveIdleParameter(const KernelGraphPtr &top_graph, const AnfNodePtr &parameter) {
    MS_EXCEPTION_IF_NULL(top_graph);
    auto erase_item_from_vec = [](std::vector<AnfNodePtr> vec, const AnfNodePtr &item) -> std::vector<AnfNodePtr> {
      for (auto iter = vec.begin(); iter != vec.end();) {
        if (*iter == item) {
          iter = vec.erase(iter);
        } else {
          ++iter;
        }
      }
      return vec;
    };
    top_graph->set_parameters(erase_item_from_vec(top_graph->parameters(), parameter));
    top_graph->set_child_graph_result(erase_item_from_vec(top_graph->child_graph_result(), parameter));
  }

  void HandleCallSite(CallSite *call_site) {
    MS_EXCEPTION_IF_NULL(call_site);
    // Update last_monad_.
    last_monad_ = call_site->last_monad;

    // The call/switch/switch_layer cnode.
    auto &cnode = call_site->cnode;
    if (CheckCallInline(cnode)) {
      auto call_graph = cnode->input(kFirstIndex);
      auto sub_kernel_graph = GetValueNodeKernelGraph(call_graph);
      std::vector<AnfNodePtr> call_inline_inputs = {NewPrimitive(prim::kPrimCallInline)};
      for (size_t i = kFirstIndex; i < common::AnfAlgo::GetInputNum(cnode); i++) {
        call_inline_inputs.emplace_back(common::AnfAlgo::GetInputNode(cnode, i));
      }
      auto call_inline = kernel_graph_->NewCNode(call_inline_inputs);
      MS_EXCEPTION_IF_NULL(call_inline);
      call_inline->set_abstract(cnode->abstract());
      common::AnfAlgo::SetNodeAttr(kAttrKernelGraph, MakeValue(sub_kernel_graph), call_inline);
      ReplaceNode(cnode, call_inline);
      return;
    }

    // Get branches of the call_site.
    // for call, there is one branch;
    // for switch, the first one is true branch;
    // for switch_layer, the first one is 0 branch.
    auto &branches = call_site->callees;

    // Link arguments and find labels for branches.
    std::vector<KernelGraphPtr> graphes;
    std::vector<uint32_t> labels;
    graphes.reserve(branches.size());
    labels.reserve(branches.size());
    for (auto &[graph, args] : branches) {
      MS_EXCEPTION_IF_NULL(graph);
      graphes.push_back(graph);
      labels.push_back(GetGraphLabel(graph));
    }

    // For the same call, their internal assignments cannot be cross-run.
    auto iter = call_last_monad_.find(graphes);
    if (iter != call_last_monad_.end()) {
      if (last_monad_ != nullptr && iter->second != nullptr) {
        last_monad_ = MakeDepend(last_monad_, iter->second);
      } else if (last_monad_ == nullptr) {
        last_monad_ = iter->second;
      }
    }

    bool monad_update = false;
    for (auto &[graph, args] : branches) {
      MS_EXCEPTION_IF_NULL(graph);
      auto linked_args = LinkArguments(args, graph);
      if (linked_args != nullptr) {
        monad_ = UpdateState(GetMonad(), linked_args);
        monad_update = true;
      }
    }
    if (!monad_update) {
      monad_ = last_monad_;
    }

    // Assign label indexes if required.
    AssignLabelIndexes(call_site);

    // For Switch, we reverse the graphes and labels, so that the false branch
    // is the first one, since for kernel LabelSwitch, false is the first branch.
    if (common::AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimSwitch)) {
      std::reverse(graphes.begin(), graphes.end());
      std::reverse(labels.begin(), labels.end());
    }

    // Create LabelGoto or LabelSwitch node.
    auto label_goto_switch = MakeLabelGotoSwitch(cnode, graphes, labels);
    call_site->conversion_cnode = label_goto_switch;
    if (call_site->recursive) {
      common::AnfAlgo::SetNodeAttr(kAttrRecursive, prim::kValueOne, label_goto_switch);
    }

    // Setup return label and output if required.
    if (call_site->return_label != kNoLabel) {
      auto label_node = LabelSet(call_site->return_label);
      AnfNodePtr output = call_site->out_param;
      MS_EXCEPTION_IF_NULL(output);
      const bool is_single_call = call_site->label_indexes.empty();
      if (is_single_call) {
        // For single call, let output depend on the label node,
        // this ensures the return label is set before output is used.
        output = MakeDepend(output, label_node);
      } else {
        // For multi-return call, assign result from temp parameter to
        // output parameter, this prevent result be overwritten by next call.
        auto tmp_param = context_.GetTempParameter(output->abstract());
        if (common::AnfAlgo::CheckPrimitiveType(output, prim::kPrimMakeTuple)) {
          output = AssignAll(output, tmp_param, false, false, true);
        } else {
          RemoveIdleParameter(context_.TopGraph(), output);
          output = TensorMove(tmp_param, false);
        }
        monad_ = UpdateState(GetMonad(), output);
      }
      // Replace the the call/switch node with the output.
      ReplaceNode(cnode, output);
      call_last_monad_[graphes] = monad_;
      return;
    }

    // If no return label required, it should be a tail call.
    if (!call_site->tail) {
      MS_LOG(EXCEPTION) << "Return label not set for non-tail call " << cnode->DebugString();
    }
    // For tail calls, replace origin call node with label_goto/label_switch.
    ReplaceNode(cnode, label_goto_switch);
    kernel_graph_->set_end_goto(label_goto_switch);
    call_last_monad_[graphes] = monad_;
  }

  // Assign label indexes to label parameters for a call site.
  void AssignLabelIndexes(const CallSite *call_site) {
    MS_EXCEPTION_IF_NULL(call_site);
    for (auto &[label_param, label_index] : call_site->label_indexes) {
      auto index_value = GetIndexValueNode(label_index);
      auto assign = Assign(label_param, index_value, false, false, false);
      monad_ = UpdateState(GetMonad(), assign);
    }
  }

  // Create or reuse ValueNode for the index.
  ValueNodePtr GetIndexValueNode(uint32_t index) {
    auto iter = index_nodes_.find(index);
    if (iter != index_nodes_.cend()) {
      // Reuse ValueNode for same index.
      return iter->second;
    }
    // Create a new ValueNode on top graph for the index.
    auto &top_graph = context_.TopGraph();
    std::vector<int64_t> data = {static_cast<int64_t>(index)};
    auto tensor = std::make_shared<tensor::Tensor>(data, kInt32);
    MS_EXCEPTION_IF_NULL(top_graph);
    MS_EXCEPTION_IF_NULL(tensor);
    auto value_node = top_graph->NewValueNode(tensor->ToAbstract(), tensor);
    top_graph->AddValueNodeToGraph(value_node);
    (void)index_nodes_.emplace(index, value_node);
    return value_node;
  }

  // Replace a node with new node in current kernel graph.
  // We also replace the arguments used for sub-graph calls.
  void ReplaceNode(const AnfNodePtr &old_node, const AnfNodePtr &new_node) {
    MS_EXCEPTION_IF_NULL(new_node);
    MS_EXCEPTION_IF_NULL(old_node);
    MS_EXCEPTION_IF_NULL(kernel_graph_);
    kernel_graph_->ReplaceNode(old_node, new_node);
    for (auto &call_site : call_info_.call_sites) {
      for (auto &callee : call_site.callees) {
        std::replace(callee.args.begin(), callee.args.end(), old_node, new_node);
      }
    }
  }

  // Make a label_goto or label_switch for a Call/Switch/SwitchLayer node.
  CNodePtr MakeLabelGotoSwitch(const CNodePtr &cnode, const std::vector<KernelGraphPtr> &graphes,
                               const std::vector<uint32_t> &labels) {
    MS_EXCEPTION_IF_NULL(cnode);
    // Create LabelGoto or LabelSwitch according the cnode type.
    const bool is_call = common::AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimCall);
    auto label_goto_switch = (is_call ? LabelGoto(labels.front()) : LabelSwitch(cnode->input(1), labels));

    // Set child graph attribute for the LabelGoto or LabelSwitch node.
    SetChildGrapAttr(label_goto_switch, graphes);

    // Mark the label_switch node is for 'switch_layer' if it is.
    if (common::AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimSwitchLayer)) {
      common::AnfAlgo::SetNodeAttr(kAttrSwitchLayer, prim::kValueOne, label_goto_switch);
    }
    return label_goto_switch;
  }

  // Handle return points.
  // use label_goto for single return point;
  // use label_switch for multi return points.
  void HandleReturnPoints() {
    auto &return_points = call_info_.return_points;
    // No return points.
    if (return_points.empty()) {
      return;
    }
    if (call_info_.return_monad_ != nullptr) {
      monad_ = call_info_.return_monad_;
    }
    // Assign output according the return points.
    AssignOutput(return_points);
    // Single return point.
    if (return_points.size() == 1) {
      // Insert label_goto for return.
      auto &return_point = return_points.front();
      MS_EXCEPTION_IF_NULL(return_point.call_site);
      auto return_goto = LabelGoto(return_point.call_site->return_label);
      common::AnfAlgo::SetNodeAttr(kAttrReturn, prim::kValueOne, return_goto);
      kernel_graph_->set_end_goto(return_goto);
      return;
    }
    // Multi return points.
    std::vector<uint32_t> return_labels;
    return_labels.reserve(return_points.size());
    // Get return labels from return points.
    (void)std::transform(return_points.begin(), return_points.end(), std::back_inserter(return_labels),
                         [](const ReturnPoint &return_point) { return return_point.call_site->return_label; });
    // Insert label_switch for multi return points.
    auto &label_param = call_info_.label_param;
    MS_EXCEPTION_IF_NULL(label_param);
    auto return_switch = LabelSwitch(label_param, return_labels);
    common::AnfAlgo::SetNodeAttr(kAttrReturn, prim::kValueOne, return_switch);
    if (!call_info_.recursive) {
      common::AnfAlgo::SetNodeAttr(kAttrMultiCallEnd, prim::kValueOne, return_switch);
    }
    kernel_graph_->set_end_goto(return_switch);
    context_.SetSubGraphMultiCall(true);
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
    auto assign_output = AssignAll(out_param, kernel_graph_->output(), false, false, true);
    monad_ = UpdateState(GetMonad(), assign_output);
  }

  // Link actual arguments to graph's formal arguments.
  // 1. for multi-args:
  //   r = Call(fg, arg1, arg2, u)
  // linked arguments:
  //   r1 = Assign(para1, arg1, c)
  //   r2 = Assign(para2, arg2, c)
  //   tuple = MakeTuple(r1, r2, u)
  // 2. for single-arg:
  //   r = Call(fg, arg)
  // linked arguments:
  //   r = Assign(para1, arg1, c)
  // 3. for empty-arg:
  //   r = Call(fg)
  // linked arguments return null.
  AnfNodePtr LinkArguments(const std::vector<AnfNodePtr> &args, const KernelGraphPtr &graph) {
    MS_EXCEPTION_IF_NULL(graph);
    auto &paras = graph->inputs();
    if (args.size() != paras.size()) {
      MS_LOG(EXCEPTION) << "Wrong arg number! " << graph->ToString() << " " << args.size() << " != " << paras.size();
    }
    // If no argument, return null.
    if (args.empty()) {
      return nullptr;
    }
    // We do not eliminate argument Assign for recursive graphs.
    const bool keep = IsRecursive(graph);
    // Single argument.
    if (args.size() == 1) {
      auto &value = args.front();
      if (HasAbstractMonad(value) || paras.front() == value) {
        // No assign for single monad argument, return it.
        return value;
      }
      return AssignAll(paras.front(), value, true, keep, false);
    }
    // Multi arguments.
    AnfNodePtrList tuple_inputs;
    tuple_inputs.reserve(args.size() + 1);
    (void)tuple_inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    for (size_t i = 0; i < args.size(); ++i) {
      auto &value = args.at(i);
      if (HasAbstractMonad(value)) {
        // No assign for monad arguments.
        (void)tuple_inputs.emplace_back(value);
        continue;
      }
      // Assign general arguments.
      auto &target = paras.at(i);
      if (target == value) {
        continue;
      }
      (void)tuple_inputs.emplace_back(AssignAll(target, value, true, keep, false));
    }
    auto new_tuple = kernel_graph_->NewCNode(tuple_inputs);
    MS_EXCEPTION_IF_NULL(new_tuple);
    // Set abstract for the MakeTuple node.
    abstract::AbstractBasePtrList element_abstracts;
    (void)std::transform(tuple_inputs.begin() + 1, tuple_inputs.end(), std::back_inserter(element_abstracts),
                         [](const AnfNodePtr &input) { return input->abstract(); });
    new_tuple->set_abstract(std::make_shared<abstract::AbstractTuple>(element_abstracts));
    return new_tuple;
  }

  // Return true if the graph is involved with recursive calls.
  bool IsRecursive(const KernelGraphPtr &kg) { return context_.call_info_map[kg].recursive; }

  // For some cnode, attributes may set to primitive instance, so we create a new prim instance for each cnode.
  static AnfNodePtr NewPrimitive(const PrimitivePtr &prim) {
    return NewValueNode(std::make_shared<Primitive>(prim->name()));
  }

  AnfNodePtr GetLinkMonad() {
    if (last_monad_ != nullptr) {
      return last_monad_;
    }
    return GetMonad();
  }

  // Make a tensor move cnode.
  CNodePtr TensorMove(const AnfNodePtr &source, bool link) {
    auto monad = (link ? GetLinkMonad() : GetMonad());
    auto tensor_move_prim = std::make_shared<Primitive>(prim::kPrimTensorMove->name());
    auto tensor_move = NewValueNode(tensor_move_prim);
    MS_EXCEPTION_IF_NULL(tensor_move_prim);
    MS_EXCEPTION_IF_NULL(tensor_move);
    auto cnode = kernel_graph_->NewCNode(std::vector<AnfNodePtr>{tensor_move, source, monad});
    MS_EXCEPTION_IF_NULL(cnode);
    cnode->set_abstract(source->abstract());
    return cnode;
  }

  // Make a assign cnode.
  CNodePtr Assign(const AnfNodePtr &target, const AnfNodePtr &source, bool link, bool keep, bool output) {
    auto monad = (link ? GetLinkMonad() : GetMonad());
    auto assign_prim = std::make_shared<Primitive>(prim::kPrimAssign->name());
    MS_EXCEPTION_IF_NULL(assign_prim);
    if (link) {
      // Mark this assign is to link real argument to formal argument.
      assign_prim->set_attr(LINK, prim::kValueOne);
    }
    if (keep) {
      // Mark this assign should not be eliminated.
      assign_prim->set_attr(KEEP, prim::kValueOne);
    }
    if (output) {
      // Mark this assign is used for output parameter.
      assign_prim->set_attr(OUTPUT, prim::kValueOne);
    }
    auto assign = NewValueNode(assign_prim);
    if (!IsCompatible(target->abstract(), source->abstract())) {
      MS_LOG(WARNING) << "Assign: " << target->DebugString() << " has different abstract() with "
                      << source->DebugString() << ", [ " << target->abstract()->ToString()
                      << " != " << source->abstract()->ToString() << " ], need insert CastOp.";
      if (AnfAlgo::GetOutputTensorNum(target) != kSingleOutput ||
          AnfAlgo::GetOutputTensorNum(source) != kSingleOutput) {
        MS_LOG(EXCEPTION) << "Assign: " << target->DebugString() << " or " << source->DebugString()
                          << " has multi outputs.";
      }
      std::vector<AnfNodePtr> cast_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimCast->name())),
                                             source};
      auto cast_node = kernel_graph_->NewCNode(cast_inputs);
      auto origin_shape = AnfAlgo::GetOutputDetailShape(source, kFirstOutput);
      auto shape = AnfAlgo::GetOutputDetailShape(target, kFirstOutput);
      if (!common::IsEqual(origin_shape, shape)) {
        MS_LOG(EXCEPTION) << "Assign: " << target->DebugString() << " and " << source->DebugString()
                          << " has different shape, source shape: " << origin_shape->ToString()
                          << ", target shape: " << shape->ToString();
      }
      auto type_id = common::AnfAlgo::GetOutputInferDataType(target, kFirstOutput);
      common::AnfAlgo::SetOutputTypeAndDetailShape({type_id}, {shape}, cast_node.get());
      common::AnfAlgo::SetNodeAttr(kAttrDstType, TypeIdToType(type_id), cast_node);
      cast_node->set_scope(source->scope());
      auto cnode = kernel_graph_->NewCNode({assign, target, cast_node, monad});
      cnode->set_abstract(target->abstract());
      return cnode;
    }
    auto cnode = kernel_graph_->NewCNode({assign, target, source, monad});
    MS_EXCEPTION_IF_NULL(cnode);
    cnode->set_abstract(target->abstract());
    return cnode;
  }

  // AissgnAll support tuple to tuple assign.
  AnfNodePtr AssignAll(const AnfNodePtr &target, const AnfNodePtr &source, bool link, bool keep, bool output) {
    MS_EXCEPTION_IF_NULL(target);
    MS_EXCEPTION_IF_NULL(source);
    if (!common::AnfAlgo::CheckPrimitiveType(target, prim::kPrimMakeTuple)) {
      // Assign single value.
      return Assign(target, source, link, keep, output);
    }
    // Assign tuple.
    auto source_abs = source->abstract();
    MS_EXCEPTION_IF_NULL(source_abs);
    if (!common::AnfAlgo::CheckPrimitiveType(source, prim::kPrimMakeTuple) &&
        source_abs->isa<abstract::AbstractTuple>()) {
      MS_EXCEPTION_IF_NULL(kernel_graph_);
      auto make_tuple = kernel_graph_->TransTupleToMakeTuple(source);
      return AssignAll(target, make_tuple, link, keep, output);
    }

    std::vector<AnfNodePtr> targets = common::AnfAlgo::GetAllOutput(target);
    std::vector<AnfNodePtr> sources = common::AnfAlgo::GetAllOutput(source);
    if (targets.size() != sources.size()) {
      MS_LOG(EXCEPTION) << "Target size " << targets.size() << " != source size " << sources.size();
    }
    AnfNodePtrList tuple_inputs;
    auto source_item_with_index = common::AnfAlgo::VisitKernelWithReturnType(source, 0);
    MS_EXCEPTION_IF_NULL(source_item_with_index.first);
    auto source_cnode = source_item_with_index.first->cast<CNodePtr>();
    auto target_cnode = target->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(source_cnode);
    MS_EXCEPTION_IF_NULL(target_cnode);
    if (!common::AnfAlgo::CheckPrimitiveType(source_cnode, prim::kPrimMakeTuple)) {
      MS_LOG(WARNING) << "Source : " << source_cnode->DebugString() << " is not MakeTuple.";
    }
    (void)tuple_inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    for (size_t i = 1; i < target_cnode->inputs().size(); ++i) {
      if (common::AnfAlgo::IsTupleOutput(target_cnode->input(i))) {
        (void)tuple_inputs.emplace_back(AssignAll(target_cnode->input(i), source_cnode->input(i), link, keep, output));
      } else {
        (void)tuple_inputs.emplace_back(Assign(target_cnode->input(i), source_cnode->input(i), link, keep, output));
      }
    }
    auto new_tuple = kernel_graph_->NewCNode(tuple_inputs);
    MS_EXCEPTION_IF_NULL(new_tuple);
    // Set abstract for the MakeTuple node.
    abstract::AbstractBasePtrList element_abstracts;
    (void)std::transform(tuple_inputs.begin() + 1, tuple_inputs.end(), std::back_inserter(element_abstracts),
                         [](const AnfNodePtr &input) { return input->abstract(); });
    new_tuple->set_abstract(std::make_shared<abstract::AbstractTuple>(element_abstracts));
    return new_tuple;
  }

  // Insert UpdateState after input node.
  AnfNodePtr UpdateState(const AnfNodePtr &state, const AnfNodePtr &input) {
    auto update_state = NewValueNode(prim::kPrimUpdateState);
    auto update_state_cnode = kernel_graph_->NewCNode({update_state, state, input});
    update_state_cnode->set_abstract(state->abstract());
    return update_state_cnode;
  }

  // Make entry label for current graph.
  // from:
  //   def sub_graph(x, y):
  //     return add(x, y)
  // to:
  //   def sub_graph(x, y, c):
  //     c = LabelSet(c) : entry_label
  //     return add(x, y)
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
    MS_EXCEPTION_IF_NULL(depend);
    auto depend_cnode = kernel_graph_->NewCNode({depend, origin, input});
    MS_EXCEPTION_IF_NULL(depend_cnode);
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
    MS_EXCEPTION_IF_NULL(cnode);
    common::AnfAlgo::SetNodeAttr(kAttrLabelIndex, MakeValue(label_id), cnode);
    cnode->set_abstract(monad->abstract());
    monad_ = cnode;
    return cnode;
  }

  // Make a LabelSet node.
  CNodePtr LabelSet(uint32_t label_id) {
    auto monad = GetMonad();
    auto label_set = NewPrimitive(prim::kPrimLabelSet);
    auto cnode = kernel_graph_->NewCNode({label_set, monad});
    common::AnfAlgo::SetNodeAttr(kAttrLabelIndex, MakeValue(label_id), cnode);
    cnode->set_abstract(monad->abstract());
    monad_ = cnode;
    return cnode;
  }

  // Make a LabelSwitch node.
  CNodePtr LabelSwitch(const AnfNodePtr &cond, const std::vector<uint32_t> &labels) {
    auto monad = GetMonad();
    auto label_switch = NewPrimitive(prim::kPrimLabelSwitch);
    MS_EXCEPTION_IF_NULL(label_switch);
    auto cnode = kernel_graph_->NewCNode({label_switch, cond, monad});
    MS_EXCEPTION_IF_NULL(cnode);
    auto label_list = MakeValue(labels);
    common::AnfAlgo::SetNodeAttr(kAttrLabelSwitchList, label_list, cnode);
    cnode->set_abstract(monad->abstract());
    monad_ = cnode;
    return cnode;
  }

  // Set child graph attribute for label_goto/label_switch node.
  static void SetChildGrapAttr(const AnfNodePtr &node, const std::vector<KernelGraphPtr> &graphs) {
    common::AnfAlgo::SetNodeAttr(kAttrChildGraph, MakeValue(graphs), node);
  }

  // Make a StackInit node.
  static CNodePtr StackInit(const KernelGraphPtr &kg) {
    MS_EXCEPTION_IF_NULL(kg);
    auto monad = AnfAlgo::MakeMonadValueNode(kg);
    auto stack_init = NewPrimitive(prim::kPrimStackInit);
    auto cnode = kg->NewCNode({stack_init, monad});
    common::AnfAlgo::SetNodeAttr(kAttrIndex, MakeValue<int64_t>(0), cnode);
    cnode->set_abstract(monad->abstract());
    return cnode;
  }

  // Make a StackDestroy node.
  static CNodePtr StackDestroy(const KernelGraphPtr &kg) {
    MS_EXCEPTION_IF_NULL(kg);
    auto monad = AnfAlgo::MakeMonadValueNode(kg);
    auto stack_destroy = NewPrimitive(prim::kPrimStackDestroy);
    auto cnode = kg->NewCNode({stack_destroy, monad});
    common::AnfAlgo::SetNodeAttr(kAttrIndex, MakeValue<int64_t>(0), cnode);
    cnode->set_abstract(monad->abstract());
    return cnode;
  }

  // Make a StackPush node.
  CNodePtr StackPush(const AnfNodePtr &input) {
    auto monad = AnfAlgo::MakeMonadValueNode(kernel_graph_);
    auto stack_push = NewPrimitive(prim::kPrimStackPush);
    auto cnode = kernel_graph_->NewCNode({stack_push, input, monad});
    common::AnfAlgo::SetNodeAttr(kAttrIndex, MakeValue<int64_t>(0), cnode);
    auto op_name = std::to_string(kernel_graph_->graph_id()) + "_stack_push_" + std::to_string(name_index_++);
    common::AnfAlgo::SetNodeAttr(kAttrStackOpName, MakeValue(op_name), cnode);
    cnode->set_abstract(monad->abstract());
    return cnode;
  }

  // Make a StackPop node.
  CNodePtr StackPop() {
    auto monad = AnfAlgo::MakeMonadValueNode(kernel_graph_);
    auto stack_pop = NewPrimitive(prim::kPrimStackPop);
    auto cnode = kernel_graph_->NewCNode({stack_pop, monad});
    common::AnfAlgo::SetNodeAttr(kAttrIndex, MakeValue<int64_t>(0), cnode);
    auto op_name = std::to_string(kernel_graph_->graph_id()) + "_stack_pop_" + std::to_string(name_index_++);
    common::AnfAlgo::SetNodeAttr(kAttrStackOpName, MakeValue(op_name), cnode);
    cnode->set_abstract(monad->abstract());  // need to refresh output's abstract().
    return cnode;
  }

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

  // The index of stackops name.
  uint32_t name_index_;

  // The flag which indicates to insert stackops.
  bool need_stackops_;

  // For the same call, the monad at the end of the function needs to be recorded.
  std::map<std::vector<KernelGraphPtr>, AnfNodePtr> call_last_monad_;
};

constexpr size_t kAssignTargetIndex = 1;
constexpr size_t kAssignSourceIndex = 2;

class ExecuteOrderGenerator {
 public:
  class Context : public BaseContext {};
  ExecuteOrderGenerator(Context &context, KernelGraphPtr graph) : context_(context), graph_(std::move(graph)) {}
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

  static uint32_t FindMaxLabelId(const std::vector<CNodePtr> &nodes) {
    uint32_t max_label = 0;
    for (auto &node : nodes) {
      if (common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimLabelSet)) {
        auto label_id = common::AnfAlgo::GetNodeAttr<uint32_t>(node, kAttrLabelIndex);
        max_label = std::max(label_id, max_label);
      }
    }
    return max_label;
  }

  void HandleLabelSwitch(const AnfNodePtr &node, std::vector<uint32_t> *labels, std::vector<uint32_t> *switch_labels,
                         std::multimap<uint32_t, uint32_t> *labels_multimap) {
    MS_EXCEPTION_IF_NULL(node);
    MS_EXCEPTION_IF_NULL(labels);
    MS_EXCEPTION_IF_NULL(switch_labels);
    bool is_new_labels = false;
    auto label_list = common::AnfAlgo::GetNodeAttr<std::vector<uint32_t>>(node, kAttrLabelSwitchList);
    std::vector<uint32_t> new_labels;
    new_labels.reserve(label_list.size());
    for (auto label_id : label_list) {
      auto iter = std::find_if(labels->begin(), labels->end(), [label_id](auto id) { return id == label_id; });
      // Use new label if find repeated label.
      if (iter == labels->end()) {
        (void)new_labels.emplace_back(label_id);
        (void)labels->emplace_back(label_id);
        continue;
      }
      (void)new_labels.emplace_back(++max_label_);
      (void)labels_multimap->emplace(*iter, max_label_);
      (void)labels->emplace_back(label_id);
      is_new_labels = true;
    }
    (void)switch_labels->insert(switch_labels->cend(), new_labels.cbegin(), new_labels.cend());
    if (is_new_labels) {
      common::AnfAlgo::SetNodeAttr(kAttrLabelSwitchList, MakeValue(new_labels), node);
    }
  }

  void HandleLabelGoto(const AnfNodePtr &node, std::vector<uint32_t> *labels, std::vector<uint32_t> *switch_labels,
                       std::multimap<uint32_t, uint32_t> *labels_multimap) {
    MS_EXCEPTION_IF_NULL(node);
    MS_EXCEPTION_IF_NULL(labels);
    MS_EXCEPTION_IF_NULL(switch_labels);
    auto label_id = common::AnfAlgo::GetNodeAttr<uint32_t>(node, kAttrLabelIndex);
    auto iter = std::find(switch_labels->begin(), switch_labels->end(), label_id);
    if (iter == switch_labels->end()) {
      (void)labels->emplace_back(label_id);
      return;
    }
    common::AnfAlgo::SetNodeAttr(kAttrLabelIndex, MakeValue(++max_label_), node);
    (void)labels_multimap->emplace(*iter, max_label_);
    (void)labels->emplace_back(max_label_);
  }

  // Unfold Repeated Labels, avoid same label in labelswitches.
  void UnfoldRepeatedLabels() {
    auto nodes = graph_->execution_order();
    std::vector<uint32_t> labels;
    std::vector<uint32_t> switch_labels;
    std::multimap<uint32_t, uint32_t> labels_multimap;
    max_label_ = FindMaxLabelId(nodes);
    for (auto &node : nodes) {
      if (common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimLabelSwitch)) {
        HandleLabelSwitch(node, &labels, &switch_labels, &labels_multimap);
        continue;
      }
      if (common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimLabelGoto)) {
        HandleLabelGoto(node, &labels, &switch_labels, &labels_multimap);
        continue;
      }
    }
    InsertLabelSet(&nodes, labels_multimap);
    graph_->set_label_num(max_label_ + 1);
    graph_->set_execution_order(nodes);
  }

  void InsertLabelSet(std::vector<CNodePtr> *nodes, const std::multimap<uint32_t, uint32_t> &labels_multimap) {
    MS_EXCEPTION_IF_NULL(nodes);
    for (auto labels : labels_multimap) {
      auto old_label = labels.first;
      auto new_label = labels.second;
      auto iter = std::find_if(nodes->begin(), nodes->end(), [old_label](auto node) {
        if (!common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimLabelSet)) {
          return false;
        }
        auto label_id = common::AnfAlgo::GetNodeAttr<uint32_t>(node, kAttrLabelIndex);
        return label_id == old_label;
      });
      if (iter == nodes->end()) {
        MS_LOG(EXCEPTION) << "Not found labelset:" << old_label;
      }
      auto label_set = NewValueNode(std::make_shared<Primitive>(prim::kPrimLabelSet->name()));
      auto cnode = graph_->NewCNode({label_set});
      common::AnfAlgo::CopyNodeAttrs(*iter, cnode);
      common::AnfAlgo::SetNodeAttr(kAttrLabelIndex, MakeValue(new_label), cnode);
      auto monad = graph_->NewValueNode(kUMonad->ToAbstract(), kUMonad);
      cnode->set_abstract(monad->abstract());
      (void)device::ascend::SelectKernelInfo(cnode);
      (void)nodes->insert(iter, cnode);
    }
  }

  static void AppendGraphOrder(std::vector<CNodePtr> *execution_order, const KernelGraphPtr &graph) {
    auto &order = graph->execution_order();
    (void)execution_order->insert(execution_order->cend(), order.cbegin(), order.cend());
  }

  static bool HasSubGraphs(const CNodePtr &cnode) {
    return (cnode && common::AnfAlgo::HasNodeAttr(kAttrChildGraph, cnode));
  }

  static std::vector<KernelGraphPtr> GetSubGraphs(const CNodePtr &cnode) {
    return common::AnfAlgo::GetNodeAttr<std::vector<KernelGraphPtr>>(cnode, kAttrChildGraph);
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
        if (!common::AnfAlgo::HasNodeAttr(kAttrSwitchLayer, cnode)) {
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
        common::AnfAlgo::EraseNodeAttr(kAttrChildGraph, cnode);
      }
    }
    // Save generated execution order into the graph.
    graph_->set_execution_order(std::move(execution_order));
  }

  std::set<CNodePtr> GetAllNodes(std::map<CNodePtr, const size_t> *search_list) {
    MS_EXCEPTION_IF_NULL(search_list);
    const auto &all_graphs = context_.visited_graphs();
    std::set<CNodePtr> all_nodes;
    for (const auto &graph : all_graphs) {
      auto out = graph->get_return();
      MS_EXCEPTION_IF_NULL(out);
      (void)search_list->emplace(out->cast<CNodePtr>(), 0);
      auto nodes = TopoSort(out);
      for (auto &node : nodes) {
        MS_EXCEPTION_IF_NULL(node);
        auto cnode = node->cast<CNodePtr>();
        if (cnode != nullptr) {
          (void)all_nodes.insert(cnode);
        }
      }
    }
    return all_nodes;
  }

  static const AnfNodePtr &GetRealNode(const AnfNodePtr &input) {
    MS_EXCEPTION_IF_NULL(input);
    if (IsPrimitiveCNode(input, prim::kPrimLoad) || IsPrimitiveCNode(input, prim::kPrimDepend)) {
      return input->cast<CNodePtr>()->inputs().at(1);
    }
    return input;
  }

  static void RemoveSameInputsAssigns(std::vector<CNodePtr> *exec_order) {
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
    std::map<CNodePtr, const size_t> search_list;
    for (size_t i = 0; i < exec_order.size(); i++) {
      (void)search_list.emplace(exec_order[i], i);
    }

    // Remove assigns that target and source are same.
    RemoveSameInputsAssigns(&exec_order);

    // Get all nodes and all graphs
    std::set<CNodePtr> all_nodes = GetAllNodes(&search_list);
    const auto &all_graphs = context_.visited_graphs();

    // Count parameter write times by check all assign nodes.
    auto param_write_times = CountParameterAssigns(search_list, exec_order);

    // Erase redundant assigns.
    for (auto iter = exec_order.begin(); iter != exec_order.end();) {
      auto &node = *iter;
      MS_EXCEPTION_IF_NULL(node);
      // We only try to erase argument link assign nodes,
      // other assign nodes are skipped.
      if (IsOptimizableAssign(node)) {
        auto &target = node->inputs().at(kAssignTargetIndex);
        MS_EXCEPTION_IF_NULL(target);
        auto para = param_write_times.find(target);
        if (para != param_write_times.end() && para->second.first == 1) {
          // Check source of the Assign.
          auto &source = node->inputs().at(kAssignSourceIndex);
          MS_EXCEPTION_IF_NULL(source);
          if (source->isa<Parameter>()) {
            auto it = param_write_times.find(source);
            const auto index = search_list[node];
            if (it != param_write_times.end() && it->second.first > 0 && it->second.second > index) {
              // Skip if Assign source is a parameter and be written in other place.
              ++iter;
              continue;
            }
          }
          // If target only write once, and source not be written,
          // replace target with source and erase the Assign node.
          MS_EXCEPTION_IF_NULL(target->func_graph());
          auto kg = target->func_graph()->cast<KernelGraphPtr>();
          MS_EXCEPTION_IF_NULL(kg);
          kg->ReplaceNode(target, source);

          // replace parameter in graph input
          for (const auto &g : all_graphs) {
            MS_EXCEPTION_IF_NULL(g);
            auto child_graph_inputs = g->MutableInputs();
            std::replace(child_graph_inputs->begin(), child_graph_inputs->end(), target, source);
            MS_LOG(DEBUG) << "Replace parameter " << target->DebugString() << " by " << source->DebugString()
                          << " in graph " << g->graph_id() << " inputs";
          }

          // replace parameter in node
          for (const auto &iter_node : all_nodes) {
            MS_EXCEPTION_IF_NULL(iter_node);
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
  std::map<AnfNodePtr, std::pair<int, size_t>> CountParameterAssigns(
    const std::map<CNodePtr, const size_t> &search_list, const std::vector<CNodePtr> &exec_order) {
    auto ref_map = graph_->GetRefMap();
    std::multimap<AnfNodePtr, std::tuple<size_t, AnfNodePtr, size_t>> ref_multimap;
    std::set<AnfNodePtr> root_inputs(graph_->inputs().begin(), graph_->inputs().end());
    (void)std::transform(ref_map.begin(), ref_map.end(), std::inserter(ref_multimap, ref_multimap.end()),
                         [](const std::pair<std::pair<AnfNodePtr, size_t>, std::pair<AnfNodePtr, size_t>> &p)
                           -> std::pair<AnfNodePtr, std::tuple<size_t, AnfNodePtr, size_t>> {
                           return {p.first.first, {p.first.second, p.second.first, p.second.second}};
                         });
    auto validate_ref_parameter = [](AnfNodePtr node) -> AnfNodePtr {
      MS_EXCEPTION_IF_NULL(node);
      if (node->isa<CNode>() && common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimTransData)) {
        auto cnode = node->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(cnode);
        auto first_input = common::AnfAlgo::VisitKernelWithReturnType(cnode->input(kFirstDataInputIndex), 0, true);
        MS_EXCEPTION_IF_NULL(first_input.first);
        return first_input.first;
      }
      return node;
    };

    // Find all graph input parameters.
    std::map<AnfNodePtr, std::pair<int, size_t>> param_write_times;
    const auto &all_graphs = context_.visited_graphs();
    for (const auto &graph : all_graphs) {
      MS_EXCEPTION_IF_NULL(graph);
      for (auto &input : graph->inputs()) {
        MS_EXCEPTION_IF_NULL(input);
        if (input->isa<Parameter>()) {
          (void)param_write_times.emplace(input, std::make_pair(0, 0));
        }
      }
    }

    // Search all refnodes for parameter write assigns.
    for (auto &node : exec_order) {
      if (ref_multimap.find(node) == ref_multimap.end()) {
        // if node is not refnode which cannot write param, skip it.
        continue;
      }
      std::set<AnfNodePtr> refed_parameters;
      for (auto [iter, end] = ref_multimap.equal_range(node); iter != end; ++iter) {
        (void)refed_parameters.insert(validate_ref_parameter(std::get<1>(iter->second)));
      }
      MS_EXCEPTION_IF_NULL(node);
      for (auto &in : node->inputs()) {
        auto visit_node = common::AnfAlgo::VisitKernelWithReturnType(in, 0).first;
        visit_node = validate_ref_parameter(visit_node);
        MS_EXCEPTION_IF_NULL(visit_node);
        if (!visit_node->isa<Parameter>() || root_inputs.find(visit_node) != root_inputs.end()) {
          continue;
        }
        if (refed_parameters.find(visit_node) != refed_parameters.end()) {
          auto iter = param_write_times.find(visit_node);
          if (iter != param_write_times.end()) {
            // Found a parameter writer, count it.
            ++(iter->second.first);
            if (search_list.find(node) == search_list.end()) {
              MS_LOG(EXCEPTION) << "node: " << node->DebugString() << " cannot found in search list.";
            }
            iter->second.second = search_list.at(node);
          }
        }
      }
    }
    return param_write_times;
  }

  // Check if a node is an assign for argument link and can be optimized.
  [[nodiscard]] static bool IsOptimizableAssign(const AnfNodePtr &node) {
    auto cnode = dyn_cast<CNode>(node);
    if (cnode == nullptr) {
      return false;
    }
    auto prim = GetValueNode<PrimitivePtr>(cnode->inputs().at(0));
    if (!IsPrimitiveEquals(prim, prim::kPrimAssign)) {
      return false;
    }
    MS_EXCEPTION_IF_NULL(prim);
    return (prim->GetAttr(LINK) == prim::kValueOne) && (prim->GetAttr(KEEP) != prim::kValueOne);
  }

  // Erase LabelGoto and LabelSet
  void EraseLabel() {
    // Find used labels (as jump target).
    std::set<uint32_t> label_used;
    auto exec_order = graph_->execution_order();
    for (auto iter = exec_order.begin(); iter != exec_order.end();) {
      auto &node = *iter;
      if (IsPrimitiveCNode(node, prim::kPrimLabelSwitch)) {
        auto labels = common::AnfAlgo::GetNodeAttr<std::vector<uint32_t>>(node, kAttrLabelSwitchList);
        for (auto label : labels) {
          (void)label_used.insert(label);
        }
      } else if (IsPrimitiveCNode(node, prim::kPrimLabelGoto)) {
        auto label = common::AnfAlgo::GetNodeAttr<uint32_t>(node, kAttrLabelIndex);
        auto next = std::next(iter);
        if (next != exec_order.end() && IsPrimitiveCNode(*next, prim::kPrimLabelSet)) {
          // The LabelGoto that jump to next node can be removed.
          auto next_label = common::AnfAlgo::GetNodeAttr<uint32_t>(*next, kAttrLabelIndex);
          if (next_label == label) {
            iter = exec_order.erase(iter);
            continue;
          }
        }
        (void)label_used.insert(label);
      }
      ++iter;
    }
    // Erase unused LabelSet nodes.
    for (auto iter = exec_order.begin(); iter != exec_order.end();) {
      auto &node = *iter;
      if (IsPrimitiveCNode(node, prim::kPrimLabelSet)) {
        auto label = common::AnfAlgo::GetNodeAttr<uint32_t>(node, kAttrLabelIndex);
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
  kernel_graph_->set_recursive_call(context.HasRecursiveCall());
  kernel_graph_->set_subgraph_multi_call(context.HasSubgraphMultiCall());
  MS_LOG(DEBUG) << "Ascend auto-monad finish.";
#ifdef ENABLE_DUMP_IR
  DumpGraphForDebug(kernel_graph_);
#endif
}

void AscendAutoMonad::GenerateExecuteOrder() {
  MS_LOG(DEBUG) << "Ascend generate execute order start.";
  ExecuteOrderGenerator::Context context;
  ExecuteOrderGenerator generator(context, kernel_graph_.get());
  generator.Run();
  MS_LOG(DEBUG) << "Ascend generate execute order finish.";
#ifndef ENABLE_SECURITY
  DumpExecuteOrder(kernel_graph_);
#endif
}
}  // namespace mindspore::session
