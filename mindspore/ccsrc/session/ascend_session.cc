/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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
#include "session/ascend_session.h"
#include <algorithm>
#include <map>
#include <tuple>
#include <set>
#include <list>
#include "operator/ops.h"
#include "ir/tensor.h"
#include "ir/anf.h"
#include "common/trans.h"
#include "device/kernel_runtime.h"
#include "device/ascend/kernel_select_ascend.h"
#include "device/ascend/kernel_build_ascend.h"
#include "device/ascend/ascend_kernel_runtime.h"
#include "device/ascend/ascend_device_address.h"
#include "pre_activate/ascend/ascend_backend_optimization.h"
#include "pre_activate/common/common_backend_optimization.h"
#include "device/kernel_adjust.h"
#include "device/ascend/ascend_stream_assign.h"
#include "device/ascend/ascend_label_assign.h"
#include "predict/predict.h"
#include "session/anf_runtime_algorithm.h"
#include "ir/scalar.h"
#include "debug/anf_ir_dump.h"
#include "debug/anf_ir_utils.h"
#include "debug/draw.h"
#include "common/utils.h"
#include "pre_activate/common/helper.h"
#include "device/kernel_runtime_manager.h"
#include "kernel/tbe/tbe_python_funcs.h"
#include "utils/config_manager.h"
#include "utils/base_ref_extends.h"

namespace mindspore {
namespace session {
const size_t kInvalidIndex = SIZE_MAX;
namespace {
void DumpGraphExeOrder(const std::vector<CNodePtr> &execution_order, const std::string &tag = "") {
  MS_LOG(INFO) << "Dump execution_order size " << execution_order.size();
  MS_LOG(INFO) << "[index][stream_label][graph_id][node string]";
  int i = 0;
  for (auto &cnode : execution_order) {
    MS_EXCEPTION_IF_NULL(cnode);
    MS_LOG(INFO) << "[ " << i << "]"
                 << "[" << AnfAlgo::GetStreamDistinctionLabel(cnode.get()) << "]"
                 << "[" << AnfAlgo::GetGraphId(cnode.get()) << "]"
                 << "[" << cnode->DebugString() << "]";
    i++;
  }

  std::stringstream buf;
  buf << "================== execution order ==================\n";
  if (!tag.empty()) {
    buf << tag << "\n";
  }
  buf << "execution_order size: " << execution_order.size() << "\n";
  i = 0;
  for (auto &cnode : execution_order) {
    MS_EXCEPTION_IF_NULL(cnode);
    buf << i << ":\n";
    buf << "\t" << cnode->DebugString() << "\n";
    buf << "\t" << AnfAlgo::GetStreamDistinctionLabel(cnode.get()) << "\n";
    buf << "\t" << AnfAlgo::GetGraphId(cnode.get()) << "\n";
    i++;
  }
  buf << "================== execution order ==================\n";
  // std::cout << buf.str() << std::endl;
}

void DumpGraphInputArgs(const VectorRef &args) {
  MS_LOG(INFO) << "Args size[%lu]" << args.size();
  for (size_t i = 0; i < args.size(); i++) {
    if (utils::isa<AnfNodePtr>(args[i])) {
      auto anf = utils::cast<AnfNodePtr>(args[i]);
      MS_EXCEPTION_IF_NULL(anf);
      MS_LOG(INFO) << "Parameter arg" << i << " = [%s]" << anf->DebugString();
    } else if (utils::isa<ValuePtr>(args[i])) {
      auto value = utils::cast<ValuePtr>(args[i]);
      MS_EXCEPTION_IF_NULL(value);
      MS_LOG(INFO) << "Tensor arg" << i << " = " << value->ToString();
    } else {
      MS_LOG(INFO) << "Unknown arg" << i << " = " << args[i].ToString();
    }
  }
}

void SetStreamDistinctionLabel(const KernelGraphPtr &graph, uint32_t label, bool is_override) {
  MS_EXCEPTION_IF_NULL(graph);
  if (is_override || graph->stream_distinction_label() == kInvalidDistincLabel) {
    graph->set_stream_distinction_label(label);
  }
}

std::vector<BaseRef> GetRealArgs(const KernelGraphPtr graph, const VectorRef &args) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> graph_inputs = graph->inputs();
  auto valid_inputs = graph->valid_inputs();
  size_t real_args_size = 0;
  std::vector<BaseRef> real_args = {};
  for (size_t i = 0; i < args.size(); i++) {
    if (utils::isa<AnfNodePtr>(args[i])) {
      auto tmp_args = AnfAlgo::GetAllOutput(utils::cast<AnfNodePtr>(args[i]), {prim::kPrimTupleGetItem});
      for (auto &real_arg : tmp_args) {
        auto anf_node = utils::cast<AnfNodePtr>(real_arg);
        MS_EXCEPTION_IF_NULL(anf_node);
        auto abstract = anf_node->abstract();
        MS_EXCEPTION_IF_NULL(abstract);
        // create multiple parameters if is a tuple output real kernel
        if (abstract->isa<abstract::AbstractTuple>() &&
            !AnfAlgo::CheckPrimitiveType(anf_node, prim::kPrimTupleGetItem)) {
          auto tuple_abstract = abstract->cast<abstract::AbstractTuplePtr>();
          MS_EXCEPTION_IF_NULL(tuple_abstract);
          real_args_size += tuple_abstract->size();
          continue;
        }
        real_args_size += 1;
        real_args.push_back(real_arg);
      }
    } else {
      real_args_size += 1;
      real_args.push_back(args[i]);
    }
  }
  if (graph_inputs.size() != valid_inputs.size()) {
    MS_LOG(EXCEPTION) << "graph_inputs.size(): " << graph_inputs.size()
                      << ", valid_inputs.size(): " << valid_inputs.size() << " not equal";
  }
  if (real_args_size != graph_inputs.size()) {
    for (size_t j = 0; j < valid_inputs.size(); j++) {
      if (valid_inputs[j]) {
        MS_LOG(INFO) << "index: " << j << ", nodes: " << graph_inputs[j]->DebugString();
      }
    }
    MS_LOG(WARNING) << "real_args_size: " << real_args_size << ", graph_inputs.size(): " << graph_inputs.size()
                    << " not equal";
  }
  return real_args;
}

std::vector<CNodePtr> GetCNodes(const std::vector<AnfNodePtr> &anf_nodes) {
  std::vector<CNodePtr> cnodes = {};
  size_t i = 0;
  for (const auto &anf : anf_nodes) {
    MS_LOG(INFO) << "apply_list[" << i++ << "] = " << anf->DebugString();
    MS_EXCEPTION_IF_NULL(anf);
    if (anf->isa<CNode>()) {
      cnodes.push_back(anf->cast<CNodePtr>());
    }
  }
  return cnodes;
}

static std::vector<std::vector<CNodePtr>> GetChildList(const std::vector<CNodePtr> &cnodes,
                                                       const std::set<PrimitivePtr> &cut_prims) {
  size_t after_cut_index = 0;
  std::vector<std::vector<CNodePtr>> ret;
  for (size_t i = 0; i < cnodes.size(); ++i) {
    bool is_cut_node = false;
    for (auto &prim : cut_prims) {
      if (AnfAlgo::CheckPrimitiveType(cnodes[i], prim)) {
        is_cut_node = true;
        break;
      }
    }
    if (is_cut_node) {
      // is call and not switch call,cut to 3 lists
      if (!AnfAlgo::CheckPrimitiveType(cnodes[i], prim::kPrimCall)) {
        // if is not a call,cut to 2 lists
        ret.emplace_back(cnodes.begin() + after_cut_index, cnodes.begin() + i);
        after_cut_index = i;
      } else if (!AnfAlgo::IsSwitchCall(cnodes[i])) {
        ret.emplace_back(cnodes.begin() + after_cut_index, cnodes.begin() + i);
        ret.emplace_back(1, cnodes[i]);
        after_cut_index = i + 1;
        continue;
      }
    }
    // get last child graph list
    if (AnfAlgo::CheckPrimitiveType(cnodes[i], prim::kPrimReturn)) {
      ret.emplace_back(cnodes.begin() + after_cut_index, cnodes.end());
      continue;
    }
  }
  return ret;
}

static void BindCallArgsWithParameter(const std::vector<AnfNodePtr> &parameters, const std::vector<AnfNodePtr> &args,
                                      KernelGraph *child_graph) {
  MS_EXCEPTION_IF_NULL(child_graph);
  MS_LOG(INFO) << "Start bind parameter of child graph:" << child_graph->graph_id();
  if (args.empty()) {
    return;
  }
  if (parameters.size() != args.size()) {
    MS_LOG(EXCEPTION) << "Graph:" << child_graph->graph_id() << " parameters size:" << parameters.size()
                      << " and args size:" << args.size() << " not equal!";
  }
  child_graph->SetExecOrderByDefault();
  for (size_t i = 0; i < parameters.size(); i++) {
    if (args[i] == parameters[i]) {
      child_graph->SetRealInput(parameters[i], args[i]);
      MS_LOG(INFO) << "Parameter and arg are same.";
      continue;
    }
    child_graph->SetRealInput(parameters[i], args[i]);
  }
}

// if a call has kernel input, it's a child graph split from ME, so these kernel input should be set into real input of
// graph.For example, call input = (prim,graph,kernel1,kernel2),then real_input = [kernel1,kernel2]
static void UpdateRealInput(NotNull<KernelGraphPtr> graph) {
  auto call_nodes = graph->FindNodeByPrimitive(prim::kPrimCall);
  for (auto &call_node : call_nodes) {
    MS_EXCEPTION_IF_NULL(call_node);
    auto child_graphs = AnfAlgo::GetCallNodeKernelGraph(call_node);
    if (child_graphs.size() == 1) {
      MS_EXCEPTION_IF_NULL(child_graphs[0]);
      std::vector<AnfNodePtr> real_args =
        std::vector<AnfNodePtr>(call_node->inputs().begin() + 2, call_node->inputs().end());
      std::vector<AnfNodePtr> child_inputs = child_graphs[0]->inputs();
      BindCallArgsWithParameter(child_inputs, real_args, child_graphs[0].get());
      call_node->set_inputs(std::vector<AnfNodePtr>(call_node->inputs().begin(), call_node->inputs().begin() + 2));
    } else if (child_graphs.size() == 2) {
      auto get_partial_args = [&](size_t input_index) -> std::vector<AnfNodePtr> {
        auto switch_node = call_node->input(1);
        MS_EXCEPTION_IF_NULL(switch_node);
        auto switch_cnode = switch_node->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(switch_cnode);
        auto partial = switch_cnode->input(input_index);
        MS_EXCEPTION_IF_NULL(partial);
        auto partial_cnode = partial->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(partial_cnode);
        auto ret = std::vector<AnfNodePtr>(partial_cnode->inputs().begin() + 2, partial_cnode->inputs().end());
        partial_cnode->set_inputs(
          std::vector<AnfNodePtr>(partial_cnode->inputs().begin(), partial_cnode->inputs().begin() + 2));
        return ret;
      };
      BindCallArgsWithParameter(child_graphs[0]->inputs(), get_partial_args(2), child_graphs[0].get());
      BindCallArgsWithParameter(child_graphs[1]->inputs(), get_partial_args(3), child_graphs[1].get());
    }
  }
}

static void RecurseToUpdateCallRealInput(NotNull<KernelGraphPtr> graph,
                                         const NotNull<std::set<KernelGraphPtr> *> memo) {
  memo->insert(graph.get());
  MS_LOG(INFO) << "Start graph id:" << graph->graph_id();
  for (auto &child_graph : graph->child_graph_order()) {
    if (memo->find(child_graph) != memo->end()) {
      MS_LOG(INFO) << "Child graph:" << child_graph->graph_id()
                   << ",parent graph:" << graph->parent_graph()->graph_id();
      continue;
    }
    RecurseToUpdateCallRealInput(NOT_NULL(child_graph), memo);
  }
  // this action should from bottom to top
  graph->UpdateCallRealInput();
}
}  // namespace

GraphId AscendSession::CompileGraph(const AnfNodePtrList &lst, const AnfNodePtrList &outputs) {
  MS_LOG(INFO) << "start";
  // construct graph, if successfully, graph_sum_ + 1
  auto graph = ConstructKernelGraph(lst, outputs);
  auto graph_id = graph->graph_id();
  MS_LOG(INFO) << "Compile graph " << graph_id << " success";
  return graph_id;
}

GraphId AscendSession::CompileGraph(NotNull<FuncGraphPtr> func_graph) {
  MS_LOG(INFO) << "start";
  std::vector<KernelGraphPtr> all_graphs;
  auto root_graph = ConstructKernelGraph(func_graph, &all_graphs);
  BackendOptimization(all_graphs);
  // split switch
  SplitGraphs(NOT_NULL(root_graph));
  // insert goto labels and label_sets
  LinkChildGraphs(NOT_NULL(root_graph));
  // resource initialize
  InitRuntimeResource();
  // assign label
  AssignLabel(NOT_NULL(root_graph));
  // recurse compile child root_graph
  std::set<KernelGraphPtr> memo;
  RecurseCompileGraph(NOT_NULL(root_graph), NOT_NULL(&memo));
  // root root_graph valiate,include genearte execute order and so on
  RootGraphExecutorValidate(NOT_NULL(root_graph));
  // adjust kernel
  AdjustKernel(root_graph);
  // assign stream
  AssignStream(NOT_NULL(root_graph));
  // insert profiling point
  device::KernelAdjust::GetInstance().Profiling(NOT_NULL(root_graph.get()));
  // build kernel
  BuildKernel(root_graph);
  // alloc mem
  MemoryAlloc(root_graph.get());
  // task generate
  GenerateTaskInfo(root_graph);
  // load task into device
  LoadTask(root_graph);
  // return the root_graph id to backend
  auto graph_id = root_graph->graph_id();
  return graph_id;
}

void AscendSession::SetFinalGraphSummaryFlag(const std::shared_ptr<KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto graph_order = GetGraphOrder(kernel_graph->graph_id());
  for (auto graph_id : graph_order) {
    auto child_graph = GetGraph(graph_id);
    if (child_graph == nullptr) {
      continue;
    }
    if (child_graph->summary_node_exist()) {
      kernel_graph->set_summary_node_exist(true);
      return;
    }
  }
  kernel_graph->set_summary_node_exist(false);
}

void AscendSession::BuildGraph(GraphId graph_id) {
  MS_LOG(INFO) << "start";
  auto graph = GetGraph(graph_id);
  MS_EXCEPTION_IF_NULL(graph);
  // resource initialize
  InitRuntimeResource();
  // multiple graph handle
  if (graph_id == final_graph_id_) {
    if (!graph->executable()) {
      return;
    }
    // insert assigns to child graph
    InsertAllAssigns();
    // insert switch and active to child graph
    MergeSwitchCompile();
    SetFinalGraphSummaryFlag(graph);
    // OptChildGraphs
    auto graph_order = GetGraphOrder(final_graph_id_);
    auto &graph_type = GetGraphOrderType(final_graph_id_);
    for (size_t i = 0; i < graph_order.size(); i++) {
      if (graph_type[i] == BRANCH_END || graph_type[i] == BRANCH_START) {
        continue;
      }
      MS_LOG(INFO) << "Start build child  graph " << graph_order[i];
      auto child_graph = GetGraph(graph_order[i]);
      CompileChildGraph(child_graph);
    }
    GetSummaryNodes(graph.get());
    // merge child graph
    MergeGraphExecOrder();
  } else {
    auto single_graph = GetGraph(graph_id);
    CompileChildGraph(single_graph);
    // set the distinction label of single graph
    single_graph->set_stream_distinction_label(graph_id);
    single_graph->UpdateExecuteKernelStreamLabel();
  }
  // adjust execution order because  merge child graph and other special operations
  AdjustKernel(graph);
  // Assign streams for control sink and hccl and so on
  AssignStream(NOT_NULL(graph));

  device::KernelAdjust::GetInstance().Profiling(NOT_NULL(graph.get()));
  // build kernel if node is cnode
  BuildKernel(graph);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->precompile_only()) {
    MS_LOG(INFO) << "Precompile only, stop in build kernel step";
  } else {
    // alloc memory, including static memory and dynamic memory
    MemoryAlloc(graph.get());
    // generate task info for task sink mode
    GenerateTaskInfo(graph);
    // load task info to device if it is sink mode
    LoadTask(graph);
  }
  // sync the inital const tensor to device
  SyncInitialTenosrToDevice();
  ExportChildGraphs(graph_id);
  MS_LOG(INFO) << "end";
}

void AscendSession::CompileChildGraph(const KernelGraphPtr &child_graph) {
  MS_EXCEPTION_IF_NULL(child_graph);
  MS_LOG(INFO) << "CompileChildGraph " << child_graph->ToString();
  opt::AscendBackendIRFusionOptimization(child_graph);
  opt::AscendBackendFuseBasicOpt(child_graph, true);
  opt::AscendBackendGraphKernelOpt(child_graph, true);
  child_graph->SetExecOrderByDefault();
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->save_graphs_flag();
  auto save_graphs_path = context_ptr->save_graphs_path();
  if (save_graphs_path.empty()) {
    save_graphs_path = ".";
  }
  if (save_graphs) {
    std::string file_path =
      save_graphs_path + "/" + "select_kernel_before" + "_graph_" + std::to_string(child_graph->graph_id()) + ".ir";
    DumpIR(file_path, child_graph);
  }
  // select kernel build info
  SelectKernel(*child_graph);
  if (save_graphs) {
    std::string file_path =
      save_graphs_path + "/" + "select_kernel_after" + "_graph_" + std::to_string(child_graph->graph_id()) + ".ir";
    DumpIR(file_path, child_graph);
  }
  // convert kernel Graph to model
  predictmodel::StepConvertGraph(child_graph);
  // optimize graph
  HardwareOptimize(child_graph);
  // assign static memory of parameters
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->AssignStaticMemoryInput(child_graph.get());
  runtime_instance->AssignStaticMemoryValueNode(child_graph.get());
}

void AscendSession::RunGraph(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs,
                             VectorRef *const outputs) {
  MS_LOG(INFO) << "start";
  auto kernel_graph = GetGraph(graph_id);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  // if none of child graph and no anf output exists
  if (!kernel_graph->executable()) {
    MS_LOG(INFO) << "No child graph has anf output";
    UpdateOutputs(kernel_graph, outputs, inputs);
    return;
  }
  // load input data from user input
  LoadInputData(kernel_graph, inputs);
  // convert inputs to model
  predictmodel::StepConvertWeight(inputs);
  {
    py::gil_scoped_release release;
    // run task on device
    ExecTask(kernel_graph);
  }
  // get result from device
  UpdateOutputs(kernel_graph, outputs, inputs);
  // summary
  Summary(kernel_graph.get());
  // dump used for debug
  Dump(kernel_graph);
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::RunOpHardwareOptimize(const std::shared_ptr<session::KernelGraph> &kernel_graph) const {
  MS_LOG(INFO) << "Start";
  // data layout optimization
  opt::RunOpAscendDataLayout(kernel_graph);
  // mixed precision optimization
  opt::AscendMixPrecision(kernel_graph);
  MS_LOG(INFO) << "Finish";
}

void AscendSession::RunOpExecTask(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  bool ret_ok = runtime_instance->LaunchKernel(kernel_graph.get());
  if (!ret_ok) {
    MS_LOG(EXCEPTION) << "run task error!";
  }
  MS_LOG(INFO) << "Finish!";
}

bool AscendSession::GraphCacheExist(const GraphInfo &graph_info) const {
  if (run_op_graphs_.find(graph_info) != run_op_graphs_.end()) {
    return true;
  }

  return false;
}

void AscendSession::BuildOp(const OpRunInfo &op_run_info, const GraphInfo &graph_info,
                            const std::vector<tensor::TensorPtr> &input_tensors, const std::vector<int> &tensors_mask) {
  MS_LOG(INFO) << "Build op " << op_run_info.op_name << " start !";
  if (GraphCacheExist(graph_info)) {
    MS_LOG(INFO) << "Build op " << op_run_info.op_name << " graph cache has existed !";
    return;
  }

  // construct graph include one op
  auto graph = ConstructSingleOpGraph(op_run_info, input_tensors, tensors_mask);
  MS_EXCEPTION_IF_NULL(graph);
  opt::RunOpAscendBackendIRFusionOptimization(graph);
  // kernel select
  SelectKernel(*graph);
  // optimize
  RunOpHardwareOptimize(graph);
  // init runtime resource
  InitRuntimeResource();
  // build kernel
  RunOpAdjustKernel(graph);
  BuildKernel(graph);
  run_op_graphs_[graph_info] = graph;
  MS_LOG(INFO) << "Build op " << op_run_info.op_name << " finish !";
}

py::tuple AscendSession::RunOp(const OpRunInfo &op_run_info, const GraphInfo &graph_info,
                               const std::vector<tensor::TensorPtr> &input_tensors) {
  auto graph = run_op_graphs_[graph_info];
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Run op " << op_run_info.op_name << " start!";
  // malloc mem
  RunOpMemoryAlloc(input_tensors, graph.get());
  // load input data to device
  LoadInputData(graph, input_tensors);
  // run op
  RunOpExecTask(graph);
  // get output
  VectorRef outputs;
  UpdateOutputs(graph, &outputs, input_tensors);
  // trans output to tuple
  auto output_tensors = TransformBaseRefListToTuple(outputs);
  if (!utils::isa<PyObjectRef>(output_tensors) ||
      !py::isinstance<py::tuple>(utils::cast<PyObjectRef>(output_tensors).object_)) {
    MS_LOG(EXCEPTION) << "The output tensors should be a tuple !";
  }
  py::object tuple_obj = utils::cast<PyObjectRef>(output_tensors).object_;
  py::tuple tuple_tensors = py::cast<py::tuple>(tuple_obj);
  RunOpMemoryClear(graph.get());
  MS_LOG(INFO) << "Run op " << op_run_info.op_name << " finish!";
  return tuple_tensors;
}

// compile graph steps
void AscendSession::SelectKernel(const KernelGraph &kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  size_t raise_precision_count = 0;
  size_t reduce_precision_count = 0;
  for (const auto &cnode : kernel_graph.execution_order()) {
    auto status = device::ascend::SelectKernelInfo(cnode);
    if (status == device::ascend::kStatusRaisePrecision) {
      raise_precision_count++;
    } else if (status == device::ascend::kStatusReducePrecision) {
      reduce_precision_count++;
    }
    MS_LOG(INFO) << "Select ApplyKernel: " << cnode->DebugString();
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->execution_mode() == kGraphMode) {
    if (raise_precision_count > 0) {
      MS_LOG(WARNING) << "There has " << raise_precision_count
                      << " node/nodes used raise precision to selected the kernel!";
    }
    if (reduce_precision_count > 0) {
      MS_LOG(WARNING) << "There has " << reduce_precision_count
                      << " node/nodes used reduce precision to selected the kernel!";
    }
  }
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::InitRuntimeResource() {
  MS_LOG(INFO) << "Start!";
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  if (!runtime_instance->Init()) {
    MS_LOG(EXCEPTION) << "Kernel runtime init error.";
  }
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::HardwareOptimize(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  device::ascend::KernelPreBuild(kernel_graph.get());
  MS_LOG(INFO) << "HardwareOptimize start!";
  opt::AscendBackendOptimization(kernel_graph);
  opt::AscendGraphKernelCommonProcess(kernel_graph);
  opt::AscendBackendFuseBasicOpt(kernel_graph, false);
  opt::AscendBackendAddAtomicClean(kernel_graph);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
  MS_LOG(INFO) << "HardwareOptimize Finish!";
}

void AscendSession::AdjustKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  opt::HideNopNode(kernel_graph.get());
  // Insert CLearZero op
  // prepare for next step from json get atomic info
  BuildKernel(kernel_graph);
  device::ascend::KernelBuildPreprocess(kernel_graph.get());
  device::KernelAdjust::GetInstance().InsertSwitchLoop(kernel_graph);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->save_graphs_flag();
  auto save_graphs_path = context_ptr->save_graphs_path();
  if (save_graphs_path.empty()) {
    save_graphs_path = ".";
  }
  if (save_graphs) {
    std::string file_path = save_graphs_path + "/" + "after_adjust_kernel.ir";
    DumpIR(file_path, kernel_graph);
  }
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::RunOpAdjustKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  opt::HideNopNode(kernel_graph.get());
  // Insert CLearZero op
  // prepare for next step from json get atomic info
  BuildKernel(kernel_graph);
  device::ascend::KernelBuildPreprocess(kernel_graph.get());
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::AssignStream(NotNull<KernelGraphPtr> kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  device::ascend::AscendStreamAssign::GetInstance().AssignStream(kernel_graph);
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::AssignLabel(NotNull<KernelGraphPtr> kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  device::ascend::AscendLabelAssign::GetInstance().AssignLabel(kernel_graph);
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::BuildKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  struct timeval start_time, end_time;
  (void)gettimeofday(&start_time, nullptr);
  auto ret = device::ascend::KernelBuild(kernel_graph.get());
  if (!ret) {
    MS_LOG(EXCEPTION) << "Kernel build error.";
  }
  (void)gettimeofday(&end_time, nullptr);
  const uint64_t kUSecondInSecond = 1000000;
  uint64_t cost = kUSecondInSecond * static_cast<uint64_t>(end_time.tv_sec - start_time.tv_sec);
  cost += static_cast<uint64_t>(end_time.tv_usec - start_time.tv_usec);
  MS_LOG(INFO) << "KernelBuild run in  " << PRIu64 << " us " << cost;
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::MemoryAlloc(KernelGraph *kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  MS_EXCEPTION_IF_NULL(kernel_graph);
  opt::RemoveNopNode(kernel_graph);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->AssignMemory(kernel_graph);
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::RunOpMemoryAlloc(const std::vector<tensor::TensorPtr> &input_tensors,
                                     KernelGraph *kernel_graph) const {
  MS_LOG(INFO) << "Start memory alloc!";
  MS_EXCEPTION_IF_NULL(kernel_graph);
  opt::RemoveNopNode(kernel_graph);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->RunOpAssignMemory(input_tensors, kernel_graph);
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::RunOpMemoryClear(KernelGraph *kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->RunOpClearMemory(kernel_graph);
}

void AscendSession::GenerateTaskInfo(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  (void)device::KernelAdjust::GetInstance().StepLoadCtrlInputs(kernel_graph);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  bool ret_ok = runtime_instance->GenTask(kernel_graph.get());
  if (!ret_ok) {
    MS_LOG(EXCEPTION) << "Generate task error!";
  }
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::LoadTask(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  bool ret_ok = runtime_instance->LoadTask(kernel_graph.get());
  if (!ret_ok) {
    MS_LOG(EXCEPTION) << "Load task error!";
  }
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::ExecTask(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  bool ret_ok = runtime_instance->Run(kernel_graph.get());
  if (!ret_ok) {
    MS_LOG(EXCEPTION) << "run task error!";
  }
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::Dump(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  MS_LOG(INFO) << "Start!";
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  (void)runtime_instance->DumpData(kernel_graph.get());
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::ExportChildGraphs(const GraphId graph_id) {
#ifdef ENABLE_DUMP_IR
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->save_graphs_flag();
  if (!save_graphs) {
    return;
  }
  auto save_graphs_path = context_ptr->save_graphs_path();
  if (save_graphs_path.empty()) {
    save_graphs_path = ".";
  }
  if (graph_id == final_graph_id_) {
    const auto &graph_order = GetGraphOrder(final_graph_id_);
    const auto &graph_type = GetGraphOrderType(final_graph_id_);
    for (size_t i = 0; i < graph_order.size(); i++) {
      if (graph_type[i] == BRANCH_END || graph_type[i] == BRANCH_START) {
        continue;
      }
      const auto child_graph = GetGraph(graph_order[i]);
      MS_LOG(DEBUG) << "Start export child graph " << graph_order[i];
      MS_EXCEPTION_IF_NULL(child_graph);
      std::string file_path = save_graphs_path + "/graph_build_" + std::to_string(child_graph->graph_id()) + ".ir";
      DumpIR(file_path, child_graph, true);
      DumpIRProto(child_graph, "vm_build_" + std::to_string(child_graph->graph_id()));
      MS_LOG(DEBUG) << "End export child graph " << graph_order[i];
    }
  }
#endif
}

GraphId AscendSession::SetFinalGraphInput(const std::vector<AnfNodePtr> &args) {
  MS_LOG(INFO) << "Start! Args size " << args.size();
  auto final_graph = NewKernelGraph();
  final_graph_id_ = final_graph->graph_id();
  MS_LOG(INFO) << "Create a new final graph" << final_graph_id_ << " success";
  // init private variables and bind them with final_graph_id
  graph_execute_orders_[final_graph_id_] = std::vector<GraphId>();
  graph_order_types_[final_graph_id_] = std::vector<GraphType>();
  for (const auto &parameter : args) {
    MS_EXCEPTION_IF_NULL(parameter);
    if (!parameter->isa<Parameter>()) {
      MS_LOG(EXCEPTION) << parameter->DebugString() << " is not a parameter type!";
    }
    AnfNodePtr parameter_backend = nullptr;
    // if function return UINT_MAX,the parameter is not exist in child graph
    auto parameter_belong_graph_id = GetGraphIdByNode(parameter);
    if (parameter_belong_graph_id == kInvalidGraphId) {
      parameter_backend = CreateNewParameterFromParameter(parameter, true, final_graph.get());
      final_graph->FrontBackendlMapAdd(parameter, parameter_backend);
      MS_LOG(INFO) << "New parameter" << parameter->DebugString() << "in final_graph";
    } else {
      // parametr is a parameter of child graph
      auto graph = GetGraph(parameter_belong_graph_id);
      MS_EXCEPTION_IF_NULL(graph);
      MS_LOG(INFO) << "Reuse parameter [" << parameter->DebugString() << "] of child graph ["
                   << parameter_belong_graph_id << "]";
      parameter_backend = graph->GetBackendAnfByFrontAnf(parameter);
      // add parameter in backend to final graph inputs
      auto final_graph_inputs = final_graph->MutableInputs();
      MS_EXCEPTION_IF_NULL(final_graph_inputs);
      final_graph_inputs->push_back(parameter_backend);
    }
    MS_EXCEPTION_IF_NULL(parameter_backend);
    MS_LOG(INFO) << "parameter backend " << parameter_backend->DebugString() << " belong_graph_id "
                 << AnfAlgo::GetGraphId(parameter_backend.get());
  }
  MS_LOG(INFO) << "End final_graph_id " << final_graph_id_;
  return final_graph_id_;
}

void AscendSession::RecurseGetSummaryNodes(KernelGraph *graph,
                                           std::map<std::string, std::pair<AnfNodePtr, int>> *summary) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(summary);
  // if final graph have no child graph
  auto graph_order_iter = graph_execute_orders_.find(graph->graph_id());
  if (graph_order_iter == graph_execute_orders_.end()) {
    SessionBasic::GetSummaryNodes(graph);
    auto summary_nodes = graph->summary_nodes();
    (*summary).insert(summary_nodes.begin(), summary_nodes.end());
    return;
  }
  // for every child graph, find summary nodes
  auto graph_order = GetGraphOrder(graph->graph_id());
  for (size_t i = 0; i < graph_order.size(); i++) {
    auto child_graph = GetGraph(graph_order[i]);
    if (child_graph == nullptr) {
      continue;
    }
    SessionBasic::GetSummaryNodes(child_graph.get());
    auto child_graph_summary = child_graph->summary_nodes();
    (*summary).insert(child_graph_summary.begin(), child_graph_summary.end());
    RecurseGetSummaryNodes(child_graph.get(), summary);
  }
  graph->set_summary_nodes(*summary);
}

void AscendSession::GetSummaryNodes(KernelGraph *graph) {
  MS_LOG(DEBUG) << "Update summary Start";
  MS_EXCEPTION_IF_NULL(graph);
  auto summary_nodes = graph->summary_nodes();
  std::map<std::string, std::pair<AnfNodePtr, int>> summary;
  summary.insert(summary_nodes.begin(), summary_nodes.end());
  RecurseGetSummaryNodes(graph, &summary);
  graph->set_summary_nodes(summary);
  MS_LOG(DEBUG) << "Update summary end size: " << summary.size();
}

AnfNodePtr AscendSession::CreateFakeOutput(GraphId fake_graph_id, const AnfNodePtr &true_output) {
  auto fake_graph = GetGraph(fake_graph_id);
  MS_EXCEPTION_IF_NULL(fake_graph);
  auto output_item_with_index = AnfAlgo::VisitKernelWithReturnType(true_output, 0);
  auto create_parameter = [&](const AbstractBasePtr &abstract) -> AnfNodePtr {
    auto parameter = fake_graph->NewParameter();
    MS_EXCEPTION_IF_NULL(parameter);
    parameter->set_abstract(abstract);
    auto new_parameter = fake_graph->NewParameter(parameter);
    // Add new parameter to the graph input of fake_graph to sure that all parameters will be allocated memory.
    auto graph_inputs = fake_graph->MutableInputs();
    MS_EXCEPTION_IF_NULL(graph_inputs);
    graph_inputs->push_back(new_parameter);
    return new_parameter;
  };
  auto create_parameter_from_cnode = [&](const AnfNodePtr &cnode, size_t output_idx) -> AnfNodePtr {
    MS_EXCEPTION_IF_NULL(cnode);
    auto abstract = cnode->abstract();
    MS_EXCEPTION_IF_NULL(abstract);
    // create multiple parameters if is a tuple output real kernel
    if (abstract->isa<abstract::AbstractTuple>()) {
      auto tuple_abstract = abstract->cast<abstract::AbstractTuplePtr>();
      MS_EXCEPTION_IF_NULL(tuple_abstract);
      MS_LOG(INFO) << "Tuple size [" << tuple_abstract->size() << "]";
      return create_parameter((*tuple_abstract)[output_idx]);
    }
    return create_parameter(cnode->abstract());
  };
  if (AnfAlgo::CheckPrimitiveType(output_item_with_index.first, prim::kPrimMakeTuple)) {
    std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
    auto make_tuple = output_item_with_index.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(make_tuple);
    for (size_t i = 1; i < make_tuple->inputs().size(); i++) {
      auto input = make_tuple->inputs()[i];
      make_tuple_inputs.push_back(CreateFakeOutput(fake_graph_id, input));
    }
    return fake_graph->NewCNode(make_tuple_inputs);
  }
  return create_parameter_from_cnode(output_item_with_index.first, output_item_with_index.second);
}

void AscendSession::SetFinalGraphOutput(const AnfNodePtr &node) {
  // get the backend anf node related to the output node of front
  auto output_from_graph_id = GetGraphIdByNode(node);
  auto output_from_graph = GetGraph(output_from_graph_id);
  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(INFO) << "Set the output[" << node->DebugString() << "] of graph[" << output_from_graph_id
               << "] to final graph";
  MS_EXCEPTION_IF_NULL(output_from_graph);
  auto final_graph = GetGraph(final_graph_id_);
  MS_EXCEPTION_IF_NULL(final_graph);
  // if output is from final graph,it remarks no child graph exist
  if (final_graph_id_ == output_from_graph_id) {
    MS_LOG(INFO) << "No child graph,output is " << node->DebugString();
    final_graph->set_output(ConstructOutput({node}, final_graph));
    final_graph->set_executable(false);
    return;
  }
  final_graph->set_output(output_from_graph->output());
}

void AscendSession::SetFinalGraphOutput(const ValuePtr &value) {
  auto value_node = NewValueNode(value);
  auto kernel_info = std::make_shared<device::KernelInfo>();
  value_node->set_kernel_info(kernel_info);
  value_node->set_abstract(abstract::FromValue(value));
  auto final_graph = GetGraph(final_graph_id_);
  MS_EXCEPTION_IF_NULL(final_graph);
  final_graph->set_output(final_graph->NewCNode({NewValueNode(prim::kPrimMakeTuple), value_node}));
  final_graph->set_executable(false);
  MS_LOG(INFO) << "Not anf output[" << value->ToString() << "]";
}

void AscendSession::SetFinalGraphOutput(const VectorRef &vec_output) {
  for (auto &output : vec_output) {
    if (utils::isa<AnfNodePtr>(output)) {
      auto output_anf_node = utils::cast<AnfNodePtr>(output);
      SetFinalGraphOutput(output_anf_node);
    } else if (utils::isa<ValuePtr>(output)) {
      auto value = utils::cast<ValuePtr>(output);
      SetFinalGraphOutput(value);
    } else {
      MS_LOG(EXCEPTION) << "Unknown output type:" << output.ToString();
    }
  }
}

void AscendSession::SetFinalGraphOutput(const BaseRef &output) {
  if (utils::isa<AnfNodePtr>(output)) {
    auto output_anf_node = utils::cast<AnfNodePtr>(output);
    SetFinalGraphOutput(output_anf_node);
  } else if (utils::isa<ValuePtr>(output)) {
    auto value = utils::cast<ValuePtr>(output);
    SetFinalGraphOutput(value);
  } else if (utils::isa<VectorRef>(output)) {
    auto vec_output = utils::cast<VectorRef>(output);
    SetFinalGraphOutput(vec_output);
  } else {
    MS_LOG(EXCEPTION) << "Unknown output type:" << output.ToString();
  }
}

KernelGraphPtr AscendSession::GetGraph(mindspore::GraphId graph_id) {
  auto it = graphs_.find(graph_id);
  if (it == graphs_.end()) {
    MS_LOG(WARNING) << "Can't find graph " << graph_id;
    return nullptr;
  }
  return it->second;
}

void AscendSession::InsertSwitchToGraph(GraphId condition_graph_id, GraphId true_graph_id) {
  MS_LOG(INFO) << "Start!";
  MS_LOG(INFO) << "Condition graph id[" << condition_graph_id << "],true graph id[" << true_graph_id << "]";
  auto condition_graph = GetGraph(condition_graph_id);
  MS_EXCEPTION_IF_NULL(condition_graph);
  tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt32, std::vector<int>{1});
  int32_t *val = nullptr;
  val = static_cast<int32_t *>(tensor->data_c(true));
  MS_EXCEPTION_IF_NULL(val);
  *val = 0;
  auto value_node = std::make_shared<ValueNode>(tensor);
  value_node->set_abstract(abstract::FromValue(tensor, false));
  auto counter_const = condition_graph->NewValueNode(value_node);
  condition_graph->AddValueNodeToGraph(counter_const);
  // create a new switch op
  auto switch_primitive = std::make_shared<Primitive>("StreamSwitch");
  auto cond_output_it = condition_output_.find(condition_graph_id);
  if (cond_output_it == condition_output_.end()) {
    MS_LOG(EXCEPTION) << "Can't find condition graph" << condition_graph_id;
  }
  auto cond_output_kernel =
    AnfAlgo::VisitKernel(condition_graph->GetBackendAnfByFrontAnf(cond_output_it->second), 0).first;
  MS_EXCEPTION_IF_NULL(cond_output_kernel);
  std::vector<AnfNodePtr> inputs = {NewValueNode(switch_primitive), cond_output_kernel, counter_const};
  CNodePtr switch_node = condition_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(switch_node);
  switch_node->set_abstract(std::make_shared<abstract::AbstractNone>());
  AnfAlgo::SetGraphId(condition_graph_id, switch_node.get());
  // set attr: cond_ RT_GREATER
  AnfAlgo::SetNodeAttr(kAttrSwitchCondition, MakeValue<int>(static_cast<int>(RT_GREATER)), switch_node);
  // set attr:data_type
  AnfAlgo::SetNodeAttr(kAttrDataType, MakeValue<int>(static_cast<int>(RT_SWITCH_INT64)), switch_node);
  // set attr:true branch graph id ,which is same to stream distinction label
  AnfAlgo::SetNodeAttr(kAttrTrueBranchStream, MakeValue<uint32_t>(true_graph_id), switch_node);
  // append switch at the end of condition graph
  auto return_node = condition_graph->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  InsertControlDependToGraph(condition_graph_id, return_node->input(1), switch_node);
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::CopyOutputOfIf(GraphId false_graph_id) {
  auto &graph_execute_order = GetGraphOrder(final_graph_id_);
  auto &graph_order_type = GetGraphOrderType(final_graph_id_);
  auto false_index = ExecOrderOfChildGraph(final_graph_id_, false_graph_id);
  if (false_index == kInvalidIndex || false_index == 0) {
    return;
  }
  for (int i = SizeToInt(false_index) - 1; i >= 0; i--) {
    size_t graph_index = IntToSize(i);
    if (graph_index >= graph_execute_order.size()) {
      MS_LOG(EXCEPTION) << "Graph index[" << graph_index << "] out of range[" << graph_execute_order.size() << "]";
    }
    if (graph_order_type[graph_index] == COMMON_GRAPH) {
      auto true_last_id = graph_execute_order[graph_index];
      MS_LOG(INFO) << "The last graph of if true branch is " << true_last_id;
      auto true_last = GetGraph(true_last_id);
      auto final_graph = GetGraph(final_graph_id_);
      MS_EXCEPTION_IF_NULL(final_graph);
      auto false_last = GetGraph(false_graph_id);
      MS_EXCEPTION_IF_NULL(true_last);
      MS_EXCEPTION_IF_NULL(false_last);
      MS_LOG(INFO) << "The last graph of false branch is " << false_graph_id;
      // create fake output
      auto fake_output_graph = NewKernelGraph();
      graph_execute_order.push_back(fake_output_graph->graph_id());
      graph_order_type.push_back(COMMON_GRAPH);
      fake_output_graph->set_output(CreateFakeOutput(fake_output_graph->graph_id(), final_graph->output()));
      final_graph->set_output(fake_output_graph->output());
      InsertMultipleAssignToGraph(true_last_id, true_last->output(), final_graph->output());
      InsertMultipleAssignToGraph(false_graph_id, false_last->output(), final_graph->output());
      // insert stream active for loop sink
      auto context_ptr = MsContext::GetInstance();
      MS_EXCEPTION_IF_NULL(context_ptr);
      if (context_ptr->enable_task_sink() && context_ptr->loop_sink_flag() &&
          ConfigManager::GetInstance().iter_num() > 1) {
        // insert active in true graph, another active will be inserted in kernel adjust
        InsertStreamActiveToGraph(true_last_id, kSecondStreamSwitchLabel);
      }
      break;
    }
  }
}

void AscendSession::SwitchCompile(GraphId cond_graph_id, GraphId true_graph_id, GraphId false_graph_id,
                                  const AnfNodePtr &output) {
  if (switches_.find(cond_graph_id) != switches_.end()) {
    MS_LOG(WARNING) << "Condition graph" << cond_graph_id << " has been set before ";
    return;
  }
  switches_[cond_graph_id] = std::pair<GraphId, GraphId>(true_graph_id, false_graph_id);
  condition_output_[cond_graph_id] = output;
  MS_LOG(INFO) << "New switch compile " << cond_graph_id << " " << true_graph_id << " " << false_graph_id;
  // set the type of condition graph
  auto cond_graph_index = ExecOrderOfChildGraph(final_graph_id_, cond_graph_id);
  auto &graph_order_type = GetGraphOrderType(final_graph_id_);
  if (cond_graph_index >= graph_order_type.size()) {
    MS_LOG(EXCEPTION) << "cond_graph_index " << cond_graph_index << " out of range " << graph_order_types_.size();
  }
  graph_order_type[cond_graph_index] = CONDITION_GRAPH;
  // update distinction label of false graph,update before merge to sure the distinction
  if (false_graph_id != kInvalidGraphId) {
    // false graph and condition in graph same stream
    auto condition_graph = GetGraph(cond_graph_id);
    MS_EXCEPTION_IF_NULL(condition_graph);
    SetStreamDistinctionLabel(GetGraph(false_graph_id), condition_graph->stream_distinction_label(), true);
    // if false graph is a condition graph and has been switch compiled before,it's false should be updated again
    auto cond_it = switches_.find(false_graph_id);
    while (cond_it != switches_.end() && cond_it->second.second != kInvalidGraphId) {
      cond_graph_id = cond_it->first;
      false_graph_id = cond_it->second.second;
      condition_graph = GetGraph(cond_graph_id);
      if (condition_graph == nullptr) {
        continue;
      }
      SetStreamDistinctionLabel(GetGraph(false_graph_id), condition_graph->stream_distinction_label(), true);
      cond_it = switches_.find(false_graph_id);
    }
  }
}  // namespace session

void AscendSession::MergeSwitchCompile() {
  auto graph_execute_order = GetGraphOrder(final_graph_id_);
  auto &graph_order_type = GetGraphOrderType(final_graph_id_);
  for (auto switch_compile : switches_) {
    auto cond_graph_id = switch_compile.first;
    auto true_graph_id = switch_compile.second.first;
    auto false_graph_id = switch_compile.second.second;
    MS_LOG(INFO) << "Switch compile: " << cond_graph_id << " " << true_graph_id << " " << false_graph_id;
    auto condition_graph = GetGraph(cond_graph_id);
    auto final_graph = GetGraph(final_graph_id_);
    MS_EXCEPTION_IF_NULL(condition_graph);
    MS_EXCEPTION_IF_NULL(final_graph);
    // insert switch to condition graph
    InsertSwitchToGraph(cond_graph_id, true_graph_id);
    auto cond_graph_index = ExecOrderOfChildGraph(final_graph_id_, cond_graph_id);
    auto prev_graph_id = kInvalidGraphId;
    // if condition graph is the first graph and final graph has assign op,then the final graph is the common graph
    if (cond_graph_index == 0 && !final_graph->execution_order().empty()) {
      prev_graph_id = final_graph_id_;
      // set the distinction label of final graph
      SetStreamDistinctionLabel(final_graph, final_graph_id_, true);
      // if condition graph is not the first graph
    } else if ((cond_graph_index - 1 < graph_execute_order.size()) &&
               (graph_order_type[cond_graph_index - 1] == COMMON_GRAPH)) {
      prev_graph_id = graph_execute_order[cond_graph_index - 1];
    }
    // insert stream active to common graph
    if (prev_graph_id != kInvalidGraphId) {
      InsertStreamActiveToGraph(prev_graph_id, condition_graph->stream_distinction_label());
    }
    // if this is a 'if' condition
    auto it = while_condition_graphs_.find(cond_graph_id);
    if (it == while_condition_graphs_.end()) {
      CopyOutputOfIf(false_graph_id);
    } else {
      // if it is a while,insert a stream active to true graph
      GraphId from_graph = it->second;
      InsertStreamActiveToGraph(from_graph, condition_graph->stream_distinction_label());
    }
  }
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::InsertAllAssigns() {
  std::vector<std::pair<AnfNodePtr, AnfNodePtr>> assigns;
  for (auto assign : assigns_) {
    auto front_anf = std::get<0>(assign);
    auto to_graph_id = std::get<1>(assign);
    auto input_idx = std::get<2>(assign);
    auto to_graph = GetGraph(to_graph_id);
    MS_EXCEPTION_IF_NULL(to_graph);
    std::vector<AnfNodePtr> graph_inputs = to_graph->inputs();
    if (input_idx >= graph_inputs.size()) {
      MS_LOG(EXCEPTION) << "input_index " << input_idx << " out of range size " << graph_inputs.size();
    }
    auto backend_parameter = graph_inputs[input_idx];
    assigns.emplace_back(std::pair<AnfNodePtr, AnfNodePtr>(front_anf, backend_parameter));
  }
  // erase the repeat assign
  std::set<std::pair<AnfNodePtr, AnfNodePtr>> inserted_nodes;
  for (auto &assign : assigns) {
    auto front_anf = assign.first;
    auto backend_parameter = assign.second;
    auto from_graph_id = GetGraphIdByNode(front_anf);
    auto from_graph = GetGraph(from_graph_id);
    MS_EXCEPTION_IF_NULL(from_graph);
    auto backend_arg = from_graph->GetBackendAnfByFrontAnf(front_anf);
    if (inserted_nodes.find(assign) == inserted_nodes.end()) {
      InsertAssignToGraph(from_graph_id, backend_arg, backend_parameter);
      (void)inserted_nodes.insert(assign);
    }
  }
}

// insert active to graph
void AscendSession::SetActive(GraphId from, GraphId to) {
  if (while_condition_graphs_.find(to) != while_condition_graphs_.end()) {
    MS_LOG(WARNING) << " to " << to << " has been exits in map,from " << from << ",exist from "
                    << while_condition_graphs_[to];
    return;
  }
  MS_LOG(INFO) << "From " << from << " to " << to;
  auto &graph_order = GetGraphOrder(final_graph_id_);
  auto &graph_type = GetGraphOrderType(final_graph_id_);
  std::vector<GraphId> graph_order_new;
  std::vector<GraphType> graph_type_new;
  for (size_t i = 0; i < graph_order.size(); i++) {
    auto graph_id = graph_order[i];
    graph_order_new.push_back(graph_id);
    graph_type_new.push_back(graph_type[i]);
    if (from == graph_id) {
      graph_order_new.push_back(kInvalidGraphId);
      graph_type_new.push_back(BRANCH_END);
    }
  }
  graph_order = graph_order_new;
  graph_type = graph_type_new;
  // set the graph type of condition graph
  graph_type[ExecOrderOfChildGraph(final_graph_id_, to)] = CONDITION_GRAPH;
  // record the condition graph into while condition set
  while_condition_graphs_[to] = from;
}

void AscendSession::SetChildGraphParameter(const AnfNodePtr &front_anf, GraphId to_graph_id, size_t input_idx) {
  MS_LOG(INFO) << "Start!";
  MS_EXCEPTION_IF_NULL(front_anf);
  auto from_graph_id = GetGraphIdByNode(front_anf);
  auto from_graph = GetGraph(from_graph_id);
  MS_EXCEPTION_IF_NULL(from_graph);
  auto to_graph = GetGraph(to_graph_id);
  MS_EXCEPTION_IF_NULL(to_graph);
  std::vector<AnfNodePtr> graph_inputs = to_graph->inputs();
  if (input_idx >= graph_inputs.size()) {
    MS_LOG(EXCEPTION) << "input_index " << input_idx << " out of range size " << graph_inputs.size();
  }
  auto backend_parameter = graph_inputs[input_idx];
  MS_EXCEPTION_IF_NULL(backend_parameter);
  auto backend_arg = from_graph->GetBackendAnfByFrontAnf(front_anf);
  MS_LOG(INFO) << "Set node[" << front_anf->DebugString() << "] of graph[" << from_graph_id << "]to node["
               << backend_parameter->DebugString() << "] of graph[" << AnfAlgo::GetGraphId(backend_parameter.get())
               << "]";
  // a node should not assign to itself
  if (backend_arg.get() == backend_parameter.get()) {
    return;
  }
  // if arg is the the parameter of child graph,it is parameter of final graph too
  if (front_anf->isa<Parameter>()) {
    MS_EXCEPTION_IF_NULL(backend_arg);
    MS_LOG(INFO) << "Reuse node [" << backend_arg->DebugString() << "], old node[" << backend_parameter->DebugString()
                 << "] will be replaced.";
    to_graph->ReplaceNode(NOT_NULL(backend_parameter), NOT_NULL(backend_arg));
    return;
  }
  MS_LOG(INFO) << "Assign of node" << backend_arg->DebugString() << " of graph " << from_graph_id << " to node"
               << backend_parameter->DebugString() << "of graph " << to_graph_id;
  assigns_.emplace_back(std::tuple<AnfNodePtr, GraphId, size_t>(front_anf, to_graph_id, input_idx));
}

void AscendSession::SetChildGraphParameter(const tensor::TensorPtr &front_tensor, GraphId to_graph_id,
                                           size_t input_idx) {
  MS_LOG(INFO) << "Start!";
  std::pair<GraphId, size_t> graph_input_pair(to_graph_id, input_idx);
  initial_tenosrs_[graph_input_pair] = front_tensor;
  MS_LOG(INFO) << "Finish!";
}

void AscendSession::UpdateGraphOrder(GraphId to_graph_id) {
  MS_LOG(INFO) << "to_graph_id " << to_graph_id;
  auto &graph_order = GetGraphOrder(final_graph_id_);
  auto &graph_type = GetGraphOrderType(final_graph_id_);
  for (size_t i = 0; i < graph_order.size(); i++) {
    if (graph_order[i] == to_graph_id) {
      return;
    }
  }
  // if graph is not in graph order,add it to graph order
  SetStreamDistinctionLabel(GetGraph(to_graph_id), to_graph_id, false);
  graph_order.push_back(to_graph_id);
  graph_type.push_back(COMMON_GRAPH);
  for (size_t i = 0; i < graph_order.size(); i++) {
    MS_LOG(INFO) << "Index " << i << ",graph_id " << graph_order[i] << ",graph_type" << graph_type[i];
  }
}

size_t AscendSession::SetChildGraphInput(const KernelGraphPtr &graph, const AnfNodePtr &node, size_t input_index) {
  auto output_num = AnfAlgo::GetOutputTensorNum(node);
  if (output_num > 1 && !AnfAlgo::CheckPrimitiveType(node, prim::kPrimTupleGetItem)) {
    return input_index + output_num;
  }
  auto valid_inputs = graph->valid_inputs();
  if (valid_inputs[input_index]) {
    SetChildGraphParameter(node, graph->graph_id(), input_index);
  } else {
    MS_LOG(DEBUG) << "Invalid input arg: " << node->DebugString();
  }
  return ++input_index;
}

size_t AscendSession::SetChildGraphInput(const KernelGraphPtr &graph, const ValuePtr &value, size_t input_index) {
  MS_EXCEPTION_IF_NULL(value);
  if (!value->isa<Tensor>()) {
    MS_LOG(EXCEPTION) << "Value Node should be a tensor, unexpected value: " << value->ToString();
  }
  SetChildGraphParameter(value->cast<TensorPtr>(), graph->graph_id(), input_index);
  return ++input_index;
}

size_t AscendSession::SetChildGraphInput(const KernelGraphPtr &graph, const VectorRef &vec_args, size_t input_index) {
  auto index = input_index;
  for (auto &arg : vec_args) {
    if (utils::isa<AnfNodePtr>(arg)) {
      // arg is a anf node
      auto node = utils::cast<AnfNodePtr>(arg);
      index = SetChildGraphInput(graph, node, input_index);
    } else if (utils::isa<ValuePtr>(arg)) {
      // arg is a tensor
      auto value = utils::cast<ValuePtr>(arg);
      index = SetChildGraphInput(graph, value, input_index);
    } else {
      MS_LOG(EXCEPTION) << "Unexpected arg type " << arg.ToString();
    }
  }
  return index;
}

void AscendSession::SetChildGraphInput(GraphId g, const VectorRef &args) {
  MS_LOG(INFO) << "Set input of graph " << g;
  auto to_graph = GetGraph(g);
  MS_EXCEPTION_IF_NULL(to_graph);
  DumpGraphInputArgs(args);
  UpdateGraphOrder(g);
  auto &graph_inputs = to_graph->inputs();
  auto real_args = GetRealArgs(to_graph, args);
  size_t input_index = 0;
  for (size_t i = 0; i < real_args.size(); i++) {
    if (input_index >= graph_inputs.size()) {
      MS_LOG(EXCEPTION) << "input_index " << input_index << " out of range size " << graph_inputs.size();
    }
    auto &real_arg = real_args[i];
    if (utils::isa<AnfNodePtr>(real_arg)) {
      // arg is a anf node
      auto node = utils::cast<AnfNodePtr>(real_arg);
      input_index = SetChildGraphInput(to_graph, node, input_index);
    } else if (utils::isa<ValuePtr>(real_arg)) {
      // arg is a tensor
      auto value = utils::cast<ValuePtr>(real_arg);
      input_index = SetChildGraphInput(to_graph, value, input_index);
    } else if (utils::isa<VectorRef>(real_arg)) {
      // arg is a VectorRef
      auto vec_args = utils::cast<VectorRef>(real_arg);
      input_index = SetChildGraphInput(to_graph, vec_args, input_index);
    } else {
      MS_LOG(EXCEPTION) << "Unexpected arg type " << real_arg.ToString();
    }
  }
  MS_LOG(INFO) << "Finish!";
}

GraphId AscendSession::GetGraphIdByNode(const AnfNodePtr &front_anf) const {
  for (const auto &graph_item : graphs_) {
    auto graph = graph_item.second;
    MS_EXCEPTION_IF_NULL(graph);
    // if front_anf is a parameter,the backend parameter may have two
    if (graph->GetBackendAnfByFrontAnf(front_anf) != nullptr) {
      return graph_item.first;
    }
  }
  MS_EXCEPTION_IF_NULL(front_anf);
  MS_LOG(DEBUG) << "front_anf " << front_anf->DebugString() << " is not exist in any graph";
  return kInvalidGraphId;
}

void AscendSession::MergeGraphExecOrder() {
  MS_LOG(INFO) << "Start!";
  // merge graph order
  auto &graph_order = GetGraphOrder(final_graph_id_);
  auto &graph_type = GetGraphOrderType(final_graph_id_);
  auto final_graph = GetGraph(final_graph_id_);
  MS_EXCEPTION_IF_NULL(final_graph);
  if (graph_order.empty()) {
    MS_LOG(WARNING) << "Graph output is a lonely variable not linked to any op!";
    return;
  }
  if (graph_order.size() > 1) {
    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    if (!context_ptr->enable_task_sink()) {
      MS_LOG(EXCEPTION) << "Control sink network should run with task-sink mode!";
    }
  }
  // if first graph is common,the final graph has no label,then set the stream of final graph same with the first graph
  SetStreamDistinctionLabel(final_graph, graph_order[0], false);
  std::vector<CNodePtr> final_exec_order = final_graph->execution_order();
  KernelGraphPtr last_graph = nullptr;
  for (size_t i = 0; i < graph_order.size(); i++) {
    auto graph_id = graph_order[i];
    if (graph_type[i] == BRANCH_END || graph_type[i] == BRANCH_START) {
      continue;
    }
    auto child_graph = GetGraph(graph_id);
    last_graph = child_graph;
    MS_EXCEPTION_IF_NULL(child_graph);
    auto exec_order = child_graph->execution_order();
    MS_LOG(INFO) << "Merge graph,graph_id " << graph_id;
    (void)std::transform(exec_order.begin(), exec_order.end(), std::back_inserter(final_exec_order),
                         [&](CNodePtr node) -> CNodePtr {
                           AnfAlgo::SetStreamDistinctionLabel(child_graph->stream_distinction_label(), node.get());
                           return node;
                         });
    // add all value nodes of child graphs to final graph
    for (auto &value_node : child_graph->graph_value_nodes()) {
      final_graph->AddValueNodeToGraph(value_node);
    }
    // copy ref map to final graph
    auto child_ref_map = child_graph->GetRefMap();
    for (auto &item : child_ref_map) {
      if (final_graph->IsInRefOutputMap(item.first)) {
        MS_LOG(EXCEPTION) << "The ref pair is already in final graph!";
      }
      final_graph->AddRefCorrespondPairs(item.first, item.second);
    }
  }
  // set final_exec_order into final graph
  MS_EXCEPTION_IF_NULL(final_graph);
  DumpGraphExeOrder(final_exec_order);
  final_graph->set_execution_order(final_exec_order);
}

void AscendSession::InsertAssignToGraph(GraphId graph_id, const AnfNodePtr &from, const AnfNodePtr &to) {
  MS_EXCEPTION_IF_NULL(from);
  MS_EXCEPTION_IF_NULL(to);
  if (AnfAlgo::OutputAddrExist(from, 0) && AnfAlgo::OutputAddrExist(to, 0) &&
      AnfAlgo::GetOutputAddr(from, 0) == AnfAlgo::GetOutputAddr(to, 0)) {
    return;
  }
  if (from.get() == to.get()) {
    return;
  }
  MS_LOG(INFO) << "Insert assign to graph " << graph_id << " from " << from->DebugString() << " to "
               << to->DebugString();
  auto graph = graphs_[graph_id];
  MS_EXCEPTION_IF_NULL(graph);
  // config inputs of assign node
  std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>("Assign")), to, from};
  // generate a new cnode
  auto assign_node = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(assign_node);
  assign_node->set_abstract(to->abstract());
  // append the assign at the end of from graph
  InsertDependToGraph(graph_id, assign_node);
}

void AscendSession::InsertMultipleAssignToGraph(GraphId graph_id, const AnfNodePtr &from, const AnfNodePtr &to) {
  std::vector<AnfNodePtr> from_outputs = AnfAlgo::GetAllOutput(from, {prim::kPrimTupleGetItem});
  std::vector<AnfNodePtr> to_outputs = AnfAlgo::GetAllOutput(to, {prim::kPrimTupleGetItem});
  MS_LOG(INFO) << "Insert assigns from [" << AnfAlgo::GetGraphId(from.get()) << "] to ["
               << AnfAlgo::GetGraphId(to.get()) << "]";
  if (from_outputs.size() != to_outputs.size()) {
    MS_LOG(INFO) << "From[" << from->DebugString(5) << "] to[" << to->DebugString(5) << "]";
    MS_LOG(EXCEPTION) << "From outputs size[" << from_outputs.size() << "] is not equal to to outputs size["
                      << to_outputs.size() << "]";
  }
  for (size_t i = 0; i < from_outputs.size(); i++) {
    InsertAssignToGraph(graph_id, from_outputs[i], to_outputs[i]);
  }
}

void AscendSession::InsertStreamActiveToGraph(GraphId graph_id, uint32_t actived_stream) {
  MS_LOG(INFO) << "Insert stream_active from " << graph_id << " to " << actived_stream;
  auto from_graph = GetGraph(graph_id);
  MS_EXCEPTION_IF_NULL(from_graph);
  std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>("StreamActive"))};
  auto active_node = from_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(active_node);
  active_node->set_abstract(std::make_shared<abstract::AbstractNone>());
  // set the active stream id into the attr of active node
  std::vector<uint32_t> active_index_value = {};
  active_index_value.push_back(actived_stream);
  AnfAlgo::SetNodeAttr(kAttrActiveStreamList, MakeValue<std::vector<uint32_t>>(active_index_value), active_node);
  // append the active node at the end of from graph
  auto return_node = from_graph->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  InsertControlDependToGraph(graph_id, return_node->input(1), active_node);
}

void AscendSession::InsertDependToGraph(GraphId graph_id, const AnfNodePtr &attch_node) {
  AscendControlParser::InsertDependToGraph(NOT_NULL(GetGraph(graph_id)), NOT_NULL(attch_node));
}

void AscendSession::InsertControlDependToGraph(GraphId graph_id, const AnfNodePtr &first_node,
                                               const AnfNodePtr &second_node) {
  AscendControlParser::InsertControlDependToGraph(NOT_NULL(GetGraph(graph_id)), NOT_NULL(first_node),
                                                  NOT_NULL(second_node));
}

size_t AscendSession::ExecOrderOfChildGraph(GraphId final_graph, GraphId child_graph) {
  auto &graph_order = GetGraphOrder(final_graph);
  for (size_t i = 0; i < graph_order.size(); i++) {
    if (child_graph == graph_order[i]) {
      return i;
    }
  }
  return kInvalidIndex;
}

std::vector<GraphId> &AscendSession::GetGraphOrder(GraphId final_graph_id) {
  auto graph_order_iter = graph_execute_orders_.find(final_graph_id);
  if (graph_order_iter == graph_execute_orders_.end()) {
    MS_LOG(EXCEPTION) << "Final graph" << final_graph_id << "has no child graph";
  }
  return graph_order_iter->second;
}

// get graph order type vector by graph id
std::vector<GraphType> &AscendSession::GetGraphOrderType(GraphId final_graph_id) {
  auto graph_type_iter = graph_order_types_.find(final_graph_id);
  if (graph_type_iter == graph_order_types_.end()) {
    MS_LOG(EXCEPTION) << "Final graph" << final_graph_id << "has no graph_order_types_";
  }
  return graph_type_iter->second;
}

void AscendSession::SyncInitialTenosrToDevice() {
  for (auto &item : initial_tenosrs_) {
    auto to_graph_id = item.first.first;
    auto input_idx = item.first.second;
    auto front_tensor = item.second;
    auto to_graph = GetGraph(to_graph_id);
    MS_EXCEPTION_IF_NULL(to_graph);
    std::vector<AnfNodePtr> graph_inputs = to_graph->inputs();
    if (input_idx >= graph_inputs.size()) {
      MS_LOG(EXCEPTION) << "input_index " << input_idx << " out of range size " << graph_inputs.size();
    }
    auto backend_parameter = graph_inputs[input_idx];
    // sync data from host to device
    MS_EXCEPTION_IF_NULL(front_tensor);
    size_t tensor_size = front_tensor->data().nbytes();
    auto addr = AnfAlgo::GetOutputAddr(backend_parameter, 0);
    MS_EXCEPTION_IF_NULL(addr);
    if (!addr->SyncHostToDevice(trans::GetRuntimePaddingShape(backend_parameter, 0), tensor_size,
                                front_tensor->data_type(), front_tensor->data_c(false))) {
      MS_LOG(EXCEPTION) << "Tensor SyncHostToDevice fail!";
    }
  }
}

static void ConstructSplitedGraphOutput(const KernelGraphPtr &new_kernel_graph, const std::vector<CNodePtr> &list) {
  // count the output of every anf node
  std::set<AnfNodePtr> has_output_nodes;
  for (auto &anf_node : list) {
    MS_EXCEPTION_IF_NULL(anf_node);
    for (auto &input : anf_node->inputs()) {
      (void)has_output_nodes.insert(input);
    }
  }

  auto make_tuple_primitve = NewValueNode(std::make_shared<Primitive>(prim::kPrimMakeTuple->name()));
  std::vector<AnfNodePtr> make_tuple_inputs = {make_tuple_primitve};
  int output_idx = 0;
  MS_EXCEPTION_IF_NULL(new_kernel_graph);
  for (auto &anf_node : list) {
    if (AnfAlgo::CheckPrimitiveType(anf_node, prim::kPrimReturn)) {
      new_kernel_graph->set_return(anf_node);
    }
    if (has_output_nodes.find(anf_node) == has_output_nodes.end()) {
      MS_LOG(INFO) << "Output[" << output_idx++ << "]:" << anf_node->DebugString();
      make_tuple_inputs.push_back(anf_node);
    }
  }
  if (new_kernel_graph->get_return() == nullptr) {
    new_kernel_graph->set_output(new_kernel_graph->NewCNode(make_tuple_inputs));
  }
}

std::vector<AnfNodePtr> AscendSession::ConstructSplitedGraph(const KernelGraphPtr &new_kernel_graph,
                                                             const std::vector<CNodePtr> &list) {
  MS_EXCEPTION_IF_NULL(new_kernel_graph);
  MS_LOG(INFO) << "start contruct splited kernel graph:" << new_kernel_graph->graph_id();
  MS_LOG(INFO) << "Construct input of kernel graph:" << new_kernel_graph->graph_id();
  std::vector<AnfNodePtr> call_node_inputs;
  std::vector<AnfNodePtr> new_graph_inputs;
  // create new parameter from cnode
  for (auto &anf_node : list) {
    MS_EXCEPTION_IF_NULL(anf_node);
    auto cnode = anf_node->cast<CNodePtr>();
    for (size_t input_idx = 1; input_idx < cnode->inputs().size(); input_idx++) {
      auto input = cnode->inputs()[input_idx];
      MS_EXCEPTION_IF_NULL(input);
      AnfNodePtr new_parameter = nullptr;
      // value node consider move to new graph
      if (input->isa<ValueNode>()) {
        cnode->set_input(input_idx, input);
        continue;
      } else if (input->isa<Parameter>()) {
        // parameter reuse and should attention mulptiple use of one parameter
        cnode->set_input(input_idx, input);
        new_parameter = input;
      } else if (AnfAlgo::GetGraphId(input.get()) != new_kernel_graph->graph_id()) {
        // if is cnode and not in current child graph
        new_parameter = CreateNewParameterFromCNode(input, true, new_kernel_graph.get());
        cnode->set_input(input_idx, new_parameter);
      } else {
        // if is a cnode and in current graph
        continue;
      }
      // if mulptiple use of one parameter or cnode, only set one parameter in graph inputs and one arg in call node
      // args
      if (std::find(call_node_inputs.begin(), call_node_inputs.end(), new_parameter) == call_node_inputs.end()) {
        new_graph_inputs.push_back(new_parameter);
        call_node_inputs.push_back(input);
      }
    }
  }
  // set graph inputs of new graph
  auto graph_inputs = new_kernel_graph->MutableInputs();
  MS_EXCEPTION_IF_NULL(graph_inputs);
  graph_inputs->clear();
  std::copy(new_graph_inputs.begin(), new_graph_inputs.end(), std::back_inserter(*graph_inputs));

  MS_LOG(INFO) << "Construct output of kernel graph:" << new_kernel_graph->graph_id();
  ConstructSplitedGraphOutput(new_kernel_graph, list);
  MS_LOG(INFO) << "end";
  return call_node_inputs;
}

void AscendSession::BackendOptimization(const std::vector<KernelGraphPtr> &all_graphs) {
  MS_LOG(INFO) << "Start BackendCommonOptimization";
  for (auto &graph : all_graphs) {
    opt::BackendCommonOptimization(graph);
  }
  MS_LOG(INFO) << "End.";
}

void AscendSession::SplitGraphs(NotNull<KernelGraphPtr> root_graph) {
  std::set<KernelGraphPtr> memo;
  // if root graph output is a call node ,the root graph is condition graph of 'if' sentence
  auto root_graph_output = AnfAlgo::VisitKernelWithReturnType(root_graph->output(), 0).first;
  if (AnfAlgo::CheckPrimitiveType(root_graph_output, prim::kPrimCall)) {
    SplitGraph(root_graph, {prim::kPrimReturn});
    for (auto &child_graph : root_graph->child_graph_order()) {
      RecurseSplitGraph(NOT_NULL(child_graph), NOT_NULL(&memo));
    }
  } else {
    RecurseSplitGraph(root_graph, NOT_NULL(&memo));
  }
  memo.clear();
  // add maketuple to the end of the last child graph to suit old process
  auto output_graph = root_graph->child_graph_order().empty() ? root_graph : root_graph->child_graph_order().back();
  auto make_tuple = output_graph->NewCNode(
    {NewValueNode(std::make_shared<Primitive>(prim::kPrimMakeTuple->name())), output_graph->output()});
  output_graph->set_output(make_tuple);
  // replace the real input if the real input is a call
  RecurseToUpdateCallRealInput(root_graph, NOT_NULL(&memo));
}

AnfNodePtr AscendSession::BindNewCallToNewGraph(NotNull<KernelGraphPtr> graph,
                                                const std::vector<CNodePtr> &child_graph_list) {
  // if child graph list only has a call ,then return the exist call
  if (child_graph_list.size() == 1 && AnfAlgo::CheckPrimitiveType(child_graph_list[0], prim::kPrimCall)) {
    return child_graph_list[0];
  }
  // create new child graph
  auto child_graph = NewKernelGraph();
  MS_EXCEPTION_IF_NULL(child_graph);
  // create new value node to bind child graph
  auto graph_value_node = graph->NewValueNode(NewValueNode(child_graph));
  std::vector<AnfNodePtr> new_call_input = {NewValueNode(std::make_shared<Primitive>(prim::kPrimCall->name())),
                                            graph_value_node};
  // set the graph id of all node of child graph
  for (auto &child_graph_node : child_graph_list) {
    AnfAlgo::SetGraphId(child_graph->graph_id(), child_graph_node.get());
  }
  auto call_node_args = ConstructSplitedGraph(child_graph, child_graph_list);
  std::copy(call_node_args.begin(), call_node_args.end(), std::back_inserter(new_call_input));
  auto new_call = graph->NewCNode(new_call_input);
  AnfAlgo::SetNodeAttr("graph_id", MakeValue(graph->graph_id()), new_call);
  return new_call;
}

void AscendSession::SplitGraph(NotNull<KernelGraphPtr> graph, const std::set<PrimitivePtr> &cut_prims) {
  MS_LOG(INFO) << "Start,graph_id:" << graph->graph_id();
  auto apply_list = GetCNodes(TopoSort(graph->get_return()));
  // update the root graph child graph order
  AscendControlParser::UpdateChildGraphOrder(graph);
  // get child list from current graph
  std::vector<std::vector<CNodePtr>> child_graph_lists = GetChildList(apply_list, cut_prims);
  if (child_graph_lists.size() > 1) {
    std::list<AnfNodePtr> depend_input = {};
    for (size_t call_index = 0; call_index < child_graph_lists.size(); call_index++) {
      auto call_node = BindNewCallToNewGraph(graph, child_graph_lists[call_index]);
      MS_EXCEPTION_IF_NULL(call_node);
      // if call node is the last call of true graph,no need create child graph after that
      auto child_graphs = AnfAlgo::GetCallNodeKernelGraph(call_node->cast<CNodePtr>());
      depend_input.push_front(call_node);
      if (child_graphs.size() == 1 && child_graphs[0] == graph->parent_graph()) {
        break;
      }
    }
    depend_input.push_front(graph->NewValueNode(NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name()))));
    auto depend = graph->NewCNode(std::vector<AnfNodePtr>(depend_input.begin(), depend_input.end()));
    auto new_return_primitive =
      graph->NewValueNode(NewValueNode(std::make_shared<Primitive>(prim::kPrimReturn->name())));
    graph->set_return(graph->NewCNode({new_return_primitive, depend}));
    AnfNodePtr pre_call_node = nullptr;
    AnfNodePtr cur_call_node = nullptr;
    auto iter = depend_input.begin();
    for (++iter; iter != depend_input.end(); ++iter) {
      pre_call_node = cur_call_node;
      cur_call_node = *iter;
      if (pre_call_node != nullptr && cur_call_node != nullptr) {
        AscendControlParser::InsertControlDependToGraph(graph, NOT_NULL(cur_call_node), NOT_NULL(pre_call_node));
      }
    }
  }
  AscendControlParser::UpdateChildGraphOrder(graph);
  UpdateRealInput(graph);
  MS_LOG(INFO) << "split graph[" << graph->graph_id() << "] end";
  // recurse to split child graph
}

void AscendSession::RecurseSplitGraph(NotNull<KernelGraphPtr> graph, const NotNull<std::set<KernelGraphPtr> *> memo) {
  memo->insert(graph.get());
  SplitGraph(graph, {prim::kPrimCall});
  for (auto &child_graph : graph->child_graph_order()) {
    if (memo->find(child_graph) == memo->end()) {
      RecurseSplitGraph(NOT_NULL(child_graph), memo);
    }
  }
}

void AscendSession::LinkChildGraphs(NotNull<KernelGraphPtr> graph) { AscendControlParser::LinkGraph(graph); }

void AscendSession::RootGraphExecutorValidate(NotNull<KernelGraphPtr> graph) {
  AscendControlParser::ExecutorValidate(graph);
}

void AscendSession::RecurseCompileGraph(NotNull<KernelGraphPtr> graph, const NotNull<std::set<KernelGraphPtr> *> memo) {
  memo->insert(graph.get());
  CompileChildGraph(graph);
  for (auto child_graph : graph->child_graph_order()) {
    if (memo->find(child_graph) != memo->end()) {
      continue;
    }
    RecurseCompileGraph(NOT_NULL(child_graph), memo);
  }
}
}  // namespace session
}  // namespace mindspore
