/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#include "pipeline/jit/pipeline.h"

#include <sstream>
#include <map>
#include <unordered_map>
#include <cstdlib>
#include <algorithm>

#include "ir/param_info.h"
#include "pipeline/jit/pass.h"
#include "pipeline/jit/parse/data_converter.h"
#include "frontend/optimizer/ad/dfunctor.h"
#include "debug/anf_ir_dump.h"
#include "debug/anf_ir_utils.h"
#include "utils/config_manager.h"
#include "utils/convert_utils.h"
#include "utils/context/context_extends.h"
#include "vm/segment_runner.h"
#include "frontend/parallel/context.h"
#include "frontend/parallel/graph_util/get_parallel_info.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "debug/trace.h"
#include "pipeline/pynative/pynative_execute.h"
#include "frontend/optimizer/py_pass_manager.h"
#include "pybind_api/pybind_patch.h"
#include "backend/kernel_compiler/cpu/random_op_cpu_kernel.h"

#if (ENABLE_CPU && (ENABLE_D || ENABLE_GPU))
#include "frontend/parallel/ps/common.h"
#include "frontend/parallel/ps/util.h"
#include "frontend/parallel/ps/worker.h"
#endif

#if (ENABLE_GE || ENABLE_D)
#include "pipeline/jit/pipeline_ge.h"
#include "transform/graph_ir/convert.h"
#include "transform/graph_ir/df_graph_manager.h"
#endif

namespace mindspore {
// namespace to support intermediate representation definition
namespace pipeline {
using Tensor = mindspore::tensor::Tensor;
using MetaTensor = mindspore::tensor::MetaTensor;
using TensorOrderMap = std::map<std::string, std::shared_ptr<Tensor>>;
using mindspore::abstract::AbstractTensor;
using mindspore::abstract::AbstractTensorPtr;
using mindspore::abstract::AbstractTuple;
using mindspore::abstract::AbstractTuplePtr;

const char IR_TYPE_ANF[] = "anf_ir";
const char IR_TYPE_ONNX[] = "onnx_ir";
const char IR_TYPE_MINDIR[] = "mind_ir";

ExecutorPyPtr ExecutorPy::executor_ = nullptr;
std::mutex ExecutorPy::instance_lock_;

std::unordered_map<abstract::AbstractBasePtrList, int, abstract::AbstractBasePtrListHasher,
                   abstract::AbstractBasePtrListEqual>
  g_args_cache;

namespace {
std::string GetBaseNameForIR(int stage_idx, const std::string &action_name) {
  std::ostringstream oss;
  auto ms_context = MsContext::GetInstance();
  if (ms_context == nullptr) {
    MS_LOG(EXCEPTION) << "ms_context is nullptr";
  }
  auto save_graphs_path = ms_context->save_graphs_path();
  if (save_graphs_path.empty()) {
    save_graphs_path = ".";
  }
  oss << save_graphs_path << "/" << stage_idx << "_" << action_name;
  return oss.str();
}
}  // namespace

py::tuple GenerateKey(const std::string &name, const std::unordered_map<std::string, py::object> &defaults) {
  MS_LOG(DEBUG) << "GenerateKey args size:" << defaults.size();
  abstract::AbstractBasePtrList args_spec;

  for (auto arg : defaults) {
    if (py::isinstance<py::module>(arg.second)) {
      MS_LOG(EXCEPTION) << "GenerateKey failed, argument input should not be py::module";
    }
    ValuePtr converted = nullptr;
    if (!parse::ConvertData(arg.second, &converted)) {
      MS_LOG(EXCEPTION) << "GenerateKey convert arg failed";
    }
    args_spec.push_back(abstract::FromValue(converted, true));
  }
  if (g_args_cache.count(args_spec) == 0) {
    static int key = 0;
    MS_LOG(INFO) << "Start new args and compile key:" << key;
    g_args_cache[args_spec] = key++;
  }
  auto argSpec = py::tuple(2);
  argSpec[0] = name;
  argSpec[1] = g_args_cache[args_spec];
  return argSpec;
}

py::bool_ VerifyInputSignature(const py::list input_signature, const py::tuple inputs) {
  MS_LOG(DEBUG) << "Verify args size:" << inputs.size();
  if (inputs.size() != input_signature.size()) {
    MS_LOG(ERROR) << "Signature size not equal to args size";
    return false;
  }

  size_t count = 0;
  for (auto arg_obj : inputs) {
    if (py::isinstance<Tensor>(arg_obj)) {
      MS_LOG(DEBUG) << "Verify Tensor";
      std::shared_ptr<Tensor> m_tensor = arg_obj.cast<std::shared_ptr<Tensor>>();
      if (m_tensor == nullptr) {
        MS_LOG(ERROR) << "Verify Tensor error, get ptr is null";
        return false;
      }
      std::shared_ptr<MetaTensor> sig = input_signature[count].cast<std::shared_ptr<MetaTensor>>();
      std::vector<int> sig_shape = sig->shape();
      TypePtr sig_type = sig->Dtype();

      std::vector<int> tensor_shape = m_tensor->shape_c();
      if (tensor_shape != sig_shape) {
        MS_LOG(ERROR) << "Python input shape is incompatible with input_signature";
        return false;
      }

      if (*m_tensor->Dtype() != *sig_type) {
        MS_LOG(ERROR) << "Python input type(" << m_tensor->Dtype()->ToString() << ") incompatible with input_signature("
                      << sig_type->ToString() << ")";
        return false;
      }
    }
    count++;
  }

  return true;
}

ExecutorPy::ExecutorPy() {}

ResourcePtr ExecutorPy::GetResource(const std::string &phase) {
  MS_LOG(DEBUG) << "Phase size:" << info_.size();
  if (info_.count(phase) == 0) {
    return nullptr;
  }
  return info_[phase]->resource;
}

FuncGraphPtr ExecutorPy::GetFuncGraph(const std::string &phase) {
  if (info_.count(phase) == 0) {
    MS_LOG(EXCEPTION) << "No phase in executor:" << GetPhasePrefix(phase);
  }
  return info_[phase]->func_graph;
}

std::size_t ExecutorPy::ArgListSize(const std::string &phase) {
  if (info_.count(phase) == 0) {
    MS_LOG(EXCEPTION) << "No phase in executor:" << GetPhasePrefix(phase);
  }
  return info_[phase]->arg_list_size;
}

compile::VmEvalFuncPtr ExecutorPy::GetVmEvalFunc(const std::string &phase) {
  ResourcePtr res = GetResource(phase);
  MS_EXCEPTION_IF_NULL(res);
  if (res->results().find(kOutput) != res->results().end() && res->results()[kOutput].is<compile::VmEvalFuncPtr>()) {
    return res->results()[kOutput].cast<compile::VmEvalFuncPtr>();
  }
  MS_LOG(ERROR) << "GetVmEvalFunc vm model can't find kOutput:" << kOutput;
  return nullptr;
}

bool ExecutorPy::HasCompiled(const std::string &phase) const {
  if (info_.count(phase) == 0) {
    return false;
  }
  return true;
}

py::bytes ExecutorPy::GetFuncGraphProto(const std::string &phase, const std::string &ir_type) {
  FuncGraphPtr fg_ptr = GetFuncGraph(phase);
  if (fg_ptr == nullptr) {
    for (auto &item : info_) {
      MS_LOG(DEBUG) << "Phase key is: " << item.first;
    }
    MS_LOG(EXCEPTION) << "Can not find func graph " << phase;
  }

  if (ir_type == IR_TYPE_ANF) {
    std::string proto_str = GetFuncGraphProtoString(fg_ptr);
    if (proto_str.empty()) {
      MS_LOG(EXCEPTION) << "Graph proto is empty.";
    }
    return proto_str;
  }

  if (ir_type == IR_TYPE_ONNX) {
    std::string proto_str = GetOnnxProtoString(fg_ptr);
    if (proto_str.empty()) {
      MS_LOG(EXCEPTION) << "Graph proto is empty.";
    }
    return proto_str;
  }

  if (ir_type == IR_TYPE_MINDIR) {
    std::string proto_str = GetBinaryProtoString(fg_ptr);
    if (proto_str.empty()) {
      MS_LOG(EXCEPTION) << "Graph proto is empty.";
    }
    return proto_str;
  }

  MS_LOG(EXCEPTION) << "Unknown ir type: " << ir_type;
}

py::dict ExecutorPy::GetParameterLayout(const std::string &phase) {
  MS_LOG(DEBUG) << "GetParameterLayout!";
  std::string layout_graph = phase + kStepParallelGraph;
  auto graph = GetFuncGraph(layout_graph);
  return mindspore::parallel::GetParameterLayout(graph);
}

py::dict ExecutorPy::GetCNodeStrategy(const std::string &phase) {
  MS_LOG(DEBUG) << "GetCNodeStrategy!";
  std::string layout_graph = phase + kStepParallelGraph;
  auto graph = GetFuncGraph(layout_graph);
  return mindspore::parallel::GetCNodeStrategy(graph);
}

py::dict ExecutorPy::GetAllreduceFusion(const std::string &phase) {
  MS_LOG(INFO) << "GetAllreduceFusion!";
  auto graph = GetFuncGraph(phase);
  return mindspore::parallel::GetAllreduceFusion(graph);
}

void ExecutorPy::DelNetRes(const std::string &id) {
#ifdef ENABLE_GE
  FinalizeBackend();
#endif
  if (executor_ != nullptr) {
    bool flag = false;
    auto tmp_info = info_;
    for (auto &item : tmp_info) {
      if (item.first.find(id) != string::npos) {
        MS_LOG(DEBUG) << "Delete network res:" << item.first;
        item.second = nullptr;
        (void)info_.erase(item.first);
        flag = true;
      }
    }

    MS_LOG(DEBUG) << "Delete flag:" << flag;
#ifdef ENABLE_GE
    if (flag && info_.size() == 0) {
      // because Ge only support one Session exist at the same time ,so we delete the old one
      transform::DfGraphManager::GetInstance().DeleteGraphRunner();
      transform::DfGraphManager::GetInstance().EraseAnfGraph();
      transform::DfGraphManager::GetInstance().DeleteGeSession();
    }
#endif
  }
}

void ExecutorPy::ClearRes() {
  MS_LOG(INFO) << "Clean executor resource!";
  executor_ = nullptr;
}

ExecutorPy::~ExecutorPy() {
  MS_LOG(INFO) << "Release Executor!";
  ConfigManager::GetInstance().ResetConfig();
}

std::map<std::string, std::pair<PrimitivePyPtr, std::string>> ExecutorPy::FetchInfoForQuantExport(
  const std::string &phase_s) {
  FuncGraphPtr func_graph = info_[phase_s]->resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_LOG(DEBUG) << "FetchInfoForQuantExport func graph(" << func_graph->ToString() << ") phase(" << phase_s << ")!";
  std::map<std::string, std::pair<PrimitivePyPtr, std::string>> fake_quant_table;
  auto filter = [](AnfNodePtr node) {
    return !(IsPrimitiveCNode(node, prim::kPrimConv2D) || IsPrimitiveCNode(node, prim::kPrimMatMul) ||
             IsPrimitiveCNode(node, prim::kPrimDepthwiseConv2dNative));
  };
  std::vector<AnfNodePtr> nodes = DeepScopedGraphSearchWithFilter(func_graph->get_return(), AlwaysInclude, filter);
  auto is_quant_cnode = [](AnfNodePtr node) {
    return IsPrimitiveCNode(node, prim::kPrimFakeQuantPerLayer) ||
           IsPrimitiveCNode(node, prim::kPrimFakeQuantPerChannel);
  };
  for (auto node : nodes) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr || cnode->size() != 3) {
      continue;
    }
    auto x = cnode->input(1);
    auto weight = cnode->input(2);
    if (!is_quant_cnode(weight)) {
      continue;
    }
    // get parameter weight's name
    cnode = weight->cast<CNodePtr>();
    auto weight_node = cnode->input(2);
    if (!weight_node->isa<Parameter>()) {
      continue;
    }
    auto weight_name = weight_node->cast<ParameterPtr>()->name();
    // find the fakequant from input
    int count = 0;
    const int max_depth = 5;
    while (!is_quant_cnode(x)) {
      if (count >= max_depth) {
        break;
      }
      cnode = x->cast<CNodePtr>();
      if (cnode == nullptr || cnode->size() <= 1) {
        break;
      }
      x = cnode->input(1);
      count += 1;
    }
    if (x->isa<Parameter>()) {
      fake_quant_table[weight_name] = std::make_pair(nullptr, "input");
    }
    // get the fakequant parameter minq's name
    if (!is_quant_cnode(x)) {
      continue;
    }
    cnode = x->cast<CNodePtr>();
    if (cnode == nullptr || cnode->size() != 4) {
      continue;
    }
    auto fakequant_min_node = cnode->input(2);
    if (!fakequant_min_node->isa<Parameter>()) {
      continue;
    }
    auto fakequant_min_node_name = fakequant_min_node->cast<ParameterPtr>()->name();
    auto quant_op_value = cnode->input(0)->cast<ValueNodePtr>()->value();
    if (!quant_op_value->isa<PrimitivePy>()) {
      continue;
    }
    auto quant_op = quant_op_value->cast<PrimitivePyPtr>();
    fake_quant_table[weight_name] = std::make_pair(quant_op, fakequant_min_node_name);
  }

  return fake_quant_table;
}

void ExecutorPy::SaveCompiledGraph(const std::string &phase_s) {
  // save the graph to ExecutorPy
  FuncGraphPtr func_graph = info_[phase_s]->resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(parallel::ParallelContext::GetInstance());
  std::string parallel_mode = parallel::ParallelContext::GetInstance()->parallel_mode();

  MS_LOG(INFO) << "Save compiled func graph(" << func_graph->ToString() << ") phase(" << phase_s << ")!";
  info_[phase_s]->func_graph = func_graph;
  if ((func_graph != nullptr) && func_graph->has_flag(parallel::AUTO_PARALLEL) &&
      ((parallel_mode == parallel::AUTO_PARALLEL) || (parallel_mode == parallel::SEMI_AUTO_PARALLEL))) {
    MS_LOG(DEBUG) << "Save model parallel parameter layout graph!";
    func_graph = info_[phase_s]->resource->results()[kStepParallelGraph].cast<FuncGraphPtr>();
    ExecutorInfoPtr executor_info = std::make_shared<ExecutorInfo>();
    std::string layout_graph = phase_s + kStepParallelGraph;
    executor_info->func_graph = func_graph;
    info_[layout_graph] = executor_info;
  } else {
    MS_LOG(DEBUG) << "Save model parallel parameter layout graph null!";
  }
  MS_LOG(INFO) << "End save compiled func graph!";
}

void ExecutorPy::GetGeBackendPolicy() const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::string backend = ms_context->backend_policy();
  if (backend != "ge") {
    MS_LOG(EXCEPTION) << backend << " backend policy is not supported under ge backend!";
  }
}

bool IsPhaseExportAir(const std::string &phase_s) {
  auto phase_to_export = "export.air";
  return phase_s.rfind(phase_to_export) != std::string::npos;
}

std::vector<ActionItem> GetPipline(const ResourcePtr &resource, const std::string &phase_s, bool use_vm) {
  bool is_air = IsPhaseExportAir(phase_s);

  std::string backend = MsContext::GetInstance()->backend_policy();

#if (ENABLE_CPU && (ENABLE_D || ENABLE_GPU))
  if (mindspore::parallel::ps::Util::IsParamServerMode()) {
    mindspore::parallel::ps::Util::SetInternalEnvVar();
  }
  if (parallel::ps::Util::IsRoleOfPServer()) {
    resource->results()[kBackend] = compile::CreateBackend();
    return PServerPipeline();
  }
  if (parallel::ps::Util::IsRoleOfScheduler()) {
    return PSchedulerPipeline();
  }
#endif

  if (use_vm && backend != "ge" && !is_air) {
    // Create backend and session
    auto backend_ptr = compile::CreateBackend();
    // Connect session to debugger
    backend_ptr->SetDebugger();
    resource->results()[kBackend] = backend_ptr;
    return VmPipeline();
  }
  return GePipeline();
}

bool ExecutorPy::CompileInner(const py::object &obj, const py::tuple &args, const py::object &phase, bool use_vm) {
  MS_LOG(DEBUG) << "Start ExecutorPy compile!";
  if ((!py::isinstance<py::str>(phase))) {
    MS_LOG(ERROR) << "Arg phase must be string.";
    return false;
  }
  // check the arg valid?
  if (py::isinstance<py::none>(obj)) {
    MS_LOG(ERROR) << "Find error: parse obj is None.";
    return false;
  }
#ifdef ENABLE_GE
  GetGeBackendPolicy();
#endif
  ExecutorInfoPtr executor_info = std::make_shared<ExecutorInfo>();
  std::string phase_s = py::cast<std::string>(phase);
  MS_LOG(INFO) << "ExecutorPy compile phase:" << phase_s << "!";
  ResourcePtr resource = std::make_shared<Resource>(obj);

  auto p_actions = GetPipline(resource, phase_s, use_vm);
  std::shared_ptr<Pipeline> pip = std::make_shared<Pipeline>(resource, FilterActions(p_actions, phase_s));

  // get the parameters items and add the value to args_spec
  abstract::AbstractBasePtrList args_spec;
  std::size_t size = args.size();
  for (std::size_t i = 0; i < size; i++) {
    ValuePtr converted = nullptr;
    bool succ = parse::ConvertData(args[i], &converted);
    if (!succ) {
      MS_LOG(EXCEPTION) << "Args convert error";
    }
    bool broaden = true;
    args_spec.push_back(abstract::FromValue(converted, broaden));
  }

  resource->set_args_spec(args_spec);
  executor_info->arg_list_size = size;
  executor_info->resource = resource;
  info_[phase_s] = executor_info;
  pip->Run();

  // save the run graph func to MsPipeLine
  SaveCompiledGraph(phase_s);

  resource->Clean();
  // Reclaim all resource used by optimizer;
  ReclaimOptimizer();

  MS_LOG(INFO) << "End ExecutorPy compile!";
  return true;
}

std::vector<ActionItem> ExecutorPy::FilterActions(const std::vector<ActionItem> &actions, const std::string &phase) {
  // filter action after validate when 'export'.
  if (GetPhasePrefix(phase).rfind("export", 0) == std::string::npos) {
    return actions;
  }
  MS_LOG(INFO) << "Phase is '" << phase << "', filter out actions after stage 'validate'";
  std::vector<ActionItem> filtered_actions;
  for (const auto &item : actions) {
    filtered_actions.emplace_back(item);
    if (item.first == "validate") {
      break;
    }
  }
  return filtered_actions;
}

void ExecutorPy::ReleaseResource(const py::object &phase) {
  ResourcePtr res = GetResource(py::cast<std::string>(phase));
  if (res != nullptr) {
    res->Clean();
  }
  // Reclaim all resource used by optimizer;
  ReclaimOptimizer();
}

static std::string PrintArgs(const py::tuple &args) {
  py::print(args);
  return "";
}

bool ExecutorPy::Compile(const py::object &obj, const py::tuple &args, const py::object &phase, bool use_vm) {
  bool ret_value = false;

  try {
    MS_LOG(DEBUG) << PrintArgs(args);
    ret_value = CompileInner(obj, args, phase, use_vm);
  } catch (const py::error_already_set &ex) {
    // print function call stack info before release
    std::ostringstream oss;
    trace::TraceGraphEval();
    trace::GetEvalStackInfo(oss);
    // call py::print to output function call stack to STDOUT, in case of output the log to file, the user can see
    // these info from screen, no need to open log file to find these info
    py::print(oss.str());
    MS_LOG(ERROR) << oss.str();
    ReleaseResource(phase);

    // re-throw this exception to Python interpreter to handle it
    throw(py::error_already_set(ex));
  } catch (const py::type_error &ex) {
    ReleaseResource(phase);
    throw py::type_error(ex);
  } catch (const py::value_error &ex) {
    ReleaseResource(phase);
    throw py::value_error(ex);
  } catch (const py::index_error &ex) {
    ReleaseResource(phase);
    throw py::index_error(ex);
  } catch (const py::attribute_error &ex) {
    ReleaseResource(phase);
    throw py::attribute_error(ex);
  } catch (const std::exception &ex) {
    ReleaseResource(phase);
    // re-throw this exception to Python interpreter to handle it
    throw(std::runtime_error(ex.what()));
  } catch (...) {
    ReleaseResource(phase);
    std::string exName(abi::__cxa_current_exception_type()->name());
    MS_LOG(EXCEPTION) << "Error occurred when compile graph. Exception name: " << exName;
  }

  return ret_value;
}

#ifdef ENABLE_LOAD_ANF_IR
// get MindSpore Intermediate Representation File
std::string GetMsIrFile(void) {
  std::string file;
  const char *path = getenv("MS_IR_FILE");
  if (path == nullptr) {
    return file;
  }

  char real_path[PATH_MAX] = {0};
  if (realpath(path, real_path) == nullptr) {
    MS_LOG(ERROR) << "MS IR path error, " << path;
    return file;
  }
  file = real_path;
  return file;
}

void RunPipelineAction(const ActionItem &action, pipeline::ResourcePtr resource, bool *result) {
  MS_EXCEPTION_IF_NULL(resource);
  MS_EXCEPTION_IF_NULL(result);

  std::string ir_file = GetMsIrFile();
  (void)parse::python_adapter::set_python_scoped();
  if (ir_file.empty()) {
    *result = action.second(resource);
    return;
  }

  // when in loading anf ir mode, action `parse` do nothing
  if (action.first == "parse") {
    return;
  }

  // load MindSpore IR from file
  if (action.first == "symbol_resolve") {
    MS_LOG(DEBUG) << action.first << " read ir file: " << ir_file;
    std::vector<FuncGraphPtr> graphs = ImportIR(ir_file);
    if (graphs.size() == 0) {
      MS_LOG(EXCEPTION) << action.first << " read ir file " << ir_file << " failed as no graph found";
    }
    auto manager = resource->manager();
    MS_EXCEPTION_IF_NULL(manager);
    for (auto &graph : graphs) {
      manager->AddFuncGraph(graph);
    }
    resource->set_func_graph(graphs[0]);
    return;
  }

  // do normal action when not in `parse` and `symbol_resolve` stage
  *result = action.second(resource);
}
#endif

void Pipeline::Run() {
  MS_LOG(INFO) << "Pipeline run";
  MS_EXCEPTION_IF_NULL(resource_);
  FuncGraphPtr user_graph = nullptr;

  WITH(MsProfile::GetProfile())[&user_graph, this]() {
    int i = 0;
    for (auto &action : actions_) {
#ifdef ENABLE_TIMELINE
      DumpTime &dump_time = DumpTime::GetInstance();
      dump_time.Record(action.first, GetTime(), true);
#endif
      bool result = true;
      WITH(MsProfile::GetProfile()->Step(action.first))[&result, &action, this]() {
        MS_LOG(DEBUG) << "Action " << action.first << " start ...";
#ifdef ENABLE_LOAD_ANF_IR
        RunPipelineAction(action, resource_, &result);
#else
        result = action.second(resource_);
#endif
        MS_LOG(DEBUG) << "Action " << action.first << " end.";
      };
      if (!result) {
        MS_LOG(EXCEPTION) << "Pipeline running to end, failed in step:" << action.first;
      }
      if (MsContext::GetInstance()->save_graphs_flag() && resource_->func_graph() != nullptr) {
        auto graph = resource_->func_graph();
        if (graph != nullptr) {
          user_graph = graph;
          std::string base_name = GetBaseNameForIR(i, action.first);

          // generate IR file in dot format, which can be converted to svg file using graphviz dot command
          draw::Draw(base_name + ".dot", graph);
          // generate IR file in human readable format
          DumpIR(base_name + ".ir", graph);
          // generate IR file in a heavily commented format, which can also be reloaded
          ExportIR(base_name + ".dat", std::to_string(i), graph);
        }
#ifdef MS_DEBUG
        // Dump graph cnode list
        MS_LOG(INFO) << "Show CNode list after " << action.first;
        graph->DumpCNodeList();
#endif
      }
      if (resource_->func_graph() != nullptr) {
        auto func_graph = resource_->func_graph();
        if (func_graph->has_flag(GRAPH_FLAG_HAS_EFFECT)) {
          func_graph->EraseUnusedNodeInOrder();
          func_graph->CheckOrder();
          for (auto fg : func_graph->func_graphs_used_total()) {
            MS_LOG(DEBUG) << "Check order graph " << fg->ToString() << ".";
            fg->EraseUnusedNodeInOrder();
            fg->CheckOrder();
          }
        }
      }
      i++;
#ifdef ENABLE_TIMELINE
      dump_time.Record(action.first, GetTime(), false);
#endif
    }
  };
#ifdef ENABLE_PROFILE
  MsProfile::Print();
  MsProfile::Reset();
#endif

  if (MsContext::GetInstance()->save_graphs_flag() && (user_graph != nullptr)) {
    std::string user_graph_file = GetFilePathName("ModelDigraph.dot");
    MS_LOG(DEBUG) << "Save user graph to: " << user_graph_file;
    draw::DrawUserFuncGraph(user_graph_file, user_graph);
  }
  MS_LOG(INFO) << "End";
}

void ProcessVmArgInner(const py::tuple &args, const ResourcePtr &res, VectorRef *const arg_list) {
  std::size_t size = args.size();

  for (std::size_t i = 0; i < size; i++) {
    py::object arg = args[i];
    auto ms_context = MsContext::GetInstance();
    if (ms_context->backend_policy() == kMsConvert && py::isinstance<py::array>(arg)) {
      MS_LOG(EXCEPTION) << "The " << i << "th arg is numpy array, not tensor.";
    }
    ValuePtr converted = nullptr;
    bool succ = parse::ConvertData(arg, &converted);
    if (!succ) {
      MS_LOG(EXCEPTION) << "The " << i << "th arg convert failed.";
    }
    if (MsContext::GetInstance()->execution_mode() == 0 && !converted->isa<tensor::Tensor>()) {
      MS_EXCEPTION(TypeError) << "For 'graph mode', the " << i << "th arg: " << converted->ToString()
                              << " is not tensor.";
    }
    arg_list->push_back(converted);
  }

  MS_EXCEPTION_IF_NULL(res);
  auto graph = res->func_graph();
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> graph_params = graph->parameters();
  std::size_t graph_params_size = graph_params.size();
  if ((*arg_list).size() != graph_params_size) {
    // maybe some default parameter
    for (std::size_t i = (*arg_list).size(); i < graph_params_size; i++) {
      MS_EXCEPTION_IF_NULL(graph_params[i]);
      auto param_ptr = (graph_params[i])->cast<ParameterPtr>();
      if (!param_ptr->has_default()) {
        MS_LOG(EXCEPTION) << "Parameter[" << i << "] has no default param";
      }
      if (!param_ptr->default_param()->isa<Tensor>()) {
        MS_LOG(EXCEPTION) << "Parameter[" << param_ptr->ToString()
                          << "] is not initialized, need to call `.init_data()`";
      }
      arg_list->push_back(param_ptr->default_param());
    }
  }
}

void ExecutorPy::ProcessVmArg(const py::tuple &args, const std::string &phase, VectorRef *const arg_list) {
  ProcessVmArgInner(args, GetResource(phase), arg_list);
}

py::object ExecutorPy::Run(const py::tuple &args, const py::object &phase) {
  std::size_t size = args.size();
  if (!py::isinstance<py::str>(phase)) {
    MS_LOG(EXCEPTION) << "Run failed, phase input is not a str";
  }
  auto phase_s = py::cast<std::string>(phase);
  std::string backend = MsContext::GetInstance()->backend_policy();
#ifdef ENABLE_GE
  if (backend == "ge") {
    return ExecDFGraph(info_, args, phase_s);
  }
#else
  if (backend == "ms" || backend == "ge") {
    auto ret_val = std::make_shared<py::object>();
    if (info_.count(phase_s) != 0 && info_[phase_s]->func_graph != nullptr) {
      if (IsGraphOutputValueNodeOrParameter(info_[phase_s]->func_graph->output(), args, ret_val)) {
        return *ret_val;
      }
    }
    if (backend == "ge") {
      if (args.size() > 0) {
        return args[0];
      }
      return args;
    }
  }
#endif
  std::size_t full_arg_size = ArgListSize(phase_s);
  if (size > full_arg_size) {
    MS_LOG(WARNING) << "The arg num : size = " << size << ". full_arg_size = " << full_arg_size;
  }
  VectorRef arg_list;
  ProcessVmArg(args, phase_s, &arg_list);

  compile::VmEvalFuncPtr run = GetVmEvalFunc(phase_s);
  if (run == nullptr) {
    MS_LOG(EXCEPTION) << "Can't find run graph func for " << phase_s;
  }

  MS_LOG(DEBUG) << "Eval run" << backend;
  BaseRef value = (*run)(arg_list);
  MS_LOG(DEBUG) << "Run end";
  return BaseRefToPyData(value);
}

FuncGraphPtr ExecutorPy::BuildGraph(const py::dict &init_params, const std::string &phase,
                                    const py::object &broadcast_params) {
#if (ENABLE_GE || ENABLE_D)
  return BuildDFGraph(info_, init_params, phase, broadcast_params);
#else
  return nullptr;
#endif
}

void ExecutorPy::UpdataParamNodeDefaultInput(const std::string &phase,
                                             const std::unordered_map<std::string, tensor::TensorPtr> &params_value) {
  FuncGraphPtr func_graph = info_[phase]->resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_LOG(DEBUG) << "UpdataParamNodeDefaultInput for func graph(" << func_graph->ToString() << ") phase(" << phase
                << ")!";
  auto &params = func_graph->parameters();
  for (const auto &param : params) {
    MS_EXCEPTION_IF_NULL(param);
    auto param_cast = param->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_cast);
    auto iter = params_value.find(param_cast->name());
    if (iter != params_value.end()) {
      param_cast->set_default_param(iter->second);
    }
  }
}

void ExecutorPy::RunInitGraph(const py::dict &init_params, const std::string &phase) {
#if ENABLE_GE
  RunGEInitGraph(init_params, phase);
#endif
}

bool InitExecDataset(const std::string &queue_name, int64_t iter_num, int64_t batch_size,
                     const std::vector<TypePtr> &types, const std::vector<std::vector<int64_t>> &shapes,
                     const std::vector<int64_t> &input_indexes, const std::string &phase, bool need_run) {
  std::string name = MsContext::GetInstance()->backend_policy();
#ifndef NO_DLIB
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!context::IsTsdOpened(ms_context) || !context::IsGeInited(ms_context)) {
    (void)InitBackend();
  }
#endif
  if (iter_num == -1) {
    iter_num = INT32_MAX;
  }
  if (name == kMsConvert || name == kMsVm) {
    return InitExecDatasetVm(queue_name, iter_num, batch_size, types, shapes, input_indexes, need_run);
  }
#if ENABLE_GE
  return InitExecDatasetGe(queue_name, iter_num, batch_size, types, shapes, input_indexes, phase);
#else
  std::string backend = MsContext::GetInstance()->backend_policy();
  if (backend == "ge") {
    return true;
  }
#endif
  return false;
}

bool InitExecDatasetVm(const std::string &queue_name, int64_t size, int64_t batch_size,
                       const std::vector<TypePtr> &types, const std::vector<std::vector<int64_t>> &shapes,
                       const std::vector<int64_t> &input_indexes, bool need_run) {
  MS_LOG(INFO) << "Start InitDataSet Entry";
  std::vector<int> int_input_indexes;
  (void)std::transform(input_indexes.begin(), input_indexes.end(), std::back_inserter(int_input_indexes),
                       [](int64_t item) { return static_cast<int>(item); });
  std::vector<std::vector<int>> int_shapes;
  (void)std::transform(shapes.begin(), shapes.end(), std::back_inserter(int_shapes),
                       [](const std::vector<int64_t> &item) {
                         std::vector<int> vector_item;
                         (void)std::transform(item.begin(), item.end(), std::back_inserter(vector_item),
                                              [](int64_t inner_item) { return static_cast<int>(inner_item); });
                         return vector_item;
                       });
  auto p_init = std::make_shared<Primitive>("InitDataSetQueue");
  p_init->set_attr("queue_name", MakeValue(queue_name));
  p_init->set_attr("size", MakeValue(static_cast<int>(size)));
  p_init->set_attr("batch_size", MakeValue(static_cast<int>(batch_size)));
  p_init->set_attr("types", MakeValue(types));
  p_init->set_attr("shapes", MakeValue(int_shapes));
  p_init->set_attr("input_indexes", MakeValue(int_input_indexes));

  const std::vector<std::string> empty_str_list;
  p_init->set_attr("input_names", MakeValue(empty_str_list));
  p_init->set_attr("output_names", MakeValue(empty_str_list));

  FuncGraphPtr func_graph = std::make_shared<FuncGraph>();
  auto app_init = std::make_shared<CNode>(AnfNodePtrList{NewValueNode(p_init)}, func_graph);
  func_graph->set_output(app_init);
  auto manager = MakeManager();
  manager->AddFuncGraph(func_graph);

  // AbstractNone indicates there is no output for this apply node.
  auto abstract_none = std::make_shared<abstract::AbstractNone>();
  app_init->set_abstract(abstract_none);

  auto backend = compile::CreateBackend();
  MS_EXCEPTION_IF_NULL(backend);
  auto convert_fn = backend->convert_fn();
  MS_EXCEPTION_IF_NULL(convert_fn);
  // Convert CNodeList to LinConvertResult.
  ConfigManager::GetInstance().set_iter_num(1);
  auto runner = convert_fn({app_init}, "");
  if (MsContext::GetInstance()->execution_mode() != kPynativeMode) {
    backend->Link(runner.graph_id);
  }
  ConfigManager::GetInstance().set_iter_num(size);

  if (!(*runner.run)) {
    // empty function
    MS_LOG(EXCEPTION) << "Backend " << backend->name() << " unsupported tdt dataset.";
  }

  // launch init dataset runner without inputs and outputs
  VectorRef args;
  auto fn = runner.run;
  if (need_run) {
    (void)(*fn)(args);
  }
  MS_LOG(DEBUG) << "InitDataSetVm End.";
  return true;
}

bool InitRandomNormal(float mean, float stddev, std::vector<int64_t> out_shape, int64_t seed,
                      const py::object &output_tensor) {
  if (out_shape.size() == 0) {
    std::cout << "output data shape is error" << std::endl;
  }
  int64_t total_count = 1;
  for (uint32_t i = 0; i < out_shape.size(); i++) {
    total_count *= out_shape[i];
  }
  uint32_t thread_num = 16;
  if (total_count <= thread_num) {
    thread_num = 1;
  }
  auto temp = py::cast<std::shared_ptr<Tensor>>(output_tensor);
  float *start_ptr = reinterpret_cast<float *>(temp->data_c());
  if (start_ptr == nullptr) {
    std::cout << "start_ptr is nullptr" << std::endl;
    return false;
  }
  int64_t batchSize = total_count / thread_num;
  std::vector<std::thread> threads(thread_num);
  mindspore::kernel::PhiloxGenerator generator = mindspore::kernel::PhiloxGenerator(seed);
  if (thread_num != 1) {
    for (uint32_t i = 0; i < thread_num - 1; i++) {
      float *offset_ptr = start_ptr + batchSize * i;
      threads[i] = std::thread(mindspore::kernel::FillRandoms<
                                 mindspore::kernel::NormalDistribution<mindspore::kernel::PhiloxGenerator, float>>,
                               generator, offset_ptr, batchSize, i);
    }
    float *offset_ptr = start_ptr + batchSize * (thread_num - 1);
    threads[thread_num - 1] = std::thread(
      mindspore::kernel::FillRandoms<mindspore::kernel::NormalDistribution<mindspore::kernel::PhiloxGenerator, float>>,
      generator, offset_ptr, total_count - (thread_num - 1) * batchSize, thread_num - 1);
  } else {
    threads[0] = std::thread(
      mindspore::kernel::FillRandoms<mindspore::kernel::NormalDistribution<mindspore::kernel::PhiloxGenerator, float>>,
      generator, start_ptr, total_count, 0);
  }
  for (uint32_t i = 0; i < thread_num; i++) {
    threads[i].join();
  }
  return true;
}

void ResetOpId() { mindspore::id_generator::reset_id(); }

void InitHccl() {
#ifdef ENABLE_GE
  (void)InitBackend();
#else
  mindspore::parse::python_adapter::set_python_env_flag(true);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  (void)context::OpenTsd(ms_context);
  uint32_t device_id = ms_context->device_id();
  std::string device_name = ms_context->device_target();
  ms_context->set_enable_hccl(true);
  if (ms_context->backend_policy() == "ms" && ms_context->device_target() == kAscendDevice) {
    auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(device_name, device_id);
    MS_EXCEPTION_IF_NULL(runtime_instance);
    if (!runtime_instance->Init()) {
      MS_LOG(ERROR) << "Kernel runtime init error.";
      return;
    }
  }
#endif
}

void FinalizeHccl() {
#ifdef ENABLE_GE
  (void)FinalizeBackend();
#else
  device::KernelRuntimeManager::Instance().ClearRuntimeResource();
#endif
}

void ExportGraph(const std::string &file_name, const std::string &, const std::string &phase) {
#if (ENABLE_GE || ENABLE_D)
  ExportDFGraph(file_name, phase);
#else
  MS_EXCEPTION(ValueError) << "Only MindSpore with Ascend backend support exporting file in 'AIR' format.";
#endif
}

void ReleaseGeTsd() {
  auto context_ptr = MsContext::GetInstance();
  if (context_ptr != nullptr) {
    (void)context::FinalizeGe(context_ptr, true);
    (void)context::CloseTsd(context_ptr, true);
  }
}

void InitBackend() {
  // set python env flag
  mindspore::parse::python_adapter::set_python_env_flag(true);
  // open tsd before ge initialize
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!context::OpenTsd(ms_context)) {
    MS_LOG(EXCEPTION) << "Open tsd failed";
  }
  (void)context::InitGe(ms_context);
}

void FinalizeBackend() {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  (void)context::FinalizeGe(context_ptr);
  (void)context::CloseTsd(context_ptr);
}

void ClearResAtexit() {
  MS_LOG(DEBUG) << "Pipeline clear all resource";
  pynative::ClearPyNativeSession();
  session::ClearPythonParasMap();
  device::KernelRuntimeManager::Instance().ClearRuntimeResource();
#if (ENABLE_CPU && (ENABLE_D || ENABLE_GPU))
  if (mindspore::parallel::ps::Util::IsParamServerMode()) {
    if (parallel::ps::Util::IsRoleOfWorker()) {
      parallel::ps::Worker<float>::GetInstance().Finalize();
    }
  }
#endif
  ad::g_k_prims.clear();

  abstract::ClearPrimEvaluatorMap();
  compile::ClearConvertCache();
  pipeline::GetMethodMap().clear();
  pipeline::GetAttrMap().clear();
  pipeline::ExecutorPy::ClearRes();
  pipeline::ReclaimOptimizer();
  pynative::PynativeExecutor::GetInstance()->ClearRes();
  opt::python_pass::PyPassManager::GetInstance()->ClearRes();
#ifdef ENABLE_GE
  transform::DfGraphManager::GetInstance().ClearGraph();
  transform::DfGraphConvertor::get_adpt_map().clear();
#endif
  ReleaseGeTsd();
  parse::python_adapter::ResetPythonScope();
}
}  // namespace pipeline
}  // namespace mindspore
