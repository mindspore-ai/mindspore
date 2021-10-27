/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include <memory>
#include <sstream>
#include <map>
#include <unordered_map>
#include <cstdlib>
#include <algorithm>
#include <iomanip>

#include "ir/param_info.h"
#include "pipeline/jit/pass.h"
#include "pipeline/jit/parse/data_converter.h"
#include "frontend/optimizer/ad/dfunctor.h"
#include "pipeline/jit/static_analysis/async_eval_result.h"
#include "debug/anf_ir_dump.h"
#include "debug/dump_proto.h"
#include "debug/anf_ir_utils.h"
#include "debug/common.h"
#include "utils/config_manager.h"
#include "utils/convert_utils.h"
#include "utils/convert_utils_py.h"
#include "utils/context/context_extends.h"
#include "vm/segment_runner.h"
#include "frontend/parallel/context.h"
#include "frontend/parallel/graph_util/get_parallel_info.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "backend/session/executor_manager.h"
#include "debug/trace.h"
#include "debug/draw.h"
#include "pipeline/pynative/pynative_execute.h"
#include "frontend/optimizer/py_pass_manager.h"
#include "pybind_api/pybind_patch.h"
#include "utils/shape_utils.h"
#include "utils/info.h"
#include "load_mindir/load_model.h"
#include "frontend/optimizer/ad/prim_bprop_optimizer.h"
#include "runtime/hardware/device_context_manager.h"
#include "utils/crypto.h"
#include "utils/comm_manager.h"
#if ((defined ENABLE_CPU) && (!defined _WIN32))
#include "ps/constants.h"
#include "ps/util.h"
#include "ps/worker.h"
#include "ps/ps_cache/ps_data/ps_data_prefetch.h"
#include "ps/ps_cache/ps_cache_manager.h"
#include "fl/server/server.h"
#include "fl/worker/fl_worker.h"
#endif

#if ((defined ENABLE_GE) || (defined ENABLE_D))
#include "pipeline/jit/pipeline_ge.h"
#include "transform/graph_ir/convert.h"
#include "transform/graph_ir/df_graph_manager.h"
#include "transform/graph_ir/op_adapter_map.h"
#include "runtime/device/ascend/profiling/profiling_manager.h"
#include "runtime/device/ascend/distribute/ascend_collective.h"
#endif
#ifdef ENABLE_DUMP_IR
#include "debug/rdr/running_data_recorder.h"
#include "debug/rdr/recorder_manager.h"
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

#ifdef ENABLE_D
#ifndef ENABLE_SECURITY
using mindspore::device::ascend::ProfilingManager;
#endif
using HcclCollectiveGroup = mindspore::device::ascend::collective::HcclCollectiveGroup;
#endif

const char IR_TYPE_ANF[] = "anf_ir";
const char IR_TYPE_ONNX[] = "onnx_ir";
const char IR_TYPE_MINDIR[] = "mind_ir";

GraphExecutorPyPtr GraphExecutorPy::executor_ = nullptr;
std::mutex GraphExecutorPy::instance_lock_;
#ifdef ENABLE_DEBUGGER
bool GraphExecutorPy::debugger_terminate_ = false;
bool GraphExecutorPy::exit_success_ = false;
#endif

std::unordered_map<abstract::AbstractBasePtrList, uint64_t, abstract::AbstractBasePtrListHasher,
                   abstract::AbstractBasePtrListEqual>
  g_args_cache;

namespace {
constexpr char kCompileCacheFilePath[] = "compile_cache.mindir";
#ifdef ENABLE_DUMP_IR
std::string GetBaseNameForIR(int64_t stage_idx, const std::string &action_name) {
  std::ostringstream oss;
  int spaces = 2;
  oss << std::setfill('0') << std::setw(spaces) << stage_idx << "_" << action_name;
  return oss.str();
}
#endif

AbstractBasePtr ArgsToAbstract(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  bool broaden = value->isa<MetaTensor>() ||
                 (MsContext::GetInstance()->get_param<bool>(MS_CTX_GRAD_FOR_SCALAR) && value->isa<Scalar>());

  return abstract::FromValue(value, broaden);
}

bool CheckArgValid(const py::handle &arg) {
  if (py::isinstance<py::list>(arg) || py::isinstance<py::tuple>(arg)) {
    auto vector_arg = py::cast<py::list>(arg);
    return std::all_of(vector_arg.begin(), vector_arg.end(), CheckArgValid);
  }

  if (py::isinstance<py::dict>(arg)) {
    auto dict_arg = py::cast<py::dict>(arg);
    return std::all_of(dict_arg.begin(), dict_arg.end(), [](const auto &pair) { return CheckArgValid(pair.second); });
  }

  return py::isinstance<py::int_>(arg) || py::isinstance<py::float_>(arg) || py::isinstance<py::none>(arg) ||
         py::isinstance<Number>(arg) || (py::isinstance<Tensor>(arg) && !py::hasattr(arg, "__parameter__"));
}

std::string GetCompileExceptionInfo() {
  std::ostringstream oss;
  trace::GetTraceStackInfo(oss);
  return oss.str();
}

void SetLoopCount(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  auto func_graph = resource->func_graph();
  if (func_graph != nullptr && func_graph->manager() != nullptr) {
    auto manager = func_graph->manager();
    size_t graph_nums = manager->func_graphs().size();
    int64_t loop_size = ConfigManager::GetInstance().iter_num();
    const auto context_ptr = MsContext::GetInstance();
    if (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice) {
      resource->set_vm_loop(!context_ptr->get_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK), loop_size);
    } else if (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kGPUDevice) {
      bool run_with_mind_rt = graph_nums == 1 || context_ptr->get_param<bool>(MS_CTX_ENABLE_MINDRT);
      resource->set_vm_loop(!run_with_mind_rt, loop_size);
    }
    MS_LOG(INFO) << "Change vm_loop_flag to " << resource->vm_loop_flag() << ", set loop_size to " << loop_size;
  }
}

void GetCachedFuncGraph(const ResourcePtr &resource, const std::string &queue_name) {
  MS_EXCEPTION_IF_NULL(resource);
  auto realpath = Common::CreatePrefixPath(kCompileCacheFilePath);
  if (!realpath.has_value()) {
    MS_LOG(EXCEPTION) << "Get real path failed. filename=" << kCompileCacheFilePath;
  }
  std::ifstream f(realpath.value());
  bool cache_file_existed = f.good();
  f.close();
  if (!cache_file_existed) {
    MS_LOG(WARNING) << "The compilation cache file '" << realpath.value()
                    << "' dose not exist. Execute all the compilation actions.";
    return;
  }
  MS_LOG(INFO) << "Use the compilation cache \"" << realpath.value() << "\" and execute the backend actions only.";
  FuncGraphPtr fg = mindspore::LoadMindIR(realpath.value());
  if (fg == nullptr) {
    MS_LOG(EXCEPTION) << "Failed to load the compilation cache file: " << realpath.value();
  }
  FuncGraphManagerPtr mng = fg->manager();
  if (mng == nullptr) {
    auto res_mng = resource->manager();
    MS_EXCEPTION_IF_NULL(res_mng);
    res_mng->AddFuncGraph(fg);
    fg->set_manager(res_mng);
  }
  auto cnodes = fg->GetOrderedCnodes();
  for (auto cnode : cnodes) {
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (prim != nullptr && prim->HasAttr("shared_name")) {
      prim->set_attr("shared_name", MakeValue(queue_name));
      break;
    }
  }
  resource->set_func_graph(fg);
}

void CacheFuncGraph(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  auto realpath = Common::CreatePrefixPath(kCompileCacheFilePath);
  if (!realpath.has_value()) {
    MS_LOG(EXCEPTION) << "Get real path failed. filename=" << kCompileCacheFilePath;
  }

  ChangeFileMode(realpath.value(), S_IRWXU);
  std::ofstream fout(realpath.value());
  if (!fout.is_open()) {
    MS_LOG(EXCEPTION) << "Open cache file '" << realpath.value() << "' failed!" << ErrnoToString(errno);
  }
  FuncGraphPtr fg = resource->func_graph();
  mind_ir::ModelProto fg_model = GetBinaryProto(fg, true);
  if (!fg_model.SerializeToOstream(&fout)) {
    MS_LOG(EXCEPTION) << "Failed to cache the graph to file " << realpath.value();
  }
  fout.close();
  ChangeFileMode(realpath.value(), S_IRUSR);
}
}  // namespace

void CheckArgsValid(const py::tuple &args) {
  for (size_t i = 0; i < args.size(); i++) {
    if (!CheckArgValid(args[i])) {
      MS_EXCEPTION(TypeError)
        << "The inputs types of the outermost network support bool, int, float, None, tensor, "
           "mstype.Number(mstype.bool, mstype.int, mstype.float, mstype.uint), "
           "and tuple or list containing only these types, and dict whose values are these types, but the "
        << i << "th arg type is " << args[i].get_type() << ", value is '" << py::str(args[i]) << "'.";
    }
  }
}

py::object GenerateArgumentsKey(const std::unordered_map<std::string, py::object> &args) {
  MS_LOG(DEBUG) << "GenerateArgumentsKey args size:" << args.size();
  abstract::AbstractBasePtrList args_spec;

  for (const auto &arg : args) {
    if (py::isinstance<py::module>(arg.second)) {
      MS_LOG(EXCEPTION) << "GenerateArgumentsKey failed, argument input should not be py::module";
    }
    ValuePtr converted = nullptr;
    if (!parse::ConvertData(arg.second, &converted)) {
      MS_LOG(EXCEPTION) << "GenerateArgumentsKey convert arg failed";
    }
    args_spec.push_back(ArgsToAbstract(converted));
  }

  uint64_t key;
  auto iter = g_args_cache.find(args_spec);
  if (iter == g_args_cache.end()) {
    static uint64_t key_counter = 0;
    key = key_counter;
    ++key_counter;
    g_args_cache[args_spec] = key;
    MS_LOG(INFO) << "Generate a new compile key for new args, key: " << key;
  } else {
    key = iter->second;
  }
  return py::int_(key);
}

py::bool_ VerifyInputSignature(const py::list &input_signature, const py::tuple &inputs) {
  MS_LOG(DEBUG) << "Verify args size:" << inputs.size();
  if (inputs.size() != input_signature.size()) {
    MS_LOG(ERROR) << "Signature size not equal to args size";
    return false;
  }

  size_t count = 0;
  for (auto arg_obj : inputs) {
    if (py::isinstance<Tensor>(arg_obj)) {
      MS_LOG(DEBUG) << "Verify Tensor";
      auto m_tensor = arg_obj.cast<std::shared_ptr<Tensor>>();
      if (m_tensor == nullptr) {
        MS_LOG(ERROR) << "Verify Tensor error, get ptr is null";
        return false;
      }
      auto sig = input_signature[count].cast<std::shared_ptr<MetaTensor>>();
      ShapeVector sig_shape = sig->shape();
      TypePtr sig_type = sig->Dtype();

      ShapeVector tensor_shape = m_tensor->shape_c();
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

void PipelineRDRProcess(const FuncGraphPtr &graph, const std::vector<ActionItem> &actions, const ActionItem &action,
                        size_t i) {
  MS_LOG(INFO) << "Recording FuncGraph in pipeline using RDR.";
  std::string name = GetBaseNameForIR(SizeToLong(i), action.first);
  if (graph != nullptr) {
    auto graph_clone = BasicClone(graph);
    if (graph_clone != nullptr) {
      DumpGraphParams dump_params = {false, static_cast<int>(kTopStack)};
      if (i == actions.size()) {
        dump_params.dump_mode = static_cast<int>(kWholeStack);
      }
      (void)mindspore::RDR::RecordAnfGraph(SUBMODULE_ID, name, graph_clone, dump_params, ".ir");
    } else {
      MS_LOG(WARNING) << "Clone FuncGraph failed in pipeline, no FuncGraph recording in RDR.";
    }
  } else {
    MS_LOG(WARNING) << "Pipeline Resource has no FuncGraph, no FuncGraph recording in RDR";
  }
  MS_LOG(INFO) << "Recording FuncGraph in pipeline end.";
}

GraphExecutorPy::GraphExecutorPy() {}

ResourcePtr GraphExecutorPy::GetResource(const std::string &phase) {
  MS_LOG(DEBUG) << "Phase size:" << info_.size();
  if (info_.count(phase) == 0) {
    return nullptr;
  }
  return info_[phase]->resource;
}

FuncGraphPtr GraphExecutorPy::GetFuncGraph(const std::string &phase) {
  if (info_.count(phase) == 0) {
    MS_LOG(EXCEPTION) << "No executor info. found for phase: " << phase;
  }
  return info_[phase]->func_graph;
}

FuncGraphPtr GraphExecutorPy::GetGradGraph(const std::string &phase) {
  if (phase.empty()) {
    MS_LOG(EXCEPTION) << "The input phase is empty.";
  }
  if (info_.count(phase) == 0) {
    MS_LOG(EXCEPTION) << "No phase in executor:" << phase;
  }

  auto execute_info = info_[phase];
  MS_EXCEPTION_IF_NULL(execute_info);
  auto grad_graph = execute_info->grad_graph;
  MS_EXCEPTION_IF_NULL(grad_graph);
  return grad_graph;
}

void GraphExecutorPy::SetGradGraph(const FuncGraphPtr &grad_graph, const std::string &phase) {
  if (phase.empty()) {
    MS_LOG(EXCEPTION) << "The input phase is empty.";
  }
  if (info_.count(phase) == 0) {
    MS_LOG(EXCEPTION) << "No phase in executor: " << phase;
  }

  auto execute_info = info_[phase];
  MS_EXCEPTION_IF_NULL(execute_info);
  if (execute_info->grad_graph != nullptr) {
    MS_LOG(DEBUG) << "The grad graph has existed, phase is: " << phase;
  }
  MS_EXCEPTION_IF_NULL(grad_graph);
  execute_info->grad_graph = grad_graph;
}

compile::VmEvalFuncPtr GraphExecutorPy::GetVmEvalFunc(const std::string &phase) {
  ResourcePtr res = GetResource(phase);
  MS_EXCEPTION_IF_NULL(res);
  if (res->results().find(kOutput) != res->results().end() && res->results()[kOutput].is<compile::VmEvalFuncPtr>()) {
    return res->results()[kOutput].cast<compile::VmEvalFuncPtr>();
  }
  MS_LOG(ERROR) << "GetVmEvalFunc vm model can't find kOutput:" << kOutput;
  return nullptr;
}

bool GraphExecutorPy::HasCompiled(const std::string &phase) const {
  if (info_.count(phase) == 0) {
    return false;
  }
  return true;
}

py::bytes GraphExecutorPy::GetFuncGraphProto(const std::string &phase, const std::string &ir_type) {
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
      MS_LOG(EXCEPTION) << "Export ANF format model failed.";
    }
    return proto_str;
  }

  if (ir_type == IR_TYPE_ONNX) {
    std::string proto_str = GetOnnxProtoString(fg_ptr);
    if (proto_str.empty()) {
      MS_LOG(EXCEPTION) << "Export ONNX format model failed.";
    }
    return proto_str;
  }

  if (ir_type == IR_TYPE_MINDIR) {
    std::string proto_str = GetBinaryProtoString(fg_ptr);
    if (proto_str.empty()) {
      MS_LOG(EXCEPTION) << "Export MINDIR format model failed.";
    }
    return proto_str;
  }

  MS_LOG(EXCEPTION) << "Unknown ir type: " << ir_type;
}

py::dict GraphExecutorPy::GetParameterLayout(const std::string &phase) {
  MS_LOG(DEBUG) << "GetParameterLayout!";
  std::string layout_graph = phase + kStepParallelGraph;
  auto graph = GetFuncGraph(layout_graph);
  return mindspore::parallel::GetParameterLayout(graph);
}

py::dict GraphExecutorPy::GetCNodeStrategy(const std::string &phase) {
  MS_LOG(DEBUG) << "GetCNodeStrategy!";
  return stra_dict_[phase];
}

py::list GraphExecutorPy::GetParallelParameterNameList(const std::string &phase) {
  std::string param_graph = phase + kStepParallelGraph;
  auto graph = GetFuncGraph(param_graph);
  return mindspore::parallel::GetParallelParameterNameList(graph);
}

void GraphExecutorPy::SetCNodeStrategy(const std::string &name, const parallel::Strategys &strategy) {
  MS_LOG(DEBUG) << "SetCNodeStrategy!";
  stra_dict_[phase_][py::str(name)] = strategy;
}

size_t GraphExecutorPy::GetNumOpsInfo(const std::string &phase) {
  MS_LOG(DEBUG) << "GetNumOpsInfo!";
  return phase_to_num_op_info_[phase];
}

void GraphExecutorPy::SetNumOpsInfo(size_t num_ops) {
  MS_LOG(DEBUG) << "SetNumOpsInfo!";
  phase_to_num_op_info_[phase_] = num_ops;
}

py::dict GraphExecutorPy::GetAllreduceFusion(const std::string &phase) {
  MS_LOG(INFO) << "GetAllreduceFusion!";
  auto graph = GetFuncGraph(phase);
  return mindspore::parallel::GetAllreduceFusion(graph);
}

// Not support multi thread, not support nested call too.
// Here using nested_called flg to avoid nested call.
void GraphExecutorPy::DelNetRes(const std::string &id) {
  static bool nested_called = false;
  if (nested_called) {
    return;
  }
  nested_called = true;
#ifdef ENABLE_GE
  FinalizeBackend();
#else
  ConfigManager::GetInstance().ResetIterNum();
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
  nested_called = false;
}

void GraphExecutorPy::ClearRes() {
  MS_LOG(INFO) << "Clean executor resource!";
  executor_ = nullptr;
}

GraphExecutorPy::~GraphExecutorPy() {
  MS_LOG(INFO) << "Release Executor!";
  ConfigManager::GetInstance().ResetConfig();
}

void GraphExecutorPy::GetWeightInfo(
  const CNodePtr &root_node, const AnfNodePtr &weight_node,
  std::map<std::string, std::pair<PrimitivePyAdapterPtr, std::string>> *fake_quant_table) {
  MS_EXCEPTION_IF_NULL(root_node);
  MS_EXCEPTION_IF_NULL(fake_quant_table);
  std::string weight_name;
  auto x = root_node->input(1);
  MS_EXCEPTION_IF_NULL(x);
  if (IsPrimitiveCNode(weight_node, prim::kPrimLoad)) {
    weight_name = weight_node->cast<CNodePtr>()->input(1)->cast<ParameterPtr>()->name();
  } else {
    auto para = weight_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(para);
    weight_name = para->name();
  }
  // find the fakequant from input
  int64_t count = 0;
  const int64_t max_depth = 5;
  CNodePtr cnode = nullptr;
  auto is_quant_cnode = [](const AnfNodePtr &node) {
    return IsPrimitiveCNode(node, prim::kPrimFakeQuantPerLayer) ||
           IsPrimitiveCNode(node, prim::kPrimFakeQuantPerChannel) ||
           IsPrimitiveCNode(node, prim::kPrimFakeLearnedScaleQuantPerLayer) ||
           IsPrimitiveCNode(node, prim::kPrimFakeLearnedScaleQuantPerChannel);
  };
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
  if (x->isa<Parameter>() || IsPrimitiveCNode(x, prim::kPrimLoad)) {
    (*fake_quant_table)[weight_name] = std::make_pair(nullptr, "input");
  }
  // get the fakequant parameter minq's name
  if (!is_quant_cnode(x)) {
    return;
  }
  cnode = x->cast<CNodePtr>();
  constexpr size_t expect_input_size = 4;
  if (cnode == nullptr || IsPrimitiveCNode(cnode, prim::kPrimLoad) || cnode->size() != expect_input_size) {
    return;
  }
  const size_t fakequant_index = 2;
  auto fakequant_min_node = cnode->input(fakequant_index);
  if (!fakequant_min_node->isa<Parameter>() && !IsPrimitiveCNode(fakequant_min_node, prim::kPrimLoad)) {
    return;
  }
  std::string fakequant_min_node_name;
  if (IsPrimitiveCNode(fakequant_min_node, prim::kPrimLoad)) {
    fakequant_min_node_name = fakequant_min_node->cast<CNodePtr>()->input(1)->cast<ParameterPtr>()->name();
  } else {
    auto param = fakequant_min_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param);
    fakequant_min_node_name = param->name();
  }
  auto quant_op_value = cnode->input(0)->cast<ValueNodePtr>()->value();
  MS_EXCEPTION_IF_NULL(quant_op_value);
  if (!quant_op_value->isa<PrimitivePy>()) {
    return;
  }
  auto quant_op = quant_op_value->cast<PrimitivePyPtr>();
  (*fake_quant_table)[weight_name] = std::make_pair(quant_op->adapter(), fakequant_min_node_name);
}

std::map<std::string, std::pair<PrimitivePyAdapterPtr, std::string>> GraphExecutorPy::FetchInfoForQuantExport(
  const std::string &phase) {
  FuncGraphPtr func_graph = info_[phase]->resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_LOG(DEBUG) << "FetchInfoForQuantExport func graph(" << func_graph->ToString() << ") phase(" << phase << ")!";
  std::map<std::string, std::pair<PrimitivePyAdapterPtr, std::string>> fake_quant_table;
  auto filter = [](const AnfNodePtr &node) {
    return !(IsPrimitiveCNode(node, prim::kPrimConv2D) || IsPrimitiveCNode(node, prim::kPrimMatMul) ||
             IsPrimitiveCNode(node, prim::kPrimDepthwiseConv2dNative));
  };
  std::vector<AnfNodePtr> nodes = DeepScopedGraphSearchWithFilter(func_graph->get_return(), AlwaysInclude, filter);
  auto is_quant_cnode = [](const AnfNodePtr &node) {
    return IsPrimitiveCNode(node, prim::kPrimFakeQuantPerLayer) ||
           IsPrimitiveCNode(node, prim::kPrimFakeQuantPerChannel) ||
           IsPrimitiveCNode(node, prim::kPrimFakeLearnedScaleQuantPerLayer) ||
           IsPrimitiveCNode(node, prim::kPrimFakeLearnedScaleQuantPerChannel);
  };
  const size_t root_node_size = 3;
  const size_t weight_index = 2;
  for (const auto &node : nodes) {
    auto root_node = node->cast<CNodePtr>();
    if (root_node == nullptr || root_node->size() != root_node_size) {
      continue;
    }
    auto weight = root_node->input(weight_index);
    if (!is_quant_cnode(weight)) {
      auto tuple_node = weight->cast<CNodePtr>();
      if (tuple_node != nullptr) {
        auto fake_node = tuple_node->input(1);
        if (!is_quant_cnode(fake_node)) {
          continue;
        } else {
          weight = fake_node;
        }
      }
    }
    // get parameter weight's name
    auto cnode = weight->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto weight_node = cnode->input(weight_index);
    if (!weight_node->isa<Parameter>() && !IsPrimitiveCNode(weight_node, prim::kPrimLoad)) {
      continue;
    }
    GetWeightInfo(root_node, weight_node, &fake_quant_table);
  }
  return fake_quant_table;
}

void GraphExecutorPy::SaveCompiledGraph(const std::string &phase) {
  // save the graph to GraphExecutorPy
  FuncGraphPtr func_graph = info_[phase]->resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(parallel::ParallelContext::GetInstance());
  std::string parallel_mode = parallel::ParallelContext::GetInstance()->parallel_mode();

  MS_LOG(INFO) << "Save compiled func graph(" << func_graph->ToString() << ") phase(" << phase << ")!";
  info_[phase]->func_graph = func_graph;
  if ((func_graph != nullptr) && func_graph->has_flag(parallel::AUTO_PARALLEL) &&
      ((parallel_mode == parallel::AUTO_PARALLEL) || (parallel_mode == parallel::SEMI_AUTO_PARALLEL))) {
    MS_LOG(DEBUG) << "Save model parallel parameter layout graph!";
    func_graph = info_[phase]->resource->results()[kStepParallelGraph].cast<FuncGraphPtr>();
    ExecutorInfoPtr executor_info = std::make_shared<ExecutorInfo>();
    std::string layout_graph = phase + kStepParallelGraph;
    executor_info->func_graph = func_graph;
    info_[layout_graph] = executor_info;
  } else {
    MS_LOG(DEBUG) << "Save model parallel parameter layout graph null!";
  }
  MS_LOG(INFO) << "End save compiled func graph!";
}

void GraphExecutorPy::GetGeBackendPolicy() const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::string backend = ms_context->backend_policy();
  if (backend != "ge") {
    MS_LOG(EXCEPTION) << backend << " backend policy is not supported under ge backend!";
  }
}

bool IsPhaseExportAir(const std::string &phase) {
  auto phase_to_export = "export.air";
  return phase.rfind(phase_to_export) != std::string::npos;
}

bool IsPhaseTrain(const std::string &phase) {
  const std::string phase_to_train = "train";
  return phase.rfind(phase_to_train) != std::string::npos;
}

bool IsPhaseLoadFromMindIR(const std::string &phase) {
  const std::string mindir_graph = "graph_load_from_mindir";
  return phase.rfind(mindir_graph) != std::string::npos;
}

std::vector<ActionItem> GetPipeline(const ResourcePtr &resource, const std::string &phase, bool use_vm) {
  MS_EXCEPTION_IF_NULL(resource);
  bool is_air = IsPhaseExportAir(phase);

  std::string backend = MsContext::GetInstance()->backend_policy();

#if ((defined ENABLE_CPU) && (!defined _WIN32))
  const std::string &server_mode = ps::PSContext::instance()->server_mode();
  if ((server_mode == ps::kServerModeFL || server_mode == ps::kServerModeHybrid) &&
      ps::PSContext::instance()->is_server()) {
    return ServerPipeline();
  }
  if (ps::PSContext::instance()->is_server()) {
    resource->results()[kBackend] = compile::CreateBackend();
    return PServerPipeline();
  }
  if (ps::PSContext::instance()->is_scheduler()) {
    return PSchedulerPipeline();
  }
#endif

  if (use_vm && backend != "ge" && !is_air) {
    compile::SetMindRTEnable();
    // Create backend.
    auto backend_ptr = compile::CreateBackend();
#ifdef ENABLE_DEBUGGER
    // Connect session to debugger
    backend_ptr->SetDebugger();
#endif
    resource->results()[kBackend] = backend_ptr;
    // If the 'use_frontend_compile_cache' context has been set true and the cache is read successfully,
    // do the backend actions only.
    if (IsPhaseTrain(phase) && MsContext::GetInstance()->get_param<bool>(MS_CTX_LOAD_COMPILE_CACHE) &&
        resource->func_graph() != nullptr) {
      return BackendPipeline();
    }
    if (IsPhaseLoadFromMindIR(phase)) {
      return MindIRPipeline();
    }
    return VmPipeline();
  }
  return GePipeline();
}

bool GraphExecutorPy::CompileInner(const py::object &source_obj, const py::tuple &args, const py::object &phase_obj,
                                   bool use_vm, const std::string &queue_name) {
  // Check if the phase is valid.
  if ((!py::isinstance<py::str>(phase_obj))) {
    MS_LOG(ERROR) << "The `phase` must be string.";
    return false;
  }
  // Check if the function or net is valid.
  if (py::isinstance<py::none>(source_obj)) {
    MS_LOG(ERROR) << "The source object to compile should not be None.";
    return false;
  }
  // Check if the args of function or net is valid.
  CheckArgsValid(args);

  auto phase = py::cast<std::string>(phase_obj);
  MS_LOG(INFO) << "Start compiling, phase: " << phase << ".";
  MS_LOG(DEBUG) << "Compiling source: {" << py::str(source_obj)
                << "}\n\n Args: " << py::str(const_cast<py::tuple &>(args));

#ifdef ENABLE_GE
  GetGeBackendPolicy();
#endif
  ExecutorInfoPtr executor_info = std::make_shared<ExecutorInfo>();
  ResourcePtr resource = std::make_shared<Resource>(source_obj);

  if (MsContext::GetInstance()->get_param<bool>(MS_CTX_LOAD_COMPILE_CACHE)) {
#ifdef ENABLE_PROFILE
    double t1 = GetTime();
#endif
    GetCachedFuncGraph(resource, queue_name);
#ifdef ENABLE_PROFILE
    double t2 = GetTime();
    MsProfile::StatTime("LoadCachedFuncGraph", t2 - t1);
#endif
  }

  phase_ = phase;
  auto actions = GetPipeline(resource, phase, use_vm);
  std::shared_ptr<Pipeline> pip = std::make_shared<Pipeline>(resource, FilterActions(actions, phase));

  // Get the parameters items and add the value to args_spec.
  abstract::AbstractBasePtrList args_spec;
  std::size_t size = args.size();
  for (std::size_t i = 0; i < size; i++) {
    ValuePtr converted = nullptr;
    bool succ = parse::ConvertData(args[i], &converted);
    if (!succ) {
      MS_LOG(EXCEPTION) << "Fail to convert the " << i << "th argument, args[" << i << "]: " << py::str(args[i]);
    }
    args_spec.push_back(ArgsToAbstract(converted));
  }
  resource->set_args_spec(args_spec);
  executor_info->arg_list_size = size;
  executor_info->resource = resource;
  info_[phase] = executor_info;
  pip->Run(phase);

  // Save the compiled graph to MsPipeLine.
  SaveCompiledGraph(phase);

  opt::python_pass::PyPassManager::GetInstance()->ClearPipelineRes();
  abstract::AnalysisContext::ClearContext();
  // Reclaim all resource used by optimizer.
  ReclaimOptimizer();
  resource->Clean();

  MS_LOG(INFO) << "Finish compiling.";
  return true;
}

std::vector<ActionItem> GraphExecutorPy::FilterActions(const std::vector<ActionItem> &actions,
                                                       const std::string &phase) {
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

void GraphExecutorPy::ReleaseResource(const py::object &phase) {
  ResourcePtr res = GetResource(py::cast<std::string>(phase));
  if (res != nullptr) {
    res->Clean();
  }
  // Reclaim all resource used by optimizer;
  ReclaimOptimizer();
}

bool GraphExecutorPy::Compile(const py::object &source_obj, const py::tuple &args, const py::object &phase, bool use_vm,
                              const std::string &queue_name) {
  bool ret_value = false;
  try {
    ret_value = CompileInner(source_obj, args, phase, use_vm, queue_name);
  } catch (const py::error_already_set &ex) {
    if (!StaticAnalysisException::Instance().HasException()) {
      // print function call stack info before release
      std::string exception_info = GetCompileExceptionInfo();
      if (!exception_info.empty()) {
        MS_LOG(ERROR) << exception_info;
      }
    }
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
  } catch (const py::key_error &ex) {
    ReleaseResource(phase);
    throw py::key_error(ex);
  } catch (const py::attribute_error &ex) {
    ReleaseResource(phase);
    throw py::attribute_error(ex);
  } catch (const py::name_error &ex) {
    ReleaseResource(phase);
    throw py::name_error(ex);
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

void CacheValidateFuncGraph(const std::string &phase, const ResourcePtr &resource) {
  if (IsPhaseTrain(phase) && MsContext::GetInstance()->get_param<bool>(MS_CTX_SAVE_COMPILE_CACHE)) {
#ifdef ENABLE_PROFILE
    double t1 = GetTime();
#endif
    CacheFuncGraph(resource);
#ifdef ENABLE_PROFILE
    double t2 = GetTime();
    MsProfile::StatTime("SaveCacheFuncGraph", t2 - t1);
#endif
  }
}

void Pipeline::Run(const std::string &phase) {
  MS_LOG(INFO) << "Pipeline run";
  MS_EXCEPTION_IF_NULL(resource_);
  FuncGraphPtr user_graph = nullptr;

  WITH(MsProfile::GetProfile())[&user_graph, &phase, this]() {
    size_t i = 0;
    for (auto &action : actions_) {
#ifdef ENABLE_TIMELINE
      DumpTime &dump_time = DumpTime::GetInstance();
      dump_time.Record(action.first, GetTime(), true);
#endif
      bool result = true;
      WITH(MsProfile::GetProfile()->Step(action.first))[&result, &action, this]() {
        MS_LOG(DEBUG) << "Action " << action.first << " start ...";
        result = action.second(resource_);
        MS_LOG(DEBUG) << "Action " << action.first << " end.";
      };
      if (action.first == "task_emit") {
        SetLoopCount(resource_);
      } else if (action.first == "validate") {
        CacheValidateFuncGraph(phase, resource_);
      }
      if (!result) {
        MS_LOG(EXCEPTION) << "Pipeline running to end, failed in step:" << action.first;
      }

      FuncGraphPtr graph = resource_->func_graph();
#ifdef ENABLE_DUMP_IR
      if (mindspore::RecorderManager::Instance().RdrEnable()) {
        PipelineRDRProcess(graph, actions_, action, i);
      }
      if (MsContext::GetInstance()->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG) && graph != nullptr) {
        user_graph = graph;
        std::string base_name = GetBaseNameForIR(SizeToLong(i), action.first);

        // generate IR file in dot format, which can be converted to svg file using graphviz dot command
        draw::Draw(base_name + ".dot", graph);
        // generate IR file in human readable format
        if (i == actions_.size() - 1) {
          DumpIR(base_name + ".ir", graph, false, kWholeStack);
        } else {
          DumpIR(base_name + ".ir", graph, false, kTopStack);
        }
        // generate IR file in a heavily commented format, which can also be reloaded
        ExportIR(base_name + ".dat", graph);
      }
#endif
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

#ifdef ENABLE_DUMP_IR
  if (MsContext::GetInstance()->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG) && (user_graph != nullptr)) {
    draw::DrawUserFuncGraph("ModelDigraph.dot", user_graph);
  }
#endif
  MS_LOG(INFO) << "End";
}

void ProcessVmArgInner(const py::tuple &args, const ResourcePtr &res, VectorRef *const arg_list) {
  MS_EXCEPTION_IF_NULL(arg_list);
  std::size_t size = args.size();
  bool arg_list_inited = !arg_list->empty();
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
    if (!arg_list_inited) {
      arg_list->push_back(converted);
      continue;
    }
    if (i >= arg_list->size()) {
      MS_LOG(EXCEPTION) << "i:" << i << " output of range:" << arg_list->size();
    }
    (*arg_list)[i] = converted;
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
      MS_EXCEPTION_IF_NULL(param_ptr);
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

void GraphExecutorPy::ProcessVmArg(const py::tuple &args, const std::string &phase, VectorRef *const arg_list) {
  ProcessVmArgInner(args, GetResource(phase), arg_list);
}

#ifdef ENABLE_DEBUGGER
void GraphExecutorPy::TerminateDebugger() {
  if (debugger_terminate_) {
    MS_LOG(INFO) << "Terminate debugger and clear resources!";
    ClearResAtexit();
    if (exit_success_) {
      exit(0);
    } else {
      exit(1);
    }
  }
}
#endif

py::object GraphExecutorPy::Run(const py::tuple &args, const py::object &phase_obj) {
  // Mindspore debugger notify main thread to exit after one step, and will not run next step
#ifdef ENABLE_DEBUGGER
  TerminateDebugger();
#endif
  std::size_t size = args.size();
  if (!py::isinstance<py::str>(phase_obj)) {
    MS_LOG(EXCEPTION) << "Run failed, phase input is not a str";
  }
  auto phase = py::cast<std::string>(phase_obj);
  std::string backend = MsContext::GetInstance()->backend_policy();
#ifdef ENABLE_GE
  if (backend == "ge") {
    return ExecDFGraph(info_, args, phase);
  }
#else
  auto ret_val = std::make_shared<py::object>();
  if (info_.count(phase) != 0 && info_[phase]->func_graph != nullptr) {
    if (IsGraphOutputValueNodeOrParameter(info_[phase]->func_graph->output(), args, ret_val)) {
      // Check the input arg must be Tensor when backend is "ms".
      if (MsContext::GetInstance()->backend_policy() == kMsConvert) {
        for (std::size_t i = 0; i < size; i++) {
          ValuePtr converted = nullptr;
          if (!parse::ConvertData(args[i], &converted)) {
            MS_LOG(EXCEPTION) << "The " << i << "th arg convert failed.";
          }
        }
      }
      return *ret_val;
    }
  }
  if (backend == "ge") {
    // Virtual output constructed for test cases.
    if (!args.empty()) {
      return args[0];
    }
    return args;
  }
#endif
  auto iter = info_.find(phase);
  if (iter == info_.end()) {
    MS_LOG(EXCEPTION) << "No executor info. found for phase: " << phase;
  }
  auto &execute_info = iter->second;
  MS_EXCEPTION_IF_NULL(execute_info);
  if (size > execute_info->arg_list_size) {
    MS_LOG(WARNING) << "The arg num : size = " << size << ". full_arg_size = " << execute_info->arg_list_size;
  }
  ProcessVmArg(args, phase, &execute_info->arg_list);
  // Start to run phase.
  compile::VmEvalFuncPtr run = GetVmEvalFunc(phase);
  if (run == nullptr) {
    MS_LOG(EXCEPTION) << "Can't find run graph func for " << phase;
  }
  // Set loopsink size for each phase.
  bool vm_loop_flag = info_[phase]->resource->vm_loop_flag();
  int64_t loop_size = info_[phase]->resource->loop_size();
  int64_t vm_loop = 1;
  if (vm_loop_flag) {
    vm_loop = loop_size;
  } else {
    // Set the loop size in config if graphs nums is 1(is_loop_sin=True), then there will be a loop embrace
    // 'Execute(graph)' in GPUSession.
    ConfigManager::GetInstance().set_gpu_loopsink_size(loop_size);
  }
  MS_LOG(INFO) << "VM loop size " << vm_loop << ", loopsink size " << vm_loop;
  py::object ret;
  MS_LOG(DEBUG) << "Eval run" << backend;
  for (int64_t i = 0; i < vm_loop; i++) {
    BaseRef value = (*run)(execute_info->arg_list);
    ret = BaseRefToPyData(value);
  }
  MS_LOG(DEBUG) << "Run end";
  return ret;
}

FuncGraphPtr GraphExecutorPy::BuildGraph(const py::dict &init_params, const std::string &phase,
                                         const py::object &broadcast_params) {
#if ((defined ENABLE_GE) || (defined ENABLE_D))
  return BuildDFGraph(info_, init_params, phase, broadcast_params);
#else
  return nullptr;
#endif
}

void GraphExecutorPy::UpdataParamNodeDefaultInput(
  const std::string &phase, const std::unordered_map<std::string, tensor::TensorPtr> &params_value) {
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

void GraphExecutorPy::RunInitGraph(const py::dict &init_params, const std::string &phase) const {
#ifdef ENABLE_GE
  RunGEInitGraph(init_params, phase);
#endif
}

void GraphExecutorPy::PyExePath(const py::object &py_exe_path) {
  if (!py::isinstance<py::str>(py_exe_path)) {
    MS_LOG(EXCEPTION) << "Failed, py_exe_path input is not a str";
  }
  auto py_exe_path_s = py::cast<std::string>(py_exe_path);
  auto ms_context = MsContext::GetInstance();
  ms_context->set_param<std::string>(MS_CTX_PYTHON_EXE_PATH, py_exe_path_s);
}

void GraphExecutorPy::KernelBuildServerDir(const py::object &kernel_build_server_dir) {
  if (!py::isinstance<py::str>(kernel_build_server_dir)) {
    MS_LOG(EXCEPTION) << "Failed, kernel_build_server_dir input is not a str";
  }
  auto kernel_build_server_dir_s = py::cast<std::string>(kernel_build_server_dir);
  auto ms_context = MsContext::GetInstance();
  ms_context->set_param<std::string>(MS_CTX_KERNEL_BUILD_SERVER_DIR, kernel_build_server_dir_s);
}

bool InitExecDataset(const std::string &queue_name, int64_t iter_num, int64_t batch_size,
                     const std::vector<TypePtr> &types, const std::vector<std::vector<int64_t>> &shapes,
                     const std::vector<int64_t> &input_indexes, const std::string &phase, bool need_run) {
  std::string name = MsContext::GetInstance()->backend_policy();
#ifndef NO_DLIB
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!context::IsTsdOpened(ms_context) || !context::IsGeInited(ms_context)) {
    InitPipeline();
  }
#endif
  if (iter_num == -1) {
    iter_num = INT32_MAX;
  }
  if (name == kMsConvert || name == kMsVm) {
    return InitExecDatasetVm(queue_name, iter_num, batch_size, types, shapes, input_indexes, need_run);
  }
#ifdef ENABLE_GE
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
#if ((defined ENABLE_CPU) && (!defined _WIN32))
  if ((ps::PSContext::instance()->is_ps_mode()) && (!ps::PSContext::instance()->is_worker())) {
    return true;
  }
#endif
  MS_LOG(INFO) << "Start InitDataSet Entry";
  mindspore::parse::python_adapter::set_python_env_flag(true);
  ShapeVector int_input_indexes;
  (void)std::transform(input_indexes.begin(), input_indexes.end(), std::back_inserter(int_input_indexes),
                       [](int64_t item) { return static_cast<int64_t>(item); });
  std::vector<ShapeVector> int_shapes;
  (void)std::transform(shapes.begin(), shapes.end(), std::back_inserter(int_shapes),
                       [](const std::vector<int64_t> &item) {
                         ShapeVector vector_item;
                         (void)std::transform(item.begin(), item.end(), std::back_inserter(vector_item),
                                              [](int64_t inner_item) { return static_cast<int64_t>(inner_item); });
                         return vector_item;
                       });
  auto p_init = std::make_shared<Primitive>("InitDataSetQueue");
  p_init->set_attr("queue_name", MakeValue(queue_name));
  p_init->set_attr("size", MakeValue(static_cast<int64_t>(size)));
  p_init->set_attr("batch_size", MakeValue(static_cast<int64_t>(batch_size)));
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
  // Before the graph compiling, need reset the iter num.
  ConfigManager::GetInstance().ResetIterNum();
#ifdef ENABLE_DUMP_IR
  mindspore::RDR::ResetRecorder();
#endif

  compile::SetMindRTEnable();
  auto backend = compile::CreateBackend();
  MS_EXCEPTION_IF_NULL(backend);
  // The data set graph compiling and running of mindRT.
  if (MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
    const auto &mindrt_backend = std::dynamic_pointer_cast<compile::MindRTBackend>(backend);
    MS_EXCEPTION_IF_NULL(mindrt_backend);
    auto &actor_info = mindrt_backend->CompileGraphs(func_graph);
    VectorRef args;
    if (need_run) {
      VectorRef outputs;
      mindrt_backend->RunGraph(actor_info, args, &outputs);
    }
    ConfigManager::GetInstance().set_iter_num(size);
    return true;
  }

  auto convert_fn = backend->convert_fn();
  MS_EXCEPTION_IF_NULL(convert_fn);
  // Convert CNodeList to LinConvertResult.
  auto segment = std::make_shared<GraphSegment>(std::vector<AnfNodePtr>{app_init}, false);
  auto runner = convert_fn(segment, "");
  ConfigManager::GetInstance().set_iter_num(size);
  // PS cache does not support loop sink.
#if ((defined ENABLE_CPU) && (!defined _WIN32))
  if (ps::PSContext::instance()->is_worker() && ps::PsDataPrefetch::GetInstance().cache_enable()) {
    ps::PsDataPrefetch::GetInstance().CreateDataChannel(queue_name, LongToSize(size));
    ConfigManager::GetInstance().set_iter_num(1);
  }
#endif

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
}  // namespace pipeline

void ResetOpId() { mindspore::id_generator::reset_id(); }

void InitHccl() {
#ifdef ENABLE_GE
  (void)InitPipeline();
#else
  mindspore::parse::python_adapter::set_python_env_flag(true);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  uint32_t device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
#if ENABLE_D
  bool task_sink = true;
  auto single_op = common::GetEnv(kGraphOpRun);
  auto enable_mem_scheduler = common::GetEnv(kEnableMemScheduler);
  if (single_op == "1" || enable_mem_scheduler == "1") {
    task_sink = false;
  }
  auto mode = ms_context->get_param<int>(MS_CTX_EXECUTION_MODE);
  if (!task_sink && mode == kGraphMode) {
    MS_LOG(INFO) << "mpi collective init.";
    if (!HcclCollectiveGroup::instance().InitCollective()) {
      MS_LOG(EXCEPTION) << "Mpi init failed, please check if mpirun is used correctly.";
    }
    device_id = IntToUint(HcclCollectiveGroup::instance().GetDeviceId());
    ms_context->set_param<uint32_t>(MS_CTX_DEVICE_ID, device_id);
    ms_context->set_param<bool>(MS_CTX_ENABLE_TASK_SINK, false);
  }
#endif
  std::string device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  ms_context->set_param<bool>(MS_CTX_ENABLE_HCCL, true);
  if (ms_context->backend_policy() == "ms" &&
      ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice) {
    auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(device_name, device_id);
    MS_EXCEPTION_IF_NULL(runtime_instance);
#ifndef ENABLE_SECURITY
    runtime_instance->PreInit();
#endif
    (void)context::OpenTsd(ms_context);
    if (!runtime_instance->Init()) {
      MS_LOG(EXCEPTION) << "Runtime init failed.";
    }
  } else {
    (void)context::OpenTsd(ms_context);
  }
#endif
#if (defined ENABLE_D)
#ifndef ENABLE_SECURITY
  if (!ProfilingManager::GetInstance().IsProfiling()) {
    ProfilingManager::GetInstance().SetHcclEnabledBefProfilingEnabled();
  }
#endif
#endif
}

void FinalizeHccl() {
#ifdef ENABLE_GE
  (void)FinalizeBackend();
#else
  session::ExecutorManager::Instance().Clear();
  device::KernelRuntimeManager::Instance().ClearRuntimeResource();
#endif
}

uint32_t GetHcclRankId() {
  uint32_t rank_id = 0;
  bool ret = CommManager::GetInstance().GetRankID("", &rank_id);
  if (!ret) {
    MS_LOG(ERROR) << "Get rank id failed, return rank id " << rank_id << " as default.";
  }
  return rank_id;
}

uint32_t GetHcclRankSize() {
  uint32_t rank_size = 0;
  bool ret = CommManager::GetInstance().GetRankSize("", &rank_size);
  if (!ret) {
    MS_LOG(ERROR) << "Get rank size failed, return rank size " << rank_size << " as default.";
  }
  return rank_size;
}

void ExportGraph(const std::string &file_name, const std::string &, const std::string &phase) {
#if ((defined ENABLE_GE) || (defined ENABLE_D))
  ExportDFGraph(file_name, phase);
#else
  MS_EXCEPTION(ValueError) << "Only support export file in 'AIR' format with Ascend backend.";
#endif
}

FuncGraphPtr LoadMindIR(const std::string &file_name, char *dec_key, const size_t key_len,
                        const std::string &dec_mode) {
  auto func_graph =
    mindspore::LoadMindIR(file_name, false, reinterpret_cast<unsigned char *>(dec_key), key_len, dec_mode);
#ifdef ENABLE_DUMP_IR
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
  if (save_graphs) {
    DumpIR("load.ir", func_graph);
  }
#endif
  return func_graph;
}

void ReleaseGeTsd() {
  auto context_ptr = MsContext::GetInstance();
  if (context_ptr != nullptr) {
    (void)context::FinalizeGe(context_ptr, true);
    (void)context::CloseTsd(context_ptr, true);
  }
}

#ifndef ENABLE_SECURITY
void StartUpProfiling() {
#ifdef ENABLE_D
  if (!ProfilingManager::GetInstance().IsProfiling()) {
    return;
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);

  MS_LOG(INFO) << "Startup profiling";
  // Start up profiling before OpenTsd
  uint32_t device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  std::string device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (ms_context->backend_policy() == "ms" &&
      ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice) {
    auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(device_name, device_id);
    MS_EXCEPTION_IF_NULL(runtime_instance);
    runtime_instance->PreInit();
  }
#endif
}
#endif

void InitPipeline() {
  // set python env flag
  mindspore::parse::python_adapter::set_python_env_flag(true);
#ifndef ENABLE_SECURITY
  // Startup profiling before open tsd
  StartUpProfiling();
#endif
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
#if ((defined ENABLE_CPU) && (!defined _WIN32))
  if (ps::PSContext::instance()->is_ps_mode() && ps::PSContext::instance()->is_worker()) {
    if (ps::PsDataPrefetch::GetInstance().cache_enable()) {
      ps::ps_cache_instance.Finalize();
    }
    MS_LOG(INFO) << "Start finalizing worker.";
    const std::string &server_mode = ps::PSContext::instance()->server_mode();
    if ((server_mode == ps::kServerModeFL || server_mode == ps::kServerModeHybrid)) {
      fl::worker::FLWorker::GetInstance().Finalize();
    } else {
      ps::Worker::GetInstance().Finalize();
    }
  }
#endif
#ifdef ENABLE_DUMP_IR
  mindspore::RDR::ResetRecorder();
#endif
  session::ExecutorManager::Instance().Clear();
  device::KernelRuntimeManager::Instance().ClearRuntimeResource();
  runtime::GraphScheduler::GetInstance().Clear();
  device::DeviceContextManager::GetInstance().ClearDeviceContexts();
  ad::g_k_prims.clear();
  ad::ClearKPynativeCellStaticRes();
  ad::PrimBpropOptimizer::GetPrimBpropOptimizerInst().Clear();

  abstract::ClearPrimEvaluatorMap();
  pipeline::GetMethodMap().clear();
  pipeline::GetAttrMap().clear();
  pipeline::GraphExecutorPy::ClearRes();
  pipeline::ReclaimOptimizer();
  pynative::PynativeExecutor::GetInstance()->ClearRes();
  opt::python_pass::PyPassManager::GetInstance()->ClearRes();
#ifdef ENABLE_GE
  transform::DfGraphManager::GetInstance().ClearGraph();
  transform::OpAdapterMap::get().clear();
#else
  ConfigManager::GetInstance().ResetIterNum();
#endif
  ReleaseGeTsd();
  parse::python_adapter::ResetPythonScope();
  abstract::AnalysisResultCacheMgr::GetInstance().Clear();
  abstract::AnalysisContext::ClearContext();
  abstract::AnalysisSchedule::GetInstance().Stop();
#ifdef ENABLE_DEBUGGER
  Debugger::GetInstance()->Reset();
#endif
  g_args_cache.clear();
  // clean static variable to prevent from crash. As static variable is released after
  // Python threads is released.
  parse::data_converter::ClearObjectCache();
  parse::Parser::CleanParserResource();
  parse::CleanDataClassToClassMap();
  trace::ClearTraceStack();
}

py::bytes PyEncrypt(char *plain_data, size_t plain_len, char *key, size_t key_len, const std::string &enc_mode) {
  size_t encrypt_len;
  auto encrypt_data = mindspore::Encrypt(&encrypt_len, reinterpret_cast<Byte *>(plain_data), plain_len,
                                         reinterpret_cast<Byte *>(key), key_len, enc_mode);
  if (encrypt_data == nullptr) {
    MS_EXCEPTION(ValueError) << "Encrypt failed";
  }
  auto py_encrypt_data = py::bytes(reinterpret_cast<char *>(encrypt_data.get()), encrypt_len);
  return py_encrypt_data;
}

py::bytes PyDecrypt(const std::string &encrypt_data_path, char *key, size_t key_len, const std::string &dec_mode) {
  size_t decrypt_len;
  auto decrypt_data =
    mindspore::Decrypt(&decrypt_len, encrypt_data_path, reinterpret_cast<Byte *>(key), key_len, dec_mode);
  if (decrypt_data == nullptr) {
    MS_LOG(ERROR) << "Decrypt failed";
    return py::none();
  }
  auto py_decrypt_data = py::bytes(reinterpret_cast<char *>(decrypt_data.get()), decrypt_len);
  return py_decrypt_data;
}

bool PyIsCipherFile(const std::string &file_path) { return mindspore::IsCipherFile(file_path); }
}  // namespace pipeline
}  // namespace mindspore
