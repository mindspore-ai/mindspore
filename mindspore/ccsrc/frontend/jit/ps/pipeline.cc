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

#include "frontend/jit/ps/pipeline.h"

#include <memory>
#include <map>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <unordered_map>
#include <functional>

#include "pybind_api/pybind_patch.h"
#include "pybind11/pybind11.h"
#include "frontend/jit/ps/action.h"
#include "frontend/jit/ps/pass.h"

#include "ir/func_graph_cloner.h"

#include "frontend/optimizer/irpass.h"
#include "include/utils/tensor_py.h"
#include "include/utils/parallel_context.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/step_auto_parallel.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/allreduce_fusion/step_allreduce_fusion.h"
#include "frontend/parallel/pass/handle_group_info.h"
#include "frontend/parallel/step_assigned_parallel.h"

#include "utils/config_manager.h"

#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "utils/phase.h"
#include "utils/interpret_node_recorder.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "mindspore/ccsrc/utils/ir_dump/dump_proto.h"
#include "frontend/jit/ps/fallback.h"
#include "mindspore/ccsrc/utils/ir_dump/draw.h"
#include "include/backend/backend_manager/backend_manager.h"
#include "runtime/hardware_abstract/utils.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/cluster/init.h"
#include "tools/profiler/profiling.h"
#include "tools/profiler/profiler.h"
#include "tools/profiler/mstx/mstx_impl.h"

namespace mindspore {
// namespace to support intermediate representation definition
namespace pipeline {
using MetaTensor = mindspore::tensor::MetaTensor;
using MetaSparseTensor = mindspore::tensor::MetaSparseTensor;
using mindspore::abstract::AbstractTuple;
using mindspore::abstract::AbstractTuplePtr;
using DeviceTensor = mindspore::device::DeviceAddress;

namespace {

bool CheckAllTensor(const ValueTuplePtr &value_tuple) {
  MS_EXCEPTION_IF_NULL(value_tuple);
  auto elements = value_tuple->value();
  for (auto element : elements) {
    MS_EXCEPTION_IF_NULL(element);
    if (!(element->isa<ValueTuple>() && CheckAllTensor(element->cast<ValueTuplePtr>())) &&
        !(element->isa<MetaTensor>())) {
      return false;
    }
  }
  return true;
}

bool Mutable(const py::object &obj, const ValuePtr &value) {
  // If a tensor has been set const arg, it should not be mutable.
  if (value->isa<MetaTensor>()) {
    constexpr char const_arg_attr[] = "const_arg";
    if (py::hasattr(obj, const_arg_attr) && py::cast<bool>(py::getattr(obj, const_arg_attr))) {
      return false;
    }
  }
  constexpr char mutable_attr[] = "__ms_mutable__";
  return py::hasattr(obj, mutable_attr) && py::cast<bool>(py::getattr(obj, mutable_attr));
}

bool CheckAndConvertToVariableLenSequence(const py::object &obj, AbstractBasePtr abs) {
  constexpr char variable_len_attr[] = "__ms_dynamic_len__";
  bool dynamic_len = (py::hasattr(obj, variable_len_attr) && py::cast<bool>(py::getattr(obj, variable_len_attr)));
  if (!dynamic_len) {
    return false;
  }
  if (!abs->isa<abstract::AbstractSequence>()) {
    MS_EXCEPTION(TypeError) << "For mutable, when the dynamic_len the True, the first input should be"
                            << " list or tuple, but got: " << abs->ToString();
  }
  auto abs_seq = abs->cast<abstract::AbstractSequencePtr>();
  abs_seq->CheckAndConvertToDynamicLenSequence();
  return true;
}

bool TensorArgMutable(const py::object &obj, const ValuePtr &value) {
  if (!value->isa<MetaTensor>()) {
    return false;
  }
  constexpr char const_arg_attr[] = "const_arg";
  return !py::hasattr(obj, const_arg_attr) || !py::cast<bool>(py::getattr(obj, const_arg_attr));
}

bool EnableTupleBroaden(const ValuePtr &value, bool enable_tuple_broaden) {
  return enable_tuple_broaden && value->isa<ValueTuple>() && CheckAllTensor(value->cast<ValueTuplePtr>());
}

bool GradForScalar(const ValuePtr &value) {
  return (MsContext::GetInstance()->get_param<bool>(MS_CTX_GRAD_FOR_SCALAR) ||
          common::GetCompileConfig("GRAD_FOR_SCALAR") == "1") &&
         value->isa<Scalar>();
}

void AddManager(const FuncGraphManagerPtr &manager, const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<FuncGraph>()) {
    auto fg = value->cast<FuncGraphPtr>();
    manager->AddFuncGraph(fg);
  }
  if (value->isa<ValueSequence>()) {
    auto value_sequence = value->cast<ValueSequencePtr>();
    for (const auto &elem : value_sequence->value()) {
      AddManager(manager, elem);
    }
  }
  if (value->isa<ValueDictionary>()) {
    for (const auto &elem : value->cast<ValueDictionaryPtr>()->value()) {
      AddManager(manager, elem.second);
    }
  }
}
}  // namespace

void AddManagerForFuncGraphArgs(const ResourcePtr &resource, const ValuePtrList &arguments) {
  auto manager = resource->manager();
  MS_EXCEPTION_IF_NULL(manager);
  for (const auto &arg : arguments) {
    AddManager(manager, arg);
  }
}

AbstractBasePtr ArgsToAbstract(const py::object &arg, const ValuePtr &value, bool enable_tuple_broaden) {
  bool broaden = TensorArgMutable(arg, value) || Mutable(arg, value) || value->isa<MetaSparseTensor>() ||
                 EnableTupleBroaden(value, enable_tuple_broaden) || GradForScalar(value);
  auto ret = abstract::ToAbstract(value, nullptr, nullptr);
  if (broaden) {
    ret = AbstractBroaden(ret);
  }
  auto is_dynamic_len = CheckAndConvertToVariableLenSequence(arg, ret);
  if (fallback::EnableFallbackListDictInplace() && !broaden && !is_dynamic_len) {
    // Attach corresponding list python object for constant list input.
    fallback::AttachPyObjToAbs(ret, arg, false);
  }
  return ret;
}

void SetLoopCount(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  auto func_graph = resource->func_graph();
  if (func_graph != nullptr && func_graph->manager() != nullptr) {
    auto manager = func_graph->manager();
    size_t graph_nums = manager->func_graphs().size();
    int64_t loop_size = ConfigManager::GetInstance().iter_num();
    const auto context_ptr = MsContext::GetInstance();
    bool enable_mind_rt = context_ptr->get_param<bool>(MS_CTX_ENABLE_MINDRT);
    if (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice) {
      resource->set_vm_loop(!(context_ptr->get_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK) || enable_mind_rt), loop_size);
    } else if (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kGPUDevice) {
      bool run_with_mind_rt = graph_nums == 1 || enable_mind_rt;
      resource->set_vm_loop(!run_with_mind_rt, loop_size);
    }
    MS_LOG(INFO) << "Change vm_loop_flag to " << resource->vm_loop_flag() << ", set loop_size to " << loop_size;
  }
}

void ResetId(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  auto need_dump = common::GetCompileConfig("DUMP_VALIDATE_BEFORE_RESET_ID");
  if (context->CanDump(kIntroductory) && need_dump == "1") {
    FuncGraphPtr graph = resource->func_graph();
    MS_EXCEPTION_IF_NULL(graph);
    DumpIR("validate_before_reset_id.ir", graph, true, kWholeStack);
  }
#endif
  mindspore::id_generator::reset_id();
  const auto &all_nodes = TopoSort(resource->func_graph()->get_return(), SuccDeeperSimple);
  auto ge_mode = AnfAlgo::IsBackendGe();
  for (const auto &node : all_nodes) {
    if (node != nullptr && node->isa<CNode>()) {
      const auto &cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      cnode->set_fullname_with_scope("");
      if (!ge_mode) {
        (void)cnode->fullname_with_scope();
      }
    }
  }
}

std::string GetObjDesc(const py::object &source) {
  std::string obj_desc;
  if (py::hasattr(source, parse::PYTHON_PARSE_METHOD)) {
    auto cell_class_name = source.attr("__class__").attr("__name__");
    auto jit_name = source.attr(parse::PYTHON_PARSE_METHOD);
    obj_desc = "'" + py::cast<std::string>(cell_class_name) + "." + py::cast<std::string>(jit_name) + "'";
  } else {
    if (py::hasattr(source, "__name__")) {
      auto jit_name = source.attr("__name__");
      obj_desc = "'" + py::cast<std::string>(jit_name) + "'";
    } else if (py::isinstance<Cell>(source)) {
      auto cell_class_name = source.attr("__class__").attr("__name__");
      obj_desc = "'" + py::cast<std::string>(cell_class_name) + ".construct'";
    } else {
      MS_EXCEPTION(TypeError) << "The source object is invalid: " << py::str(source);
    }
  }
  return obj_desc;
}

bool IsPhaseExport(const std::string &phase) {
  constexpr auto export_str = "export";
  return phase.compare(0, strlen(export_str), export_str) == 0;
}

bool IsPhaseLoadFromMindIR(const std::string &phase) {
  const std::string mindir_graph = "graph_load_from_mindir";
  return phase.rfind(mindir_graph) != std::string::npos;
}

void SetHookForArgAbstract(const ResourcePtr &resource, const py::object &arg, abstract::AbstractBasePtr abs) {
  if (tensor::IsTensorPy(arg)) {
    auto tensor = tensor::ConvertToTensor(arg);
    MS_EXCEPTION_IF_NULL(tensor);
    const auto hooks = parse::ResolveTensorHooks(resource, tensor);
    if (hooks != nullptr) {
      MS_LOG(DEBUG) << "Set hooks for arg: " << py::str(arg) << ", abstract: " << abs << ".";
      abs->set_user_data(TENSOR_HOOK_MAP, hooks);
    }
  } else {
    MS_LOG(DEBUG) << "arg: " << py::str(arg) << " is not a Tensor, we only support arg of type Tensor now.";
  }
}

void CacheFuncGraph(const ResourcePtr &resource) {
  if (!resource->EnableCompileCache()) {
    return;
  }
  {
    MsProfileStatGuard stat_guard("SaveCacheFuncGraph", "compile_cache", true);
    resource->CacheFuncGraph();
  }
}

void CheckInterpretNodeLineInfos() {
  auto &py_interpret_nodes = InterpretNodeRecorder::GetInstance().PyInterpretNodes();
  auto &py_execute_nodes = InterpretNodeRecorder::GetInstance().PyExecuteNodes();
  if (py_interpret_nodes.empty() && py_execute_nodes.empty()) {
    return;
  }

  std::stringstream ss;
  ss << "Found unsupported syntax in graph mode, those codes would be fallen back to Python interpreter:\n";
  // Dump for PyInterpret.
  ss << "----------------------------------------\n";
  ss << " After Parser Phase (total: " << py_interpret_nodes.size() << ")\n";
  ss << "----------------------------------------\n";
  size_t num = 1;
  for (const auto &node : py_interpret_nodes) {
    const auto line_info = trace::GetDebugInfoStr(node->debug_info());
    ss << "# No. " << num << ":\n" << line_info << "\n";
    ++num;
  }
  ss << "\n";
  // Dump for PyExecute.
  ss << "----------------------------------------\n";
  ss << " After Optimizer Phase (total: " << py_execute_nodes.size() << ")\n";
  ss << "----------------------------------------\n";
  num = 1;
  for (const auto &node : py_execute_nodes) {
    ss << "# No. " << num << ":\n";
    const auto &cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    const auto &weak_script_node = cnode->weak_input(1);
    const auto &script_node = weak_script_node.lock();
    MS_EXCEPTION_IF_NULL(script_node);
    const auto &script = GetValueNode<StringImmPtr>(script_node);
    // Usually the script is a value node.
    std::string script_str;
    if (script != nullptr) {
      script_str = script->value();
    } else {
      const auto &script_abs = script_node->abstract();
      if (script_abs != nullptr) {
        const auto script_abs_scalar = script_abs->cast<abstract::AbstractScalarPtr>();
        auto script_value = script_abs_scalar->BuildValue();
        MS_EXCEPTION_IF_NULL(script_value);
        auto script_value_str = script_value->cast<StringImmPtr>();
        MS_EXCEPTION_IF_NULL(script_value_str);
        script_str = script_value_str->value();
      }
    }
    if (!script_str.empty()) {
      ss << "Script: " << script_str << "\n\n";
    } else {
      ss << "Node: " << node->DebugString() << "\n\n";
    }
    const auto line_info = trace::GetDebugInfoStr(node->debug_info());
    ss << line_info << "\n";
    ++num;
  }
  ss << "\n";
  ss << "----------------------------------------\n";

  // Print the codes run in JIT Fallback.
  if (common::GetEnv("MS_DEV_FALLBACK_DUMP_NODE") == "1") {
    MS_LOG(ERROR) << ss.str();
  } else {
    MS_LOG(INFO) << ss.str();
  }
  InterpretNodeRecorder::GetInstance().Clear();
}

#ifdef ENABLE_DUMP_IR
std::string GetBaseNameForIR(int64_t stage_idx, const std::string &action_name) {
  std::ostringstream oss;
  int spaces = 2;
  oss << std::setfill('0') << std::setw(spaces) << stage_idx << "_" << action_name;
  return oss.str();
}

void RecordIR(const size_t action_index, const size_t action_size, const std::string &action_name,
              const FuncGraphPtr &graph, FuncGraphPtr *user_graph) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory) && graph != nullptr) {
    *user_graph = graph;
    std::string base_name = GetBaseNameForIR(SizeToLong(action_index), action_name);

    // Generate IR file in human-readable format
    static const auto switch_order = (common::GetEnv("MS_DEV_SAVE_GRAPHS_SORT_MODE") == "1");
    if (switch_order) {
      ExportIR(base_name + ".ir", graph);
    } else {
      DumpIR(base_name + ".ir", graph, true, kWholeStack);
    }
    if (context->CanDump(kFully)) {
      draw::Draw(base_name + ".dot", graph);
    }
  }
}
#endif

void SaveGraphForReadability(const std::string &action_name, const FuncGraphPtr &graph, const ResourcePtr &resource) {
  if (graph != nullptr && action_name.find("optimize") != string::npos) {
#ifdef ENABLE_DUMP_IR
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    if (context->CanDump(kIntroductory)) {
      DumpIRProto(graph, action_name);
    }
#endif
    resource->set_optimize_graph(graph);
  }
}

void Pipeline::Run() {
  MS_LOG(INFO) << "Pipeline run";
  MS_EXCEPTION_IF_NULL(resource_);
  FuncGraphPtr user_graph = nullptr;
  const std::string last_compile_action = kValidate;
  const std::string last_compile_action_for_compile_cache = kBackendPass;
  bool already_print_profile = false;
  static const auto compile_profile_finish_action = common::GetCompileConfig("COMPILE_PROFILE_FINISH_ACTION");
  ProfileExecute(MsProfile::GetProfile(), [this, &user_graph, &last_compile_action,
                                           &last_compile_action_for_compile_cache, &already_print_profile]() {
    size_t i = 0;
    for (auto &action : actions_) {
      std::string action_name = action.first;
      MSTX_RANGE_GUARD(action_name.c_str(), nullptr, profiler::MSTX_DOMAIN_MODEL_PREPARATION);
      MsProfileStatGuard stat_guard(std::move(action_name), "compile_action", true);
#ifdef ENABLE_TIMELINE
      DumpTime &dump_time = DumpTime::GetInstance();
      dump_time.Record(action.first, GetTime(), true);
#endif
      ProcessStatus::GetInstance().RecordStart(action.first);
      uint64_t start_time = profiler::GetClockSyscnt();
      bool result = true;
      ProfileExecute(MsProfile::GetProfile()->Step(action.first), [&result, &action, this]() {
        MS_LOG(INFO) << "Status record: start " << action.first << " action.";
        result = action.second(resource_);
        MS_LOG(INFO) << "Status record: end " << action.first << " action.";
        if (IS_OUTPUT_ON(mindspore::kInfo)) {
          auto func_graph = resource_->func_graph();
          MS_EXCEPTION_IF_NULL(func_graph);
          auto manager = func_graph->manager();
          MS_EXCEPTION_IF_NULL(manager);
          MS_LOG(INFO) << "Extra status record: total func graphs: " << manager->func_graphs().size()
                       << ", total nodes: " << manager->all_nodes().size();
        }
        if (common::GetCompileConfig("CHECK_PASS_NODE_SCOPE") == "1") {
          const auto &new_all_nodes = TopoSort(resource_->func_graph()->return_node(), SuccDeeperSimple);
          for (const auto &node : new_all_nodes) {
            validator::ValidateScope(node, action.first);
          }
        }
      });
      (void)profiler::CollectHostInfo(kCompiler, action.first, action.first, start_time, profiler::GetClockSyscnt(), 0);
      ProcessStatus::GetInstance().RecordEnd();
      if (!result) {
        MS_LOG(INTERNAL_EXCEPTION) << "Pipeline running to end, failed in step:" << action.first;
      }

      if (EnabledProfile() && compile_profile_finish_action == action.first) {
        ProfileExecuteBreak(MsProfile::GetProfile());
        MsProfile::Print();
        already_print_profile = true;
      }

      if (action.first == kTaskEmit) {
        SetLoopCount(resource_);
      } else if (action.first == last_compile_action) {
        CheckInterpretNodeLineInfos();
        ResetId(resource_);
      } else if (action.first == last_compile_action_for_compile_cache) {
        CacheFuncGraph(resource_);
      }
      FuncGraphPtr graph = resource_->func_graph();
#ifdef ENABLE_DUMP_IR
      RecordIR(i, actions_.size(), action.first, graph, &user_graph);
#endif
      SaveGraphForReadability(action.first, graph, resource_);
      i++;
#ifdef ENABLE_TIMELINE
      dump_time.Record(action.first, GetTime(), false);
#endif
    }
  });

  if (EnabledProfile()) {
    if (!already_print_profile) {
      MsProfile::Print();
    }
    MsProfile::Reset();
  }

#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory) && (user_graph != nullptr)) {
    if (context->CanDump(kFully)) {
      draw::DrawUserFuncGraph("ModelDigraph.dot", user_graph);
    }
  }
  if (common::GetEnv("DUMP_PARALLEL_INFO") == "1") {
    std::unordered_map<std::string, std::vector<uint32_t>> group_map;
    if (distributed::collective::CollectiveManager::instance()->initialized()) {
      group_map = distributed::collective::CollectiveManager::instance()->get_group_map();
    }
    if (parallel::g_device_manager == nullptr) {
      MS_LOG(WARNING) << "parallel::g_device_manager is not initialized. Skip dump parallel info.";
    } else {
      auto global_rank_id = parallel::g_device_manager->global_rank();
      DumpParallelJson("dump_parallel_info_" + std::to_string(global_rank_id) + ".json", resource_->func_graph(),
                       global_rank_id, group_map);
    }
  }
#endif
  MS_LOG(INFO) << "End";
}

std::string GetJitLevel() {
  const auto &jit_config = PhaseManager::GetInstance().jit_config();
  auto iter = jit_config.find("jit_level");
  if (iter != jit_config.end()) {
    return iter->second;
  }
  return "";
}
}  // namespace pipeline
}  // namespace mindspore
