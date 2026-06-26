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
#include "include/common/utils/parallel_context.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/step_auto_parallel.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/allreduce_fusion/step_allreduce_fusion.h"
#include "frontend/parallel/pass/handle_group_info.h"
#include "frontend/parallel/step_assigned_parallel.h"

#include "include/common/utils/config_manager.h"

#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "utils/phase.h"
#include "utils/interpret_node_recorder.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "mindspore/ccsrc/utils/ir_dump/dump_proto.h"
#include "frontend/jit/ps/fallback.h"
#include "mindspore/ccsrc/utils/ir_dump/draw.h"
#include "backend/backend_manager/backend_manager.h"
#include "runtime/hardware_abstract/utils.h"
#include "runtime/hardware_abstract/device_context/device_context.h"
#include "runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/backend/distributed/init.h"
#include "tools/profiler/profiling.h"
#include "tools/profiler/profiler.h"

#include "frontend/operator/py_execute_py.h"  // Only include one-time in the whole project.

namespace mindspore {
// namespace to support intermediate representation definition
namespace pipeline {
using Tensor = mindspore::tensor::Tensor;
using MetaTensor = mindspore::tensor::MetaTensor;
using MetaSparseTensor = mindspore::tensor::MetaSparseTensor;
using CSRTensor = mindspore::tensor::CSRTensor;
using COOTensor = mindspore::tensor::COOTensor;
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

bool CheckArgValid(const py::handle &arg) {
  if (py::isinstance<py::list>(arg) || py::isinstance<py::tuple>(arg)) {
    auto vector_arg = py::cast<py::list>(arg);
    return std::all_of(vector_arg.begin(), vector_arg.end(), CheckArgValid);
  }

  if (py::isinstance<py::dict>(arg)) {
    auto dict_arg = py::cast<py::dict>(arg);
    return std::all_of(dict_arg.begin(), dict_arg.end(), [](const auto &pair) { return CheckArgValid(pair.second); });
  }

  if (tensor::IsTensorPy(arg)) {
    auto tensor = tensor::ConvertToTensor(arg);
    MS_EXCEPTION_IF_NULL(tensor);
    if (tensor->data_type() == kNumberTypeBool) {
      MS_LOG(INFO) << "It is not recommended to use a tensor of bool data type as network input, which may cause "
                   << "operator compilation failure. For more details, please refer to the FAQ at "
                   << "https://mindspore.cn/search?[AddN]%20input(kNumberTypeBool.";
    }
  }

  return py::isinstance<py::int_>(arg) || py::isinstance<py::float_>(arg) || py::isinstance<py::none>(arg) ||
         py::isinstance<Number>(arg) || py::isinstance<py::str>(arg) || tensor::IsTensorPy(arg) ||
         py::isinstance<CSRTensor>(arg) || py::isinstance<COOTensor>(arg);
}

void RecordInitStatus() {
  static bool printed = false;
  if (!printed) {
    MS_LOG(INFO) << "Status record: system init.";
    printed = true;
  }
}

std::string ToOrdinal(const size_t &i) {
  auto suffix = "th";
  if (i == kIndex1) {
    suffix = "st";
  } else if (i == kIndex2) {
    suffix = "nd";
  } else if (i == kIndex3) {
    suffix = "rd";
  }
  return std::to_string(i) + suffix;
}

kernel::PyExecuteOutputUserDataPtr GetUserDataFromAddress(const py::object &res) {
  const auto allow_fallback_runtime = (fallback::GetJitSyntaxLevel() >= kCompatible);
  if (!allow_fallback_runtime) {
    return nullptr;
  }

  if (tensor::IsTensorPy(res)) {
    auto res_tensor = tensor::ConvertToTensor(res);
    MS_EXCEPTION_IF_NULL(res_tensor);
    if (res_tensor->has_user_data(kernel::PyExecuteOutputUserData::key)) {
      return res_tensor->GetUserData().get<kernel::PyExecuteOutputUserData>(kernel::PyExecuteOutputUserData::key);
    }
  }
  return nullptr;
}

template <typename T>
py::object GetVectorRefPyDataWithAbstract(const VectorRef &value_list, const abstract::AbstractSequencePtr &seq_abs) {
  auto value_size = value_list.size();
  auto ret = T(value_size);

  const auto allow_fallback_runtime = (fallback::GetJitSyntaxLevel() >= kCompatible);
  size_t ref_idx = 0;
  for (size_t i = 0; i < seq_abs->size(); ++i) {
    auto elem_abs = seq_abs->elements()[i];
    if (elem_abs->isa<abstract::AbstractNone>() && !allow_fallback_runtime) {
      continue;
    }
    ret[ref_idx] = BaseRefToPyDataWithUserData(value_list[ref_idx], elem_abs);
    ref_idx++;
  }
  if (ref_idx != value_size) {
    MS_LOG(EXCEPTION) << "The size of elements (excluding None) should be equal to " << value_size << ", but got "
                      << ref_idx;
  }
  return ret;
}

py::object GetVectorRefPyData(const VectorRef &value_list, const AbstractBasePtr &abs) {
  if (abs == nullptr || abs->isa<abstract::AbstractCSRTensor>() || abs->isa<abstract::AbstractCOOTensor>() ||
      abs->isa<abstract::AbstractAny>()) {
    return BaseRefToPyData(value_list, abs);
  }
  // Need to consider AbstractAny with vector ref scene later.
  if (!abs->isa<abstract::AbstractSequence>()) {
    MS_LOG(EXCEPTION) << "Can not convert vector ref with abstract " << abs->ToString();
  }
  auto seq_abs = abs->cast<abstract::AbstractSequencePtr>();
  if (seq_abs->dynamic_len()) {
    return BaseRefToPyData(value_list, abs);
  }
  if (seq_abs->isa<abstract::AbstractTuple>()) {
    return GetVectorRefPyDataWithAbstract<py::tuple>(value_list, seq_abs);
  }
  return GetVectorRefPyDataWithAbstract<py::list>(value_list, seq_abs);
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

py::object GetSelfFromArgs(const py::object &args) {
  if (!py::isinstance<py::tuple>(args)) {
    return py::object();
  }
  auto args_tuple = py::cast<py::tuple>(args);
  if (args_tuple.size() == 0) {
    return py::object();
  }
  py::object first_arg = args_tuple[0];
  if (!py::isinstance<Cell>(first_arg)) {
    return py::object();
  }
  return first_arg;
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

void CheckArgsValid(const py::object &source, const py::tuple &args) {
  if (!IS_OUTPUT_ON(mindspore::kInfo)) {
    return;
  }
  for (size_t i = 0; i < args.size(); i++) {
    if (!CheckArgValid(args[i])) {
      MS_LOG(INFO) << "The " << ToOrdinal(i + 1) << " arg type is " << args[i].get_type() << ", value is '"
                   << py::str(args[i]) << "'.";
    }
  }
}

py::bool_ VerifyInputSignature(const py::list &input_signature, const py::tuple &inputs) {
  MS_LOG(DEBUG) << "Verify args size:" << inputs.size();
  if (inputs.size() != input_signature.size()) {
    MS_LOG(ERROR) << "Signature size not equal to args size";
    return false;
  }

  size_t count = 0;
  for (auto arg_obj : inputs) {
    std::shared_ptr<Tensor> m_tensor = nullptr;
    bool is_tensor = false;
    if (tensor::IsTensorPy(arg_obj)) {
      m_tensor = tensor::ConvertToTensor(arg_obj);
      is_tensor = true;
    }
    if (is_tensor && m_tensor == nullptr) {
      MS_LOG(ERROR) << "Verify Tensor error, get ptr is null";
      return false;
    }

    if (m_tensor != nullptr) {
      MS_LOG(DEBUG) << "Verify Tensor";
      auto sig = tensor::ConvertToTensor(input_signature[count]);
      MS_EXCEPTION_IF_NULL(sig);
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

bool InitExecDataset(const std::string &queue_name, int64_t iter_num, int64_t batch_size,
                     const std::vector<TypePtr> &types, const std::vector<std::vector<int64_t>> &shapes,
                     const std::vector<int64_t> &input_indexes, const std::string &, bool need_run) {
  if (UseSimulationApi()) {
    return true;
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::string name = ms_context->backend_policy();
#ifdef WITH_BACKEND
  if (ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice) {
    if (ms_context->get_param<uint32_t>(MS_CTX_TSD_REF) <= 0) {
      InitPipeline();
    }
  }
#endif

  if (name == "ms" || name == "vm" || name == "ge") {
#ifdef WITH_BACKEND
    if (iter_num == -1) {
      iter_num = INT32_MAX;
    }
    bool status = InitExecDatasetVm(queue_name, iter_num, batch_size, types, shapes, input_indexes, need_run);
    return status;
#endif
  }
  return name == "ge" ? true : false;
}

bool InitExecDatasetVm(const std::string &queue_name, int64_t size, int64_t batch_size,
                       const std::vector<TypePtr> &types, const std::vector<std::vector<int64_t>> &shapes,
                       const std::vector<int64_t> &input_indexes, bool need_run) {
  PROF_START(InitExecDatasetVm);
  MS_LOG(INFO) << "Start InitDataSet Entry";
  mindspore::python_adapter::set_python_env_flag(true);
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
  auto app_init = std::make_shared<CNode>(AnfNodeWeakPtrList({NewValueNode(p_init)}), func_graph);
  func_graph->set_output(app_init);
  auto manager = MakeManager();
  manager->AddFuncGraph(func_graph);

  // AbstractNone indicates there is no output for this apply node.
  auto abstract_none = std::make_shared<abstract::AbstractNone>();
  app_init->set_abstract(abstract_none);
  // Before the graph compiling, need reset the iter num.
  ConfigManager::GetInstance().ResetIterNum();

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  MS_LOG(DEBUG) << "Enable mindRT.";
  context_ptr->set_param<bool>(MS_CTX_ENABLE_MINDRT, true);
  context_ptr->Refresh();

  VectorRef args;
  if (need_run) {
    VectorRef outputs;
    const auto &backend_jit_config = backend::BackendJitConfig::ParseBackendJitConfig();
    auto backend_ret =
      backend::BackendManager::GetInstance().Build(func_graph, backend_jit_config, backend_jit_config.backend);
    backend::BackendManager::GetInstance().Run(backend_ret.first, backend_ret.second, args, &outputs);
  }
  ConfigManager::GetInstance().set_iter_num(queue_name, size);
  PROF_END(InitExecDatasetVm);
  return true;
}

std::string GetJitLevel() {
  const auto &jit_config = PhaseManager::GetInstance().jit_config();
  auto iter = jit_config.find("jit_level");
  if (iter != jit_config.end()) {
    return iter->second;
  }
  return "";
}

void ResetOpId() { mindspore::id_generator::reset_id(); }
void ResetOpIdWithOffset() { mindspore::id_generator::reset_id_with_offset(); }

void InitHccl() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  ms_context->set_param<bool>(MS_CTX_ENABLE_HCCL, true);
#ifdef WITH_BACKEND
  auto backend = ms_context->backend_policy();
  if (backend == "ge") {
    if (!mindspore::distributed::Initialize()) {
      MS_LOG(EXCEPTION) << "InitHccl failed.";
    }
    InitPipeline();
    return;
  }
#endif
  mindspore::python_adapter::set_python_env_flag(true);
  std::string device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (ms_context->backend_policy() == "ms" && device_name == kAscendDevice) {
    if (!mindspore::distributed::Initialize()) {
      MS_LOG(EXCEPTION) << "InitHccl failed.";
    }
  }
}

void InitHccl(int64_t timeout, uint32_t world_size, uint32_t node_id, distributed::cluster::TCPStoreClientPtr store,
              std::string group_name) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  ms_context->set_param<bool>(MS_CTX_ENABLE_HCCL, true);
#ifdef WITH_BACKEND
  auto backend = ms_context->backend_policy();
  if (backend == "ge") {
    if (!mindspore::distributed::Initialize(timeout, world_size, node_id, store, group_name)) {
      MS_LOG(EXCEPTION) << "InitHccl failed.";
    }
    InitPipeline();
    return;
  }
#endif
  mindspore::python_adapter::set_python_env_flag(true);
  std::string device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (ms_context->backend_policy() == "ms" && device_name == kAscendDevice) {
    if (!mindspore::distributed::Initialize(timeout, world_size, node_id, store, group_name)) {
      MS_LOG(EXCEPTION) << "InitHccl failed.";
    }
  }
}

void FinalizeHccl() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
#ifdef WITH_BACKEND
  auto backend = ms_context->backend_policy();
  if (backend == "ge") {
    return;
  }
#endif
  device::DeviceContextManager::GetInstance().ClearDeviceContexts();
  device::DeviceContextManager::GetInstance().UnloadPlugin();
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

FuncGraphPtr LoadMindIR(const std::string &file_name, const char *dec_key, const size_t key_len,
                        const std::string &dec_mode, const py::object decrypt) {
  FuncGraphPtr func_graph = nullptr;
  if (dec_mode == "Customized") {
    py::bytes key_bytes(dec_key);
    py::bytes model_stream = decrypt(file_name, key_bytes);
    std::string model_string(model_stream);

    MindIRLoader mindir_loader;
    func_graph = mindir_loader.LoadMindIR(model_string.c_str(), model_string.size());
  } else {
    MindIRLoader mindir_loader(false, reinterpret_cast<const unsigned char *>(dec_key), key_len, dec_mode, false);
    func_graph = mindir_loader.LoadMindIR(file_name);
  }
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    DumpIR("load.ir", func_graph);
  }
#endif
  return func_graph;
}

FuncGraphPtr SplitMindIR(const std::string &file_name) {
  MS_LOG(INFO) << "Start split mindir";
  FuncGraphPtr func_graph = nullptr;
  MindIRLoader mindir_loader;
  func_graph = mindir_loader.LoadMindIR(file_name);
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Load MindIR file failed. Please check model file.";
    return nullptr;
  }
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    DumpIR("load.ir", func_graph);
  }
#endif
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  parallel_context->Reset();
  parallel_context->set_parallel_mode(parallel::kAutoParallel);
  parallel_context->set_strategy_search_mode(parallel::kRecursiveProgramming);
  parallel_context->set_direct_split(true);
  parallel_context->set_full_batch(true);
  parallel_context->set_group_ckpt_save_file("group_info");

  FuncGraphManagerPtr func_graph_manager = func_graph->manager();

  MS_LOG(INFO) << "func_graph_manager is not null";
  if (func_graph_manager == nullptr) {
    std::vector<FuncGraphPtr> graphs{func_graph};
    func_graph_manager = std::make_shared<FuncGraphManager>(graphs);
    func_graph_manager->AddFuncGraph(func_graph);
  }
  pipeline::ResourcePtr resource = std::make_shared<pipeline::Resource>();
  resource->set_manager(func_graph_manager);

  // Get the parameters items and add the value to args_abs.
  auto params = func_graph->parameters();
  auto inputs = func_graph->get_inputs();
  for (std::size_t i = 0; i < inputs.size(); i++) {
    auto input = inputs[i]->abstract();
    (void)parallel::ExtendInputArgsAbstractShape(input, i);
  }
  parallel::StepAutoParallel(func_graph, NULL);
  parallel::StepParallel(func_graph, NULL);
  parallel::StepAllreduceFusion(func_graph, NULL);
  resource->set_func_graph(func_graph);
  resource->set_manager(func_graph->manager());
  opt::irpass::OptimizeIRPassLib irpass;
  opt::OptPassConfig virtual_dataset = opt::OptPassConfig({irpass.virtual_dataset_eliminate_});
  opt::OptPassConfig virtual_output = opt::OptPassConfig({irpass.virtual_output_eliminate_});

  opt::OptPassGroupMap map_parallel_eliminate(
    {{"virtual_dataset", virtual_dataset}, {"virtual_output", virtual_output}});

  auto split_pass_opts = opt::Optimizer::MakeOptimizer("map_parallel_eliminate", resource, map_parallel_eliminate);
  ProfileExecute(MsProfile::GetProfile()->Step("split_pass_opts"),
                 [&split_pass_opts, &func_graph]() { func_graph = split_pass_opts->step(func_graph, true); });

  AbstractBasePtrList args_abs_list;
  (void)std::transform(params.begin(), params.end(), std::back_inserter(args_abs_list),
                       [](const AnfNodePtr &p) -> AbstractBasePtr { return p->abstract(); });
  func_graph = pipeline::Renormalize(resource, func_graph, args_abs_list);

  resource->set_args_abs(args_abs_list);

  MindIRExporter mindir_exporter;
  mindir_exporter.ExportProto(func_graph, "split_net", nullptr);

  parallel::HandleGroupInfo();

  return func_graph;
}

FuncGraphPtr SplitDynamicMindIR(const std::string &file_name, size_t device_num, size_t rank_id, bool sapp) {
  MS_LOG(INFO) << "Start split dynamic mindir for transformer network";
  FuncGraphPtr func_graph = nullptr;
  MindIRLoader mindir_loader;
  func_graph = mindir_loader.LoadMindIR(file_name);
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Load MindIR file failed. Please check model file.";
    return nullptr;
  }
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    DumpIR("load.ir", func_graph);
  }
#endif
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  parallel_context->Reset();
  parallel_context->set_parallel_mode(parallel::kAutoParallel);
  parallel_context->set_strategy_search_mode(parallel::kRecursiveProgramming);
  parallel_context->set_direct_split(true);
  parallel_context->set_full_batch(true);
  parallel_context->set_group_ckpt_save_file("group_info");

  for (size_t rank_id_iter = 0; rank_id_iter < device_num; rank_id_iter++) {
    auto tmp_func_graph = mindspore::BasicClone(func_graph);
    FuncGraphManagerPtr func_graph_manager = tmp_func_graph->manager();

    if (func_graph_manager == nullptr) {
      MS_LOG(INFO) << "func_graph_manager is null";
      std::vector<FuncGraphPtr> graphs{tmp_func_graph};
      func_graph_manager = std::make_shared<FuncGraphManager>(graphs);
      func_graph_manager->AddFuncGraph(tmp_func_graph);
    }

    auto inputs = tmp_func_graph->get_inputs();
    for (std::size_t i = 0; i < inputs.size(); i++) {
      auto input = inputs[i]->abstract();
      (void)parallel::ExtendInputArgsAbstractShape(input, i);
    }

    auto res = parallel::StepAssignedParallel(tmp_func_graph, func_graph_manager, device_num, rank_id_iter, sapp);
    if (!res) {
      MS_LOG(ERROR) << "StepAssignedParallel failed. Please check.";
      return nullptr;
    }
    pipeline::ResourcePtr resource = std::make_shared<pipeline::Resource>();
    resource->set_is_load(false);
    resource->set_manager(func_graph_manager);
    resource->set_func_graph(tmp_func_graph);
    // Get the parameters items and add the value to args_abs.
    auto params = tmp_func_graph->parameters();
    AbstractBasePtrList args_abs_list;
    (void)std::transform(params.begin(), params.end(), std::back_inserter(args_abs_list),
                         [](const AnfNodePtr &p) -> AbstractBasePtr { return p->abstract(); });
    tmp_func_graph = pipeline::Renormalize(resource, tmp_func_graph, args_abs_list);

#ifdef ENABLE_DUMP_IR
    auto re_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(re_context);
    if (re_context->CanDump(kIntroductory)) {
      string renormalize_net_name = "Renomalize_" + std::to_string(rank_id_iter) + ".ir";
      DumpIR(renormalize_net_name, tmp_func_graph);
    }
#endif

    parallel::HandleGroupInfo();
    string net_save_name = "split_net" + std::to_string(rank_id_iter);
    MindIRExporter mindir_exporter;
    res = mindir_exporter.ExportProto(tmp_func_graph, net_save_name, nullptr);
    if (!res) {
      MS_LOG(ERROR) << "Export MindIR file failed failed. Please check.";
      return nullptr;
    }
  }

  return func_graph;
}

void InitPipeline() {
  // set python env flag
  RecordInitStatus();
  mindspore::python_adapter::set_python_env_flag(true);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  CompileConfigManager::GetInstance().CollectCompileConfig();
}

void BindDeviceCtx() { device::DeviceContextManager::GetInstance().BindDeviceCtx(); }

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

py::bytes PyDecryptData(char *model_data, size_t data_size, char *key, size_t key_len, const std::string &dec_mode) {
  size_t decrypt_len;
  auto decrypt_data = mindspore::Decrypt(&decrypt_len, reinterpret_cast<Byte *>(model_data), data_size,
                                         reinterpret_cast<Byte *>(key), key_len, dec_mode);
  if (decrypt_data == nullptr) {
    MS_LOG(ERROR) << "Decrypt failed";
    return py::none();
  }
  auto py_decrypt_data = py::bytes(reinterpret_cast<char *>(decrypt_data.get()), decrypt_len);
  return py_decrypt_data;
}

bool PyIsCipherFile(const std::string &file_path) { return mindspore::IsCipherFile(file_path); }

void FinalizeCluster() {
#if defined(__linux__) && defined(WITH_BACKEND)
  if (distributed::cluster::ClusterContext::instance()->initialized()) {
    if (!distributed::cluster_exit_with_exception()) {
      MS_LOG(INFO) << "Start finalize the cluster instance.";
      // Finalize MindSpore cluster only when this process exits without any exception.
      (void)distributed::cluster::ClusterContext::instance()->Finalize(UINT32_MAX);
      MS_LOG(INFO) << "End finalize the cluster instance.";
    } else {
      (void)distributed::cluster::ClusterContext::instance()->StopThreadsOnException();
    }
  }
#endif
}

void SwapCache(const py::object &host_, const py::object &device_, const py::object &block_mapping_,
               const bool &is_device_to_host) {
  tensor::TensorPtr block_mapping = tensor::ConvertToTensor(block_mapping_);
  auto block_mapping_shape = block_mapping->shape();
  const size_t num_two = 2;
  if (block_mapping_shape.size() != num_two) {
    MS_LOG_EXCEPTION << "The shape size of Cache input mapping tensor should be 2, but got: "
                     << block_mapping_shape.size();
  }
  if (block_mapping_shape[kIndex1] != num_two) {
    MS_LOG_EXCEPTION << "The second dim of CacheKernel input mapping tensor should be 2, but got: "
                     << block_mapping_shape[0];
  }

  tensor::TensorPtr device = tensor::ConvertToTensor(device_);
  MS_EXCEPTION_IF_NULL(device);
  auto in_shape = device->shape();
  tensor::TensorPtr host = tensor::ConvertToTensor(host_);
  MS_EXCEPTION_IF_NULL(host_);
  auto type_byte = GetTypeByte(TypeIdToType(host->data_type()));
  size_t block_size_in_bytes = LongToSize(
    std::accumulate(in_shape.begin() + kIndex1, in_shape.end(), SizeToLong(type_byte), std::multiplies<int64_t>()));

  uint8_t *host_ptr = reinterpret_cast<uint8_t *>(host->data_c());
  MS_EXCEPTION_IF_NULL(host_ptr);
  auto device_addr = std::dynamic_pointer_cast<device::DeviceAddress>(device->device_address());
  MS_EXCEPTION_IF_NULL(device_addr);
  uint8_t *device_ptr = reinterpret_cast<uint8_t *>(const_cast<void *>(device_addr->GetPtr()));
  MS_EXCEPTION_IF_NULL(device_ptr);

  auto block_mapping_data = reinterpret_cast<int64_t *>(block_mapping->data_c());
  device::DeviceContextKey host_key = {device_addr->GetDeviceType(), device_addr->device_id()};
  device::DeviceContext *host_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_context);
  MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);

  host_context->device_res_manager_->SyncAllStreams();
  for (size_t i = 0; i < LongToSize(block_mapping_shape[0]); i++) {
    int64_t src_block_num = block_mapping_data[num_two * i];
    int64_t dst_block_num = block_mapping_data[num_two * i + kIndex1];
    size_t src_block_offset = LongToSize(src_block_num) * block_size_in_bytes;
    size_t dst_block_offset = LongToSize(dst_block_num) * block_size_in_bytes;
    if (is_device_to_host) {
      host_context->device_res_manager_->Copy(host_ptr + dst_block_offset, device_ptr + src_block_offset,
                                              block_size_in_bytes, device::CopyType::kD2H, device_addr->stream_id());
    } else {
      host_context->device_res_manager_->Copy(device_ptr + dst_block_offset, host_ptr + src_block_offset,
                                              block_size_in_bytes, device::CopyType::kH2D, device_addr->stream_id());
    }
  }
}

py::object BaseRefToPyDataWithUserData(const BaseRef &value, const AbstractBasePtr &abs) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kGraphExecutorPy, runtime::ProfilerEvent::kOutputProcess,
                                     "BaseRefToPyData");
  const auto allow_fallback_runtime = (fallback::GetJitSyntaxLevel() >= kCompatible);
  if (!allow_fallback_runtime) {
    return BaseRefToPyData(value, abs);
  }
  if (utils::isa<ValuePtr>(value)) {
    // Do not use abs as input to BaseRefToPyData, since the res need to be a tensor to get user data.
    auto res = BaseRefToPyData(value);
    const auto user_data = GetUserDataFromAddress(res);
    if (user_data != nullptr) {
      return user_data->obj;
    } else {
      MS_LOG(DEBUG) << "user data is empty";
    }
  } else if (utils::isa<VectorRef>(value)) {
    auto vec_ref = utils::cast<VectorRef>(value);
    return GetVectorRefPyData(vec_ref, abs);
  }
  return BaseRefToPyData(value, abs);
}

bool RunJitPipeline() {
  bool is_auto_parallel = (parallel::ParallelContext::GetInstance()->parallel_mode() == parallel::kSemiAutoParallel ||
                           parallel::ParallelContext::GetInstance()->parallel_mode() == parallel::kAutoParallel);
  if (is_auto_parallel || common::GetEnv("MS_DEV_JIT_PIPELINE") == "0") {
    return false;
  }
  return true;
}

std::string DumpFuncGraph(const py::object &obj) {
  auto func_graph = obj.cast<FuncGraphPtr>();
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "The python object is not a func_graph";
    return "";
  }
  std::ostringstream out_oss;
  DumpIR(out_oss, func_graph, false, kOff, false);
  return out_oss.str();
}

void PreJit(const py::object &args, const py::object &kwargs) {
  const auto &self = GetSelfFromArgs(args);
  parse::Parser::InitParserEnvironment(self);
}
}  // namespace pipeline
}  // namespace mindspore
