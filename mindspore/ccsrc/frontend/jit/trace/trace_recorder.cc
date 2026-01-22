/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "mindspore/ccsrc/frontend/jit/trace/trace_recorder.h"

#include <algorithm>
#include <mutex>
#include <utility>
#include <vector>

#include "ir/tensor_new.h"
#include "ir/dtype/tensor_type.h"
#include "utils/ms_context.h"
#include "utils/trace_info.h"
#include "frontend/operator/composite/do_signature.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "primitive/sequence_ops.h"
#include "primitive/structure_ops.h"
#include "primitive/framework_ops.h"
#include "include/frontend/jit/ps/executor/jit_executor_py.h"
#include "include/frontend/jit/trace/trace_recorder_interface.h"
#include "frontend/jit/ps/parse/parse_base.h"
#include "frontend/jit/ps/static_analysis/static_analysis.h"
#include "frontend/jit/ps/parse/resolve.h"
#include "frontend/jit/ps/static_analysis/prim.h"
#include "frontend/operator/ops_front_infer_function.h"
#include "include/utils/tensor_py.h"
#include "include/utils/pynative/grad_state.h"
#include "include/utils/pynative/adapter.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "frontend/jit/ps/parse/function_block.h"
#include "frontend/jit/ps/parse/data_converter.h"
#include "include/frontend/pybind_api.h"

namespace py = pybind11;
namespace mindspore {
constexpr size_t kPrimResultIndex = 0;
constexpr size_t kFileNameIndex = 1;
constexpr size_t kLineNumberIndex = 2;
constexpr size_t kArgsIndex = 3;
namespace trace {
namespace {
abstract::AbstractBasePtr GetAbstract(const py::object &obj) {
  ValuePtr val = nullptr;
  parse::ConvertData(obj, &val);
  MS_EXCEPTION_IF_NULL(val);
  const auto &abs = abstract::ToAbstract(val, nullptr, nullptr);
  MS_EXCEPTION_IF_NULL(abs);
  return abs;
}

std::string GetPyObjId(const py::object &obj) {
  py::object py_obj_str = python_adapter::CallPyFn(parse::PYTHON_MOD_PARSE_MODULE, parse::PYTHON_MOD_GET_OBJ_ID, obj);
  if (py::isinstance<py::none>(py_obj_str)) {
    MS_LOG(INTERNAL_EXCEPTION) << "The object has no id(), [" << py::str(obj.get_type()) << "] " << py::str(obj);
  }
  return py_obj_str.cast<std::string>();
}

bool IsMutable(const py::object &obj) {
  constexpr char mutable_attr[] = "__ms_mutable__";
  return py::hasattr(obj, mutable_attr) && py::cast<bool>(py::getattr(obj, mutable_attr));
}

CNodePtr GenerateCNode(const FuncGraphPtr &func_graph, const PrimitivePtr &prim, const AnfNodePtrList &args_inputs) {
  auto node_inputs = args_inputs;
  if (ops::IsPrimitiveFunction(prim->name())) {
    auto primitive = std::make_shared<Primitive>(prim->name());
    primitive->AddAttr("Converted", MakeValue(true));
    const auto &new_prim = std::make_shared<prim::DoTransPrimitiveFunction>(primitive);
    (void)node_inputs.insert(node_inputs.cbegin(), NewValueNode(new_prim));
  } else {
    prim->AddAttr("Converted", MakeValue(true));
    (void)node_inputs.insert(node_inputs.cbegin(), NewValueNode(prim));
  }
  return func_graph->NewCNodeInOrder(node_inputs);
}

py::object SyncTensor(const py::object &obj) {
  if (tensor::IsTensorPy(obj)) {
    const auto &tensor = tensor::ConvertToTensor(obj);
    MS_EXCEPTION_IF_NULL(tensor);
    auto cpu_tensor = tensor->cpu();
    py::object new_tensor_obj = PackTensorToPyObject(cpu_tensor);
    return new_tensor_obj;
  } else if (py::isinstance<py::tuple>(obj)) {
    const py::tuple &obj_tuple = py::cast<py::tuple>(obj);
    py::tuple new_tuple_obj(obj_tuple.size());
    for (size_t i = 0; i < obj_tuple.size(); ++i) {
      auto new_obj = SyncTensor(obj_tuple[i]);
      new_tuple_obj[i] = new_obj;
    }
    return new_tuple_obj;
  } else if (py::isinstance<py::list>(obj)) {
    const py::list &obj_list = py::cast<py::list>(obj);
    py::list new_list_obj(obj_list.size());
    for (size_t i = 0; i < obj_list.size(); ++i) {
      auto new_obj = SyncTensor(obj_list[i]);
      new_list_obj[i] = new_obj;
    }
    return new_list_obj;
  }
  return obj;
}

DebugInfoPtr GenerateDebugInfos(const py::list &file_names, const py::list &linenos, const std::string &name = "") {
  if (file_names.size() == 0 || linenos.size() == 0 || file_names.size() != linenos.size()) {
    MS_LOG(EXCEPTION) << "Wrong line info list size, " << file_names.size() << ", " << linenos.size();
  }
  DebugInfoPtr debug_info = nullptr;
  TraceInfoPtr trace_info = nullptr;
  for (size_t i = file_names.size(); i > 0; --i) {
    const auto &file_name = file_names[i - 1];
    const auto &lineno = linenos[i - 1];
    const auto location = std::make_shared<Location>(py::cast<std::string>(file_name), py::cast<int>(lineno), 0,
                                                     py::cast<int>(lineno), 0, "", std::vector<std::string>());
    debug_info = std::make_shared<DebugInfo>(location);
    if (trace_info != nullptr) {
      debug_info->set_trace_info(trace_info);
    } else if (!name.empty()) {  // Set name for root debug info.
      debug_info->set_name(name);
    }
    trace_info = MakeTraceInfo<TraceOpt>(debug_info);
  }
  return debug_info;
}

AnfNodePtr GetInt(const py::object &obj, const DebugInfoPtr &, bool set_abstract) {
  MS_LOG(DEBUG) << "Constant int64_t: " << py::str(obj);
  const auto &value_node = NewValueNode(py::cast<int64_t>(obj));
  if (set_abstract) {
    value_node->set_abstract(GetAbstract(obj));
  }
  return value_node;
}

AnfNodePtr GetFloat(const py::object &obj, const DebugInfoPtr &, bool set_abstract) {
  MS_LOG(DEBUG) << "Constant float: " << py::str(obj);
  auto data = py::cast<float>(obj);
  const auto &value_node = NewValueNode(data);
  auto fp32_val = value_node->value()->cast<FP32ImmPtr>();
  if (fp32_val != nullptr) {
    MS_LOG(DEBUG) << "Set float64 value to FP32Imm.";
    fp32_val->set_prim_value(py::cast<double>(obj));
  }
  if (set_abstract) {
    value_node->set_abstract(GetAbstract(obj));
  }
  return value_node;
}

AnfNodePtr GetStr(const py::object &obj, const DebugInfoPtr &, bool set_abstract) {
  MS_LOG(DEBUG) << "Constant str: " << py::str(obj);
  const auto &value_node = NewValueNode(py::cast<std::string>(obj));
  if (set_abstract) {
    value_node->set_abstract(GetAbstract(obj));
  }
  return value_node;
}

AnfNodePtr GetNone(const py::object &obj, const DebugInfoPtr &, bool set_abstract) {
  MS_LOG(DEBUG) << "Constant none: " << py::str(obj);
  const auto &value_node = NewValueNode(kNone);
  if (set_abstract) {
    value_node->set_abstract(GetAbstract(obj));
  }
  return value_node;
}

AnfNodePtr GetEllipsis(const py::object &obj, const DebugInfoPtr &, bool set_abstract) {
  MS_LOG(DEBUG) << "Constance ellipsis: " << py::str(obj);
  const auto &value_node = NewValueNode(kEllipsis);
  if (set_abstract) {
    value_node->set_abstract(GetAbstract(obj));
  }
  return value_node;
}

AnfNodePtr GetType(const py::object &obj, const DebugInfoPtr &, bool set_abstract) {
  MS_LOG(DEBUG) << "Constance type: " << py::str(obj);
  const auto &type_node = NewValueNode(obj.cast<TypePtr>());
  if (set_abstract) {
    type_node->set_abstract(GetAbstract(obj));
  }
  return type_node;
}

AnfNodePtr GetBool(const py::object &obj, const DebugInfoPtr &, bool set_abstract) {
  MS_LOG(DEBUG) << "Constant bool: " << py::str(obj);
  const auto &value_node = NewValueNode(py::cast<bool>(obj));
  if (set_abstract) {
    value_node->set_abstract(GetAbstract(obj));
  }
  return value_node;
}

AnfNodePtr GetSlice(const py::object &obj, const DebugInfoPtr &, bool set_abstract) {
  MS_LOG(DEBUG) << "Constant slice: " << py::str(obj);
  const auto &value_node = NewValueNode(parse::ConvertSlice(obj));
  if (set_abstract) {
    value_node->set_abstract(GetAbstract(obj));
  }
  return value_node;
}

// Convert the data according to instance type
using InstanceCheckFunc = std::function<bool(const py::object &)>;
using InstanceConvertFunc = std::function<AnfNodePtr(const py::object &, const DebugInfoPtr &, bool)>;
class DataConvertFunc {
 public:
  explicit DataConvertFunc(InstanceConvertFunc convert_func) : convert_func_(std::move(convert_func)) {}

  virtual ~DataConvertFunc() = default;

  virtual bool Matched(const py::object &obj) = 0;

  AnfNodePtr GetConstNode(const py::object &obj, const DebugInfoPtr &debug_info, bool set_abstract) {
    if (convert_func_ == nullptr) {
      MS_LOG(INTERNAL_EXCEPTION) << "convert func is null";
    }
    return convert_func_(obj, debug_info, set_abstract);
  }

 private:
  InstanceConvertFunc convert_func_ = nullptr;
};

// Convert the data according to instance type
template <typename T>
class ByTypeDataConvertFunc : public DataConvertFunc {
 public:
  explicit ByTypeDataConvertFunc(const InstanceConvertFunc &convert_func)
      : DataConvertFunc(convert_func), check_func_(py::isinstance<T>) {}

  ~ByTypeDataConvertFunc() override = default;

  bool Matched(const py::object &obj) override { return check_func_ != nullptr ? check_func_(obj) : false; }

 private:
  InstanceCheckFunc check_func_ = nullptr;
};

using DataConvertFuncPtr = std::shared_ptr<DataConvertFunc>;

static const std::vector<DataConvertFuncPtr> &GetDataConvertFuncs() {
  // Convert data by python object type.
  static const std::vector<DataConvertFuncPtr> data_convert_funcs{
    std::make_shared<ByTypeDataConvertFunc<py::bool_>>(GetBool),
    std::make_shared<ByTypeDataConvertFunc<py::int_>>(GetInt),
    std::make_shared<ByTypeDataConvertFunc<py::float_>>(GetFloat),
    std::make_shared<ByTypeDataConvertFunc<py::str>>(GetStr),
    std::make_shared<ByTypeDataConvertFunc<py::none>>(GetNone),
    std::make_shared<ByTypeDataConvertFunc<py::ellipsis>>(GetEllipsis),
    std::make_shared<ByTypeDataConvertFunc<py::slice>>(GetSlice),
    std::make_shared<ByTypeDataConvertFunc<Type>>(GetType),
  };
  return data_convert_funcs;
}
}  // namespace

py::object DefaultOutput() { return PackTensorToPyObject(tensor::from_scalar(0.0)); }

PrimitivePtr ChangeInplacePrim(const PrimitivePtr &prim) {
  // Some primitives generated by pynative are not fitting graph process and need to be skipped.
  static std::unordered_map<std::string, PrimitivePtr> PrimitivesNeedToChange{{"InplaceIndexPut", nullptr},
                                                                              {"InplaceCopy", nullptr},
                                                                              {"InplaceMul", prim::kPrimMul},
                                                                              {"InplaceAddsExt", prim::kPrimAddExt}};
  auto it = PrimitivesNeedToChange.find(prim->name());
  if (it != PrimitivesNeedToChange.end()) {
    return it->second;
  }
  return prim;
}

void Capture(const py::args &args, py::object *res) {
  if (!IsTracing()) {
    return;
  }
  *res = CaptureRun(py::args(py::tuple(args[1])), *res, args[0]);
}

void CapturePy(PyObject *args, PyObject **res) {
  if (!IsTracing()) {
    return;
  }
  py::args py_args = py::reinterpret_borrow<py::args>(args);
  py::object py_res = py::reinterpret_borrow<py::object>(*res);
  py::object capture_res = CaptureRun(py::args(py::tuple(py_args[1])), py_res, py_args[0]);
  *res = capture_res.ptr();
}

void Capture(const py::list &args, const PrimitivePtr &prim, py::object *res) {
  if (!IsTracing()) {
    return;
  }
  auto jit_context = python_adapter::CallPyFn("mindspore.graph.jit_context", "jit_context");
  std::string method = "prepare_op";
  py::tuple op_info = jit_context.attr(method.c_str())(prim->name(), res, *args);
  const py::object &prim_res = op_info[kPrimResultIndex];
  const py::list &file_names = op_info[kFileNameIndex];
  const py::list &linenos = op_info[kLineNumberIndex];
  const py::tuple &inputs = op_info[kArgsIndex];
  const auto debug_info = GenerateDebugInfos(file_names, linenos);
  trace::TraceRecorder::GetInstance()->ProcessNewNode(prim, prim_res, debug_info, py::args(inputs), false);
  *res = prim_res;
}

void CapturePy(PyObject *args, const PrimitivePtr &prim, PyObject **res) {
  if (!IsTracing()) {
    return;
  }
  py::list py_args = py::reinterpret_borrow<py::list>(args);
  py::object py_res = py::reinterpret_borrow<py::object>(*res);
  auto jit_context = python_adapter::CallPyFn("mindspore.graph.jit_context", "jit_context");
  std::string method = "prepare_op";
  py::tuple op_info = jit_context.attr(method.c_str())(prim->name(), py_res, *py_args);
  const py::object &prim_res = op_info[kPrimResultIndex];
  const py::list &file_names = op_info[kFileNameIndex];
  const py::list &linenos = op_info[kLineNumberIndex];
  const py::tuple &inputs = op_info[kArgsIndex];
  const auto debug_info = GenerateDebugInfos(file_names, linenos);
  trace::TraceRecorder::GetInstance()->ProcessNewNode(prim, prim_res, debug_info, py::args(inputs), false);
  *res = prim_res.ptr();
}

void CaptureResolveOperation(const py::tuple &args, const std::string &named_primitive, py::object *res) {
  if (!IsTracing()) {
    return;
  }
  auto jit_context = python_adapter::CallPyFn("mindspore.graph.jit_context", "jit_context");
  std::string method = "prepare_op";
  py::tuple op_info = jit_context.attr(method.c_str())(named_primitive, res, *args);
  const py::object &prim_res = op_info[kPrimResultIndex];
  const py::list &file_names = op_info[kFileNameIndex];
  const py::list &linenos = op_info[kLineNumberIndex];
  const py::tuple &inputs = op_info[kArgsIndex];
  auto trope =
    python_adapter::GetPyObjAttr(python_adapter::GetPyModule("mindspore.graph._parse.resources"), "trope_ns");
  parse::NameSpacePtr name_space = std::make_shared<parse::NameSpace>("CommonOPS", trope);
  parse::SymbolPtr symbol = std::make_shared<parse::Symbol>(named_primitive);
  MS_LOG(DEBUG) << "name_space: " << name_space->ToString() << ", symbol: " << symbol->ToString();
  const auto debug_info = GenerateDebugInfos(file_names, linenos);
  trace::TraceRecorder::GetInstance()->ProcessNewResolveNode(name_space, symbol, prim_res, debug_info, py::args(inputs),
                                                             false);
  *res = prim_res;
}

void Capture(const std::vector<py::object> &args_vec, const PrimitivePtr &prim, py::object *res) {
  if (!IsTracing()) {
    return;
  }
  py::tuple args(args_vec.size());
  for (size_t i = 0; i < args_vec.size(); i++) {
    args[i] = args_vec[i];
  }
  auto jit_context = python_adapter::CallPyFn("mindspore.graph.jit_context", "jit_context");
  std::string method = "prepare_op";
  py::tuple op_info = jit_context.attr(method.c_str())(prim->name(), res, *args);
  const py::object &prim_res = op_info[kPrimResultIndex];
  const py::list &file_names = op_info[kFileNameIndex];
  const py::list &linenos = op_info[kLineNumberIndex];
  const py::tuple &inputs = op_info[kArgsIndex];
  const auto debug_info = GenerateDebugInfos(file_names, linenos);
  trace::TraceRecorder::GetInstance()->ProcessNewNode(prim, prim_res, debug_info, py::args(inputs), false);
  *res = prim_res;
}

void CapturePy(const std::vector<PyObject *> &args_vec, const PrimitivePtr &prim, PyObject **res) {
  if (!IsTracing()) {
    return;
  }
  py::tuple args(args_vec.size());
  for (size_t i = 0; i < args_vec.size(); ++i) {
    args[i] = py::reinterpret_borrow<py::object>(args_vec[i]);
  }
  py::object py_res = py::reinterpret_borrow<py::object>(*res);
  auto jit_context = python_adapter::CallPyFn("mindspore.graph.jit_context", "jit_context");
  std::string method = "prepare_op";
  py::tuple op_info = jit_context.attr(method.c_str())(prim->name(), py_res, *args);
  const py::object &prim_res = op_info[kPrimResultIndex];
  const py::list &file_names = op_info[kFileNameIndex];
  const py::list &linenos = op_info[kLineNumberIndex];
  const py::tuple &inputs = op_info[kArgsIndex];
  const auto debug_info = GenerateDebugInfos(file_names, linenos);
  trace::TraceRecorder::GetInstance()->ProcessNewNode(prim, prim_res, debug_info, py::args(inputs), false);
  *res = prim_res.ptr();
}

py::object CaptureRun(const py::args &args, const py::object &res, const py::object &prim_py) {
  // Capture node from trace func.
  auto jit_context = python_adapter::CallPyFn("mindspore.graph.jit_context", "jit_context");
  std::string method = "run_op";
  return jit_context.attr(method.c_str())(prim_py, res, *args);
}

bool IsTracing() { return trace::TraceRecorder::GetInstance()->BuildingTraceGraph(); }

void PassNodeInArg(PyObject *origin_obj, PyObject *new_obj) {
  if (IsTracing()) {
    auto origin_object_py = py::reinterpret_borrow<py::object>(origin_obj);
    auto new_object_py = py::reinterpret_borrow<py::object>(new_obj);
    trace::TraceRecorder::GetInstance()->PassNode(origin_object_py, new_object_py);
  }
}

bool Compiled() {
  auto jit_context = python_adapter::CallPyFn("mindspore.graph.jit_context", "jit_context");
  if (py::isinstance<py::none>(jit_context)) {
    return false;
  }
  std::string compile_flag = "compiled";
  py::bool_ compiled = jit_context.attr(compile_flag.c_str());
  return py::cast<bool>(compiled);
}

void TraceRecorder::Clear() {
  // Clear the AnfNode in python object.
  py_obj_node_map_.clear();
  std::stack<FuncGraphPtr>().swap(graph_stack_);
  side_effect_nodes_.clear();
  args_ = py::tuple();
  phase_.clear();
}

void TraceRecorder::PassNode(const py::object &origin_obj, const py::object &new_obj) {
  auto origin_node = GetNode(origin_obj, nullptr);
  if (origin_node == nullptr) {
    return;
  }
  SetNode(new_obj, origin_node, origin_node->debug_info());
}

FuncGraphPtr TraceRecorder::InitTopGraph(const DebugInfoPtr &debug_info) {
  if (!graph_stack_.empty()) {
    Clear();
    (void)python_adapter::CallPyFn("mindspore.graph.jit_context", "set_jit_context", py::none());
    MS_LOG(EXCEPTION) << "The maximum nesting level for using the jit trace and jit ast decorators is one."
                      << " Please check the current code for its nesting usage.";
  }
  auto fg_debug_info = std::make_shared<GraphDebugInfo>(MakeTraceInfo<TraceOpt>(debug_info));
  const auto new_graph = std::make_shared<FuncGraph>(std::move(fg_debug_info));
  graph_stack_.push(new_graph);
  return new_graph;
}

void TraceRecorder::BeginGraph(const py::object &func_name, const py::object &phase, const py::list &file_names,
                               const py::list &linenos, const py::args &args) {
  phase_ = py::cast<std::string>(phase);
  args_ = args;
  // Normalize the name and set as debug name.
  auto function_name = py::cast<std::string>(func_name);
  std::replace(function_name.begin(), function_name.end(), '.', '_');
  function_name += "__trace_";
  const auto debug_info = GenerateDebugInfos(file_names, linenos, function_name);
  const auto new_graph = InitTopGraph(debug_info);
  MS_LOG(DEBUG) << "Start build graph, " << new_graph << "/" << new_graph->ToString() << ", arg size: " << args.size()
                << ", args: " << py::str(py::cast<py::object>(args)) << ", phase_: " << phase_;
  for (size_t i = 0; i < args.size(); ++i) {
    const auto &param = new_graph->add_parameter();
    std::stringstream param_name_buffer;
    param_name_buffer << "arg" << i;
    const auto &param_name = param_name_buffer.str();
    param->set_name(param_name);
    if (param->debug_info() != nullptr) {
      param->debug_info()->set_name(param_name);
    }
    SetNode(args[i], param, debug_info);
  }
}

FuncGraphPtr TraceRecorder::BuildEndGraph(const py::list &file_names, const py::list &linenos,
                                          const py::args &output_args, bool nested) {
  auto func_graph = graph_stack_.top();
  MS_LOG(DEBUG) << "End build graph, " << func_graph << "/" << func_graph->ToString()
                << ", output_args: " << py::str(py::cast<py::object>(output_args)) << ", phase_: " << phase_;
  const auto debug_info = GenerateDebugInfos(file_names, linenos);
  if (output_args.size() == 1) {  // Maybe function output.
    const auto &fn_res = output_args[0];
    const auto &output_node = GetNode(fn_res, debug_info);
    func_graph->set_output(output_node);
    parse::AttachIsolatedNodes(func_graph, side_effect_nodes_);
  } else {  // Definitely jit block multiple outputs.
    AnfNodePtrList make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
    for (size_t i = 0; i < output_args.size(); ++i) {
      (void)make_tuple_inputs.emplace_back(GetNode(output_args[i], debug_info));
    }
    auto make_tuple_node = func_graph->NewCNode(make_tuple_inputs);
    func_graph->set_output(make_tuple_node);
    parse::AttachIsolatedNodes(func_graph, side_effect_nodes_);
  }
  MS_EXCEPTION_IF_NULL(func_graph->return_node());
  if (func_graph->return_node()->debug_info() != nullptr) {
    func_graph->return_node()->debug_info()->set_trace_info(MakeTraceInfo<TraceOpt>(debug_info));
  }
  MS_LOG(DEBUG) << "End build graph, " << func_graph << "/" << func_graph->ToString() << ", phase_: " << phase_;
#ifdef ENABLE_DUMP_IR
  const auto &context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    DumpIR("jit_trace_" + func_graph->ToString() + ".ir", func_graph);
  }
#endif
  if (nested) {
    graph_stack_.pop();
    Clear();
  }
  return func_graph;
}

void TraceRecorder::EndGraph(const py::list &file_names, const py::list &linenos, const py::dict &jit_config,
                             const py::args &output_args) {
  const auto &func_graph = BuildEndGraph(file_names, linenos, output_args);
  // Run compile pipeline with func graph.
  auto graph_executor = pipeline::GetExecutor();
  graph_executor->SetJitConfig(jit_config);
  (void)graph_executor->CompileInner(func_graph, args_, py::dict(), phase_, true);
  MS_LOG(DEBUG) << "End compile pipeline.";
  graph_stack_.pop();
  Clear();
}

py::object TraceRecorder::RunGraph(const py::object &phase, const py::dict &jit_config, const py::tuple &args) {
  MS_LOG(DEBUG) << "Run graph, arg size: " << args.size() << ", args: " << py::str(py::cast<py::object>(args))
                << ", phase: " << phase;
  auto graph_executor = pipeline::GetExecutor();
  graph_executor->SetJitConfig(jit_config);
  MS_EXCEPTION_IF_NULL(graph_executor);
  py::object res;
  if (pynative::GradState::Get().RequiresGrad()) {
    pynative::PyNativeAdapter::SetGraphPhase(py::cast<std::string>(phase));
    FuncGraphPtr jit_fg = graph_executor->GetFuncGraph(py::cast<std::string>(phase));
    MS_EXCEPTION_IF_NULL(jit_fg);
    if (args.size() > jit_fg->parameters().size()) {
      MS_LOG(EXCEPTION) << "The number of inputs: " << args.size()
                        << " should not greater than the number of parameters,which is : "
                        << jit_fg->parameters().size()
                        << ". Please make sure all of the inputs were used in trace block.";
    }
    res = pynative::PyNativeAdapter::GradJit(args);
    // Update forward graph with fprop graph.
    FuncGraphPtr grad_jit_fg = graph_executor->GetJitGradGraph(py::cast<std::string>(phase));
    MS_EXCEPTION_IF_NULL(grad_jit_fg);
#ifdef ENABLE_DUMP_IR
    const auto &context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    if (context->CanDump(kIntroductory)) {
      DumpIR("jit_trace_run_" + jit_fg->ToString() + ".ir", jit_fg);
      DumpIR("jit_trace_run_grad_" + grad_jit_fg->ToString() + ".ir", grad_jit_fg);
    }
#endif
    MS_LOG(DEBUG) << "jit_fg: " << jit_fg->ToString() << ", modify_output: " << jit_fg->modify_output()
                  << ", grad_jit_fg: " << grad_jit_fg->ToString();
    if (jit_fg->modify_output()) {
      res = py::cast<py::tuple>(res)[0];
    }
  } else {
    res = graph_executor->Run(args, phase);
  }
  if (IS_OUTPUT_ON(mindspore::kDebug)) {
    MS_LOG(DEBUG) << "return res: " << py::str(SyncTensor(res));
  }
  return res;
}

AnfNodePtr TraceRecorder::ConvertParameterObj(const py::object &input_obj) {
  auto top_func_graph = graph_stack_.top();
  // Parameter object should not be none
  if (py::isinstance<py::none>(input_obj)) {
    MS_LOG(EXCEPTION) << "Resolve class Parameter error because obj is null.";
  }
  if (!py::hasattr(input_obj, "name")) {
    MS_LOG(EXCEPTION) << "Resolve class Parameter error: cannot find name attr for obj";
  }
  // Get the parameter name from parameter object
  auto name_attr = python_adapter::GetPyObjAttr(input_obj, "name");
  if (py::isinstance<py::none>(name_attr)) {
    MS_LOG(EXCEPTION) << "Parameter object should have name attribute";
  }
  const auto &param_name = py::cast<std::string>(name_attr);
  auto value = parse::GetParameterValue(input_obj);
  MS_LOG(DEBUG) << "Created a new weight parameter for " << top_func_graph->ToString() << ", param: " << param_name;
  return top_func_graph->AddFvParameter(param_name, value);
}

void TraceRecorder::NewFuncGraphNode(const py::tuple &info, const py::args &inputs) {
  constexpr size_t kPhaseIndex = 3;
  constexpr size_t kIsNested = 4;
  const py::bool_ &is_nested = info[kIsNested];
  if (is_nested) {
    Clear();
    (void)python_adapter::CallPyFn("mindspore.graph.jit_context", "set_jit_context", py::none());
    MS_LOG(EXCEPTION) << "The maximum nesting level for using the jit trace and jit ast decorators is one."
                      << " Please check the current code for its nesting usage.";
  }
  const py::object &phase = info[kPhaseIndex];
  const py::object &prim_res = info[kPrimResultIndex];
  const py::list &file_names = info[kFileNameIndex];
  const py::list &linenos = info[kLineNumberIndex];
  auto graph_executor = pipeline::GetExecutor();
  MS_EXCEPTION_IF_NULL(graph_executor);
  FuncGraphPtr jit_fg = graph_executor->GetFuncGraph(py::cast<std::string>(phase));
  const auto debug_info = GenerateDebugInfos(file_names, linenos);
  std::pair<AnfNodePtrList, AbstractBasePtrList> input_pair = GenerateInputs(inputs, debug_info);
  AnfNodePtrList node_inputs = input_pair.first;
  const size_t fv_param_count = jit_fg->fv_param_count();
  if (fv_param_count > 0) {
    const auto &parameters = jit_fg->parameters();
    const size_t fv_position = parameters.size() - fv_param_count;
    for (size_t i = fv_position; i < parameters.size(); ++i) {
      const auto &param = parameters[i]->cast<ParameterPtr>();
      const auto &new_param = graph_stack_.top()->AddFvParameter(param->name(), param->default_param());
      (void)node_inputs.emplace_back(new_param);
    }
  }
  parse::ClearCNodeAbstract(jit_fg);
  (void)node_inputs.insert(node_inputs.cbegin(), NewValueNode(jit_fg));
  AnfNodePtr cnode = graph_stack_.top()->NewCNodeInOrder(node_inputs);
  if (cnode->debug_info() != nullptr) {
    cnode->debug_info()->set_trace_info(MakeTraceInfo<TraceOpt>(debug_info));
  }
  MS_LOG(DEBUG) << "New cnode: " << cnode->DebugString();
  SetNode(prim_res, cnode, debug_info);
}
void TraceRecorder::NewNode(const py::object &prim_obj, const py::tuple &op_info, const py::args &inputs) {
  constexpr size_t kDoSignatureIndex = 3;
  const py::object &prim_res = op_info[kPrimResultIndex];
  const py::list &file_names = op_info[kFileNameIndex];
  const py::list &linenos = op_info[kLineNumberIndex];
  const py::bool_ &do_signature = op_info[kDoSignatureIndex];
  const auto &prim_py = std::make_shared<PrimitivePy>(prim_obj);
  bool signature = py::cast<bool>(do_signature);
  const auto debug_info = GenerateDebugInfos(file_names, linenos);
  ProcessNewNode(prim_py, prim_res, debug_info, inputs, signature);
}

std::pair<AnfNodePtrList, AbstractBasePtrList> TraceRecorder::GenerateInputs(const py::args &inputs,
                                                                             const DebugInfoPtr &debug_info) {
  AnfNodePtrList node_inputs;
  AbstractBasePtrList abs_inputs;
  for (size_t i = 0; i < inputs.size(); ++i) {
    AnfNodePtr node;
    auto input_obj = inputs[i];
    bool is_parameter = py::hasattr(input_obj, "__parameter__") && tensor::IsTensorPy(input_obj);
    // When the input of a cnode is a weight, add it to the top graph.
    if (is_parameter) {
      node = ConvertParameterObj(input_obj);
    } else {
      node = GetNode(inputs[i], debug_info);
      MS_EXCEPTION_IF_NULL(node);
    }
    (void)node_inputs.emplace_back(node);
    if (node->abstract() != nullptr) {
      (void)abs_inputs.emplace_back(node->abstract());
    } else {
      (void)abs_inputs.emplace_back(GetAbstract(input_obj));
    }
    MS_LOG(DEBUG) << "Add input, " << node->DebugString();
  }
  return std::make_pair(node_inputs, abs_inputs);
}

void TraceRecorder::ProcessNewNode(const PrimitivePtr &prim, const py::object &prim_res, const DebugInfoPtr &debug_info,
                                   const py::args &inputs, bool do_signature) {
  MS_LOG(DEBUG) << "NewNode, prim: " << prim->name() << ", prim_res: [" << py::str(prim_res.get_type()) << "] "
                << GetPyObjId(prim_res) << "/" << py::str(prim_res) << ", inputs size: " << inputs.size()
                << ", inputs: " << py::str(py::cast<py::object>(inputs));
  auto new_prim = ChangeInplacePrim(prim);
  if (new_prim == nullptr) {
    return;
  }
  std::pair<AnfNodePtrList, AbstractBasePtrList> input_pair = GenerateInputs(inputs, debug_info);
  AnfNodePtrList node_inputs = input_pair.first;
  AbstractBasePtrList abs_inputs = input_pair.second;
  AnfNodePtr cnode;
  if (do_signature) {
    cnode = prim::GenerateCNodeBySignatures(graph_stack_.top(), new_prim->name(), new_prim, abs_inputs, node_inputs);
  } else {
    cnode = GenerateCNode(graph_stack_.top(), new_prim, node_inputs);
  }
  if (cnode->debug_info() != nullptr) {
    cnode->debug_info()->set_trace_info(MakeTraceInfo<TraceOpt>(debug_info));
  }
  MS_LOG(DEBUG) << "New cnode: " << cnode->DebugString();
  if (GetPrimEffectInfo(new_prim).HasEffect()) {
    side_effect_nodes_.add(cnode);
    return;
  }
  SetNode(prim_res, cnode, debug_info);
}

void TraceRecorder::ProcessNewResolveNode(const parse::NameSpacePtr &name_space, const parse::SymbolPtr &resolve_symbol,
                                          const py::object &prim_res, const DebugInfoPtr debug_info,
                                          const py::args &inputs, bool do_signature) {
  MS_LOG(DEBUG) << "NewNode, prim: " << resolve_symbol->symbol() << ", prim_res: [" << py::str(prim_res.get_type())
                << "] " << GetPyObjId(prim_res) << "/" << py::str(prim_res) << ", inputs size: " << inputs.size()
                << ", inputs: " << py::str(py::cast<py::object>(inputs));
  std::pair<AnfNodePtrList, AbstractBasePtrList> input_pair = GenerateInputs(inputs, debug_info);
  AnfNodePtrList node_inputs = input_pair.first;
  ValueNodePtr module_node = NewValueNode(name_space);
  ValueNodePtr symbol_node = NewValueNode(resolve_symbol);
  auto operation = graph_stack_.top()->NewCNodeInOrder({NewValueNode(prim::kPrimResolve), module_node, symbol_node});
  (void)node_inputs.insert(node_inputs.cbegin(), operation);
  AnfNodePtr cnode = graph_stack_.top()->NewCNodeInOrder(node_inputs);
  if (cnode->debug_info() != nullptr) {
    cnode->debug_info()->set_trace_info(MakeTraceInfo<TraceOpt>(debug_info));
  }
  MS_LOG(DEBUG) << "New cnode: " << cnode->DebugString();
  SetNode(prim_res, cnode, debug_info);
}

AnfNodePtr TraceRecorder::GetNode(const py::object &obj, const DebugInfoPtr &debug_info, bool set_abstract) {
  const auto &convert_funcs = GetDataConvertFuncs();
  for (auto &convert_func : convert_funcs) {
    if (convert_func->Matched(obj)) {
      return convert_func->GetConstNode(obj, debug_info, set_abstract);
    }
  }
  if (tensor::IsTensorPy(obj)) {
    return GetTensorNode(obj, debug_info, set_abstract);
  } else if (py::isinstance<py::tuple>(obj)) {
    const py::tuple &tuple_obj = py::cast<py::tuple>(obj);
    return GetTupleNode(tuple_obj, debug_info, set_abstract);
  } else if (py::isinstance<py::list>(obj)) {
    const py::list &list_obj = py::cast<py::list>(obj);
    return GetListNode(list_obj, debug_info, set_abstract);
  } else if (py::isinstance<py::dict>(obj)) {
    const py::dict &dict_obj = py::cast<py::dict>(obj);
    return GetDictNode(dict_obj, debug_info, set_abstract);
  } else if (py::cast<std::string>(py::str(obj.get_type())).find("dict_keys") !=
             py::cast<std::string>(py::str(obj.get_type())).npos) {
    return GetDictKeyNode(obj, debug_info, set_abstract);
  } else if (py::cast<std::string>(py::str(obj.get_type())).find("dict_values") !=
             py::cast<std::string>(py::str(obj.get_type())).npos) {
    return GetDictValueNode(obj, debug_info, set_abstract);
  } else if (py::cast<std::string>(py::str(obj.get_type())).find("dict_items") !=
             py::cast<std::string>(py::str(obj.get_type())).npos) {
    return GetDictItemNode(obj, debug_info, set_abstract);
  }
  Clear();
  MS_LOG(INTERNAL_EXCEPTION) << "Not support [" << py::str(obj.get_type()) << "] " << py::str(obj)
                             << ", line: " << trace::GetDebugInfoStr(debug_info, "", kSourceLineTipDiscard);
}

AnfNodePtr TraceRecorder::GetTensorNode(const py::object &tensor_obj, const DebugInfoPtr &debug_info,
                                        bool set_abstract) {
  const auto &tensor = tensor::ConvertToTensor(tensor_obj);
  MS_EXCEPTION_IF_NULL(tensor);
  // Get the preceding CNode firstly.
  if (tensor->has_user_data("__node__")) {
    const auto &node = tensor->user_data<AnfNode>("__node__");
    MS_EXCEPTION_IF_NULL(node);
    MS_LOG(INFO) << "Get node from [" << py::str(tensor_obj.get_type()) << "] " << GetPyObjId(tensor_obj) << "/"
                 << py::str(tensor_obj) << ", ptr: " << tensor.get() << ", " << node->DebugString()
                 << ", line: " << trace::GetDebugInfoStr(debug_info, "", kSourceLineTipDiscard);
    return node;
  }
  // No preceding node, create a ValueNode.
  MS_LOG(INFO) << "No preceding node found, to create tensor value node. [" << py::str(tensor_obj.get_type()) << "] "
               << GetPyObjId(tensor_obj) << "/" << py::str(tensor_obj) << ", ptr: " << tensor.get()
               << ", line: " << trace::GetDebugInfoStr(debug_info, "", kSourceLineTipDiscard);
  const auto &value_node = NewValueNode(tensor);
  if (set_abstract) {
    value_node->set_abstract(GetAbstract(tensor_obj));
  }
  return value_node;
}

AnfNodePtr TraceRecorder::GetTupleNode(const py::tuple &tuple_obj, const DebugInfoPtr &debug_info, bool set_abstract) {
  // Create MakeTuple CNode.
  AnfNodePtrList make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  for (size_t i = 0; i < tuple_obj.size(); ++i) {
    const auto &node = GetNode(tuple_obj[i], debug_info);
    (void)make_tuple_inputs.emplace_back(node);
  }
  const auto &make_tuple_cnode = graph_stack_.top()->NewCNodeInOrder(make_tuple_inputs);
  if (set_abstract) {
    make_tuple_cnode->set_abstract(GetAbstract(tuple_obj));
  }
  MS_LOG(DEBUG) << "Creating new maketuple node. Tuple_obj: " << py::str(py::cast<py::object>(tuple_obj))
                << ", new node: " << make_tuple_cnode->DebugString();
  return make_tuple_cnode;
}

AnfNodePtr TraceRecorder::GetListNode(const py::list &list_obj, const DebugInfoPtr &debug_info, bool set_abstract) {
  // Create MakeList CNode for py::list each time.
  AnfNodePtrList make_list_inputs = {NewValueNode(prim::kPrimMakeList)};
  for (size_t i = 0; i < list_obj.size(); ++i) {
    const auto &node = GetNode(list_obj[i], debug_info);
    (void)make_list_inputs.emplace_back(node);
  }
  const auto &list_cnode = graph_stack_.top()->NewCNodeInOrder(make_list_inputs);
  if (set_abstract) {
    list_cnode->set_abstract(GetAbstract(list_obj));
  }
  return list_cnode;
}

AnfNodePtr TraceRecorder::GetDictNode(const py::dict &dict_obj, const DebugInfoPtr &debug_info, bool set_abstract) {
  // Create MakeDict CNode for py::dict each time.
  AnfNodePtrList make_dict_inputs = {NewValueNode(prim::kPrimMakeDict)};
  std::vector<AnfNodePtr> dict_key_nodes;
  std::vector<AnfNodePtr> dict_value_nodes;
  (void)dict_key_nodes.emplace_back(NewValueNode(prim::kPrimMakeTuple));
  (void)dict_value_nodes.emplace_back(NewValueNode(prim::kPrimMakeTuple));
  for (auto item : dict_obj) {
    (void)dict_key_nodes.emplace_back(GetNode(py::cast<py::object>(item.first), debug_info));
    (void)dict_value_nodes.emplace_back(GetNode(py::cast<py::object>(item.second), debug_info));
  }
  const auto &key_cnode = graph_stack_.top()->NewCNodeInOrder(dict_key_nodes);
  const auto &value_cnode = graph_stack_.top()->NewCNodeInOrder(dict_value_nodes);
  (void)make_dict_inputs.emplace_back(key_cnode);
  (void)make_dict_inputs.emplace_back(value_cnode);
  const auto &dict_cnode = graph_stack_.top()->NewCNodeInOrder(make_dict_inputs);
  if (set_abstract) {
    dict_cnode->set_abstract(GetAbstract(dict_obj));
  }
  return dict_cnode;
}

AnfNodePtr TraceRecorder::GetDictKeyNode(const py::object &dict_keys, const DebugInfoPtr &debug_info,
                                         bool set_abstract) {
  std::vector<AnfNodePtr> dict_key_nodes;
  (void)dict_key_nodes.emplace_back(NewValueNode(prim::kPrimMakeTuple));
  py::iterable keys_iter = py::reinterpret_borrow<py::iterable>(dict_keys);
  for (auto item : keys_iter) {
    (void)dict_key_nodes.emplace_back(GetNode(py::cast<py::object>(item), debug_info));
  }
  const auto &key_cnode = graph_stack_.top()->NewCNodeInOrder(dict_key_nodes);
  if (set_abstract) {
    key_cnode->set_abstract(GetAbstract(dict_keys));
  }
  return key_cnode;
}

AnfNodePtr TraceRecorder::GetDictValueNode(const py::object &dict_values, const DebugInfoPtr &debug_info,
                                           bool set_abstract) {
  std::vector<AnfNodePtr> dict_value_nodes;
  (void)dict_value_nodes.emplace_back(NewValueNode(prim::kPrimMakeTuple));
  py::iterable keys_iter = py::reinterpret_borrow<py::iterable>(dict_values);
  for (auto item : keys_iter) {
    (void)dict_value_nodes.emplace_back(GetNode(py::cast<py::object>(item), debug_info));
  }
  const auto &value_cnode = graph_stack_.top()->NewCNodeInOrder(dict_value_nodes);
  if (set_abstract) {
    value_cnode->set_abstract(GetAbstract(dict_values));
  }
  return value_cnode;
}

AnfNodePtr TraceRecorder::GetDictItemNode(const py::object &dict_items, const DebugInfoPtr &debug_info,
                                          bool set_abstract) {
  // Create MakeDict CNode for py::dict each time.
  AnfNodePtrList make_dict_inputs = {NewValueNode(prim::kPrimMakeList)};
  for (auto item : dict_items) {
    py::tuple key_value = py::cast<py::tuple>(item);
    std::vector<AnfNodePtr> dict_key_value_nodes = {
      NewValueNode(prim::kPrimMakeTuple), GetNode(key_value[0], debug_info), GetNode(key_value[1], debug_info)};
    const auto &key_value_cnode = graph_stack_.top()->NewCNodeInOrder(dict_key_value_nodes);
    (void)make_dict_inputs.emplace_back(key_value_cnode);
  }
  const auto &dict_cnode = graph_stack_.top()->NewCNodeInOrder(make_dict_inputs);
  if (set_abstract) {
    dict_cnode->set_abstract(GetAbstract(dict_items));
  }
  return dict_cnode;
}

void TraceRecorder::SetNode(const py::object &obj, const AnfNodePtr &node, const DebugInfoPtr &debug_info,
                            bool set_abstract) {
  if (tensor::IsTensorPy(obj)) {
    const auto &tensor = tensor::ConvertToTensor(obj);
    MS_EXCEPTION_IF_NULL(tensor);
    tensor->set_user_data<AnfNode>("__node__", node);
    MS_LOG(INFO) << "Set node to [" << py::str(obj.get_type()) << "] " << GetPyObjId(obj) << "/" << py::str(obj)
                 << ", ptr: " << tensor.get() << ", " << node->DebugString()
                 << ", line: " << trace::GetDebugInfoStr(debug_info, "", kSourceLineTipDiscard);
    if (set_abstract) {
      node->set_abstract(GetAbstract(obj));
    }
    return;
  } else if (py::isinstance<py::bool_>(obj) || py::isinstance<py::int_>(obj) || py::isinstance<py::float_>(obj) ||
             py::isinstance<py::str>(obj) || py::isinstance<py::none>(obj) || py::isinstance<py::ellipsis>(obj) ||
             py::isinstance<TensorType>(obj)) {
    MS_LOG(DEBUG) << "Get Constant value: " << py::str(obj);
    if (set_abstract) {
      node->set_abstract(GetAbstract(obj));
    }
    return;
  } else if (py::isinstance<py::tuple>(obj)) {
    const py::tuple &tuple_obj = py::cast<py::tuple>(obj);
    SetTupleNode(tuple_obj, node, debug_info, set_abstract);
    return;
  } else if (py::isinstance<py::list>(obj)) {
    const py::list &list_obj = py::cast<py::list>(obj);
    SetListNode(list_obj, node, debug_info, set_abstract);
    return;
  } else if (py::isinstance<py::dict>(obj)) {
    const py::dict &dict_obj = py::cast<py::dict>(obj);
    SetDictNode(dict_obj, node, debug_info, set_abstract);
    return;
  }
  Clear();
  MS_LOG(INTERNAL_EXCEPTION) << "Not support [" << py::str(obj.get_type()) << "] " << py::str(obj)
                             << ", line: " << trace::GetDebugInfoStr(debug_info, "", kSourceLineTipDiscard);
}

void TraceRecorder::SetTupleNode(const py::tuple &tuple_obj, const AnfNodePtr &node, const DebugInfoPtr &debug_info,
                                 bool set_abstract) {
  for (size_t i = 0; i < tuple_obj.size(); ++i) {
    if (!IsMutable(tuple_obj) && !tensor::IsTensorPy(tuple_obj[i])) {
      continue;
    }
    // Create Tuple GetItem CNode.
    AnfNodePtrList tuple_getitem_inputs = {NewValueNode(prim::kPrimTupleGetItem)};
    (void)tuple_getitem_inputs.emplace_back(node);
    (void)tuple_getitem_inputs.emplace_back(NewValueNode(SizeToLong(i)));
    const auto &getitem_cnode = graph_stack_.top()->NewCNodeInOrder(tuple_getitem_inputs);
    SetNode(tuple_obj[i], getitem_cnode, debug_info, set_abstract);
  }
}

void TraceRecorder::SetListNode(const py::list &list_obj, const AnfNodePtr &node, const DebugInfoPtr &debug_info,
                                bool set_abstract) {
  if (!IsMutable(list_obj)) {
    // Not create list -> node relation, to get node by GetNode(list_obj) if need.
    if (set_abstract) {
      node->set_abstract(GetAbstract(list_obj));
    }
    // Not return, create tensor -> node relation by list items in advance.
  }
  // It's mutable list.
  for (size_t i = 0; i < list_obj.size(); ++i) {
    // Create List GetItem CNode.
    AnfNodePtrList list_getitem_inputs = {NewValueNode(prim::kPrimListGetItem)};
    (void)list_getitem_inputs.emplace_back(node);
    (void)list_getitem_inputs.emplace_back(NewValueNode(SizeToLong(i)));
    const auto &getitem_cnode = graph_stack_.top()->NewCNodeInOrder(list_getitem_inputs);
    SetNode(list_obj[i], getitem_cnode, debug_info, set_abstract);
  }
}

void TraceRecorder::SetDictNode(const py::dict &dict_obj, const AnfNodePtr &node, const DebugInfoPtr &debug_info,
                                bool set_abstract) {
  if (!IsMutable(dict_obj)) {
    // Not create dict -> node relation, to get node by GetNode(dict_obj) if need.
    if (set_abstract) {
      node->set_abstract(GetAbstract(dict_obj));
    }
    // Not return, create tensor -> node relation by dict items in advance.
  }
  for (const auto &item : dict_obj) {
    // Create dict GetItem CNode.
    AnfNodePtrList dict_getitem_inputs = {NewValueNode(prim::kPrimDictGetItem)};
    (void)dict_getitem_inputs.emplace_back(node);
    (void)dict_getitem_inputs.emplace_back(GetNode(py::cast<py::object>(item.first), debug_info));
    const auto &getitem_cnode = graph_stack_.top()->NewCNodeInOrder(dict_getitem_inputs);
    SetNode(py::cast<py::object>(item.second), getitem_cnode, debug_info, set_abstract);
  }
}

void TraceRecorder::SyncTensorNode(const py::object &old_tensor_obj, const py::object &new_tensor_obj) {
  if (!tensor::IsTensorPy(old_tensor_obj) || !tensor::IsTensorPy(new_tensor_obj)) {
    return;
  }
  const auto &old_tensor = tensor::ConvertToTensor(old_tensor_obj);
  MS_EXCEPTION_IF_NULL(old_tensor);
  if (!old_tensor->has_user_data("__node__")) {
    MS_LOG(INTERNAL_EXCEPTION) << "Has no node in tensor, [" << py::str(old_tensor_obj.get_type()) << "] "
                               << py::str(old_tensor_obj) << ", ptr: " << old_tensor.get();
  }
  const auto &node = old_tensor->user_data<AnfNode>("__node__");
  MS_EXCEPTION_IF_NULL(node);
  const auto &new_tensor = tensor::ConvertToTensor(new_tensor_obj);
  MS_EXCEPTION_IF_NULL(new_tensor);
  new_tensor->set_user_data<AnfNode>("__node__", node);
  MS_LOG(DEBUG) << "Sync node from [" << py::str(old_tensor_obj.get_type()) << "] ptr: " << old_tensor.get() << " to ["
                << py::str(new_tensor_obj.get_type()) << "] ptr: " << new_tensor.get()
                << ", node: " << node->DebugString();
}

py::object TraceRecorder::InitTraceGraphInputs(const AbstractBasePtr &abs, const AnfNodePtr &param) {
  MS_EXCEPTION_IF_NULL(abs);
  auto val = abs->BuildValue();
  bool has_value = val != nullptr && !val->ContainsValueAny();
  if (abs->isa<abstract::AbstractSequence>()) {
    param->set_abstract(abs);
    const auto &abs_seq = abs->cast<abstract::AbstractSequencePtr>()->elements();
    py::tuple tuple_node(abs_seq.size());
    for (size_t i = 0; i < abs_seq.size(); ++i) {
      auto element = param->func_graph()->NewCNodeInOrder(
        {NewValueNode(prim::kPrimTupleGetItem), param, NewValueNode(SizeToLong(i))});
      tuple_node[i] = InitTraceGraphInputs(abs_seq[i], element);
    }
    return tuple_node;
  } else if (abs->isa<abstract::AbstractTensor>() || !has_value) {
    if (!abs->isa<abstract::AbstractTensor>()) {
      MS_LOG(WARNING) << "Input should be Tensor, but get " << abs->ToString() << ".";
    }
    param->set_abstract(abs);
    auto type_ptr = abs->GetType();
    MS_EXCEPTION_IF_NULL(type_ptr);
    auto tensor_type_ptr = type_ptr->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(tensor_type_ptr);
    auto type_id = tensor_type_ptr->element()->type_id();
    auto shape_ptr = abs->GetShape();
    MS_EXCEPTION_IF_NULL(shape_ptr);
    auto shape_vec = shape_ptr->GetShapeVector();
    auto tensor_ptr = tensor::from_spec(type_id, shape_vec, device::DeviceType::kCPU);
    py::object tensorpyObject = PackTensorToPyObject(tensor_ptr);
    SetNode(tensorpyObject, param, param->debug_info());
    return tensorpyObject;
  } else {
    param->set_abstract(abs);
    auto py_data = ValueToPyData(val);
    return py_data;
  }
}

void RegTraceRecorderPy(const py::module *m) {
  (void)py::class_<TraceRecorder, std::shared_ptr<TraceRecorder>>(*m, "TraceRecorder")
    .def_static("get_instance", &TraceRecorder::GetInstance, "Get trace manager instance.")
    .def("begin_graph", &TraceRecorder::BeginGraph, "Start a new graph.")
    .def("end_graph", &TraceRecorder::EndGraph, "Finish graph building.")
    .def("run_graph", &TraceRecorder::RunGraph, "Run the built graph.")
    .def("new_node", &TraceRecorder::NewNode, "Append a new CNode into current graph.")
    .def("sync_tensor_node", &TraceRecorder::SyncTensorNode, "Sync node from a tensor to another.")
    .def("new_fg_node", &TraceRecorder::NewFuncGraphNode, "Append a new CNode of func graph into current graph.")
    .def("pass_node", &TraceRecorder::PassNode, "Pass node when the object of a input is modified");
}
}  // namespace trace
}  // namespace mindspore
