/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "pipeline/jit/pi/graph_capture/graph_analyzer.h"
#include <algorithm>
#include <unordered_set>
#include <utility>
#include <string>
#include <vector>
#include "pipeline/jit/pi/pi_jit_config.h"
#include "pipeline/jit/pi/graph_guard/infer.h"
#include "pipeline/jit/pi/graph_capture/graph.h"
#include "pipeline/jit/pi/graph_capture/special_func_infer.h"
#include "pipeline/jit/pi/graph_capture/graph_build.h"
#include "pipeline/jit/pi/graph_capture/side_effect.h"

namespace mindspore {
namespace pijit {

extern TracePtr GetTrace(ValueNode *node, bool strict, bool print, int depth, int max_depth);

const int kMsFlagSet = AObject::kMsFlagGradFunc | AObject::kMsFlagStandardFunc | AObject::kMsFlagShardFunc |
                       AObject::kMsFlagVmapFunc | AObject::kMsFlagJitFunc;
static bool IsRepeatWithoutSideEffect(ValueNode *v, bool repeat_attr_item_access);

static bool CheckBuildTupleRepeatable(ValueNode *value, bool repeat_attr_item_access) {
  for (auto i : value->getInputs()) {
    if (i->GetOpcode() == BUILD_TUPLE || !IsRepeatWithoutSideEffect(i, repeat_attr_item_access)) {
      return false;
    }
  }
  return true;
}

static bool CheckBuildSliceRepeatable(const std::vector<ValueNode *> &inputs, bool repeat_attr_item_access) {
  for (auto i : inputs) {
    if (i->GetOpcode() != LOAD_CONST) {
      return false;
    }
  }
  return true;
}

// These are operations that are repeated and have no side effects.
static bool IsRepeatWithoutSideEffect(ValueNode *v, bool repeat_attr_item_access) {
  if (IsNonLocalValue(v)) {
    return true;
  }

  AObject::Type type = v->GetVobj() ? v->GetVobj()->GetType() : AObject::kTypeAnyValue;
  switch (v->GetOpcode()) {
    case BUILD_TUPLE:
      return CheckBuildTupleRepeatable(v, repeat_attr_item_access);
    case BUILD_SLICE:
      // NOTE: mindspore can't resolve call 'slice' class
      return CheckBuildSliceRepeatable(v->getInputs(), repeat_attr_item_access);
    case BINARY_SUBSCR:
    case LOAD_ATTR:
      return type == AObject::kTypeAnyValue ? false : repeat_attr_item_access;
    case BUILD_CONST_KEY_MAP:
      return true;
    case BUILD_MAP:
      if (type == AObject::kTypeDict) {
        AbstractDict *d = static_cast<AbstractDict *>(v->GetVobj());
        return d->size() == 0 || d->KeyType() != AObject::kTypeAnyValue;
      }
      return false;
    default:
      break;
  }
  return false;
}

namespace {
/**
 * mindspore func_graph assume these unsupported value is constant, so it same as global.
 * avoid parameter unsupported error by global
 */
bool ValidateGraphParameters(ValueNode *node) {
  static const std::set<AObject::Type> unsupported_parameter = {
    AObject::kTypeAnyValue,  AObject::kTypeFunction,      AObject::kTypeBoundMethod,
    AObject::kTypePrimitive, AObject::kTypeMetaFuncGraph, AObject::kTypeCell,
  };
  AObject *info = node->GetVobj();
  if (info == nullptr) {
    return false;
  }
  return unsupported_parameter.find(info->GetType()) == unsupported_parameter.end();
}
}  // namespace

bool GraphAnalyzer::ProduceInterpretValue(ValueNode *v) {
  bool repeat_op = graph_->Config().GetBoolConfig(GraphJitConfig::kEnableOptimizeForAttrItem);
  auto &locals = GetCaptureInfo().interpret_.values;
  auto &values = GetCaptureInfo().captured_.values;
  for (auto i : v->getInputs()) {
    if (IsNonLocalValue(i) || locals.find(i) != locals.end()) {
      continue;
    }
    if (values.find(i) == values.end()) {
      MS_LOG(INTERNAL_EXCEPTION) << "capture info can't find the value [" << i->ToString() << "]";
    }
    if (!IsRepeatWithoutSideEffect(i, repeat_op)) {
      return false;
    }
    // duplicate some operations if possible
    if (ProduceInterpretValue(i)) {
      continue;
    }
    return false;
  }
  AddToEscaped(v);
  return true;
}

// if operation can't be repeated, or block has attr access side effect
// can't reorder attr access op, must be interpret all attr, item access operation
static bool CheckAttrItemSupport(ValueNode *v, bool repeat_op) {
  int op = v->GetOpcode();
  AObject::Type type = v->input(0)->GetVobj() ? v->input(0)->GetVobj()->GetType() : AObject::kTypeAnyValue;
  // item access
  if (op == BINARY_SUBSCR) {
    return type != AObject::kTypeAnyValue;
  }
  // attr access
  if (type == AObject::kTypeAnyValue || type == AObject::kTypeBoundMethod) {
    return false;
  }
  if (type == AObject::kTypeTensor && !FindTensorName(v->GetName())) {
    return false;
  }
  return true;
}

static bool CheckSideEffectedFunc(ValueNode *v) {
  std::set<std::string> funcs = {"assign", "Assign"};
  if (Utils::IsCallOp(v->GetOpcode())) {
    py::object callable = v->input(0)->GetVobj() ? v->input(0)->GetVobj()->GetPyObject() : py::object();
    return callable.ptr() != nullptr ? funcs.find(GetFuncName(callable)) != funcs.end() : true;
  }
  return false;
}

bool GraphAnalyzer::HandleCallableToGraph(AObject *f) {
  static bool known_type[AObject::kTypeCount] = {false};
  if (known_type[AObject::kTypePrimitive] == false) {
    known_type[AObject::kTypePrimitive] = true;
    known_type[AObject::kTypeCell] = true;
    known_type[AObject::kTypeMetaFuncGraph] = true;
    known_type[AObject::kTypePrimitiveFunction] = true;
  }
  if (f == nullptr) {
    return false;
  }
  // don't pass unknown callable to graph
  bool is_known_func = known_type[f->GetType()] || CheckJitConstexpr(f->GetPyObject());
  bool is_ms_support_func = f->TestMsFlag(kMsFlagSet);
  if (!is_known_func && !is_ms_support_func) {
    return false;
  }
  if (f->GetType() == AObject::kTypePrimitive) {
    PyTypeObject *tp = f->GetTypeObject();
    std::string name = (tp && tp->tp_name ? tp->tp_name : "");
    if (name == "Assign") {
      return false;
    }
  }
  return true;
}

bool GraphAnalyzer::AddToCaptured(ValueNode *v) {
  if (IsNonLocalValue(v)) {
    return true;
  }
  int op = v->GetOpcode();
  bool repeat_op = graph_->Config().GetBoolConfig(GraphJitConfig::kEnableOptimizeForAttrItem);
  if ((op == LOAD_ATTR || op == BINARY_SUBSCR) && !CheckAttrItemSupport(v, repeat_op)) {
    return false;
  }

  if (Utils::IsCallOp(v->GetOpcode())) {
    AObject *f = v->input(0)->GetVobj();
    bool can_pass = HandleCallableToGraph(f);
    if (!can_pass) {
      return false;
    }
    GetCaptureInfo().has_grad_ |= f->TestMsFlag(AObject::kMsFlagGradFunc);
  }

  auto &locals = GetCaptureInfo().interpret_.values;  // interpret values
  auto &values = GetCaptureInfo().captured_.values;   // graph produced values
  for (auto i : v->getInputs()) {
    bool produced_in_graph = values.find(i) != values.end() || IsNonLocalValue(i);
    MS_EXCEPTION_IF_CHECK_FAIL(produced_in_graph || locals.find(i) != locals.end(),
                               "check values order, all input must be generate before this value " + i->ToString());
    if (i->GetVobj() == nullptr) {
      return false;
    }
    AObject::Type type = i->GetVobj()->GetType();
    PyTypeObject *tp = i->GetVobj()->GetTypeObject();
    if (type == AObject::kTypeAnyValue && !IsMsClass(reinterpret_cast<PyObject *>(tp))) {
      // don't pass unknown object to graph
      return false;
    }
    if (type == AObject::kTypeCell && !Utils::IsCallOp(op)) {
      // don't pass a cell object that not call to graph.
      return false;
    }
  }

  GetCaptureInfo().captured_.values.insert(v);
  GetCaptureInfo().captured_.operations.push_back(v);
  return true;
}

void GraphAnalyzer::AddToEscaped(ValueNode *v) {
  MS_EXCEPTION_IF_CHECK_FAIL(GetCaptureInfo().interpret_.values.find(v) == GetCaptureInfo().interpret_.values.end(),
                             "duplicate escaped values");
  GetCaptureInfo().interpret_.values.insert(v);
  GetCaptureInfo().interpret_.operations.push_back(v);
}

extern TracePtr GetTrace(ValueNode *node, bool strict, bool print, int depth, int max_depth);

bool GraphAnalyzer::HandleSideEffectNodeForCapture(AbstractNode *capture_node) {
  auto &replace_map = graph_->GetSideEffect()->GetReplaceMap();
  bool find =
    std::any_of(replace_map.begin(), replace_map.end(),
                [&capture_node](std::pair<ValueNode *, ValueNode *> item) { return capture_node == item.second; });
  if (find) {
    return true;
  }
  for (auto &side_effect_node : graph_->GetSideEffect()->GetSideEffectNodes()) {
    if (capture_node == side_effect_node && side_effect_node->GetOpcode() == CALL_FUNCTION) {
      return true;
    }
  }
  return false;
}

bool GraphAnalyzer::TryToCapture(AbstractNode *n) {
  if (HandleSideEffectNodeForCapture(n)) {
    return true;
  }

  ValueNode *v = static_cast<ValueNode *>(n);
  AObject *o = v->GetVobj();
  if (IsNonLocalValue(v)) {
    return true;
  }

  if (Utils::IsMsUnsupported(v->GetOpcode())) {
    // if mindspore unsupported, must be interpret
  } else if (AddToCaptured(v)) {
    return true;
  }

  const int ms_flag =
    AObject::kMsFlagGradFunc | AObject::kMsFlagShardFunc | AObject::kMsFlagVmapFunc | AObject::kMsFlagJitFunc;
  if (o && o->TestMsFlag(ms_flag) && AddToCaptured(v)) {
    GetCaptureInfo().has_grad_ = o->TestMsFlag(AObject::kMsFlagGradFunc);
    return true;
  }
  if (v->GetOpcode() == STORE_ATTR || v->GetOpcode() == STORE_DEREF) {
    return false;
  }
  if (!GetCaptureInfo().captured_.values.empty() && CheckSideEffectedFunc(v)) {
    return false;
  }
  if (ProduceInterpretValue(v)) {
    return true;
  }
  if (!HasTensorOperation()) {
    CleanCapturedValue();
    AddToEscaped(v);
    return true;
  }

  if (v->GetGraph() != nullptr && this->graph_->Config().GetBoolConfig(GraphJitConfig::kLogGraphBreak)) {
    GRAPH_JIT_LOG_F("capture failed, operations is unsupported [%s] at [%U: %d]", v->ToString().c_str(),
                    v->GetGraph()->GetCodeObj()->co_filename, v->GetLineNo());
    GRAPH_JIT_LOG_F("parameters");
    for (auto &i : v->getInputs()) {
      PyObject *op = i->GetVobj() ? i->GetVobj()->GetPyObject().ptr() : nullptr;
      GRAPH_JIT_LOG_F("%s", op ? AObject::ToString(op).c_str() : "NULL");
    }
  }

  MS_LOG(DEBUG) << "---operation that depend on the graph outputs, break graph---";
  return false;
}

bool GraphAnalyzer::AnalyzeCall(CallNode *call_node) {
  if (call_node->GetSubGraph() == nullptr) {
    return false;
  }
  if (call_node->GetInlineReason() != InlineReason::kInline) {
    return false;
  }

  Graph *g = call_node->GetGraph();

  CapturedInfo back_up = info_;
  const auto &p = call_node->GetParams();
  // capture parameter handle operations
  auto iter = std::find_if(p.begin(), p.end(), [this](ValueNode *i) { return !this->TryToCapture(i); });
  // capture sub-graph
  if (iter == p.end() && AnalyzeRecursive(call_node->GetSubGraph())) {
    return true;
  }
  info_ = back_up;
  g->StopTraceAt(call_node->bci(), StopTraceReason::kStopTraceDataDependsOnGraphOut);
  return false;
}

bool GraphAnalyzer::AnalyzeRecursive(Graph *g) {
  for (auto n : g->GetTracedNodes()) {
    int bci = static_cast<ValueNode *>(n)->bci();
    if (n->GetType() == AbstractNode::Call && AnalyzeCall(static_cast<CallNode *>(n))) {
      continue;
    }
    if (bci != -1 && g->GetStopTraceBci() == bci) {
      return false;
    }
    if (!TryToCapture(n)) {
      g->StopTraceAt(bci, StopTraceReason::kStopTraceDataDependsOnGraphOut);
      return false;
    }
  }
  return true;
}

void GraphAnalyzer::CollectCapturedInputs() {
  auto &locals = GetCaptureInfo().interpret_.values;
  auto &values = GetCaptureInfo().captured_.values;
  mindspore::CompactSet<ValueNode *> inputs;
  for (ValueNode *i : GetCaptureInfo().captured_.operations) {
    for (auto input : i->getInputs()) {
      if (values.find(input) != values.end() || IsNonLocalValue(input)) {
        continue;
      }
      MS_EXCEPTION_IF_CHECK_FAIL(locals.find(input) != locals.end(), "check graph input");
      inputs.insert(input);
    }
  }
  GetCaptureInfo().captured_.inputs = {inputs.begin(), inputs.end()};
}

void GraphAnalyzer::UseDefAnalyze() {
  // UD analyze: alive nodes analysis
  std::vector<ValueNode *> aliveLocals = GetAliveLocals(graph_);
  if (!aliveLocals.empty()) {
    bool isStopAnalyze = false;
    while (!isStopAnalyze) {
      isStopAnalyze = AnalyzeAliveLocals(aliveLocals);
      if (isStopAnalyze) {
        break;
      }
      aliveLocals = GetAliveLocals(graph_);
    }
  }
}

void GraphAnalyzer::Analyze() {
  const FrameStates &enter_frame = graph_->GetFrame(0);
  GetCaptureInfo().interpret_.values.insert(enter_frame.GetLocals().begin(), enter_frame.GetLocals().end());
  AnalyzeRecursive(graph_);
  if (!HasTensorOperation()) {
    CleanCapturedValue();
  }
  UseDefAnalyze();
  if (graph_->GetStopTraceBci() != -1) {
    graph_->GetSideEffect()->CleanSideEffects(graph_->GetStopTraceBci());
  }
  CollectCapturedAndInterpret();
  CollectGraphInputs();

  need_interpret_ = true;

  if (graph_->GetStopTraceBci() != -1 || !GetCaptureInfo().interpret_.operations.empty()) {
    return;
  }
  bool support_ret = graph_->GetRetVal()->GetVobj() && graph_->GetRetVal()->GetVobj()->IsMindSporeSupportedType();
  if (!support_ret) {
    return;
  }
  PyCodeObject *co = graph_->GetCodeObj();
  const auto &args = enter_frame.GetLocals();
  int argc = co->co_argcount + co->co_kwonlyargcount;
  // check all parameters is graph supported, but here not check variable arguments
  auto end = args.begin() + argc;
  auto iter = std::find_if(args.begin(), end, [](ValueNode *i) { return !ValidateGraphParameters(i); });
  if (iter == end) {
    need_interpret_ = false;
  }
  if (!graph_->GetSideEffect()->GetSideEffectNodes().empty()) {
    need_interpret_ = true;
  }
  if (!graph_->GetSideEffect()->GetGlobalList().empty()) {
    need_interpret_ = true;
  }
}

FrameStates buildLastFrame(Graph *g) { return g->GetFrame(g->GetStopTraceBci()); }

std::vector<ValueNode *> GraphAnalyzer::GetAliveLocals(Graph *g) {
  int bci = g->GetStopTraceBci();
  if (this->graph_->Config().GetBoolConfig(GraphJitConfig::kLogGraphBreak)) {
    GRAPH_JIT_LOG_F("UD analyze: enter GetAliveLocals bci %d", bci);
  }
  std::vector<ValueNode *> outputs = g->CollectAliveNode(bci);
  mindspore::CompactSet<ValueNode *> uniques;
  for (auto output : outputs) {
    uniques.insert(output);
  }
  outputs.assign(uniques.begin(), uniques.end());
  return outputs;
}

void PrintAliveNodes(std::vector<ValueNode *> aliveNodes) {
  GRAPH_JIT_LOG_F("UD analyze: alive node size : %ld", aliveNodes.size());
  for (auto node : aliveNodes) {
    if (node) {
      GRAPH_JIT_LOG_F("UD analyze: alive node: %s", node->ToString().c_str());
    }
  }
}

bool GraphAnalyzer::AnalyzeAliveLocals(std::vector<ValueNode *> aliveNodes) {
  if (this->graph_->Config().GetBoolConfig(GraphJitConfig::kLogGraphBreak)) {
    PrintAliveNodes(aliveNodes);
  }
  bool isAllNodesSupportOutput = true;
  for (auto node : aliveNodes) {
    AObject *o = node->GetVobj();
    bool supported_type = o && o->IsMindSporeSupportedType();
    if (supported_type) {
      continue;
    }
    auto capturedLocals = info_.captured_.operations;
    if (std::find(capturedLocals.begin(), capturedLocals.end(), node) == capturedLocals.end()) {
      continue;
    }

    if (!HasTensorOperation()) {
      CleanCapturedValue();
      break;
    }

    //  reset break graph point
    isAllNodesSupportOutput = false;
    int new_break_point = node->bci();
    auto curNode = node;
    MS_EXCEPTION_IF_CHECK_FAIL(new_break_point != -1, "break point cannot be -1");
    MS_EXCEPTION_IF_NULL(curNode->GetGraph());
    if (this->graph_->Config().GetBoolConfig(GraphJitConfig::kLogGraphBreak)) {
      GRAPH_JIT_LOG_F("reset break point: %d", new_break_point);
    }
    this->graph_->StopTraceAt(new_break_point, StopTraceReason::kStopTraceDataDependsOnGraphOut);

    // re-collect captured info
    info_.clear();
    const FrameStates &enter_frame = graph_->GetFrame(0);
    GetCaptureInfo().interpret_.values.insert(enter_frame.GetLocals().begin(), enter_frame.GetLocals().end());
    (void)AnalyzeRecursive(graph_);
    break;
  }
  return isAllNodesSupportOutput;
}

static bool SkipSpecialFuncOrPrimitive(const py::object &callable) {
  if (callable.ptr() == nullptr) {
    return false;
  }
  if (CheckJitConstexpr(callable) || CheckMSConstexpr(callable)) {
    return true;
  }
  if (IsPrimitiveType<true>(Py_TYPE(callable.ptr()))) {
    std::string name = callable.attr("name").cast<std::string>();
    return GetSpecialPrimitiveInferFunc().find(name) != GetSpecialPrimitiveInferFunc().end();
  }
  return false;
}

bool GraphAnalyzer::HasTensorOperation() const {
  bool has_tensor_cal = false;
  for (auto i : info_.captured_.values) {
    AObject *value = i->GetVobj();
    int op = i->GetOpcode();
    if (Utils::IsCallOp(op)) {
      if (SkipSpecialFuncOrPrimitive(i->input(0)->GetVobj()->GetPyObject())) {
        continue;
      }
      if (value->GetType() == AObject::kTypeCFunction) {
        continue;
      }
      return true;
    }
    if (Utils::IsBinaryMathOp(op) && value->GetType() == AObject::kTypeTensor) {
      return true;
    }
  }
  return has_tensor_cal;
}

void GraphAnalyzer::CapturedInfo::Info::clear() {
  values.clear();
  inputs.clear();
  operations.clear();
  outputs.clear();
}

void GraphAnalyzer::CapturedInfo::GraphInputs::clear() {
  args.clear();
  globals.clear();
  vargs = nullptr;
  kwargs = nullptr;
}

void GraphAnalyzer::CapturedInfo::clear() {
  captured_.clear();
  interpret_.clear();
  graph_inputs_.clear();
}

std::string GraphAnalyzer::CapturedInfo::Info::ToString() {
  std::stringstream s;
  s << "values: ";
  for (auto i : values) {
    s << i->ToString() << "\n";
  }
  s << "inputs: \n";
  for (auto i : inputs) {
    s << i->ToString() << "\n";
  }
  s << "operations: \n";
  for (auto i : operations) {
    s << i->ToString() << "\n";
  }
  s << "outputs: \n";
  for (auto i : outputs) {
    s << i->ToString() << "\n";
  }
  return s.str();
}

std::string GraphAnalyzer::CapturedInfo::GraphInputs::ToString() {
  std::stringstream s;
  s << "globals: ";
  for (auto i : globals) {
    s << i->ToString() << "\n";
  }
  s << "args: \n";
  for (auto i : args) {
    s << i->ToString() << "\n";
  }
  s << "vargs: ";
  if (vargs != nullptr) {
    s << vargs->ToString();
  }
  s << "\n";
  s << "kwargs: ";
  if (kwargs != nullptr) {
    s << kwargs->ToString();
  }
  s << "\n";
  return s.str();
}

std::string GraphAnalyzer::CapturedInfo::ToString() {
  std::stringstream s;
  s << "1. captured_ info: \n";
  s << captured_.ToString();
  s << "2. interpret_ info: \n";
  s << interpret_.ToString();
  s << "3. graph_inputs_: \n";
  s << graph_inputs_.ToString();
  s << "4. has_grad_: " << has_grad_ << "\n";
  return s.str();
}

void GraphAnalyzer::CleanCapturedValue() {
  auto &locals = info_.interpret_.values;
  for (auto i : info_.captured_.operations) {
    if (locals.find(i) == locals.end()) {
      locals.insert(i);
      info_.interpret_.operations.emplace_back(i);
    }
  }
  info_.captured_.values.clear();
  info_.captured_.operations.clear();
}

static std::vector<ValueNode *> CollectGraphOutputs(const mindspore::CompactSet<ValueNode *> &interpret,
                                                    const std::vector<ValueNode *> &alive) {
  std::vector<ValueNode *> outputs;
  for (auto i : alive) {
    if (interpret.find(i) == interpret.end() && !IsNonLocalValue(i)) {
      outputs.push_back(i);
    }
  }
  return outputs;
}

void GraphAnalyzer::CollectCapturedAndInterpret() {
  CollectCapturedInputs();
  int break_bci = graph_->GetStopTraceBci();
  std::vector<ValueNode *> alive_nodes = graph_->CollectAliveNode(break_bci, &alive_locals_);

  GetCaptureInfo().captured_.outputs = CollectGraphOutputs(GetCaptureInfo().interpret_.values, alive_nodes);
  GetCaptureInfo().interpret_.inputs = graph_->GetFrame(0).GetLocals();
  GetCaptureInfo().interpret_.outputs = std::move(alive_nodes);
  auto vec = graph_->GetSideEffect()->CollectSideEffectAliveNodes();
  GetCaptureInfo().interpret_.outputs.insert(GetCaptureInfo().interpret_.outputs.end(), vec.begin(), vec.end());
}

void GraphAnalyzer::CollectGraphInputs() {
  PyCodeObject *co_ = graph_->GetCodeObj();
  auto &interpret_ = GetCaptureInfo().interpret_;
  auto &captured_ = GetCaptureInfo().captured_;
  auto &graph_inputs = GetCaptureInfo().graph_inputs_;

  // NOTE: if *vargs is cell variable, it is not parameter node
  MS_EXCEPTION_IF_CHECK_FAIL(co_->co_nlocals == static_cast<int>(interpret_.inputs.size()),
                             "interpret inputs must be same as locals");

  ValueNode *vargs = nullptr;
  ValueNode *kwargs = nullptr;
  int arg_index = co_->co_argcount + co_->co_kwonlyargcount;
  if ((co_->co_flags & CO_VARARGS) && interpret_.inputs[arg_index] != &ValueNode::kUnboundLocal) {
    vargs = interpret_.inputs[arg_index];
  }
  arg_index += (IntToSize(co_->co_flags) & CO_VARARGS) != 0;
  if ((IntToSize(co_->co_flags) & CO_VARKEYWORDS) && interpret_.inputs[arg_index] != &ValueNode::kUnboundLocal) {
    kwargs = interpret_.inputs[arg_index];
  }

  // Identify parameters and global variables
  for (auto input : captured_.inputs) {
    if (input == graph_inputs.vargs) {
      graph_inputs.vargs = vargs;
    } else if (input == graph_inputs.kwargs) {
      graph_inputs.kwargs = kwargs;
    } else if (ValidateGraphParameters(input)) {
      graph_inputs.args.push_back(input);
    } else {
      graph_inputs.globals.push_back(input);
    }
  }

  size_t inputs_count = captured_.inputs.size();
  captured_.inputs = graph_inputs.args;
  if (graph_inputs.vargs != nullptr) {
    captured_.inputs.push_back(graph_inputs.vargs);
  }
  if (graph_inputs.kwargs != nullptr) {
    captured_.inputs.push_back(graph_inputs.kwargs);
  }
  captured_.inputs.insert(captured_.inputs.end(), graph_inputs.globals.begin(), graph_inputs.globals.end());
  MS_EXCEPTION_IF_CHECK_FAIL(inputs_count == captured_.inputs.size(), "error parameters");
}

void MindGraphAnalyzer::CollectCapturedInputs() {
  auto &inputs = GetCaptureInfo().captured_.inputs;
  const FrameStates &enter_frame = graph_->GetFrame(0);
  PyCodeObject *co = graph_->GetCodeObj();
  int argc = co->co_argcount + co->co_kwonlyargcount;
  argc += SizeToInt(co->co_flags & CO_VARARGS) ? 1 : 0;
  argc += SizeToInt(co->co_flags & CO_VARKEYWORDS) ? 1 : 0;
  for (Py_ssize_t m = 0; m < argc; ++m) {
    const auto &local = enter_frame.Local(m);
    if (local == &ValueNode::kUnboundLocal) {
      continue;
    }
    inputs.push_back(local);
  }
}

void MindGraphAnalyzer::Analyze() {
  auto origin_stop_bci = graph_->GetStopTraceBci();
  UseDefAnalyze();

  const FrameStates &enter_frame = graph_->GetFrame(0);
  GetCaptureInfo().interpret_.values.insert(enter_frame.GetLocals().begin(), enter_frame.GetLocals().end());

  auto mind_graph_builder = std::static_pointer_cast<MindGraphBuilder>(graph_builder_);
  MS_EXCEPTION_IF_NULL(mind_graph_builder);
  auto func_graph_builder = mind_graph_builder->FGBuilder();
  if (func_graph_builder->graph() == nullptr) {
    // Graph build failed, add all nodes to ordered_escaped_locals.
    MS_LOG(DEBUG) << "Failed to build graph";
    GetCaptureInfo().interpret_.operations.clear();
    for (const auto &traced_node : graph_->GetTracedNodes()) {
      if (origin_stop_bci != -1 && traced_node->bci() >= origin_stop_bci) {
        break;
      }
      AddToEscaped(traced_node);
    }
    graph_->StopTraceAt(origin_stop_bci, StopTraceReason::kStopTraceDataDependsOnGraphOut);
    need_interpret_ = true;
    GetCaptureInfo().captured_.clear();
    CollectCapturedAndInterpret();
    return;
  }
  CollectCapturedAndInterpret();
  CollectGraphInputs();

  need_interpret_ = true;
  if (graph_->GetStopTraceBci() != -1 || !GetCaptureInfo().interpret_.operations.empty()) {
    return;
  }
  bool support_ret = graph_->GetRetVal()->GetVobj() && graph_->GetRetVal()->GetVobj()->IsMindSporeSupportedType();
  if (!support_ret) {
    return;
  }
  need_interpret_ = false;
}

bool MindGraphAnalyzer::AnalyzeAliveLocals(std::vector<ValueNode *> aliveNodes) {
  bool isAllNodesSupportOutput = true;
  auto mind_graph_builder = std::static_pointer_cast<MindGraphBuilder>(graph_builder_);
  MS_EXCEPTION_IF_NULL(mind_graph_builder);
  auto func_graph_builder = mind_graph_builder->FGBuilder();
  MS_EXCEPTION_IF_NULL(func_graph_builder);
  func_graph_builder->ClearOutputNodes();
  for (auto node : aliveNodes) {
    // If the value can get from local, no need to add to graph output.
    if (IsNonLocalValue(node)) {
      MS_LOG(DEBUG) << "Skip non local value used as graph return.";
      continue;
    }
    auto capturedLocals = info_.captured_.operations;
    if (std::find(capturedLocals.begin(), capturedLocals.end(), node) == capturedLocals.end()) {
      continue;
    }
    AObject *o = node->GetVobj();
    auto out_py_obj = o->GetPyObject();
    if (func_graph_builder->AddOutput(out_py_obj, true)) {
      MS_LOG(DEBUG) << "Add output success.";
      continue;
    }
    MS_LOG(DEBUG) << "Add output failed.";
    //  reset break graph point
    isAllNodesSupportOutput = false;
    int new_break_point = node->bci();
    auto curNode = node;
    if (new_break_point == -1) {
      // No node is unsupported output since no node in captured output.
      isAllNodesSupportOutput = true;
      break;
    }
    MS_EXCEPTION_IF_NULL(curNode->GetGraph());
    if (this->graph_->Config().GetBoolConfig(GraphJitConfig::kLogGraphBreak)) {
      GRAPH_JIT_LOG_F("reset break point: %d", new_break_point);
    }
    this->graph_->StopTraceAt(new_break_point, StopTraceReason::kStopTraceDataDependsOnGraphOut);
    break;
  }
  return isAllNodesSupportOutput;
}

void MindGraphAnalyzer::UpdateCapturedOrder() {
  const auto &traced_nodes = graph_->GetTracedNodes();
  auto stop_bci = graph_->GetStopTraceBci();
  if (stop_bci == -1) {
    GetCaptureInfo().captured_.operations = traced_nodes;
  } else {
    GetCaptureInfo().captured_.operations.clear();
    for (const auto &traced_node : traced_nodes) {
      if (traced_node->bci() >= stop_bci) {
        break;
      }
      GetCaptureInfo().captured_.operations.push_back(traced_node);
    }
  }
  const auto &captured_local_order = GetCaptureInfo().captured_.operations;
  mindspore::CompactSet<ValueNode *> new_capture_local_values;
  for (auto val : captured_local_order) {
    new_capture_local_values.insert(val);
  }
  GetCaptureInfo().captured_.values = new_capture_local_values;
}

void MindGraphAnalyzer::UseDefAnalyze() {
  // UD analyze: alive nodes analysis
  std::vector<ValueNode *> aliveLocals = GetAliveLocals(graph_);
  if (!aliveLocals.empty()) {
    bool stop_analyze = false;
    while (!stop_analyze) {
      UpdateCapturedOrder();
      // Add graph output according to leaf nodes.
      stop_analyze = AnalyzeAliveLocals(aliveLocals);
      if (!stop_analyze) {
        aliveLocals = GetAliveLocals(graph_);
      }
    }
  }
}

}  // namespace pijit
}  // namespace mindspore
