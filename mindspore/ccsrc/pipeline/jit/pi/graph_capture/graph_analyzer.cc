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
#include <vector>
#include "pipeline/jit/pi/pi_jit_config.h"
#include "pipeline/jit/pi/graph_guard/infer.h"
#include "pipeline/jit/pi/graph_capture/graph.h"

namespace mindspore {
namespace pijit {

extern bool CheckMSConstexpr(const py::object &func);
extern bool CheckJitConstexpr(const py::object &func);

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
    default:
      break;
  }
  return false;
}

bool GraphAnalyzer::ProduceInterpretValue(ValueNode *v) {
  bool repeat_op = graph_->Config().GetBoolConfig(GraphJitConfig::kEnableOptimizeForAttrItem);
  auto &locals = GetCaptureInfo().escaped_locals;
  auto &values = GetCaptureInfo().captured_locals.values;
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

extern bool CheckJitConstexpr(const py::object &func);
bool GraphAnalyzer::HandleCallableToGraph(AObject *f) {
  if (f == nullptr) {
    return false;
  }
  // don't pass unknown callable to graph
  bool is_known_func = f->GetType() == AObject::kTypeCell || f->GetType() == AObject::kTypePrimitive ||
                       f->GetType() == AObject::kTypeMetaFuncGraph || CheckJitConstexpr(f->GetPyObject());
  bool is_ms_support_func = f->TestMsFlag(kMsFlagSet);
  if (!is_known_func && !is_ms_support_func) {
    return false;
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

  auto &locals = GetCaptureInfo().escaped_locals;          // interpret values
  auto &values = GetCaptureInfo().captured_locals.values;  // graph produced values
  for (auto i : v->getInputs()) {
    bool produced_in_graph = values.find(i) != values.end() || IsNonLocalValue(i);
    MS_EXCEPTION_IF_CHECK_FAIL(produced_in_graph || locals.find(i) != locals.end(),
                               "check values order, all input must be generate before this value " + i->ToString());
    AObject::Type type = i->GetVobj() ? i->GetVobj()->GetType() : AObject::kTypeAnyValue;
    if (type == AObject::kTypeAnyValue) {
      // don't pass unknown object to graph
      return false;
    }
    if (type == AObject::kTypeCell && !Utils::IsCallOp(op)) {
      // don't pass a cell object that not call to graph.
      return false;
    }
  }

  GetCaptureInfo().captured_locals.values.insert(v);
  GetCaptureInfo().captured_locals.order.push_back(v);
  return true;
}

void GraphAnalyzer::AddToEscaped(ValueNode *v) {
  MS_EXCEPTION_IF_CHECK_FAIL(GetCaptureInfo().escaped_locals.find(v) == GetCaptureInfo().escaped_locals.end(),
                             "duplicate escaped values");
  GetCaptureInfo().escaped_locals.insert(v);
  GetCaptureInfo().ordered_escaped_locals.push_back(v);
}

extern TracePtr GetTrace(ValueNode *node, bool strict, bool print, int depth, int max_depth);

bool GraphAnalyzer::TryToCapture(AbstractNode *n) {
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

void GraphAnalyzer::CollectInputs() {
  auto &locals = GetCaptureInfo().escaped_locals;
  auto &values = GetCaptureInfo().captured_locals.values;
  auto &inputs = GetCaptureInfo().captured_locals.inputs;
  for (ValueNode *i : GetCaptureInfo().captured_locals.order) {
    for (auto input : i->getInputs()) {
      if (values.find(input) != values.end() || IsNonLocalValue(input)) {
        continue;
      }
      MS_EXCEPTION_IF_CHECK_FAIL(locals.find(input) != locals.end(), "check graph input");
      inputs.insert(input);
    }
  }
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
  GetCaptureInfo().escaped_locals.insert(enter_frame.GetLocals().begin(), enter_frame.GetLocals().end());
  AnalyzeRecursive(graph_);
  if (!HasTensorOperation()) {
    CleanCapturedValue();
  }
  UseDefAnalyze();
  CollectInputs();

  need_interpret_ = true;
  if (graph_->GetStopTraceBci() != -1 || !GetCaptureInfo().ordered_escaped_locals.empty()) {
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
}

FrameStates buildLastFrame(Graph *g) { return g->GetFrame(g->GetStopTraceBci()); }

std::vector<ValueNode *> GraphAnalyzer::GetAliveLocals(Graph *g) {
  std::vector<ValueNode *> liveLocals;
  int bci = g->GetStopTraceBci();
  if (bci == -1) {
    if (g->GetRetVal()) {
      liveLocals.push_back(g->GetRetVal());
    }
    return liveLocals;
  }
  const Liveness *liveness = g->GetCFG()->GetLiveness();
  if (this->graph_->Config().GetBoolConfig(GraphJitConfig::kLogGraphBreak)) {
    GRAPH_JIT_LOG_F("UD analyze: enter GetAliveLocals bci %d", bci);
  }
  BitMap lives = liveness->CollectAlive(bci);
  FrameStates f = buildLastFrame(graph_);
  std::vector<ValueNode *> locals = f.GetLocals();
  for (size_t i = 0; i < locals.size(); i++) {
    if (lives.Get(i)) {
      liveLocals.push_back(locals[i]);
    }
  }
  std::vector<ValueNode *> stacks = f.GetStacks();
  std::copy(stacks.begin(), stacks.end(), std::back_inserter(liveLocals));
  return liveLocals;
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
    auto capturedLocals = info_.captured_locals.order;
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
    ClearCapturedInfo();
    const FrameStates &enter_frame = graph_->GetFrame(0);
    GetCaptureInfo().escaped_locals.insert(enter_frame.GetLocals().begin(), enter_frame.GetLocals().end());
    (void)AnalyzeRecursive(graph_);
    break;
  }
  return isAllNodesSupportOutput;
}

bool GraphAnalyzer::HasTensorOperation() const {
  bool has_tensor_cal = false;
  for (auto i : info_.captured_locals.values) {
    AObject *value = i->GetVobj();
    int op = i->GetOpcode();
    if (Utils::IsCallOp(op)) {
      py::object callable = i->input(0)->GetVobj()->GetPyObject();
      if (callable.ptr() == nullptr) {
        return true;
      }
      if (CheckJitConstexpr(callable) || CheckMSConstexpr(callable)) {
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

void GraphAnalyzer::ClearCapturedInfo() {
  info_.escaped_locals.clear();
  info_.ordered_escaped_locals.clear();
  info_.captured_locals.inputs.clear();
  info_.captured_locals.values.clear();
  info_.captured_locals.order.clear();
}

void GraphAnalyzer::CleanCapturedValue() {
  auto &locals = info_.escaped_locals;
  for (auto i : info_.captured_locals.order) {
    if (locals.find(i) == locals.end()) {
      locals.insert(i);
      info_.ordered_escaped_locals.emplace_back(i);
    }
  }
  info_.captured_locals.values.clear();
  info_.captured_locals.order.clear();
}

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

}  // namespace pijit
}  // namespace mindspore
