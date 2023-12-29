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
namespace jit {
namespace graph {

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
  bool can_access = repeat_op && v->GetBlock() && !v->GetBlock()->HasAttrSideEffect();
  AObject::Type type = v->input(0)->GetVobj() ? v->input(0)->GetVobj()->GetType() : AObject::kTypeAnyValue;
  // item access
  if (op == BINARY_SUBSCR) {
    return type != AObject::kTypeAnyValue && can_access;
  }
  // attr access
  if (!can_access) {
    return false;
  }
  if (type == AObject::kTypeAnyValue || type == AObject::kTypeBoundMethod) {
    return false;
  }
  if (type == AObject::kTypeTensor && !FindTensorName(v->GetName())) {
    return false;
  }
  return true;
}

extern bool CheckJitConstexpr(const py::object &func);
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
    if (f == nullptr) {
      return false;
    }
    // don't pass unknown callable to graph
    bool is_known_func = f->GetType() == AObject::kTypeCell || f->GetType() == AObject::kTypePrimitive ||
                         f->GetType() == AObject::kTypeMetaFuncGraph ||
                         (f->GetType() == AObject::kTypeFunction && CheckJitConstexpr(f->GetPyObject()));
    bool is_ms_support_func = f->TestMsFlag(kMsFlagSet);
    if (!is_known_func && !is_ms_support_func) {
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

  bool supported_oper = !Utils::IsMsUnsupported(v->GetOpcode());
  bool supported_type = o && o->IsMindSporeSupportedType();
  if (!supported_oper || !supported_type) {
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
    auto tr = GetTrace(v, false, true, 0, -1);
    GRAPH_JIT_LOG_F("trace %s", tr ? tr->ToString().c_str() : "trace failed");
    GRAPH_JIT_LOG_F("capture failed, operations is unsupported [%s] at [%U: %d]", v->ToString().c_str(),
                    v->GetGraph()->GetCodeObj()->co_filename, v->GetLineNo());
    GRAPH_JIT_LOG_F("parameters");
    for (auto &i : v->getInputs()) {
      GRAPH_JIT_LOG_F("%s", i->GetVobj() ? i->GetVobj()->ToString().c_str() : "<nil>");
    }
  }

  MS_LOG(DEBUG) << "---operation that depend on the graph outputs, break graph---";
  return false;
}

bool GraphAnalyzer::AnalyzeCall(CallNode *call_node) {
  if (call_node->GetSubGraph() == nullptr) {
    return false;
  }
  if (call_node->GetInlineReason() != InlineReason::kInline &&
      call_node->GetInlineReason() != InlineReason::kInlinePartial) {
    return false;
  }

  Graph *g = call_node->GetGraph();

  CapturedInfo back_up;
  if (!kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kFeatureBreakAtInlinedFunction)) {
    back_up = info_;
  }
  const auto &p = call_node->GetParams();
  // capture parameter handle operations
  auto iter = std::find_if(p.begin(), p.end(), [this](ValueNode *i) { return !this->TryToCapture(i); });
  // capture sub-graph
  if (iter == p.end() && AnalyzeRecursive(call_node->GetSubGraph())) {
    return true;
  }
  if (!kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kFeatureBreakAtInlinedFunction)) {
    info_ = back_up;
  }
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

void GraphAnalyzer::Analyze() {
  const FrameStates &enter_frame = graph_->GetFrame(0);
  GetCaptureInfo().escaped_locals.insert(enter_frame.GetLocals().begin(), enter_frame.GetLocals().end());
  AnalyzeRecursive(graph_);
  if (!HasTensorOperation()) {
    CleanCapturedValue();
  }
  CollectInputs();
}

bool GraphAnalyzer::HasTensorOperation() const {
  bool has_tensor_cal = false;
  for (auto i : info_.captured_locals.values) {
    AObject *value = i->GetVobj();
    int op = i->GetOpcode();
    if (Utils::IsCallOp(op)) {
      return true;
    }
    if (Utils::IsBinaryMathOp(op) && value->GetType() == AObject::kTypeTensor) {
      has_tensor_cal = true;
      break;
    }
  }
  return has_tensor_cal;
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
}  // namespace graph
}  // namespace jit
}  // namespace mindspore
