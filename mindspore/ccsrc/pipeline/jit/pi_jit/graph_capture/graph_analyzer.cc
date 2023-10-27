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
#include "pipeline/jit/pi_jit/graph_capture/graph_analyzer.h"
#include <algorithm>
#include <unordered_set>
#include <vector>
#include "pipeline/jit/pi_jit/pi_jit_config.h"

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
  int op = v->GetOpcode();
  if ((op == LOAD_ATTR || op == BINARY_SUBSCR) && repeat_attr_item_access) {
    return true;
  }

  switch (op) {
    case BUILD_TUPLE:
      return CheckBuildTupleRepeatable(v, repeat_attr_item_access);
    case BUILD_SLICE:
      // NOTE: mindspore can't resolve call 'slice' class
      return CheckBuildSliceRepeatable(v->getInputs(), repeat_attr_item_access);
    default:
      break;
  }
  return false;
}

bool GraphAnalyzer::ProduceInterpretValue(ValueNode *v) {
  bool repeat_op = Config().GetBoolConfig(GraphJitConfig::kEnableOptimizeForAttrItem);
  auto &locals = GetCaptureInfo().escaped_locals;
  auto &values = GetCaptureInfo().captured_locals.values;
  for (auto i : v->getInputs()) {
    if (IsNonLocalValue(i) || locals.find(i) != locals.end()) {
      continue;
    }
    bool captured = values.find(i) != values.end();
    if (i->bci() == -1 && !captured) {
      // inlined function's parameters, graph break at function parameters build
      MS_EXCEPTION_IF_CHECK_FAIL(false, "not implement graph break at inlined function parameters build");
      return false;
    }
    MS_EXCEPTION_IF_CHECK_FAIL(captured, "check ProduceInterpretValue and TryToCapture");
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

bool GraphAnalyzer::AddToCaptured(ValueNode *v) {
  int op = v->GetOpcode();
  bool repeat_op = Config().GetBoolConfig(GraphJitConfig::kEnableOptimizeForAttrItem);
  bool attr_access = op == LOAD_ATTR || op == BINARY_SUBSCR;
  if (attr_access && (!repeat_op || !v->GetBlock() || v->GetBlock()->HasAttrSideEffect())) {
    // if operation can't be repeated, or block has attr access side effect
    // can't reorder attr access op, must be interpret all attr, item access operation
    return false;
  }

  if (Utils::IsCallOp(v->GetOpcode())) {
    AObject *f = v->input(0)->GetVobj();
    if (f == nullptr) {
      return false;
    }
    // don't pass unknown callable to graph
    bool is_known_func = f->GetType() == AObject::kTypeCell || f->GetType() == AObject::kTypePrimitive;
    bool is_ms_support_func = f->TestMsFlag(kMsFlagSet);
    if (!is_known_func && !is_ms_support_func) {
      return false;
    }
    GetCaptureInfo().must_be_graph_mode_ |= f->TestMsFlag(AObject::kMsFlagGradFunc);
  }

  auto &locals = GetCaptureInfo().escaped_locals;          // interpret values
  auto &values = GetCaptureInfo().captured_locals.values;  // graph produced values
  for (auto i : v->getInputs()) {
    bool produced_in_graph = values.find(i) != values.end() || IsNonLocalValue(i);
    MS_EXCEPTION_IF_CHECK_FAIL(produced_in_graph || locals.find(i) != locals.end(),
                               "check values order, all input must be generate before this value " + i->to_str());
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

bool GraphAnalyzer::TryToCapture(AbstractNode *n) {
  ValueNode *v = static_cast<ValueNode *>(n);
  AObject *o = v->GetVobj();
  if (IsNonLocalValue(v)) {
    return true;
  }
  if (Utils::IsMsUnsupported(v->GetOpcode()) || !v->IsMindsporeSupportedOperation()) {
    // if mindspore unsupported, must be interpret
  } else if (AddToCaptured(v)) {
    return true;
  }

  const int ms_flag =
    AObject::kMsFlagGradFunc | AObject::kMsFlagShardFunc | AObject::kMsFlagVmapFunc | AObject::kMsFlagJitFunc;
  if (o && o->TestMsFlag(ms_flag) && AddToCaptured(v)) {
    GetCaptureInfo().must_be_graph_mode_ = o->TestMsFlag(AObject::kMsFlagGradFunc);
    return true;
  }

  if (ProduceInterpretValue(v)) {
    return true;
  }
  if (!HasTensorOperation()) {
    CleanCapturedValue();
    AddToEscaped(v);
    return true;
  }
  MS_LOG(DEBUG) << "---operation that depend on the graph outputs, break graph---";
  return false;
}

bool GraphAnalyzer::AnalyzeCall(CallNode *call_node) {
  Graph *g = call_node->GetGraph();
  if (call_node->GetInlineReason() == InlineReason::kInline ||
      call_node->GetInlineReason() == InlineReason::kInlinePartial) {
    for (ValueNode *i : call_node->GetParams()) {
      TryToCapture(i);  // first captured
      i->set_bci(-1);
    }
    if (!AnalyzeRecursive(call_node->GetSubGraph())) {
      g->StopTraceAt(call_node->bci(), call_node->GetSubGraph()->GetStopTraceReason());
      return false;
    }
    return true;
  }
  // class build in graph if it's life cycle only in graph
  g->StopTraceAt(call_node->bci(), call_node->GetSubGraph()->GetStopTraceReason());
  return false;
}

bool GraphAnalyzer::AnalyzeBlock(Block *b) {
  auto stop_trace = b->GetGraph()->GetStopTraceAt();
  for (auto n : b->GetNodes()) {
    if (n->GetType() == AbstractNode::Call && (reinterpret_cast<CallNode *>(n))->GetSubGraph()) {
      if (!AnalyzeCall(reinterpret_cast<CallNode *>(n))) {
        return false;
      }
      continue;
    }
    if (stop_trace && stop_trace->bci() == (reinterpret_cast<ValueNode *>(n))->bci()) {
      return false;
    }
    if (!TryToCapture(n)) {
      b->GetGraph()->StopTraceAt((reinterpret_cast<InstrNode *>(n))->bci(),
                                 StopTraceReason::kStopTraceDataDependsOnGraphOut);
      return false;
    }
  }
  return true;
}

bool GraphAnalyzer::AnalyzeRecursive(Graph *g) {
  if (!g->GetCFG().get()) {
    return true;  // empty graph
  }
  g->GetCFG()->ClearDeadBBEdges();
  Block *b = g->GetCFG()->GetFirstBB();
  auto stop_trace = g->GetStopTraceAt();
  do {
    if (!AnalyzeBlock(b)) {
      return false;  // graph break;
    }
    if (b->succ_bbs().size() == 1) {
      Block *next = *b->succ_bbs().begin();
      if (next->pred_bbs().size() != 1) {
        MS_ASSERT(next->is_loop_head());
        return false;
      }
      MS_ASSERT(b == *next->pred_bbs().begin());
      b = next;
      continue;
    }
    break;  // return block or branch Block
  } while (true);
  // break at unsupported bytecode
  if (stop_trace && stop_trace->bci() <= b->instrs().back().bci()) {
    if (Utils::IsIfJump(stop_trace->GetOpcode()) &&
        g->GetStopTraceReason() != StopTraceReason::kStopTraceLoop_Unsupported) {
      g->StopTraceAt(stop_trace->bci(), StopTraceReason::kStopTraceIf_Unsupported);
    }
    return false;
  }
  // break at branch
  if (b && (b->GetJumpBB() || b->GetFallBB()) && !b->is_loop_head()) {
    g->StopTraceAt(b->instrs().back().bci(), StopTraceReason::kStopTraceIf_Unsupported);
    return false;
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
