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
#include "pipeline/jit/pi/graph_capture/graph.h"
#include <set>
#include <algorithm>
#include <memory>
#include <string>
#include <regex>
#include <utility>
#include "pipeline/jit/pi/common.h"

namespace mindspore {
namespace pijit {
Graph::Graph(PyCodeObject *co, PyObject *globals, const GraphJitConfig &conf)
    : ret_val_(nullptr),
      generator_result_(nullptr),
      co_(py::cast<py::object>(reinterpret_cast<PyObject *>(co))),
      f_globals_(py::cast<py::object>(globals)),
      conf_(conf),
      guard_(nullptr),
      prune_branch_count_(0) {
  stop_trace_info_ = {-1, StopTraceReason::kNonStopTrace};
  if (!co) {
    frame_states_[0] = std::make_unique<FrameStates>();  // emplace empty frame
    module_name_ = "";
    return;
  }
  cfg_ = std::make_unique<CFG>(co);
  cfg_->GenerateCFG();

  if (conf_.GetBoolConfig(GraphJitConfig::kPrintBB)) {
    GRAPH_JIT_LOG_F("%s\n\n", cfg_->DumpBBs().c_str());
  }
  if (conf_.GetBoolConfig(GraphJitConfig::kPrintCFG)) {
    cfg_->DumpCFGGraph();
  }

  auto pyname = PyDict_GetItemString(globals, "__name__");
  if (pyname) {
    module_name_ = PyUnicode_AsUTF8(pyname);
  } else {
    module_name_ = "";
    PyErr_Clear();
  }

  if (conf_.GetBoolConfig(GraphJitConfig::kLoopUnrolling)) {
    LoopFinder loop_finder(this);
    loop_finder.FormSimpleLoopInfo();
  }
  sideEffect_ = std::make_unique<SideEffect>();
}

ValueNode *Graph::NewValueNode(AObject *obj_info, int op, int arg, const std::vector<ValueNode *> &inputs,
                               const std::string &name) {
  MS_EXCEPTION_IF_CHECK_FAIL(!Utils::IsCallOp(op), "must not be call function opcode");
  ValueNode *node = this->allocator().NewNode<ValueNode>(obj_info, op, arg, inputs);
  node->SetName(name);
  node->SetGraph(this);
  ConstantInfo::CollectConstantInfo(node);
  if (node->IsConstantValue() && obj_info && CheckConstPyObject(obj_info->GetPyObject().ptr())) {
    node->SetOpcode(LOAD_CONST);
    node->SetOparg(-1);
    node->ClearInputs();
  }
  return node;
}

CallNode *Graph::NewCallNode(int op, int arg, const std::vector<ValueNode *> &inputs) {
  MS_EXCEPTION_IF_CHECK_FAIL(Utils::IsCallOp(op), "must be call function opcode");
  CallNode *node = this->allocator().NewNode<CallNode>(op, arg, inputs);
  node->SetGraph(this);
  return node;
}

/**
 * TODO: FindLoopEnd, FindLoopBegin, reset break bci
 * restore graph status. clean the variable that loop produced
 * restore frame status of break bci that override by loop analyze
 */
bool Graph::IsBreakAtLoop() const {
  int break_bci = this->GetStopTraceBci();
  if (break_bci == -1) {
    return false;
  }
  const auto &instr = this->cfg_->instr_pool();
  // find the last backward edge overlapping this break point
  int res = break_bci;
  for (int i = break_bci; i < static_cast<int>(instr.size()); ++i) {
    MS_EXCEPTION_IF_CHECK_FAIL(i == instr[i]->bci(), "!!!");
    if (instr[i]->extra_jump() != nullptr) {
      res = std::min(instr[i]->extra_jump()->bci(), res);
    }
  }
  return res != break_bci;
}

bool Graph::IsBreakAtLoopAfterUnrolling() const {
  if (!Config().GetBoolConfig(GraphJitConfig::kLoopUnrolling)) {
    return false;
  }
  if (GetStopTraceBci() == -1) {
    return false;
  }
  if (traced_nodes_.empty()) {
    return false;
  }
  if (GetStopTraceBci() > traced_nodes_.back()->bci()) {
    return false;
  }
  bool break_with_loop_unroll = false;
  for (auto node : traced_nodes_) {
    if (node->bci() >= GetStopTraceBci()) {
      break_with_loop_unroll |= node->GetBlock() != nullptr && node->GetBlock()->is_loop_body();
    }
  }
  return break_with_loop_unroll;
}

void Graph::SetFrame(int bci, const FrameStates &f) {
  // just set once, used to restore the first status if has a loop
  auto &ptr = frame_states_[bci];
  if (ptr == nullptr) {
    ptr = std::make_unique<FrameStates>(f);
  }
}

const FrameStates &Graph::GetFrame(int bci) const {
  auto iter = frame_states_.find(bci);
  MS_EXCEPTION_IF_CHECK_FAIL(iter != frame_states_.end(), "can't find frame status");
  return *(iter->second);
}

static bool CheckObjPtr(ValueNode *node) {
  return node->GetVobj() == nullptr || node->GetVobj()->GetPyObject().ptr() == nullptr;
}

TracePtr GetTrace(ValueNode *node, bool strict, bool print, int depth, int max_depth);
static bool PrepareTraceParam(ValueNode *node, TraceVector *tv, int depth, int max_depth, bool *has_unsupported,
                              bool strict, bool print) {
  const std::vector<ValueNode *> &inputs = node->getInputs();
  for (auto it : inputs) {
    auto t = GetTrace(it, strict, print, depth + 1, max_depth);
    if (t == nullptr) {
      return false;
    } else if (t->GetTraceType() == TraceType::Unsupported) {
      *has_unsupported = true;
    }
    tv->push_back(t);
  }
  return !CheckObjPtr(node);
}

static bool CheckDepth(int depth, int max_depth) { return depth < max_depth || max_depth == -1; }

TracePtr GetTrace(ValueNode *node, bool strict, bool print, int depth, int max_depth) {
  if (!CheckDepth(depth, max_depth)) {
    MS_LOG(DEBUG) << "too deep trace for guard";
    return nullptr;
  }
  if (node->GetType() == AbstractNode::Type::Call) {
    Graph *sub_graph = static_cast<CallNode *>(node)->GetSubGraph();
    if (sub_graph && sub_graph->GetRetVal() != nullptr) {
      return GetTrace(sub_graph->GetRetVal(), strict, print, depth, max_depth);
    }
  }

  PyObject *obj = node->GetVobj() ? node->GetVobj()->GetPyObject().ptr() : nullptr;
  int opcode = node->GetOpcode();
  int oparg = node->GetOparg();
  const std::string &name = node->GetName();
  const char *module_name = node->GetGraph() ? node->GetGraph()->GetModuleName() : "";

  TraceVector tv;
  bool has_unsupported = false;
  if (!PrepareTraceParam(node, &tv, depth, max_depth, &has_unsupported, strict, print)) {
    return strict ? nullptr : std::make_shared<UnsupportedTrace>(nullptr, tv, opcode, oparg);
  }
  TracePtr ret = nullptr;
  switch (node->GetType()) {
    case AbstractNode::Type::Value:
      if (!has_unsupported) {
        ret = CreateOpTrace(obj, opcode, oparg, tv, module_name, name, strict, print);
      }
      break;
    case AbstractNode::Type::Call:
      if (!has_unsupported) {
        const std::string &func_name = node->input(0)->GetName();
        ret = CreateOpTrace(obj, opcode, oparg, tv, module_name, func_name, strict, print);
      }
      break;
    case AbstractNode::Type::Param:
      return std::make_shared<RootTrace>(obj, mindspore::pijit::TraceType::Param, oparg, name);
    case AbstractNode::Type::CellVar: /* fall-through */
    case AbstractNode::Type::FreeVar:
      return std::make_shared<RootTrace>(obj, mindspore::pijit::TraceType::Param, oparg, name);
    case AbstractNode::Type::kUnbound:
      break;
    default:
      break;
  }
  return (ret == nullptr && !strict) ? std::make_shared<UnsupportedTrace>(obj, tv, opcode, oparg) : ret;
}

bool Graph::GuardValueNode(ValueNode *node) {
  if (node->IsConstantValue()) {
    return true;
  }
  TracePtr tr = this->TraceValueNode(node);
  if (tr == nullptr) {
    return false;
  }
  bool ret = guard_->GetGuard()->GuardOn(tr, mindspore::pijit::GuardLevel::GEqual);
  node->SetConstantValue(ret);
  return ret;
}

TracePtr Graph::TraceValueNode(ValueNode *node, int max_trace_depth) {
  AObject *vo = node->GetVobj();
  if (guard_ == nullptr || !vo || vo->GetPyObject().ptr() == nullptr) {
    return nullptr;
  }
  if (max_trace_depth < 0) {
    max_trace_depth = Config().getIntConfig(GraphJitConfig::GraphJitConfig::kMaxTraceDepth);
  }
  return GetTrace(node, Config().GetBoolConfig(GraphJitConfig::kStrictTrace),
                  Config().GetBoolConfig(GraphJitConfig::kPrintGuard), 0, max_trace_depth);
}

std::vector<ValueNode *> Graph::CollectAliveNode(int bci, std::vector<int> *ids, BitMap *map) const {
  if (bci == -1) {
    return {this->GetRetVal()};
  }
  BitMap alive = this->GetCFG()->GetLiveness()->CollectAlive(bci);
  std::vector<ValueNode *> result = CollectAliveNode(this->GetFrame(bci), &alive, ids);
  if (map != nullptr) {
    *map = std::move(alive);
  }
  return result;
}

std::vector<ValueNode *> Graph::CollectAliveNode(const FrameStates &last_frame, BitMap *alive, std::vector<int> *ids) {
  std::vector<ValueNode *> outputs = last_frame.GetStacks();
  // collect alive locals
  for (BitMap::Iter iter(alive, true), end(alive, false); iter != end; ++iter) {
    size_t i = *iter;
    // exclude undefined locals
    if (last_frame.Local(i) != &ValueNode::kUnboundLocal) {
      if (ids != nullptr) {
        ids->push_back(i);
      }
      outputs.push_back(last_frame.Local(i));
    } else {
      alive->Clear(i);
    }
  }
  return outputs;
}

bool Graph::GuardSequenceNodeLength(ValueNode *sequence_node, Py_ssize_t sequence_size) {
  if (sequence_node->IsConstantValue()) {
    return true;
  }
  const auto &cnst = sequence_node->GetConstantInfo();
  if (cnst != nullptr && cnst->len() != -1) {
    MS_EXCEPTION_IF_CHECK_FAIL(sequence_size == cnst->len(), "error sequence length");
    return true;
  }
  TracePtr tr = this->TraceValueNode(sequence_node);
  if (tr == nullptr) {
    return false;
  }
  const auto &guard = this->GetGuard()->GetGuard();
  bool strict = this->Config().GetBoolConfig(GraphJitConfig::kStrictTrace);

  PyObject *builtin_len = PyDict_GetItemString(PyEval_GetBuiltins(), "len");
  MS_EXCEPTION_IF_NULL(builtin_len);
  TracePtr len_func = CreateOpTrace(builtin_len, LOAD_CONST, -1, {}, "", "", strict);
  TracePtr len_trace = CreateOpTrace(py::int_(sequence_size).ptr(), CALL_FUNCTION, 1, {len_func, tr}, "", "", strict);
  guard->GuardOn(len_trace, GuardLevel::GEqual, false);

  sequence_node->MakeConstantInfo()->set_len(sequence_size);
  return true;
}

bool Graph::GuardType(ValueNode *node) {
  if (node->IsConstantValue()) {
    return true;
  }
  const auto &cnst = node->GetConstantInfo();
  if (cnst != nullptr && cnst->type() != nullptr) {
    return true;
  }
  TracePtr tr = this->TraceValueNode(node);
  if (tr == nullptr) {
    return false;
  }
  bool ret = guard_->GetGuard()->GuardOn(tr, mindspore::pijit::GuardLevel::GType);
  node->MakeConstantInfo()->set_type(node->GetVobj()->GetTypeObject());
  return ret;
}

static bool SkipGuardInlinedFunc(ValueNode *func_node) {
  if (func_node->IsConstantValue()) {
    return true;
  }
  AObject::Type value_type = func_node->GetVobj()->GetType();
  if (func_node->GetOpcode() == LOAD_ATTR) {
    AObject *src_info = func_node->input(0)->GetVobj();
    if (src_info->GetType() == AObject::kTypeTensor && value_type == AObject::kTypeBoundMethod) {
      // function from Tensor
      return true;
    }
  }
  return false;
}

bool Graph::GuardInlinedFunc(CallNode *call_node) {
  if (SkipGuardInlinedFunc(call_node->input(0))) {
    return true;
  }
  TracePtr tr = this->TraceValueNode(call_node->input(0));
  if (tr == nullptr) {
    return false;
  }
  const auto &guard = this->GetGuard()->GetGuard();
  bool strict = this->Config().GetBoolConfig(GraphJitConfig::kStrictTrace);

  AObject *callable_info = call_node->input(0)->GetVobj();
  AObject::Type func_type = callable_info->GetType();
  PyObject *callable = callable_info->GetPyObject().ptr();
  if (func_type == AObject::kTypeBoundMethod) {
    PyObject *func = PyMethod_GET_FUNCTION(callable);
    tr = CreateOpTrace(func, LOAD_ATTR, 0, {tr}, "", "__func__", strict);
    tr = CreateOpTrace(PyFunction_GET_CODE(func), LOAD_ATTR, 0, {tr}, "", "__code__", strict);
    guard->GuardOn(tr, GuardLevel::GId);
  } else if (func_type == AObject::kTypeCell || func_type == AObject::kTypeAnyValue) {
    guard->GuardOn(tr, GuardLevel::GType, false);
    call_node->input(0)->MakeConstantInfo()->set_type(callable_info->GetTypeObject());
  } else if (func_type == AObject::kTypeFunction) {
    tr = CreateOpTrace(PyFunction_GET_CODE(callable), LOAD_ATTR, 0, {tr}, "", "__code__", strict);
    guard->GuardOn(tr, GuardLevel::GId);
    call_node->input(0)->SetConstantValue(true);
  } else {
    return false;
  }
  return true;
}

static std::string TraceInferFailed(ValueNode *node) {
  std::stringstream s;
  s << node << " ";
  switch (node->GetType()) {
    case AbstractNode::Call:
    case AbstractNode::Value: {
      s << "bci " << node->bci() << " " << Utils::GetOpName(node->GetOpcode()) << " " << node->GetOparg();
      if (Utils::IsNameRelated(node->GetOpcode())) {
        s << " " << node->GetName();
      }
      break;
    }
    case AbstractNode::Param: {
      s << "Parameter " << node->GetOparg();
      break;
    }
    case AbstractNode::CellVar:
    case AbstractNode::FreeVar: {
      s << "Closure " << node->GetOparg();
      break;
    }
    case AbstractNode::kUnbound: {
      s << "(kUnboundLocal)";
      break;
    }
    default: {
      break;
    }
  }
  s << " object is ";
  PyObject *op = node->GetVobj() ? node->GetVobj()->GetPyObject().ptr() : nullptr;
  if (op != nullptr) {
    s << AObject::ToString(op);
    return s.str();
  }
  s << "<NULL>:\n";
  for (size_t i = 0; i < node->getInputs().size(); ++i) {
    s << " " << std::regex_replace(TraceInferFailed(node->input(i)), std::regex("\n"), "\n  ") << "\n";
  }
  s.seekp(-1, s.cur);
  s << " ";
  return s.str();
}

std::string Graph::ToString(int depth) const {
  std::stringstream s;
  std::string prefix(depth << 1, ' ');
  std::string code_name = co_.ptr() != nullptr ? std::string(py::str(co_.ptr())) : "<no code>";

  s << prefix << "*** Trace Nodes [" << code_name << "] ***\n";
  if (depth == 0) {
    s << prefix << "Frame:\n" << GetFrame(0).ToString();
  }

  s << prefix << "Nodes:\n";
  for (auto i : GetTracedNodes()) {
    s << prefix << i->ToString() << "\n";
    if (i->GetType() != AbstractNode::Call) {
      continue;
    }
    CallNode *node = static_cast<CallNode *>(i);
    s << prefix << "{ inline stat " << GetInlineReasonDesc(node->GetInlineReason()) << "\n";
    if (node->GetSubGraph() != nullptr) {
      s << node->GetSubGraph()->ToString(depth + 1);
    }
    s << prefix << "}\n";
  }
  s << prefix << "return: " << (GetRetVal() ? GetRetVal()->ToString() : "trace break") << "\n";
  s << prefix << GetStopTraceReasonDesc(GetStopTraceReason()) << "\n";
  s << prefix << "break bci: " << GetStopTraceBci() << "\n\n";

  if (depth == 0) {
    std::string break_info;
    if (GetRetVal()) {
      PyObject *op = GetRetVal()->GetVobj() ? GetRetVal()->GetVobj()->GetPyObject().ptr() : nullptr;
      if (op == nullptr) {
        break_info = TraceInferFailed(GetRetVal());
      }
    } else {
      break_info = this->DumpBreakInfo();
    }
    if (!break_info.empty()) {
      s << prefix << std::regex_replace(break_info, std::regex("\n"), "\n" + prefix) << "\n";
    }
  }
  return s.str();
}

void DumpUnsupportedByteCodeInfo(std::stringstream &s, int op, int arg) {
  if (op == SETUP_WITH || op == SETUP_FINALLY) {
    s << Utils::GetOpName(op) << " " << arg << " is skipped in break_graph or a exception happened.\n";
  } else {
    s << Utils::GetOpName(op) << " " << arg << " is not support.\n";
  }
}

std::string Graph::DumpBreakInfo() const {
  if (GetStopTraceBci() == -1) {
    return std::string();
  }
  std::stringstream s;
  const auto &f = GetFrame(GetStopTraceBci());
  const auto &nodes = GetTracedNodes();
  const auto &instrs = cfg_->instr_pool();
  int break_bci = GetStopTraceBci();

  s << "graph break at: " << break_bci << ":\n";
  std::vector<ValueNode *> parameters;
  if (nodes.size() == 0 || nodes.back()->bci() < break_bci) {
    // break at unsupported bytecode
    int op = instrs[break_bci]->op();
    int arg = instrs[break_bci]->arg();
    DumpUnsupportedByteCodeInfo(s, op, arg);
    switch (op) {
      case POP_JUMP_IF_FALSE:
      case POP_JUMP_IF_TRUE:
      case JUMP_IF_FALSE_OR_POP:
      case JUMP_IF_TRUE_OR_POP:
      case FOR_ITER:
      case UNPACK_SEQUENCE:
      case UNPACK_EX: {
        parameters.push_back(f.Peek(0));
        break;
      }
      case CALL_FUNCTION_EX: {
        arg = (arg & 0x01);
      }
      case CALL_FUNCTION_KW: {
        arg++;
      }
      case CALL_METHOD:
      case CALL_FUNCTION: {
        for (int i = arg; i >= 0; --i) {
          parameters.push_back(f.Peek(i));
          AObject *v = f.Peek(i)->GetVobj();
          // just print the first infer failed value
          if (v == nullptr || v->GetPyObject().ptr() == nullptr) {
            parameters = {f.Peek(i)};
            break;
          }
        }
        break;
      }
      default: {
        return s.str();
      }
    }
  } else {
    auto iter = std::find_if(nodes.begin(), nodes.end(), [break_bci](ValueNode *n) { return n->bci() == break_bci; });
    MS_EXCEPTION_IF_CHECK_FAIL(iter != nodes.end(), "can't find break info.");
    parameters.push_back(*iter);
  }
  // print traced value
  for (auto node : parameters) {
    s << TraceInferFailed(node) << "\n";
  }
  return s.str();
}

std::string FrameStates::ToString() const {
  std::stringstream s;
  s << "locals:\n";
  for (size_t i = 0; i < locals.size(); ++i) {
    if (locals[i] != &ValueNode::kUnboundLocal) {
      s << i << ": " << locals[i]->ToString() << "\n";
    }
  }
  s << "\nstacks:\n";
  std::for_each(stack.rbegin(), stack.rend(), [&s](ValueNode *i) { s << i->ToString() << "\n"; });
  s << "\ncell free:\n";
  std::for_each(cell_free.begin(), cell_free.end(), [&s](ValueNode *i) { s << i->ToString() << "\n"; });
  s << "\n";
  return s.str();
}

}  // namespace pijit
}  // namespace mindspore
