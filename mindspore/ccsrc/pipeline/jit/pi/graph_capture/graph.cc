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
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include "pipeline/jit/pi/common.h"

namespace mindspore {
namespace jit {
namespace graph {
Graph::Graph(PyCodeObject *co, PyObject *globals, const GraphJitConfig &conf)
    : ret_val_(nullptr),
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
    if (conf_.GetBoolConfig(GraphJitConfig::kPrintAfterAll)) {
      GRAPH_JIT_LOG_F("%s", DumpLoops().c_str());
    }
  }
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
  // find the last backward edge is overlap this break point
  int res = break_bci;
  for (int i = break_bci; i < static_cast<int>(instr.size()); ++i) {
    MS_EXCEPTION_IF_CHECK_FAIL(i == instr[i]->bci(), "!!!");
    if (instr[i]->extra_jump() != nullptr) {
      res = std::min(instr[i]->extra_jump()->bci(), res);
    }
  }
  return res != break_bci;
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

TracePtr GetTrace(ValueNode *node, bool strict, bool print, int depth, int max_depth);
static bool PrepareTraceParam(std::vector<ValueNode *> *inputs, TraceVector *tv, int depth, int max_depth,
                              bool *has_unsupported, bool strict, bool print) {
  for (auto it : *inputs) {
    auto t = GetTrace(it, strict, print, depth + 1, max_depth);
    if (t == nullptr) {
      return false;
    } else if (t->GetTraceType() == TraceType::Unsupported) {
      *has_unsupported = true;
    }
    tv->push_back(t);
  }
  return true;
}

static bool CheckDepth(int depth, int max_depth) { return depth < max_depth || max_depth == -1; }

static bool CheckObjPtr(ValueNode *node) {
  return node->GetVobj() == nullptr || node->GetVobj()->GetPyObject().ptr() == nullptr;
}

TracePtr GetTrace(ValueNode *node, bool strict, bool print, int depth, int max_depth) {
  if (!CheckDepth(depth, max_depth)) {
    MS_LOG(DEBUG) << "too deep trace for guard";
    return nullptr;
  }
  TraceVector tv;
  ValueNode *p = node;
  bool has_unsupported = false;
  if (!PrepareTraceParam(&(node->getInputs()), &tv, depth, max_depth, &has_unsupported, strict, print) ||
      CheckObjPtr(node)) {
    return strict ? nullptr : std::make_shared<UnsupportedTrace>(nullptr, tv, p->GetOpcode(), p->GetOparg());
  }
  TracePtr ret = nullptr;
  switch (node->GetType()) {
    case AbstractNode::Type::Value: {
      if (!has_unsupported) {
        ret = CreateOpTrace(p->GetVobj()->GetPyObject().ptr(), p->GetOpcode(), p->GetOparg(), tv,
                            node->GetGraph()->GetModuleName(), p->GetName(), strict, print);
      }
      if (ret == nullptr && !strict) {
        ret = std::make_shared<UnsupportedTrace>(p->GetVobj()->GetPyObject().ptr(), tv, p->GetOpcode(), p->GetOparg());
      }
      return ret;
    } break;
    case AbstractNode::Type::Call: {
      if (!has_unsupported) {
        if (p->GetOpcode() == CALL_FUNCTION) {
          ret = CreateOpTrace(p->GetVobj()->GetPyObject().ptr(), p->GetOpcode(), p->GetOparg(), tv,
                              node->GetGraph()->GetModuleName(), p->getInputs()[0]->GetName(), strict, print);
        } else {
          ret = CreateOpTrace(p->GetVobj()->GetPyObject().ptr(), p->GetOpcode(), p->GetOparg(), tv,
                              node->GetGraph()->GetModuleName(), p->GetName(), strict, print);
        }
      }
      if (ret == nullptr && !strict) {
        ret = std::make_shared<UnsupportedTrace>(p->GetVobj()->GetPyObject().ptr(), tv, p->GetOpcode(), p->GetOparg());
      }
      return ret;
    } break;
    case AbstractNode::Type::Param: {
      return std::make_shared<RootTrace>(p->GetVobj()->GetPyObject().ptr(), mindspore::jit::graph::TraceType::Param,
                                         p->GetOparg(), p->GetName());
    }
    case AbstractNode::Type::CellVar:
    case AbstractNode::Type::FreeVar: {
      return std::make_shared<RootTrace>(p->GetVobj()->GetPyObject().ptr(), mindspore::jit::graph::TraceType::Param,
                                         p->GetOparg(), p->GetName());
    }
    case AbstractNode::Type::Unbound:
      break;
    default:
      break;
  }
  return nullptr;
}

bool Graph::GuardValueNode(ValueNode *node) {
  AObject *vo = node->GetVobj();
  if (guard_ == nullptr || !vo || vo->GetPyObject().ptr() == nullptr) {
    return false;
  }
  if (node->GetOpcode() == LOAD_CONST) {
    return true;
  }
  TracePtr t = GetTrace(node, Config().GetBoolConfig(GraphJitConfig::kStrictTrace),
                        Config().GetBoolConfig(GraphJitConfig::kPrintGuard), 0,
                        Config().getIntConfig(GraphJitConfig::GraphJitConfig::kMaxTraceDepth));
  if (t != nullptr) {
    bool ret = guard_->GetGuard()->GuardOn(t, mindspore::jit::graph::GuardLevel::GEqual);
    if (Config().GetBoolConfig(GraphJitConfig::kGuardDetachObject)) {
      t->Detach();
    }
    return ret;
  }
  return false;
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

void Graph::print(int depth) const {
  std::string prefix(depth << 1, ' ');
  std::string code_name = co_.ptr() != nullptr ? std::string(py::str(co_.ptr())) : "<no code>";
  GRAPH_JIT_LOG_F("%s*** Trace Nodes [%s] ***\n", prefix.c_str(), code_name.c_str());

  if (depth == 0) {
    GRAPH_JIT_LOG_F("%s Frame:\n", prefix.c_str());
    const FrameStates &f = GetFrame(0);
    f.print();
  }

  GRAPH_JIT_LOG_F("%sNodes:\n", prefix.c_str());
  for (auto i : GetTracedNodes()) {
    GRAPH_JIT_LOG_F("%s%s\n", prefix.c_str(), i->ToString().c_str());
    if (i->GetType() != AbstractNode::Call) {
      continue;
    }
    CallNode *node = static_cast<CallNode *>(i);
    GRAPH_JIT_LOG_F("%s{--inline stat %s --\n", prefix.c_str(), GetInlineReasonDesc(node->GetInlineReason()).c_str());
    if (node->GetSubGraph() != nullptr) {
      node->GetSubGraph()->print(depth + 1);
    }
    GRAPH_JIT_LOG_F("%s}\n", prefix.c_str());
  }

  GRAPH_JIT_LOG_F("\n%s return: %s\n", prefix.c_str(),
                  GetRetVal() ? GetRetVal()->ToString().c_str() : "track break or no return");
  GRAPH_JIT_LOG_F("%s%s\n", prefix.c_str(), GetStopTraceReasonDesc(stop_trace_info_.reason).c_str());
  GRAPH_JIT_LOG_F("%sbreak bci: %d\n", prefix.c_str(), stop_trace_info_.bci);
  GRAPH_JIT_LOG_F("\n");
}

void FrameStates::print() const {
  GRAPH_JIT_LOG_F("locals:\n");
  int c = 0;
  for (auto i : locals) {
    GRAPH_JIT_LOG_F("%d:%s\n", c++, i == &ValueNode::UnboundLocal ? "(UnboundLocal)" : i->ToString().c_str());
  }
  GRAPH_JIT_LOG_F("\nstacks:\n");
  for (auto i : stack) {
    GRAPH_JIT_LOG_F("%d:%s\n", c++, i == &ValueNode::UnboundLocal ? "(UnboundLocal)" : i->ToString().c_str());
  }
  GRAPH_JIT_LOG_F("\ncell_free:\n");
  for (auto i : cell_free) {
    GRAPH_JIT_LOG_F("%s\n", i == &ValueNode::UnboundLocal ? "(UnboundLocal)" : i->ToString().c_str());
  }
  GRAPH_JIT_LOG_F("\n");
}

std::string Graph::DumpLoops() const {
  std::ostringstream os;
  if (loops_.empty()) {
    return os.str();
  }
  os << "*** Dump Loops on [" << py::str(co_.ptr()).cast<std::string>() << "] ***\n";
  for (const auto *lp : loops_) {
    os << lp->Dump();
  }
  os << '\n';
  return os.str();
}

}  // namespace graph
}  // namespace jit
}  // namespace mindspore
