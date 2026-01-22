/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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
#include "frontend/jit/pi/graph_capture/graph.h"
#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <regex>
#include <utility>
#include <set>
#include <unordered_map>
#include "frontend/jit/pi/graph_build/func_graph_builder.h"
#include "frontend/jit/pi/runtime.h"
#include "frontend/jit/pi/utils/utils.h"
#include "frontend/jit/pi/graph_capture/graph_build.h"
#include "frontend/jit/pi/graph_capture/abstract_wrapper.h"
#include "frontend/jit/pi/graph_guard/infer.h"
#include "frontend/jit/pi/graph_guard/cache.h"
#include "utils/convert_utils_base.h"

namespace mindspore {
namespace pijit {

class DefineMap : public GuardStatus {
 public:
  bool is_definitions_map() const override { return true; }
  std::unordered_map<const AObject *, std::set<ValueNode *>> map_;
};

class GuardBuilder {
 public:
  using DefineMapPtr = std::shared_ptr<DefineMap>;
  GuardBuilder(bool strict, int max_depth, bool print) : max_depth_(max_depth), strict_(strict), print_(print) {}

  const auto &guard() const { return guard_; }
  void SetGuard(const std::shared_ptr<OptCode> &g);

  TracePtr GetTrace(ValueNode *node, int depth = 0);
  TraceVector GetTraceClosure(ValueNode *node, bool *succ, int depth = 0);
  bool CheckDepthForTrace(TracePtr *ret, ValueNode *node, int depth);
  bool PrepareTraceParam(ValueNode *node, TraceVector *tv, int depth, bool *has_unsupported);
  TracePtr CacheTrace(ValueNode *node, TracePtr ret, const TraceVector &tv);

  std::shared_ptr<OptCode> guard_;
  DefineMapPtr definitions_;
  int max_depth_;
  bool strict_;
  bool print_;
};

Graph::~Graph() {}
Graph::Graph(PyCodeObject *co, PyObject *globals, const GraphJitConfig &conf)
    : ret_val_(nullptr),
      generator_result_(nullptr),
      co_(py::cast<py::object>(reinterpret_cast<PyObject *>(co))),
      f_globals_(py::cast<py::object>(globals)),
      conf_(conf),
      side_effect_handler_(std::make_shared<SideEffectHandler>(this)),
      func_graph_builder_(nullptr) {
  guard_builder_ = std::make_unique<GuardBuilder>(
    // save config
    Config().GetBoolConfig(GraphJitConfig::kStrictTrace), Config().getIntConfig(GraphJitConfig::kMaxTraceDepth),
    IsPiJitLogOn(LogCfg::kGuard));

  break_info_.bci_ = -1;
  break_info_.reason_ = StopTraceReason::kNonStopTrace;
  if (!co) {
    frame_states_[0] = std::make_unique<FrameStates>();  // emplace empty frame
    module_name_ = "";
    return;
  }
  cfg_ = std::make_unique<CFG>(co);
  cfg_->GenerateCFG();

  if (conf_.GetBoolConfig(GraphJitConfig::kPrintBB)) {
    GRAPH_JIT_LOG_F("%s\n\n", cfg_->ToString().c_str());
  }

  auto pyname = PyDict_GetItemString(globals, "__name__");
  if (pyname) {
    module_name_ = PyUnicode_AsUTF8(pyname);
  } else {
    module_name_ = "";
    PyErr_Clear();
  }
}

const std::shared_ptr<OptCode> &Graph::GetGuardManager() const { return guard_builder_->guard(); }
void Graph::SetGuard(const std::shared_ptr<OptCode> &g) { guard_builder_->SetGuard(g); }
void Graph::RemoveAllGuardItems() const { GetGuardManager()->GetGuard()->RemoveAllGuardItems(); }
void GuardBuilder::SetGuard(const std::shared_ptr<OptCode> &g) {
  guard_ = g;
  if (g->guard_status() == nullptr) {
    g->guard_status() = std::make_shared<DefineMapPtr::element_type>();
  }
  definitions_ = std::static_pointer_cast<DefineMapPtr::element_type>(g->guard_status());
}

bool Graph::NeedSymbolic(ValueNode *node) {
  if (Config().getIntConfig(GraphJitConfig::kSymbolic) == -1) {
    return false;
  }
  if (node->IsConstantValue()) {
    // such as LOAD_CONST,
    MS_LOG(DEBUG) << "The node already marked the constant";
    return false;
  }
  AObject::Type real_type = node->GetVobj()->GetType();
  // now, only support these types
  // for mutable tuple and list ... not implement
  const auto valid_type = {AObject::kTypeInt, AObject::kTypeFloat, AObject::kTypeBool, AObject::kTypeTensor};
  if (valid_type.end() == std::find(valid_type.begin(), valid_type.end(), real_type)) {
    return false;
  }
  JitCompileResults *jcr = GetJitCompileResults(GetCodeObj());
  MS_EXCEPTION_IF_NULL(jcr);
  if (jcr->cache().fail_guard().empty()) {
    return false;
  }
  TracePtr trace = this->TraceValueNode(node);
  auto item = jcr->cache().FindFailInfo(trace, GIType::GTEqual);
  if (item.count_ == 0) {  // can't symbolic object type
    return false;
  }
  const int symbolic_scalar = Config().getIntConfig(GraphJitConfig::kSymbolic) + 1;
  const int symbolic_tensor = Config().getIntConfig(GraphJitConfig::kSymbolic);
  const int symbolic_shape = symbolic_tensor > 0 ? symbolic_tensor + 4 : INT_MAX;
  bool is_tensor = real_type == AObject::kTypeTensor;
  // guard is not value equal guard for tensor, will lower to dynamic shape.
  int threshold = is_tensor ? (IsSpecializedGuard(item.item_) ? symbolic_tensor : symbolic_shape) : symbolic_scalar;
  MS_LOG(DEBUG) << "find failed item: !! " << item.item_->GetTrace()->ToString() << " fail count: " << item.count_
                << " > " << threshold << std::endl;
  bool is_mutable = item.count_ > threshold;
  if (!is_mutable) {
    return false;
  }
  py::object symbol_object = SymbolicFromGuard(item.item_, node->GetVobj()->GetPyObject());
  if (symbol_object.ptr() == nullptr) {
    MS_LOG(INFO) << "symbolic fail from guard: " << item.item_->ToString() << std::endl
                 << "the node is: " << node->ToString();
    return false;
  }
  PIJIT_DEBUG_LOG(LogCfg::kDynamic) << "Symbolic object: " << trace->ToString() << " at " << GetFileName(this) << ":"
                                    << node->GetLineNo();
  node->SetVobj(AObject::Convert(symbol_object));
  node->GetTrace()->SetObject(symbol_object);
  // clear compile cache for this fail item ...
  // erase fail item from cache ...
  return true;
}

bool Graph::PrepareParameter(ValueNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(DEBUG) << "node: " << node->ToString();
  bool isExpandOn = conf_.GetBoolConfig(GraphJitConfig::kExpandGraphInput);
  using PrepareHelper = std::function<bool(ValueNode *, std::vector<ValueNode *> *)>;
  static PrepareHelper prepare_oper = [isExpandOn](ValueNode *node, std::vector<ValueNode *> *oper) {
    int opcode = node->GetOpcode();
    if (opcode == LOAD_CONST || opcode == LOAD_GLOBAL || opcode == LOAD_DEREF || node->GetType() == ValueNode::Param ||
        oper->end() != std::find(oper->begin(), oper->end(), node)) {
      return true;
    }
    while (node->GetType() == ValueNode::Call) {
      auto g = static_cast<CallNode *>(node)->GetSubGraph();
      if (g == nullptr || g->GetRetVal() == nullptr) {
        return false;
      }
      node = g->GetRetVal();
    }
    bool isExpandInput =
      (isExpandOn && (GraphBuilder::GetExpandInputMap().find(node) != GraphBuilder::GetExpandInputMap().end()));
    if (opcode != LOAD_ATTR && opcode != BINARY_SUBSCR && !isExpandInput) {
      return false;
    }
    for (const auto &in : node->inputs()) {
      if (!prepare_oper(in, oper)) {
        return false;
      }
    }
    oper->push_back(node);
    return true;
  };

  prepare_.inputs_.push_back(node);
  size_t backup = prepare_.operations_.size();
  if (prepare_oper(node, &prepare_.operations_)) {
    return true;
  }
  prepare_.operations_.resize(backup);
  prepare_.inputs_.pop_back();
  MS_LOG(INFO) << "PrepareParameter failed. Node: " << node->ToString();
  return false;
}

void Graph::AddNodeInfo(ValueNode *node, AObject *obj_info, const std::string &name) {
  Graph *graph = this;

  node->SetName(name);
  node->SetGraph(graph);

  auto new_object = obj_info ? obj_info->GetPyObject().ptr() : nullptr;
  if (new_object != nullptr && !CheckConstPyObject(new_object)) {  // literal not need track
    graph->GetSideEffect()->data()->Track(new_object, node);
  }
  MS_EXCEPTION_IF_CHECK_FAIL(GetGuardManager() != nullptr, "can't find guard");
  // track definitions
  auto &defs_set = guard_builder_->definitions_->map_[obj_info];
  defs_set.insert(node);
  if (obj_info && obj_info->HasMultiVersion()) {
    for (auto i : defs_set) {
      // guard all modified variable define, recompile if changed
      guard_builder_->guard()->GetGuard()->GuardOn(TraceValueNode(i), GuardLevel::kGuardMatchIDS);
    }
  }
  if (node->GetType() == ValueNode::Param) {
    return;
  }
  ConstantInfo::CollectConstantInfo(node);
  if (node->IsConstantValue() && obj_info && CheckConstPyObject(obj_info->GetPyObject().ptr())) {
    node->SetOpcode(LOAD_CONST);
    node->SetOparg(-1);
    node->ClearInputs();
  }
}

CallNode *Graph::NewCallNode(int op, int arg, const std::vector<ValueNode *> &inputs) {
  MS_EXCEPTION_IF_CHECK_FAIL(Opcode(op).IsCall(), "must be call function opcode");
  CallNode *node = this->allocator().NewNode<CallNode>(op, arg, inputs);
  node->SetGraph(this);
  return node;
}

ValueNode *Graph::NewValueNode(AObject *obj_info, int op, int arg, const std::vector<ValueNode *> &inputs,
                               const std::string &name) {
  MS_EXCEPTION_IF_CHECK_FAIL(!Opcode(op).IsCall(), "must not be call function opcode");
  ValueNode *node = this->allocator().NewNode<ValueNode>(obj_info, op, arg, inputs);
  AddNodeInfo(node, obj_info, name);
  return node;
}

CellVarNode *Graph::NewCellNode(AObject *obj_info, int op, int arg, const std::vector<ValueNode *> &inputs,
                                const std::string &name) {
  CellVarNode *node = this->allocator().NewNode<CellVarNode>(CellVarNode::CellVar);
  AddNodeInfo(node, obj_info, name);
  for (auto i : inputs) {
    node->AddInput(i);
  }
  node->SetOpcode(op);
  node->SetOparg(arg);
  node->SetVobj(obj_info);
  return node;
}

ParamNode *Graph::NewParamNode(AObject *obj_info, int index, const std::string &name) {
  ParamNode *node = this->allocator().NewNode<ParamNode>(obj_info, index);
  AddNodeInfo(node, obj_info, name);
  params_.push_back(node);
  return node;
}

void Graph::StopTraceAt(int bci, StopTraceReason reason, const std::vector<std::string> &hints) {
  break_info_.bci_ = bci;
  break_info_.reason_ = reason;

  if (bci != -1 && ((IsPiJitLogOn(LogCfg::kGraphBreak) && !IsBreakAtCall(this)) ||
                    conf_.GetBoolConfig(GraphJitConfig::kFullGraph))) {
    std::ostringstream oss;
    oss << "Reason: " << GetStopTraceReasonDesc(reason);
    if (!hints.empty()) {
      std::for_each(hints.begin(), hints.end(), [&oss](const auto &hint) { oss << "\n  Hint: " << hint; });
    }
    oss << "\n\nFrom user code:\n";
    const auto &trace_ctx_stack = TraceManager::trace_context_stack();
    for (const auto &ctx : trace_ctx_stack) {
      if (ctx.location() != nullptr) {
        oss << ctx.location()->ToString();
      }
    }

    if (!conf_.GetBoolConfig(GraphJitConfig::kFullGraph)) {
      if (trace_ctx_stack.empty() || trace_ctx_stack.back().location() == nullptr) {
        return;
      }
      PIJIT_DEBUG_LOG(LogCfg::kGraphBreak)
        << std::endl
        << "Graph break at: " << trace_ctx_stack.back().location()->ToString() << std::endl
        << oss.str();
    } else {
      throw GraphBreakException(oss.str());
    }
  }
}

/**
 * FindLoopEnd, FindLoopBegin, reset break bci
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
  for (int i = break_bci; i < SizeToInt(instr.size()); ++i) {
    MS_EXCEPTION_IF_CHECK_FAIL(i == instr[i]->bci(), "!!!");
    if (instr[i]->extra_jump() != nullptr) {
      res = std::min(instr[i]->extra_jump()->bci(), res);
    }
  }
  return res != break_bci;
}

/**
 * Should Never Compile
 */
bool Graph::ShouldNeverCompile() const {
  if (this->IsBreakAtLoop() && !this->RestoreLoopStatus()) {
    MS_LOG(DEBUG) << "Break at loop, should never compile";
    return true;
  }

  return found_inner_class;
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

bool GuardBuilder::PrepareTraceParam(ValueNode *node, TraceVector *tv, int depth, bool *has_unsupported) {
  const std::vector<ValueNode *> &inputs = node->inputs();
  for (auto it : inputs) {
    if (it->GetTrace() != nullptr) {
      tv->push_back(it->GetTrace());
      continue;
    }
    auto t = GetTrace(it, depth + 1);
    if (t == nullptr) {
      return false;
    } else if (t->GetTraceType() == TraceType::Unsupported) {
      *has_unsupported = true;
    }
    tv->push_back(t);
  }
  return !CheckObjPtr(node);
}

bool GuardBuilder::CheckDepthForTrace(TracePtr *ret, ValueNode *node, int depth) {
  int max_depth = max_depth_;
  if (depth > max_depth && max_depth != -1) {
    MS_LOG(DEBUG) << "too deep trace for guard";
    return false;
  }
  auto ct = node->GetTrace();
  if (ct != nullptr) {
    if (ct->GetDepth() + depth > max_depth && max_depth != -1) {
      MS_LOG(DEBUG) << "too deep trace for guard";
      return false;
    } else {
      *ret = ct;
      return false;
    }
  }
  return true;
}

TracePtr GuardBuilder::CacheTrace(ValueNode *node, TracePtr ret, const TraceVector &tv) {
  auto vobj = node->GetVobj() ? node->GetVobj()->GetBaseVersion() : nullptr;
  PyObject *obj = vobj ? vobj->GetPyObject().ptr() : nullptr;
  int opcode = node->GetOpcode();
  int oparg = node->GetOparg();
  if (ret == nullptr && !strict_) {
    return std::make_shared<UnsupportedTrace>(obj, tv, opcode, oparg);
  }
  if (ret == nullptr) {
    return nullptr;
  }
  node->SetTrace(ret);
  return ret;
}

TracePtr GetTrace(ValueNode *node, bool strict, bool print, int depth, int max_depth) {
  Graph *g = node->GetGraph();
  return g == nullptr ? GuardBuilder(strict, max_depth, print).GetTrace(node) : g->TraceValueNode(node);
}

static void HandleCallArgs(TraceVector *tv, const py::object &kw) {
#if IS_PYTHON_3_11_PLUS
  if (kw.ptr() != nullptr) {
    tv->push_back(CreateOpTrace(kw.ptr(), LOAD_CONST, 0, {}));
  }
#endif
}

TracePtr GuardBuilder::GetTrace(ValueNode *node, int depth) {
  bool strict = strict_;
  bool print = print_;
  TracePtr ret = nullptr;
  if (!CheckDepthForTrace(&ret, node, depth)) {
    return ret;
  }
  if (node->GetType() == AbstractNode::Type::Call) {
    Graph *sub_graph = static_cast<CallNode *>(node)->GetSubGraph();
    if (sub_graph && sub_graph->GetRetVal() != nullptr) {
      return GetTrace(sub_graph->GetRetVal(), depth);
    }
  }

  auto vobj = node->GetVobj() ? node->GetVobj()->GetBaseVersion() : nullptr;
  PyObject *obj = vobj ? vobj->GetPyObject().ptr() : nullptr;
  int opcode = node->GetOpcode();
  int oparg = node->GetOparg();
  const std::string &name = node->GetName();
  const char *module_name = node->GetGraph() ? node->GetGraph()->GetModuleName() : "";

  TraceVector tv;
  bool has_unsupported = false;
  if (!PrepareTraceParam(node, &tv, depth, &has_unsupported)) {
    return strict ? nullptr : std::make_shared<UnsupportedTrace>(nullptr, tv, opcode, oparg);
  }
  switch (node->GetType()) {
    case AbstractNode::Type::CellVar: /* fall-through */
    case AbstractNode::Type::FreeVar: /* fall-through */
    case AbstractNode::Type::Value:
      if (!has_unsupported) {
        ret = CreateOpTrace(obj, opcode, oparg, tv, module_name, name, strict, print);
      }
      break;
    case AbstractNode::Type::Call:
      if (!has_unsupported) {
        const std::string &func_name = node->input(0)->GetName();
        HandleCallArgs(&tv, static_cast<CallNode *>(node)->kw_names());
        ret = CreateOpTrace(obj, opcode, oparg, tv, module_name, func_name, strict, print);
      }
      break;
    case AbstractNode::Type::Param:
      ret = oparg == -1 ? nullptr : std::make_shared<RootTrace>(obj, mindspore::pijit::TraceType::Param, oparg, name);
      break;
    default:
      break;
  }
  return CacheTrace(node, ret, tv);
}

static bool IsVariableNode(PyObject *obj) {
  if (obj == nullptr) {
    return false;
  }
  return IsTensorPyObject(obj) || CheckScalar(obj);
}

static bool IsAttrClosure(ValueNode *node) {
  while (node != nullptr) {
    switch (node->GetType()) {
      case AbstractNode::Type::Call:
      case AbstractNode::Type::Value: {
        int opcode = node->GetOpcode();
        if (opcode != LOAD_ATTR) {
          return false;
        } else {
          auto inputs = node->inputs();
          if (inputs.size() == 0) {
            return false;
          }
          node = inputs[0];
        }
      } break;
      case AbstractNode::Type::Param:
      case AbstractNode::Type::CellVar:
      case AbstractNode::Type::FreeVar:
      case AbstractNode::Type::kUnbound:
        return true;
      default:
        return false;
    }
  }
  return false;
}

TraceVector GuardBuilder::GetTraceClosure(ValueNode *node, bool *succ, int depth) {
  std::vector<ValueNode *> todo;
  std::map<ValueNode *, bool> done;
  todo.push_back(node);
  std::vector<TracePtr> ret;
  while (todo.size() > 0) {
    node = todo[0];
    done[node] = true;
    todo.erase(todo.begin());
    switch (node->GetType()) {
      case AbstractNode::Type::Call:
      case AbstractNode::Type::Value: {
        PyObject *obj = node->GetVobj() ? node->GetVobj()->GetPyObject().ptr() : nullptr;
        if (IsVariableNode(obj)) {
          if (IsAttrClosure(node)) {
            auto item = GetTrace(node, depth);
            if (item != nullptr) {
              ret.insert(ret.end(), item);
            } else {
              *succ = false;
              MS_LOG(DEBUG) << "too deep trace for guard";
              return {};
            }
          } else {
            auto inputs = node->inputs();
            for (auto input : inputs) {
              if (done.find(input) != done.end()) {
                continue;
              } else {
                todo.push_back(input);
              }
            }
          }
        }
      } break;
      case AbstractNode::Type::Param:
      case AbstractNode::Type::CellVar:
      case AbstractNode::Type::FreeVar:
      case AbstractNode::Type::kUnbound:
      default:
        break;
    }
  }
  return ret;
}

bool Graph::GuardValueNode(ValueNode *node, GuardLevel level) {
  if (node->IsConstantValue()) {
    return true;
  }
  TracePtr tr = this->TraceValueNode(node);
  if (tr == nullptr) {
    bool retg = this->GuardValueNodeClosure(node);
    if (retg) {
      if (level == GuardLevel::GEqual || level == GuardLevel::GId) {
        node->SetConstantValue(retg);
      }
    }
    return retg;
  }
  bool ret = GetGuardManager()->GetGuard()->GuardOn(tr, level);
  if (level == GuardLevel::GEqual || level == GuardLevel::GId) {
    node->SetConstantValue(ret);
  }
  return ret;
}

void Graph::GuardParameter(ValueNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->abstract_wrapper() == nullptr) {
    GuardType(node);
    return;
  }
  if (IsInterpretedObject(node->abstract_wrapper())) {
    MS_LOG(DEBUG) << "Got an InterpretedObject as graph parameter, guard its type: " << node->ToString();
    GuardType(node);
    return;
  }
  if (node->abstract_wrapper()->IsConstant()) {
    MS_LOG(WARNING) << "got a constant value as graph parameter. recompile if value changed: " << node->ToString();
    GuardValueNode(node, GuardLevel::GEqual);
    return;
  }
  GuardValueNode(node, GuardLevel::GDeduce);
}

void Graph::GuardGlobal(ValueNode *node) {
  if (node->GetVobj()->GetType() == AObject::kTypeTensor) {
    MS_LOG(WARNING) << "use global tensor in jit, treat as Parameter of graph";
    // not implement ...
  }
  // How to fast check Module attribute(or global varialbe) changed ?
}

bool Graph::GuardValueNodeClosure(ValueNode *node, GuardLevel level) {
  bool bSucc = true;
  auto vec = this->TraceValueNodeClosure(node, &bSucc);
  if (bSucc) {
    std::map<size_t, TracePtr> rep;
    for (auto item : vec) {
      auto id = item->Info().Id();
      if (rep.find(id) == rep.end()) {
        GetGuardManager()->GetGuard()->GuardOn(item, level);
      }
      rep[id] = item;
    }
    MS_LOG(DEBUG) << "Finish guard closure. " << pijit::ToString(node);
    return true;
  } else {
    MS_LOG(INFO) << "Trace failed, cannot add guard. " << pijit::ToString(node);
    return false;
  }
}

TracePtr Graph::TraceValueNode(ValueNode *node, int max_trace_depth) {
  AObject *vo = node->GetVobj();
  if (GetGuardManager() == nullptr || !vo || vo->GetPyObject().ptr() == nullptr) {
    MS_LOG(DEBUG) << "Cannot get trace for node: " << node->ToString();
    return nullptr;
  }
  return guard_builder_->GetTrace(node);
}

std::vector<TracePtr> Graph::TraceValueNodeClosure(ValueNode *node, bool *ret, int) {
  AObject *vo = node->GetVobj();
  if (GetGuardManager() == nullptr || !vo || vo->GetPyObject().ptr() == nullptr) {
    MS_LOG(DEBUG) << "Cannot get trace for node: " << node->ToString();
    return {};
  }
  bool succ = true;
  auto list = guard_builder_->GetTraceClosure(node, &succ);
  if (ret != nullptr) {
    *ret = succ;
  }
  return list;
}

std::vector<ValueNode *> Graph::CollectAliveNode(int bci, std::vector<int> *ids) const {
  std::vector<ValueNode *> result;
  if (bci == -1) {
    result = {this->GetRetVal()};
  } else {
    BitMap alive = this->GetCFG()->GetLiveness()->CollectAlive(bci);
    result = CollectAliveNode(this->GetFrame(bci), &alive, ids);
  }
  if (side_effect_->IsEmpty()) {
    return result;
  }
  // alive locals must be original node
  for (auto &node : result) {
    auto new_node = this->GetSideEffect()->GetSource(node);
    if (new_node->GetScope() == AbstractObjectBase::SCOPE_LOCAL ||
        new_node->GetScope() == AbstractObjectBase::SCOPE_NOT_SPECIFIED) {
      continue;
    }
    if (new_node->GetOpcode() == LOAD_ATTR) {  // transform the alive attribute source
      auto &attr_source = new_node->inputs()[0];
      attr_source = this->GetSideEffect()->GetSource(attr_source);
    }
    node = new_node;
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
  MS_EXCEPTION_IF_NULL(sequence_node);
  MS_LOG(DEBUG) << "Try to guard sequence length: " << sequence_size << ", node: " << sequence_node->ToString();
  if (sequence_node->IsConstantValue()) {
    MS_LOG(DEBUG) << "Sequence node is const value, skip guard. " << sequence_node->ToString();
    return true;
  }
  const auto &cnst = sequence_node->GetConstantInfo();
  if (cnst != nullptr && cnst->len() != -1) {
    MS_EXCEPTION_IF_CHECK_FAIL(sequence_size == cnst->len(), "error sequence length");
    MS_LOG(DEBUG) << "Sequence length is const, skip guard. " << sequence_node->ToString();
    return true;
  }
  TracePtr tr = this->TraceValueNode(sequence_node);
  if (tr == nullptr) {
    if (this->GuardValueNodeClosure(sequence_node)) {
      sequence_node->MakeConstantInfo()->set_len(sequence_size);
      return true;
    }
    return false;
  }
  const auto &guard = this->GetGuardManager()->GetGuard();
  bool strict = this->Config().GetBoolConfig(GraphJitConfig::kStrictTrace);

  PyObject *builtin_len = PyDict_GetItemString(PyEval_GetBuiltins(), "len");
  MS_EXCEPTION_IF_NULL(builtin_len);
  TracePtr len_func = CreateOpTrace(builtin_len, LOAD_CONST, -1, {}, "", "", strict);
  TracePtr len_trace = CreateOpTrace(py::int_(sequence_size).ptr(), IS_PYTHON_3_11_PLUS ? CALL : CALL_FUNCTION, 1,
                                     {len_func, tr}, "", "len", strict);
  guard->GuardOn(len_trace, GuardLevel::GEqual, true);

  sequence_node->MakeConstantInfo()->set_len(sequence_size);
  MS_LOG(DEBUG) << "Finish guard sequence length: " << sequence_node->ToString();
  return true;
}

bool Graph::GuardType(ValueNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(DEBUG) << "Try to guard type: " << node->ToString();
  if (node->IsConstantValue()) {
    MS_LOG(DEBUG) << "Node is const value, skip guard. " << node->ToString();
    return true;
  }
  const auto &cnst = node->GetConstantInfo();
  if (cnst != nullptr && cnst->type() != nullptr) {
    MS_LOG(DEBUG) << "Node type is const, skip guard. " << node->ToString();
    return true;
  }
  TracePtr tr = this->TraceValueNode(node);
  if (tr == nullptr) {
    if (this->GuardValueNodeClosure(node)) {
      node->MakeConstantInfo()->set_type(node->GetVobj()->GetTypeObject());
      return true;
    }
    return false;
  }
  bool ret = GetGuardManager()->GetGuard()->GuardOn(tr, mindspore::pijit::GuardLevel::GType);
  node->MakeConstantInfo()->set_type(node->GetVobj()->GetTypeObject());
  MS_LOG(DEBUG) << "Finish guard type, result: " << (ret ? "ok" : "failed") << ", node: " << node->ToString();
  return ret;
}

static bool SkipGuardInlinedFunc(CallNode *node) {
  ValueNode *func_node = node->input(0);
  if (func_node->IsConstantValue()) {
    MS_LOG(DEBUG) << "Func node is const value, skip guard. " << func_node->ToString();
    return true;
  }
  // Now only guard specialized function and cell
  AObject::Type func_type = func_node->GetVobj()->GetType();
  if (func_type == AObject::kTypeCell) {
    return false;
  }
  int op = func_node->GetOpcode();
  if (op == LOAD_ATTR || op == MAKE_FUNCTION) {
    MS_LOG(INFO) << "skip guard function from attribute";
    return true;
  }
  return false;
}

bool Graph::GuardInlinedFunc(CallNode *call_node) {
  MS_LOG(INFO) << "guard inlined function (sub-graph) " << call_node->input(0)->ToString();
  if (SkipGuardInlinedFunc(call_node)) {
    MS_LOG(INFO) << "skip guard function (sub-graph) " << call_node->input(0)->ToString();
    return true;
  }
  TracePtr tr = this->TraceValueNode(call_node->input(0));
  if (tr == nullptr) {
    // unknown source function. Maybe dynamic call-site, do nothing
    MS_LOG(DEBUG) << "Unknown source function, cannot add guard. " << call_node->ToString();
    return false;
  }
  const auto &guard = this->GetGuardManager()->GetGuard();
  bool strict = this->Config().GetBoolConfig(GraphJitConfig::kStrictTrace);

  AObject *callable_info = call_node->input(0)->GetVobj();
  AObject::Type func_type = callable_info->GetType();
  PyObject *callable = callable_info->GetPyObject().ptr();
  if (func_type == AObject::kTypeBoundMethod) {
    PyObject *func = PyMethod_GET_FUNCTION(callable);
    tr = CreateOpTrace(func, LOAD_ATTR, 0, {tr}, "", "__func__", strict);
    guard->GuardOn(tr, GuardLevel::GId);
  } else if (func_type == AObject::kTypeCell) {
    MS_LOG(DEBUG) << "guard inlined cell " << py::str(callable_info->GetPyObject().ptr());
    guard->GuardOn(tr, GuardLevel::GDeduce, false);
  } else if (func_type == AObject::kTypeAnyValue) {
    guard->GuardOn(tr, GuardLevel::GType, false);
    call_node->input(0)->MakeConstantInfo()->set_type(callable_info->GetTypeObject());
  } else if (func_type == AObject::kTypeFunction) {
    guard->GuardOn(tr, GuardLevel::GId);
    call_node->input(0)->SetConstantValue(true);
  } else {
    MS_LOG(DEBUG) << "Unknown callable type, cannot add guard. " << callable_info->ToString();
    return false;
  }
  return true;
}

static void PrintAnfNode(std::ostream *out, mindspore::AnfNode *anf_node, const std::string &prefix) {
  auto &s = *out;
  std::string str;
  str += anf_node->DebugString(AnfNode::DebugStringLevel::kLevel1) + " ";
  str += anf_node->abstract() == nullptr ? "<NULL>" : anf_node->abstract()->ToString();
  std::replace(str.begin(), str.end(), '\n', ' ');
  s << " AnfNode(" << anf_node << ") [" << str << "]";
}

static void PrintValueNode(std::ostream *out, FuncGraphBuilder *fg, ValueNode *node, const std::string &prefix) {
  auto &s = *out;
  s << prefix << node->ToString();
  if (fg == nullptr) {
    s << "<FuncGraphBuilder is nullptr>" << std::endl;
    return;
  }
  auto anf_node = fg->FindNodeByWrapper(node->abstract_wrapper());
  if (anf_node != nullptr) {
    bool is_any = anf_node->abstract() != nullptr &&
                  (*mindspore::kValueAny == *anf_node->abstract()->mindspore::abstract::AbstractBase::BuildValue());
    bool is_cnst = node->IsConstantValue();
    PrintAnfNode(out, anf_node.get(), prefix);
    s << " is_constant " << (is_cnst == is_any ? "==" : "!=") << " is_any";
  } else if (node->IsConstantValue()) {
    s << " Is Constant";
  } else {
    s << " Not find AnfNode";
  }
  s << std::endl;
}

static void PrintCallNode(std::ostream *os, FuncGraphBuilder *fg, CallNode *node, int depth) {
  std::string prefix(depth * kTwo, ' ');
  auto &s = *os;
  s << prefix << "{ inline stat " << GetInlineReasonDesc(node->GetInlineReason());
  if (!node->GetParams().empty()) {
    s << ", has extra operations to handle parameters:" << std::endl;
    for (const auto &ex : node->GetParams()) {
      s << prefix << "  " << ex->ToString() << std::endl;
      PrintValueNode(&s, fg, ex, prefix);
    }
  }
  s << std::endl;
  if (node->GetSubGraph() != nullptr) {
    s << node->GetSubGraph()->ToString(depth + 1);
  }
  s << prefix << "}" << std::endl;
}

void Graph::PrintFrame(std::ostream *out, const std::string &prefix) const {
  auto &s = *out;
  const auto &f = GetFrame(0);
  auto *fg = func_graph_builder_.get();
  s << "locals:" << std::endl;
  for (const auto &node : f.GetLocals()) {
    if (node != &ValueNode::kUnboundLocal) {
      PrintValueNode(&s, fg, node, prefix);
    }
  }
  s << "stacks:" << std::endl;
  for (auto iter = f.GetStacks().rbegin(), end = f.GetStacks().rend(); iter != end; ++iter) {
    PrintValueNode(&s, fg, *iter, prefix);
  }
  s << "cell free:" << std::endl;
  for (const auto &node : f.GetClosures()) {
    PrintValueNode(&s, fg, node, prefix);
  }
}

std::string Graph::ToString(int depth) const {
  std::stringstream s;
  std::string prefix(depth << 1, ' ');
  std::string code_name = co_.ptr() != nullptr ? std::string(py::str(co_.ptr())) : "<no code>";

  s << prefix << "*** Trace Nodes [" << code_name << "] ***" << std::endl;
  if (depth == 0) {
    PrintFrame(&s, prefix);
    s << std::endl;
  }

  s << prefix << "Graph " << this << " Nodes:" << std::endl;
  for (auto i : GetTracedNodes()) {
    PrintValueNode(&s, func_graph_builder().get(), i, prefix);
    if (i->GetType() == AbstractNode::Call) {
      PrintCallNode(&s, func_graph_builder().get(), static_cast<CallNode *>(i), depth);
    }
  }
  if (GetRetVal()) {
    s << prefix << "Return the Node " << GetRetVal() << std::endl;
  }
  s << prefix << GetStopTraceReasonDesc(GetStopTraceReason()) << std::endl;
  s << prefix << "break bci: " << GetStopTraceBci();
  if (GetStopTraceBci() != -1) {
    const auto &instr = cfg_->instr_pool()[GetStopTraceBci()];
    s << " " << instr->ToString() << " line " << instr->line() << " at " << PyUnicode_AsUTF8(GetCodeObj()->co_filename);
  }
  s << std::endl << std::endl;

  if (depth == 0) {
    this->DumpBreakInfo(&s);
  }
  return s.str();
}

static void TraceInferFailed(std::ostream *out, ValueNode *node, int depth = 0) {
  auto &s = *out;
  std::string prefix(IntToSize(depth) << 1, ' ');
  s << prefix << node << " ";
  switch (node->GetType()) {
    case AbstractNode::Call:
    case AbstractNode::Value:
      s << "ValueNode " << node;
      break;
    case AbstractNode::Param:
      s << "Parameter (" << node->GetOparg() << ")" << node;
      break;
    case AbstractNode::CellVar:
    case AbstractNode::FreeVar:
      s << "Closure (" << node->GetOparg() << ")" << node;
      break;
    case AbstractNode::kUnbound:
      s << "(kUnboundLocal)";
      break;
    default:
      break;
  }
  s << " object is ";
  PyObject *op = node->GetVobj() ? node->GetVobj()->GetPyObject().ptr() : nullptr;
  if (op != nullptr) {
    s << AObject::ToString(op) << std::endl;
    return;
  }
  s << "<NULL>:" << std::endl;
  for (size_t i = 0; i < node->inputs().size(); ++i) {
    TraceInferFailed(out, node->input(i), depth + 1);
  }
}

void DumpUnsupportedByteCodeInfo(std::ostream &s, Opcode op, int arg) {
  if (op == SETUP_WITH || op == SETUP_FINALLY) {
    s << op.name() << " " << arg << " is skipped in break_graph or a exception happened.\n";
  } else {
    s << op.name() << " " << arg << " is not support.\n";
  }
}

void Graph::DumpBreakInfo(std::ostream *out) const {
  auto &s = *out;
  if (GetRetVal()) {
    TraceInferFailed(out, GetRetVal());
  }
  if (GetStopTraceBci() == -1) {
    return;
  }
  const auto &f = GetFrame(GetStopTraceBci());
  const auto &nodes = GetTracedNodes();
  const auto &instrs = cfg_->instr_pool();
  int break_bci = GetStopTraceBci();

  s << "graph break at: " << break_bci << " " << instrs[break_bci]->ToString() << std::endl;
  std::vector<ValueNode *> parameters;
  if (nodes.size() == 0 || nodes.back()->bci() < break_bci) {
    // break at unsupported bytecode
    Opcode op(instrs[break_bci]->op());
    int arg = instrs[break_bci]->arg();
    DumpUnsupportedByteCodeInfo(s, op, arg);
    if (op.IsConditionJump() || op == FOR_ITER || op == UNPACK_SEQUENCE || op == UNPACK_EX) {
      arg = 0;
    } else if (op == CALL_FUNCTION_EX) {
      arg = (arg & 0x01) + 1;
    } else if (op == CALL_FUNCTION_KW) {
      arg++;
    } else {
      return;
    }
    for (int i = arg; i >= 0; --i) {
      parameters.push_back(f.Peek(i));
    }
  } else {
    auto iter = std::find_if(nodes.begin(), nodes.end(), [break_bci](ValueNode *n) { return n->bci() == break_bci; });
    if (iter != nodes.end()) {
      parameters.push_back(*iter);
    }
  }
  // print traced value
  for (auto node : parameters) {
    TraceInferFailed(out, node);
  }
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

void GraphBreakException::set_error() const {
  py::object exception_type = py::module::import("mindspore.graph._utils.pijit_context").attr("Unsupported");
  if (exception_type.ptr() == nullptr || !PyType_Check(exception_type.ptr())) {
    MS_LOG(WARNING) << "Cannot import 'Unsupported' from 'mindspore.graph._utils.pijit_context', "
                    << "use RuntimeError instead";
    PyErr_SetString(PyExc_RuntimeError, what());
  } else {
    PyErr_SetString(exception_type.ptr(), what());
  }
}

std::string GetFileName(const Graph *graph) {
  if (graph == nullptr || graph->GetCodeObj() == nullptr) {
    return "";
  }
  return PyUnicode_AsUTF8(graph->GetCodeObj()->co_filename);
}

std::string GetNameAndLocation(const Graph *graph) {
  if (graph == nullptr || graph->GetCodeObj() == nullptr) {
    return "";
  }
  std::ostringstream ss;
  PyCodeWrapper co(graph->GetCodeObj());
  ss << "'" << graph->GetCodeName() << "' " << graph << " at \"" << co.FileName() << ":" << co.FirstLine() << "\"";
  return ss.str();
}

// If the graph is break at calling subgraph, then return the CallNode at break bci, or else return nullptr.
CallNode *FindBreakAtCall(const Graph *graph) {
  int break_bci = graph->GetStopTraceBci();
  if (break_bci == -1) {
    return nullptr;
  }
  const std::vector<ValueNode *> &traced_nodes = graph->GetTracedNodes();
  auto it = std::find_if(traced_nodes.rbegin(), traced_nodes.rend(), [break_bci](ValueNode *node) {
    return node->bci() == break_bci && node->GetType() == AbstractNode::Call &&
           (static_cast<CallNode *>(node))->GetSubGraph() != nullptr;
  });
  return it != traced_nodes.rend() ? static_cast<CallNode *>(*it) : nullptr;
}

}  // namespace pijit
}  // namespace mindspore
