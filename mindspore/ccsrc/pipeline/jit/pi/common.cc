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
#include "pipeline/jit/pi/common.h"
#include <algorithm>
#include <iomanip>
#include <list>
#include <map>
#include <string>
#include <vector>
#include "pipeline/jit/pi/external.h"
#include "pipeline/jit/pi/graph_capture/code_gen.h"
#include "pipeline/jit/pi/graph_capture/graph_build.h"
#include "pipeline/jit/pi/graph_capture/graph_analyzer.h"
#include "pipeline/jit/pi/graph_compiler/compiler.h"
#include "pipeline/jit/pi/graph_compiler/cg/byte_code_generator.h"
#include "pipeline/jit/pi/graph_compiler/inliner/func_inliner.h"
#include "pipeline/jit/pi/graph_compiler/parser/byte_code_parser.h"
#include "pipeline/jit/pi/graph_compiler/utils.h"
#include "pipeline/jit/pi/utils/utils.h"
#include "pipeline/jit/pi/graph_guard/strategy.h"
#include "pipeline/jit/ps/pipeline.h"
#include "pipeline/pynative/pynative_utils.h"
#include "runtime/pynative/op_executor.h"

namespace mindspore {
namespace jit {
namespace graph {
static Py_tss_t *tss = NULL;

void AddConfigToGuard(const GraphJitConfig &c, OptGuardPtr guard);
void AddGuardForParam(const PyFrameObject *f, OptGuardPtr guard, bool detach);
static void AddGradFlagForParam(bool grad_flag, OptGuardPtr guard, bool detach);
static void CollectTraceBack(JitCompileResults *c, PyCodeObject *code, bool is_graph_mode);

std::map<TimeRecorder::RecorderType, TimeRecorder::PerfData> TimeRecorder::data_;
static std::map<uint64_t, size_t> code_size_execute_python;  // execute count, code size
static std::map<uint64_t, size_t> code_size_execute_graph;   // execute count, code size

// jit compiler initialize
static void ensureInitialize() {
  static bool init = false;
  if (init) {
    return;
  }
  init = true;
  if (tss == NULL) {
    tss = PyThread_tss_alloc();
    PyThread_tss_create(tss);
  }
  std::atexit([]() {
    if (!kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kLogPerf)) {
      return;
    }
    for (const auto &i : TimeRecorder::data_) {
      std::cout << i.first << " " << i.second.count << " times, " << (i.second.nano / TimeRecorder::scale) << " seconds"
                << std::endl;
    }
    size_t sum_code_py =
      std::accumulate(code_size_execute_python.begin(), code_size_execute_python.end(), 0,
                      [](size_t sum, const std::pair<uint64_t, size_t> &i) { return sum + (i.first * i.second); });
    size_t sum_code_graph =
      std::accumulate(code_size_execute_graph.begin(), code_size_execute_graph.end(), 0,
                      [](size_t sum, const std::pair<uint64_t, size_t> &i) { return sum + (i.first * i.second); });

    std::cout << "execute code ratio (graph / (graph + python)): "
              << (sum_code_graph / static_cast<double>(sum_code_graph + sum_code_py)) << std::endl;
  });
}

void Tracebackes::PushInlineInfo(InlineInfo info) {
  const auto &it = inline_infos_.find(info.root_name_);
  if (it != inline_infos_.cend()) {
    it->second.push_back(info);
  } else {
    std::list<InlineInfo> inlines;
    inlines.push_back(info);
    inline_infos_.emplace(info.root_name_, inlines);
  }
}

static void PrintLabel(std::stringstream &os, const std::string &str, int distance = 30) {
  os << std::left << std::setw(distance) << str << ": ";
}

std::string Tracebackes::Dump(bool is_all) const {
  std::stringstream os;
  std::string cur_name = tbs_.empty() ? "" : tbs_.back().func_name_;
  if (is_all) {
    os << "*** Dump Traceback on [" << raw_func_info_name_ << "] ***\n";
  } else {
    os << "*** Dump ByteCode After Traceback on [" << cur_name << "] ***\n";
  }
  if (tbs_.empty()) {
    return os.str();
  }
  std::list<Tracebacke> candidates;
  if (is_all) {
    candidates = tbs_;
  } else {
    // last one traceback
    candidates.emplace_back(tbs_.back());
  }
  // dump traceback list head
  int name_length = FindMaxNameLength(candidates);
  os << std::left << std::setw(name_length) << "func_name:"
     << "  -->  " << std::left << std::setw(name_length) << "changed_func:" << std::left << std::setw(10)
     << "run_mode:" << std::left << std::setw(30) << "stop_trace:" << std::left << std::setw(10)
     << "code_size:" << std::endl;
  os << "--------------------------------------------------------------------------------------\n";
  // dump traceback list content
  for (const auto &tb : candidates) {
    os << std::left << std::setw(name_length) << tb.func_name_ << "  -->  ";
    os << std::left << std::setw(name_length) << tb.changed_func_;
    if (tb.is_graph_mode_) {
      os << std::left << std::setw(10) << "[GRAPH]";
    } else {
      os << std::left << std::setw(10) << "PYNATIVE";
    }
    // dump stop trace reason
    auto it_trace = stop_trace_res_.find(tb.func_name_);
    if (it_trace != stop_trace_res_.cend()) {
      os << std::left << std::setw(30) << GetStopTraceReasonDesc(it_trace->second);
    } else {
      os << std::left << std::setw(30) << "unknown";
    }
    os << std::left << std::setw(10) << tb.code_size_ << " =====>\n";
    // dump inline info
    DumpInlineInfo(os, tb.func_name_);
  }
  os << "\n\n";
  if (is_all) {
    os << DumpSummary();
  }
  return os.str();
}

void Tracebackes::DumpInlineInfo(std::stringstream &os, const std::string &func_name) const {
  const auto &it = inline_infos_.find(func_name);
  if (it == inline_infos_.cend()) {
    return;
  }
  for (const auto &info : it->second) {
    std::string space((info.depth + 1) * 2, ' ');
    os << space << "| inline_info:" << GetInlineReasonDesc(info.res) << " line:" << info.line;
    if (!info.inline_name_.empty()) {
      os << " func_name:" << info.inline_name_;
    }
    if (info.res == InlineReason::kInline || info.res == InlineReason::kInlinePartial) {
      os << " code_size:" << info.code_size_;
    }
    os << "\n";
  }
}

std::string Tracebackes::DumpSummary() const {
  std::stringstream os;
  if (tbs_.empty()) {
    return os.str();
  }
  os << "*** Dump Summary on [" << raw_func_info_name_ << "] ***\n";
  PrintLabel(os, "traceback_num");
  os << tbs_.size() << "\n";

  std::array<int, kStopTrace_Reason_Count> stop_trace_reason_array{0};
  std::array<int, kInline_Reason_Count> inline_reason_array{0};
  int graph_mode_num = 0;
  int raw_code_size = raw_code_size_;
  int pynative_code_size = 0;
  int graph_mode_code_size = 0;
  for (const auto &tb : tbs_) {
    if (tb.is_graph_mode_) {
      graph_mode_num++;
      graph_mode_code_size += tb.code_size_;
    } else {
      pynative_code_size += tb.code_size_;
    }
    auto it_trace = stop_trace_res_.find(tb.func_name_);
    if (it_trace != stop_trace_res_.cend()) {
      // count stop trace reason
      stop_trace_reason_array[it_trace->second]++;
    }
    const auto &it_inline = inline_infos_.find(tb.func_name_);
    if (it_inline == inline_infos_.cend()) {
      continue;
    }
    for (const auto &info : it_inline->second) {
      // count inline reason
      inline_reason_array[info.res]++;
      if (info.res == InlineReason::kInline || info.res == InlineReason::kInlinePartial) {
        raw_code_size += info.code_size_;
      }
    }
  }
  PrintLabel(os, "graph_mode_num");
  os << graph_mode_num << "\n";
  PrintLabel(os, "raw_code_size(+ inline)");
  os << raw_code_size << "\n";
  PrintLabel(os, "pynative_code_size");
  os << pynative_code_size << "\n";
  PrintLabel(os, "graph_mode_code_size");
  os << graph_mode_code_size << "\n";
  os << "----------stop_trace_reason----------\n";
  for (size_t i = 0; i < stop_trace_reason_array.size(); ++i) {
    PrintLabel(os, GetStopTraceReasonDesc(static_cast<StopTraceReason>(i)));
    os << stop_trace_reason_array[i] << "\n";
  }
  os << "----------inline_reason----------\n";
  for (size_t i = 0; i < inline_reason_array.size(); ++i) {
    PrintLabel(os, GetInlineReasonDesc(static_cast<InlineReason>(i)));
    os << inline_reason_array[i] << "\n";
  }
  os << "\n\n";
  return os.str();
}

int Tracebackes::FindMaxNameLength(const std::list<Tracebacke> &tbs) const {
  int max_length = 15;
  for (const auto &tb : tbs) {
    int len1 = tb.func_name_.length();
    int len2 = tb.changed_func_.length();
    max_length = std::max(max_length, std::max(len1, len2)) + 2;
  }
  max_length = std::min(max_length, 35);
  return max_length;
}

static void freeJitCompileResults(void *jitCompileResults) {
  // maybe nullptr if other module use _PyEval_RequestCodeExtraIndex
  if (jitCompileResults == nullptr) {
    return;
  }
  // called after code object freed
  JitCompileResults *c = reinterpret_cast<JitCompileResults *>(jitCompileResults);

  for (auto &oc : c->codehub->GetOptTarget(OptOption::CreateOptionByPoint(c))) {
    PyCodeObject *co = oc->GetPythonCode();
    MS_EXCEPTION_IF_CHECK_FAIL(co == nullptr || Py_REFCNT(co) == 1, "code handler must be only one");
  }
  c->code = nullptr;
  c->codehub.reset();

  std::for_each(c->children_.begin(), c->children_.end(), [](CodeExtra *i) { i->parent_ = nullptr; });
  if (c->parent_ != nullptr) {
    auto &leaf = c->parent_->children_;
    leaf.erase(std::remove_if(leaf.begin(), leaf.end(), [c](CodeExtra *i) { return i == c; }), leaf.end());
  }
  MS_LOG(DEBUG) << __FUNCTION__ << " " << c;
  delete c;
}

static JitCompileResults *allocJitCompileResults() {
  JitCompileResults *c = new JitCompileResults();
  c->parent_ = nullptr;
  c->stat = JitCompileResults::NEVER_COMPILE;
  c->tbs = std::make_shared<Tracebackes>();
  c->codehub = std::make_shared<OptCodeHub>();
  c->conf = std::make_shared<GraphJitConfig>();
  return c;
}

JitCompileResults *getJitCompileResults(PyObject *code, bool alloc) {
  if (PyMethod_Check(code)) {
    code = PyMethod_GET_FUNCTION(code);
  }
  if (PyFunction_Check(code)) {
    code = PyFunction_GET_CODE(code);
  }
  if (!PyCode_Check(code)) {
    return NULL;
  }
  ensureInitialize();
  Py_ssize_t index = (Py_ssize_t)PyThread_tss_get(tss);
  if (index == 0) {
    index = _PyEval_RequestCodeExtraIndex(freeJitCompileResults);
    if (index == -1) {
      return NULL;
    }
    // ensure index is not 0
    PyThread_tss_set(tss, reinterpret_cast<void *>(index + 1));
  } else {
    index = index - 1;
  }

  JitCompileResults *c = NULL;
  if (!_PyCode_GetExtra(code, index, reinterpret_cast<void **>(&c))) {
    if (c != NULL) {
      return c;
    }
    if (!alloc) {
      return NULL;
    }
    c = allocJitCompileResults();
    if (c == NULL) {
      return NULL;
    }
    if (!_PyCode_SetExtra(code, index, c)) {
      MS_LOG(DEBUG) << "allocJitCompileResults " << c << " for " << std::string(py::str(code));
      return c;
    }
    freeJitCompileResults(c);
  }
  PyErr_Clear();
  return NULL;
}

static PyFrameObject *RebuildFrame(PyThreadState *tstate, PyCodeObject *co, const PyFrameObject *f) {
  int argc = f->f_code->co_argcount + f->f_code->co_kwonlyargcount;
  MS_ASSERT(co != nullptr && argc == co->co_argcount + co->co_kwonlyargcount);
  MS_ASSERT((f->f_code->co_flags & CO_VARARGS) == (co->co_flags & CO_VARARGS));
  MS_ASSERT((f->f_code->co_flags & CO_VARKEYWORDS) == (co->co_flags & CO_VARKEYWORDS));
  argc += (f->f_code->co_flags & CO_VARARGS) ? 1 : 0;
  argc += (f->f_code->co_flags & CO_VARKEYWORDS) ? 1 : 0;

  PyFrameObject *frame = PyFrame_New(tstate, co, f->f_globals, NULL);
  // copy arguments
  for (int i = 0; i < argc; i++) {
    Py_XINCREF(f->f_localsplus[i]);
    frame->f_localsplus[i] = f->f_localsplus[i];
  }
  // restore arguments from cell
  std::vector<PyObject *> cells_content(f->f_code->co_nlocals, nullptr);
  for (int i = 0; f->f_code->co_cell2arg != NULL && i < PyTuple_GET_SIZE(f->f_code->co_cellvars); ++i) {
    Py_ssize_t argi = f->f_code->co_cell2arg[i];
    if (argi != CO_CELL_NOT_AN_ARG) {
      PyObject *cell = f->f_localsplus[f->f_code->co_nlocals + i];
      cells_content[argi] = PyCell_GET(cell);
    }
  }
  // new cell
  for (int i = 0; i < PyTuple_GET_SIZE(co->co_cellvars); ++i) {
    PyObject *cell;
    if (co->co_cell2arg != NULL && co->co_cell2arg[i] != CO_CELL_NOT_AN_ARG) {
      Py_ssize_t argi = co->co_cell2arg[i];
      MS_EXCEPTION_IF_CHECK_FAIL(cells_content[argi], "Unbound local exception");
      cell = PyCell_New(cells_content[argi]);
    } else {
      cell = PyCell_New(NULL);
    }
    frame->f_localsplus[co->co_nlocals + i] = cell;
  }

  // copy closure
  for (int i = 0; i < PyTuple_GET_SIZE(co->co_freevars); ++i) {
    int a = f->f_code->co_nlocals + PyTuple_GET_SIZE(f->f_code->co_cellvars) + i;
    int b = co->co_nlocals + PyTuple_GET_SIZE(co->co_cellvars) + i;
    auto o = f->f_localsplus[a];
    Py_XINCREF(o);
    frame->f_localsplus[b] = o;
  }
  return frame;
}

static PyObject *GetClosure(const PyFrameObject *f) {
  int nfrees = PyTuple_GET_SIZE(f->f_code->co_freevars);
  if (nfrees == 0) {
    return nullptr;
  }
  PyObject *closure = PyTuple_New(nfrees);
  int idx = f->f_code->co_nlocals + PyTuple_GET_SIZE(f->f_code->co_cellvars);
  for (int i = 0; i < nfrees; ++i) {
    PyObject *o = f->f_localsplus[idx + i];
    Py_INCREF(o);
    PyTuple_SET_ITEM(closure, i, o);
  }
  return closure;
}

static PyFrameObject *PrepareCallCompiledCallable(PyThreadState *tstate, const PyFrameObject *f,
                                                  const JitCompileResults *c) {
  return RebuildFrame(tstate, c->code->GetPythonCode(), f);
}

static void GuardForFrame(PyFrameObject *frame, const OptCodePtr &oc, const GraphJitConfig &conf) {
  const char *code_name = PyUnicode_AsUTF8(frame->f_code->co_name);
  AddConfigToGuard(conf, oc->GetGuard());
  AddGuardForParam(frame, oc->GetGuard(), conf.GetBoolConfig(GraphJitConfig::kGuardDetachObject));
  AddGradFlagForParam(pynative::PyNativeExecutor::GetInstance()->grad_flag(), oc->GetGuard(),
                      conf.GetBoolConfig(GraphJitConfig::kGuardDetachObject));
  if (conf.GetBoolConfig(GraphJitConfig::kPrintGuard)) {
    GRAPH_JIT_LOG_F("Guard on %s by %s!\n", code_name, oc->GetGuard()->GetDescript().c_str());
    return;
  }
  MS_LOG(DEBUG) << "Guard on " << code_name << " by " << oc->GetGuard()->GetDescript() << "!" << std::endl;
}

static void ValidateCompiledResults(const JitCompileResults *c) {
  if (c->stat != JitCompileResults::GRAPH_CALLABLE) {
    return;
  }
  bool valid_res;
  if (c->code->GetNativeFunc()) {
    valid_res = true;
  } else {
    valid_res = c->code->GetPythonCode() != nullptr;
  }
  MS_EXCEPTION_IF_CHECK_FAIL(valid_res, "check compiled result");
}

static py::object MakeCodeFromCodeGen(CodeGenerator *cg, PyObject *default_globals) {
  PyObject *new_code = reinterpret_cast<PyObject *>(cg->MakeCodeObj());
  PyObject *used_globals = cg->GetGlobals().ptr();
  PyObject *key, *value;
  Py_ssize_t pos = 0;
  while (PyDict_Next(used_globals, &pos, &key, &value)) {
    PyObject *old = PyDict_GetItem(default_globals, key);
    MS_EXCEPTION_IF_CHECK_FAIL(old == nullptr || old == value, "duplicate global value " + std::string(py::str(key)) +
                                                                 ":" + std::string(py::str(old)) + "->" +
                                                                 std::string(py::str(value)));
    PyDict_SetItem(default_globals, key, value);
  }
  return py::reinterpret_steal<py::object>(new_code);
}

// preprocess before compile, split bytecode to sub-function
static void GraphCapture(JitCompileResults *jcr) {
  MS_EXCEPTION_IF_NULL(jcr->code);

  GraphJitConfig &conf = *jcr->conf;

  GraphBuilder g(jcr->origin_frame_);
  (void)g.BuildGraph();

  // check graph has primitive call

  GraphAnalyzer analyzer(g.GetGraph());
  analyzer.Analyze();

  // dump DFG
  if (conf.GetBoolConfig(GraphJitConfig::kPrintAfterAll)) {
    g.DumpDFG();
  }

  CodeGenerator cg(g.GetGraph(), analyzer.GetCaptureInfo());
  bool is_graph_break = cg.TryToBreakGraphIfParameterUnsupported();
  cg.CutoffBytecodesIfGraphBreak();

  py::object new_code = MakeCodeFromCodeGen(&cg, jcr->origin_frame_->f_globals);
  jcr->code->SetPythonCode(new_code);
  jcr->stat = JitCompileResults::GRAPH_CALLABLE;

  if (conf.GetBoolConfig(GraphJitConfig::kPrintAfterAll)) {
    Utils::DisFuncObject(new_code.ptr());
    GRAPH_JIT_LOG_F("\n\n");
  }

  // collect stop trace reason to traceback
  jcr->tbs->PushStopTraceRes(g.GetGraph()->GetCodeName(), g.GetGraph()->GetStopTraceReason());
  AObject::aobject_mem_pool_.Clear(__FILE__, __LINE__);

  if (is_graph_break || (cg.IsCodeChanged() && g.GetGraph()->GetStopTraceAt())) {
    // break graph interpret
  } else if (conf.GetBoolConfig(GraphJitConfig::kInterpretCapturedCode)) {
    // config interpret
  } else {
    jcr->stat = JitCompileResults::GRAPH_CAPTURED;
  }
}

static void CollectTraceBack(JitCompileResults *c, PyCodeObject *code, bool is_graph_mode) {
  if (code == nullptr) {
    code = c->origin_frame_->f_code;
  }
  std::string name = Utils::GetPyName(c->origin_frame_->f_code->co_name);
  std::string changed_name = Utils::GetPyName(code->co_name);
  int code_size = (PyBytes_GET_SIZE(code->co_code)) / sizeof(_Py_CODEUNIT);
  c->tbs->PushTbs({name, changed_name, code_size, is_graph_mode});
}

std::string GetFuncGraphPhase(const PyFrameObject &frame, const OptCodePtr &oc) {
  std::string phase = py::cast<std::string>(frame.f_code->co_filename) + "_" +
                      std::to_string(frame.f_code->co_firstlineno) + "_" + py::cast<std::string>(frame.f_code->co_name);
  if (oc != nullptr) {
    phase += oc->GetGuard()->GetDescript();
  } else {
    for (int i = 0; i < frame.f_code->co_argcount; i++) {
      PyObject *obj = PyTuple_GET_ITEM(frame.f_code->co_varnames, i);
      py::object para = py::cast<py::object>(PyDict_GetItem(frame.f_locals, obj));
      auto node = GraphUtils::ConvertPythonObjectToAnfNode(para);
      phase += "_" + node->abstract()->ToString();
    }
  }
  phase += ".pi_jit";
  return phase;
}

void AddConfigToGuard(const GraphJitConfig &c, OptGuardPtr guard) {
  std::map<std::string, bool> cfg;
  cfg[kSpecializeScalar] = c.GetBoolConfig(GraphJitConfig::kGuardSpecializeScalar);
  cfg[kSpecializeContainer] = c.GetBoolConfig(GraphJitConfig::kGuardSpecializeContainer);
  cfg[kSpecializeTensor] = c.GetBoolConfig(GraphJitConfig::kGuardSpecializeTensor);
  guard->UpdateConfig(cfg);
}

void AddGuardForParam(const PyFrameObject *f, OptGuardPtr guard, bool detach) {
  int argc = f->f_code->co_argcount + f->f_code->co_kwonlyargcount;
  PyTupleObject *vargs = NULL;
  PyDictObject *kwargs = NULL;
  if (f->f_code->co_flags & CO_VARARGS) {
    vargs = _PyTuple_CAST(f->f_localsplus[argc]);
  }
  if (f->f_code->co_flags & CO_VARKEYWORDS) {
    kwargs = reinterpret_cast<PyDictObject *>(f->f_localsplus[argc + (vargs ? 1 : 0)]);
  }
  for (int i = 0; i < argc; ++i) {
    RootTracePtr ptr = std::make_shared<RootTrace>(f->f_localsplus[i], mindspore::jit::graph::TraceType::Param, i);
    guard->GuardOn(ptr, mindspore::jit::graph::GuardLevel::GDeduce, false);
    if (detach) {
      ptr->Detach();
    }
  }
  if (vargs != NULL) {
    RootTracePtr ptr =
      std::make_shared<RootTrace>(f->f_localsplus[argc], mindspore::jit::graph::TraceType::Param, argc);
    guard->GuardOn(ptr, mindspore::jit::graph::GuardLevel::GDeduce, false);
    if (detach) {
      ptr->Detach();
    }
  }
  if (kwargs != NULL) {
    RootTracePtr ptr = std::make_shared<RootTrace>(f->f_localsplus[argc + (vargs ? 1 : 0)],
                                                   mindspore::jit::graph::TraceType::Param, argc + (vargs ? 1 : 0));
    guard->GuardOn(ptr, mindspore::jit::graph::GuardLevel::GDeduce, false);
    if (detach) {
      ptr->Detach();
    }
  }
  for (int i = 0; f->f_code->co_cell2arg && i < PyTuple_GET_SIZE(f->f_code->co_cellvars); ++i) {
    Py_ssize_t arg = f->f_code->co_cell2arg[i];
    if (arg != CO_CELL_NOT_AN_ARG) {
      RootTracePtr ptr = std::make_shared<RootTrace>(
        f->f_localsplus[f->f_code->co_nlocals + i], mindspore::jit::graph::TraceType::Param, f->f_code->co_nlocals + i);
      guard->GuardOn(ptr, mindspore::jit::graph::GuardLevel::GDeduce, false);
      if (detach) {
        ptr->Detach();
      }
    }
  }
}

static void AddGradFlagForParam(bool grad_flag, OptGuardPtr guard, bool detach) {
  CustomizedTracePtr ptr = std::make_shared<CustomizedTrace>(
    grad_flag ? Py_True : Py_False,
    [](PTraceContext context) -> PyObject * {
      auto pynative_exec = pynative::PyNativeExecutor::GetInstance();
      PyObject *ret = pynative_exec->grad_flag() ? Py_True : Py_False;
      Py_INCREF(ret);
      return ret;
    },
    [grad_flag]() -> std::string {
      return std::string("{PyNativeExecutor::GetInstance()->grad_flag == ") + std::to_string(grad_flag) +
             std::string("}(type:") + std::to_string(TraceType::Customized) + std::string(")");
    });
  guard->GuardOn(ptr, mindspore::jit::graph::GuardLevel::GEqual, true);
  if (detach) {
    ptr->Detach();
  }
}

static void CallGraphCompiler(JitCompileResults *jcr, PyFunctionObject *func, const PyFrameObject *frame) {
  std::string phase = GetFuncGraphPhase(*frame, jcr->code);
  MS_LOG(DEBUG) << "Phase is " << phase << "!";
  CallableGraph callable = mindspore::jit::graph::Compiler::Compile(*func, *frame, phase);

  ReleaseFunc rFunc = nullptr;
  if (jcr->conf->GetBoolConfig(GraphJitConfig::kAutoCleanCache)) {
    rFunc = [phase]() {
      auto graph_executor = mindspore::pipeline::GraphExecutorPy::GetInstance();
      if (graph_executor->HasCompiled(phase)) {
        py::str p(phase);
        py::set s;
        s.add(phase);
        py::object o = py::none();
        graph_executor->DelNetRes(o, s);
        MS_LOG(DEBUG) << "To release " << phase;
      }
    };
  }
  jcr->code->SetNativeFunc(phase, callable, rFunc);
  jcr->stat = JitCompileResults::GRAPH_CALLABLE;
}

static void GraphCompile(JitCompileResults *jcr, const PyFrameObject *frame) {
  // restore function object from frame
  PyObject *new_func = PyFunction_New(reinterpret_cast<PyObject *>(frame->f_code), frame->f_globals);
  Py_XSETREF(PyFunction_GET_CLOSURE(new_func), GetClosure(frame));
  PyFunctionObject *func = reinterpret_cast<PyFunctionObject *>(new_func);

  CallGraphCompiler(jcr, func, frame);

  Py_DECREF(new_func);
}

extern bool UnsupportedCodeTypeCheck(PyCodeObject *co);
static bool JitCompile(PyThreadState *tstate, JitCompileResults *c) {
  if (UnsupportedCodeTypeCheck(c->origin_frame_->f_code)) {
    return false;
  }

  const char *origin_code_name = PyUnicode_AsUTF8(c->origin_frame_->f_code->co_name);
  MS_LOG(DEBUG) << "---start compile " << origin_code_name << "---";

  // new guard code
  c->code = c->codehub->AddOptTarget(OptOption::CreateOptionByPoint(c));

  py::object frame = py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject *>(c->origin_frame_));
  if (c->stat == JitCompileResults::GRAPH_CANDIDATE) {
    TimeRecorder _time_recorder(TimeRecorder::kTimeCompileCapture,
                                kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kLogPerf));
    runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kCapture, runtime::ProfilerEvent::kCaptureProcess,
                                       "PIJitGapture");
    c->stat = JitCompileResults::GRAPH_BUILDING;
    GraphCapture(c);
    if (c->stat == JitCompileResults::GRAPH_CAPTURED) {
      PyFrameObject *f = PrepareCallCompiledCallable(tstate, c->origin_frame_, c);
      frame = py::reinterpret_steal<py::object>(reinterpret_cast<PyObject *>(f));
    }
  }

  if (c->stat == JitCompileResults::GRAPH_CAPTURED) {
    TimeRecorder _time_recorder(TimeRecorder::kTimeCompileGraph,
                                kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kLogPerf));
    runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kCapture, runtime::ProfilerEvent::kCaptureCompile,
                                       "PIJitCompile");
    c->stat = JitCompileResults::GRAPH_BUILDING;
    PyFrameObject *f = reinterpret_cast<PyFrameObject *>(frame.ptr());
    PyFrame_FastToLocals(f);
    GraphCompile(c, f);
  }

  GuardForFrame(reinterpret_cast<PyFrameObject *>(frame.ptr()), c->code, *c->conf);
  CollectTraceBack(c, c->code->GetPythonCode(), c->code->GetNativeFunc() != nullptr);

  MS_LOG(DEBUG) << "---compile " << origin_code_name << " successful---";
  if (c->conf->GetBoolConfig(GraphJitConfig::kPrintAfterAll)) {
    GRAPH_JIT_LOG_F("%s\n", c->tbs->Dump().c_str());
  }
  MS_EXCEPTION_IF_CHECK_FAIL(c->stat == JitCompileResults::GRAPH_CALLABLE, "unknown compile state");
  return true;
}

static std::vector<py::object> PackArgs(const PyFrameObject *frame) {
  const Py_ssize_t argc = frame->f_code->co_argcount + frame->f_code->co_kwonlyargcount;
  bool has_varg = frame->f_code->co_flags & CO_VARARGS;
  py::list args(argc);
  py::object vargs;
  py::object kwvargs;
  for (Py_ssize_t i = 0; i < argc; ++i) {
    args[i] = py::reinterpret_borrow<py::object>(frame->f_localsplus[i]);
  }
  if (has_varg) {
    vargs = py::reinterpret_borrow<py::object>(frame->f_localsplus[argc]);
  }
  if (frame->f_code->co_flags & CO_VARKEYWORDS) {
    kwvargs = py::reinterpret_borrow<py::object>(frame->f_localsplus[argc + has_varg]);
  }

  const Py_ssize_t ncells = PyTuple_GET_SIZE(frame->f_code->co_cellvars);
  for (Py_ssize_t i = 0; frame->f_code->co_cell2arg && i < ncells; ++i) {
    Py_ssize_t argi = frame->f_code->co_cell2arg[i];
    if (argi != CO_CELL_NOT_AN_ARG) {
      PyObject *cell = frame->f_localsplus[frame->f_code->co_nlocals + i];
      args[argi] = py::reinterpret_borrow<py::object>(PyCell_GET(cell));
    }
  }
  return {args, vargs, kwvargs};
}

static py::object CallGraph(const JitCompileResults *c, const py::object &args, const py::object &kwvargs) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kCapture, runtime::ProfilerEvent::kCaptureRunGraph,
                                     "PIJitRunGraph");
  PyObject *py_args = args.ptr();
  PyObject *py_kwvargs = kwvargs.ptr();
  PyObject *res;
  if (c->conf->GetBoolConfig(GraphJitConfig::kPerfStatistics) &&
      c->code->GetPerf(OptPerf::PerfKind::kPerfGraph)->GetStatistics()->GetTotalCount() <
        c->conf->getIntConfig(GraphJitConfig::kPerfStatisticsCount)) {
    std::function<PyObject *(PyObject * py_args, PyObject * py_kwvargs)> func = [c](PyObject *py_args,
                                                                                    PyObject *py_kwvargs) {
      auto ret = c->code->GetNativeFunc()(py_args, py_kwvargs);
      runtime::OpExecutor::GetInstance().WaitAll();
      return ret;
    };
    runtime::OpExecutor::GetInstance().WaitAll();
    res = CallFunction(c->code->GetPerf(OptPerf::PerfKind::kPerfGraph), func, py_args, py_kwvargs);
  } else {
    res = c->code->GetNativeFunc()(py_args, py_kwvargs);
  }

  if (res == NULL && !PyErr_Occurred()) {
    PyErr_SetString(PyExc_RuntimeError, "compiled graph execute failed");
  }
  return py::reinterpret_steal<py::object>(res);
}

static py::object CallCompiledCallable(PyThreadState *tstate, PyFrameObject *f, const JitCompileResults *c) {
  PyFrameObject *new_f;
  PyObject *res;
  int bci;

  if (c->code->GetPythonCode() != nullptr) {
    new_f = PrepareCallCompiledCallable(tstate, f, c);
  } else {
    Py_INCREF(f);
    new_f = f;
  }

  if (c->conf->GetBoolConfig(GraphJitConfig::kPerfStatistics) &&
      c->code->GetPerf(OptPerf::PerfKind::kPerfPyNative)->GetStatistics()->GetTotalCount() <
        c->conf->getIntConfig(GraphJitConfig::kPerfStatisticsCount)) {
    std::function<PyObject *(PyThreadState * tstate, PyFrameObject * f, int exc)> func = [](PyThreadState *tstate,
                                                                                            PyFrameObject *f, int exc) {
      auto ret = _PyEval_EvalFrameDefault(tstate, f, exc);
      runtime::OpExecutor::GetInstance().WaitAll();
      return ret;
    };
    runtime::OpExecutor::GetInstance().WaitAll();
    // use function pointer not std::function
    res = CallFunction(c->code->GetPerf(OptPerf::PerfKind::kPerfPyNative), func, tstate, new_f, 0);
  } else {
    res = _PyEval_EvalFrameDefault(tstate, new_f, 0);
  }

  code_size_execute_python[PyBytes_GET_SIZE(new_f->f_code->co_code)]++;

  bci = new_f->f_lasti;
  Py_DECREF(new_f);

  if (res == NULL && !PyErr_Occurred()) {
    PyErr_Format(PyExc_RuntimeError, "compiled function failed with unknown error, error bci %d", bci);
  }
  return py::reinterpret_steal<py::object>(res);
}

static bool PreferCallGraph(const JitCompileResults *c) {
  if (c->code->GetNativeFunc() == nullptr) {
    return false;
  }
  OptStrategy::ExecKind stat = OptStrategy::ExecKind::kExecGraph;
  if (c->conf->GetBoolConfig(GraphJitConfig::kPerfStatistics)) {
    constexpr auto kStatisticsScale = 10000.0;
    int scale_statistics = c->conf->getIntConfig(GraphJitConfig::kPerfStatisticsScale10000x);
    stat = OptStrategy::MakeExecStrategyByPerf(
      c->code->GetPerf(OptPerf::PerfKind::kPerfGraph), c->code->GetPerf(OptPerf::PerfKind::kPerfPyNative),
      c->conf->getIntConfig(GraphJitConfig::kPerfStatisticsCount), scale_statistics / kStatisticsScale);
  }
  int graph_bytecode_min = c->conf->getIntConfig(GraphJitConfig::kStaticGraphBytecodeMin);
  if (graph_bytecode_min > 0 && stat == OptStrategy::ExecKind::kExecGraph) {
    stat = OptStrategy::MakeExecStrategyByComplex(c->code->GetPythonCode(), graph_bytecode_min);
  }
  return stat == OptStrategy::ExecKind::kExecGraph;
}

static void SetExecStatus(const JitCompileResults *c, const PyFrameObject *f, bool graph_preferred) {
  bool enable_statistics = c->conf->GetBoolConfig(GraphJitConfig::kPerfStatistics);
  int graph_bytecode_min = c->conf->getIntConfig(GraphJitConfig::kStaticGraphBytecodeMin);
  if (enable_statistics || (graph_bytecode_min > 0)) {
    PyObject_SetItem(f->f_globals, reinterpret_cast<PyObject *>(f->f_code), (graph_preferred ? Py_True : Py_False));
  }
}

static py::object CallCompiledResults(PyThreadState *tstate, PyFrameObject *f, const JitCompileResults *c) {
  if (MsContext::GetInstance()->get_param<bool>(MS_CTX_PRECOMPILE_ONLY)) {
    return py::none();
  }

  ValidateCompiledResults(c);

  bool graph_preferred = PreferCallGraph(c);
  SetExecStatus(c, f, graph_preferred);

  std::vector<py::object> packed_args = PackArgs(f);
  if (packed_args[1].ptr() != nullptr) {
    PyList_Append(packed_args[0].ptr(), packed_args[1].ptr());
  }

  py::object args = py::reinterpret_steal<py::object>(PyList_AsTuple(packed_args[0].ptr()));
  py::object kwvargs = packed_args[2];
  py::object res = graph_preferred ? CallGraph(c, args, kwvargs) : CallCompiledCallable(tstate, f, c);

  if (graph_preferred) {
    code_size_execute_graph[PyBytes_GET_SIZE(f->f_code->co_code)]++;
  }

  // dump traceback
  if (c->conf->GetBoolConfig(GraphJitConfig::kPrintTraceback)) {
    // dump all traceback for the root function
    GRAPH_JIT_LOG_F("%s\n", c->tbs->Dump(true).c_str());
  }
  if (!PyErr_Occurred()) {
    c->tbs->Clear();
  }
  return res;
}

static bool CheckGuard(JitCompileResults *c, const PyFrameObject *f) {
  TimeRecorder _time_recorder(TimeRecorder::kTimeGuard, kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kLogPerf));

  c->code = nullptr;

  OptOptionPtr opt = OptOption::CreateOptionByPoint(c);
  for (auto &oc : c->codehub->GetOptTarget(opt)) {
    OptGuardPtr guard = oc->GetGuard();
    bool print_guard = c->conf->GetBoolConfig(GraphJitConfig::kPrintGuard);
    if (guard != nullptr && guard->Check(f, print_guard)) {
      c->code = oc;
      break;
    }
  }
  MS_LOG(DEBUG) << __FUNCTION__ << (c->code != nullptr ? " success !" : " failed !");
  return c->code != nullptr;
}

static bool JitCompileWithTry(PyThreadState *tstate, JitCompileResults *c) {
  TimeRecorder _time_recorder(TimeRecorder::kTimeCompile, kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kLogPerf));

  if (!c->conf->GetBoolConfig(GraphJitConfig::kCompileWithTry)) {
    return JitCompile(tstate, c);
  }

  bool compiled = false;
  try {
    compiled = JitCompile(tstate, c);
  } catch (std::exception &e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
  }
  if (PyErr_Occurred()) {
    compiled = false;
  }
  if (!compiled) {
    MS_LOG(ERROR) << "compiled failed with " << py::error_already_set().what() << " at "
                  << std::string(py::str(reinterpret_cast<PyObject *>(c->origin_frame_->f_code)));
    c->stat = JitCompileResults::NEVER_COMPILE;
    PyErr_Clear();
  }
  return compiled;
}

static py::object CodeHook(PyThreadState *tstate, JitCompileResults *c, PyFrameObject *frame) {
  bool just_compiled = false;
  switch (c->stat) {
    case JitCompileResults::NEVER_COMPILE:
      break;
    case JitCompileResults::GRAPH_CAPTURED:
      if (c->conf->GetBoolConfig(GraphJitConfig::kInterpretCapturedCode)) {
        break;
      }
    /* fallthrough */
    case JitCompileResults::GRAPH_CANDIDATE:
      MS_EXCEPTION_IF_CHECK_FAIL(c->origin_frame_ == nullptr || c->origin_frame_ == frame,
                                 "check recursive call compiling function");
      c->origin_frame_ = frame;
      if (c->conf->GetBoolConfig(GraphJitConfig::kCompileWithoutCapture)) {
        c->stat = JitCompileResults::GRAPH_CAPTURED;
      }
      if (!JitCompileWithTry(tstate, c)) {
        c->stat = JitCompileResults::NEVER_COMPILE;
        break;
      }
      just_compiled = true;
    /* fallthrough */
    case JitCompileResults::GRAPH_CALLABLE: {
      bool check_guard;
      {
        runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kCapture, runtime::ProfilerEvent::kCaptureGuard,
                                           "PIJitGuard");
        check_guard = CheckGuard(c, frame);
      }
      if (check_guard) {
        c->origin_frame_ = nullptr;
        return CallCompiledResults(tstate, frame, c);
      }
      if (!just_compiled) {
        c->stat = JitCompileResults::GRAPH_CANDIDATE;
        return CodeHook(tstate, c, frame);
      }
      MS_LOG(EXCEPTION) << "shouldn't reach here";
    }
    /* fallthrough */
    case JitCompileResults::GRAPH_BUILDING:
      MS_LOG(ERROR) << "recursive call, compiler call the code "
                    << std::string(py::str(reinterpret_cast<PyObject *>(frame->f_code))) << " which is compiling";
      break;
    default:
      MS_LOG(EXCEPTION) << "shouldn't reach here";
      break;
  }
  PyObject *res = _PyEval_EvalFrameDefault(tstate, frame, 0);
  return py::reinterpret_steal<py::object>(res);
}

static void ApplyAutoJit(PyFrameObject *f) {
  if (!kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kAutoJit)) {
    return;
  }

  PyObject *code = reinterpret_cast<PyObject *>(f->f_code);
  if (getJitCompileResults(code, false) != nullptr) {
    return;
  }

  // first reached this code
  // allocate for all code while auto jit
  (void)getJitCompileResults(code, true);
  if (!kPIJitConfigDefault.ShouldAutoJit(f)) {
    return;
  }
  (void)pi_jit_should_compile(py::cast<py::object>(code), py::dict());
}

#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION < 9)
PyObject *EvalFrame(PyFrameObject *f, int exc) {
  PyThreadState *tstate = PyThreadState_Get();

#else
PyObject *EvalFrame(PyThreadState *tstate, PyFrameObject *f, int exc) {
#endif

  // exception handler
  if (exc != 0) {
    return _PyEval_EvalFrameDefault(tstate, f, exc);
  }

  ApplyAutoJit(f);

  PyObject *code = reinterpret_cast<PyObject *>(f->f_code);
  JitCompileResults *c = getJitCompileResults(code, false);
  if (c == nullptr) {
    return _PyEval_EvalFrameDefault(tstate, f, exc);
  }
  py::object res;
  try {
    res = CodeHook(tstate, c, f);
  } catch (py::error_already_set &e) {
    MS_LOG(ERROR) << "execute failed with" << e.what() << " at "
                  << std::string(py::str(reinterpret_cast<PyObject *>(f->f_code)));

    e.restore();
  }
  if (PyErr_Occurred()) {
    res = py::object();
  }
  return res.inc_ref().ptr();
}
}  // namespace graph
}  // namespace jit
}  // namespace mindspore

namespace mindspore {

#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION == 9 || PY_MINOR_VERSION == 7)

py::bool_ pi_jit_enable() {
  PyInterpreterState *inter = PyInterpreterState_Main();
  _PyFrameEvalFunction prev = _PyInterpreterState_GetEvalFrameFunc(inter);
  if (prev != _PyEval_EvalFrameDefault) {
    return false;
  }
  mindspore::jit::graph::ensureInitialize();
  _PyInterpreterState_SetEvalFrameFunc(inter, mindspore::jit::graph::EvalFrame);
  return true;
}

py::bool_ pi_jit_disable() {
  PyInterpreterState *inter = PyInterpreterState_Main();
  _PyFrameEvalFunction prev = _PyInterpreterState_GetEvalFrameFunc(inter);
  if (prev != mindspore::jit::graph::EvalFrame) {
    return false;
  }
  _PyInterpreterState_SetEvalFrameFunc(inter, _PyEval_EvalFrameDefault);
  return true;
}

// bellowing code is used for debugging code generate, and will be remove soon
py::object test_graph_ir_code_gen(const py::object &func) {
  mindspore::jit::graph::Utils::DisFuncObject(func.ptr());
  auto byteCodeParser = std::make_shared<mindspore::jit::graph::ByteCodeParser>(func);
  mindspore::jit::graph::ir::FunctionNodePtr func_node = byteCodeParser->Parse();
  auto inliner = std::make_shared<mindspore::jit::graph::FuncInliner>(func_node);
  inliner->Run();
  auto func_obj = mindspore::jit::graph::ByteCodeGenerator::GenFunction(func_node);
  mindspore::jit::graph::Utils::DisFuncObject(func_obj.ptr());
  func_obj.inc_ref();
  return func_obj;
}

py::bool_ pi_jit_should_compile(const py::object &funcHandle, const py::object &tag) {
  PyObject *func = funcHandle.ptr();
  PyObject *code = NULL;
  if (PyFunction_Check(func)) {
    code = PyFunction_GET_CODE(func);
  } else if (PyMethod_Check(func)) {
    func = PyMethod_GET_FUNCTION(func);
    code = PyFunction_GET_CODE(func);
  } else if (PyCode_Check(func)) {
    code = func;
  } else {
    return false;
  }
  mindspore::jit::graph::JitCompileResults *c = mindspore::jit::graph::getJitCompileResults(code);
  if (c == nullptr) {
    return false;
  }
  if (c->stat != mindspore::jit::graph::JitCompileResults::NEVER_COMPILE) {
    *c->conf = mindspore::jit::graph::GraphJitConfig(tag);
    return true;
  }

  int raw_code_size = (PyBytes_GET_SIZE(reinterpret_cast<PyCodeObject *>(code)->co_code)) / sizeof(_Py_CODEUNIT);
  std::string raw_func_info_name = py::str(code).cast<std::string>();
  std::string raw_func_name = "";
  if (PyFunction_Check(func)) {
    const char *module_name = PyUnicode_AsUTF8(PyFunction_GET_MODULE(func));
    const char *s = strchr(module_name, '.');
    std::string top_module = s ? std::string(module_name, s - module_name) : module_name;
    mindspore::jit::graph::kPIJitConfigDefault.AddAllowedInlineModules(top_module);

    raw_func_name = mindspore::jit::graph::Utils::GetPyName(reinterpret_cast<PyFunctionObject *>(func)->func_qualname);
  }

  c->stat = mindspore::jit::graph::JitCompileResults::GRAPH_CANDIDATE;
  *c->conf = mindspore::jit::graph::GraphJitConfig(tag);
  *c->tbs = mindspore::jit::graph::Tracebackes(raw_func_name, raw_func_info_name, raw_code_size);
  return true;
}
#else

py::bool_ pi_jit_enable() { return py::bool_(false); }
py::bool_ pi_jit_disable() { return py::bool_(false); }
py::bool_ pi_jit_should_compile(const py::object &func, const py::object &tag) {
  MS_LOG(WARNING) << "GraphJit not support this python version " << PY_MAJOR_VERSION << '.' << PY_MINOR_VERSION
                  << " only support on python3.9 or python3.7";
  return py::bool_(false);
}
// bellowing code is used for debugging code generate, and will be remove soon
py::object test_graph_ir_code_gen(const py::object &func) { return py::bool_(false); }

#endif

}  // namespace mindspore
