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
#include "frontend/jit/pi/runtime.h"
#include <algorithm>
#include <iomanip>
#include <iterator>
#include <regex>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "pybind11/pybind11.h"
#include "include/utils/convert_utils_py.h"
#include "frontend/jit/pi/external.h"
#include "frontend/jit/pi/graph_build/parameter_manager.h"
#include "frontend/jit/pi/graph_capture/graph_build.h"
#include "frontend/jit/pi/graph_capture/graph_analyzer.h"
#include "frontend/jit/pi/graph_compiler/abstract_type_deducer.h"
#include "frontend/jit/pi/graph_compiler/cg/byte_code_generator.h"
#include "frontend/jit/pi/graph_compiler/inliner/func_inliner.h"
#include "frontend/jit/pi/graph_compiler/parser/byte_code_parser.h"
#include "frontend/jit/pi/graph_compiler/utils.h"
#include "frontend/jit/pi/utils/utils.h"
#include "frontend/jit/pi/utils/opcode_declare.h"
#include "frontend/jit/pi/graph_guard/guard.h"
#include "frontend/jit/pi/graph_guard/strategy.h"
#include "frontend/jit/pi/graph_guard/shape_ctx.h"
#include "frontend/jit/pi/capture_context.h"
#include "include/frontend/jit/ps/executor/jit_executor_py.h"
#include "frontend/jit/pi/python_adapter/py_frame.h"
#include "include/runtime/pipeline/pipeline.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "frontend/jit/pi/graph_capture/code_generator.h"
#include "utils/convert_utils_base.h"
#include "frontend/jit/pi/eval_frame_hook.h"
#include "include/utils/tensor_py.h"
#include "include/utils/pynative/grad_state.h"
#include "tools/profiler/profiler.h"
#include "frontend/jit/pi/utils/py_obj_registry.h"

namespace mindspore {
namespace pijit {

void AddConfigToGuard(const GraphJitConfig &c, OptGuardPtr guard);
static void AddGradFlagForParam(const OptGuardPtr &guard);
static void CollectTraceBack(JitCompileResults *c, PyCodeObject *code, bool is_graph_mode);

class StaticAnalysisExceptionCleaner {
 public:
  StaticAnalysisExceptionCleaner() = default;
  ~StaticAnalysisExceptionCleaner() { StaticAnalysisException::Instance().ClearException(); }
};

class RunEnvironment {
 public:
  RunEnvironment() = default;

  void fetchAndSetRunEnv(const JitCompileResults *jcr) {
    auto ms_context = MsContext::GetInstance();
    jit_level_ = ms_context->GetJitLevel();
    task_sink_ = ms_context->get_param<bool>(MS_CTX_ENABLE_TASK_SINK);

    auto jit_level = jcr->conf()->getJitLevel();
    auto grad_flag = pynative::GradState::Get().grad_flag();
    auto task_sink = jit_level == "O2" && !grad_flag;
    ms_context->set_param(MS_CTX_JIT_LEVEL, jit_level);
    ms_context->SetJitLevel(jit_level);
    ms_context->set_param<bool>(MS_CTX_ENABLE_TASK_SINK, task_sink);
    ms_context->set_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK, task_sink);
  }

  void resumePreviousRunEnv() {
    auto ms_context = MsContext::GetInstance();
    ms_context->set_param(MS_CTX_JIT_LEVEL, jit_level_);
    ms_context->SetJitLevel(jit_level_);
    ms_context->set_param<bool>(MS_CTX_ENABLE_TASK_SINK, task_sink_);
    ms_context->set_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK, task_sink_);
  }

 private:
  std::string jit_level_;
  bool task_sink_ = false;
};

static void PrintGuardPerf() {
  std::map<std::string, std::pair<size_t, size_t>> guard_info;
  std::map<std::string, std::pair<size_t, size_t>> guard_freq_info;
  std::map<std::string, std::pair<size_t, size_t>> trace_info;
  std::map<std::string, std::pair<size_t, std::vector<size_t>>> item_info;
  OptGuardPerf::GetGuardPerf()->GetGuardPerfInfo(&guard_info, &item_info, &trace_info, &guard_freq_info);
  std::cout << "Guard performance info:" << std::endl;
  std::cout << "guard, count, total time, success, fail" << std::endl;
  for (const auto &item : guard_info) {
    auto iter = guard_freq_info.find(item.first);
    if (iter != guard_freq_info.end()) {
      std::cout << "guard:" << item.first << ", " << item.second.first << ", " << item.second.second << ","
                << iter->second.first << "," << iter->second.second << std::endl;
    } else {
      std::cout << "guard:" << item.first << ", " << item.second.first << ", " << item.second.second << std::endl;
    }
  }
  std::cout << "trace, count, total time" << std::endl;
  for (const auto &item : trace_info) {
    std::cout << "trace:" << item.first << ", " << item.second.first << ", " << item.second.second << std::endl;
  }
  std::cout << "item, count, [stage time]" << std::endl;
  for (const auto &item : item_info) {
    std::cout << "item:" << item.first << "," << item.second.first << ", [";
    for (auto stage : item.second.second) {
      std::cout << stage << ",";
    }
    std::cout << "]" << std::endl;
  }
}

class GuardPerfLogger {
 public:
  ~GuardPerfLogger() {
    if (kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kLogGuardPerf)) {
      PrintGuardPerf();
    }
  }
};
GuardPerfLogger at_exit_printer;

// jit compiler initialize
static void ensureInitialize() {
  static bool init = false;
  if (init) {
    return;
  }
  init = true;
}

static void PrintLabel(std::stringstream &os, const std::string &str, int distance = 30) {
  os << std::left << std::setw(distance) << str << ": ";
}

std::string Traceback::Dump(bool is_all) const {
  constexpr auto width = 10;

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
  std::list<Element> candidates;
  if (is_all) {
    candidates = tbs_;
  } else {
    // last one traceback
    candidates.emplace_back(tbs_.back());
  }
  // dump traceback list head
  int name_length = FindMaxNameLength(candidates);
  os << std::left << std::setw(name_length) << "func_name:  -->  " << std::left << std::setw(name_length)
     << "changed_func:" << std::left << std::setw(width) << "run_mode:" << std::left << std::setw(kThree * width)
     << "stop_trace:" << std::left << std::setw(width) << "code_size:" << std::endl;
  os << "--------------------------------------------------------------------------------------\n";
  // dump traceback list content
  for (const auto &tb : candidates) {
    os << std::left << std::setw(name_length) << tb.func_name_ << "  -->  ";
    os << std::left << std::setw(name_length) << tb.changed_func_;
    if (tb.is_graph_mode_) {
      os << std::left << std::setw(width) << "[GRAPH]";
    } else {
      os << std::left << std::setw(width) << "PYNATIVE";
    }
    // dump stop trace reason
    auto it_trace = stop_trace_res_.find(tb.func_name_);
    if (it_trace != stop_trace_res_.cend()) {
      os << std::left << std::setw(kThree * width)
         << GetStopTraceReasonDesc(static_cast<StopTraceReason>(it_trace->second));
    } else {
      os << std::left << std::setw(kThree * width) << "unknown";
    }
    os << std::left << std::setw(width) << tb.code_size_ << "\n";
  }
  os << "\n\n";
  os << DumpSummary();
  return os.str();
}

std::string Traceback::DumpSummary() const {
  std::stringstream os;
  if (tbs_.empty()) {
    return os.str();
  }
  os << "*** Dump Summary on [" << raw_func_info_name_ << "] ***\n";
  PrintLabel(os, "traceback_num");
  os << tbs_.size() << "\n";

  std::array<int, kStopTrace_Reason_Count> stop_trace_reason_array{0};
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
    if (stop_trace_reason_array[i] > 0) {
      PrintLabel(os, GetStopTraceReasonDesc(static_cast<StopTraceReason>(i)));
      os << stop_trace_reason_array[i] << "\n";
    }
  }
  os << "\n\n";
  return os.str();
}

int Traceback::FindMaxNameLength(const std::list<Element> &tbs) const {
  constexpr auto name_length = kFive * (kTwo + kFive);
  int max_length = 15;
  for (const auto &tb : tbs) {
    int len1 = SizeToInt(tb.func_name_.length());
    int len2 = SizeToInt(tb.changed_func_.length());
    max_length = std::max(max_length, std::max(len1, len2)) + kTwo;
  }
  max_length = std::min(max_length, name_length);
  return max_length;
}

static void GuardForFrame(const PyFrameWrapper &f, const OptCodePtr &oc, const GraphJitConfig &conf) {
  const char *code_name = f.GetCode().Name();
  AddConfigToGuard(conf, oc->GetGuard());
  // parameter guard is delay to graph parameter build
  AddGradFlagForParam(oc->GetGuard());
  // It tooks too much time in Guard's GetDescript function when trace depth is too large.
  MS_LOG(DEBUG) << "Guard on " << code_name << " by " << oc->GetGuard()->GetDescript() << "!" << std::endl;
}

static void ValidateCompiledResults(const JitCompileResults *c) {
  if (c->stat() != JitCompileResults::GRAPH_CALLABLE) {
    return;
  }
  bool valid_res;
  if (c->code()->GetNativeFunc()) {
    valid_res = true;
  } else {
    valid_res = c->code()->GetPythonCode() != nullptr;
  }
  MS_EXCEPTION_IF_CHECK_FAIL(valid_res, "check compiled result");
}

static void MarkBreak(Graph *g) {
  TimeRecorder recorder(__FUNCTION__, kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kLogPerf));
  int break_bci = g->GetStopTraceBci();
  if (break_bci == -1) {
    return;
  }
  PyCodeObject *code;
  const auto &nodes = g->GetTracedNodes();
  if (nodes.empty()) {
    code = g->GetCodeObj();
  } else {
    auto iter = std::find_if(nodes.begin(), nodes.end(), [&break_bci](ValueNode *i) { return i->bci() >= break_bci; });
    iter -= iter == nodes.end();
    for (code = (*iter)->GetGraph()->GetCodeObj(); code == nullptr && iter != nodes.begin(); --iter) {
      code = (*iter)->GetGraph()->GetCodeObj();
    }
  }
  MS_EXCEPTION_IF_NULL(code);
  auto jcr = GetJitCompileResults(code);
  if (jcr != nullptr) {
    jcr->break_count()++;
  }
}

static void GraphCapture(JitCompileResults *jcr);

static bool TryLoopBodyReCapture(JitCompileResults *jcr, const GraphBuilderPtr &g) {
  if (jcr->conf()->GetBoolConfig(GraphJitConfig::kReCaptureLoopBody)) {
    MS_LOG(INFO) << "TryLoopBodyReCapture for " << g->GetGraph()->GetCodeName();
    LoopBodyReCaptureCodeGenerator loopBodyCg(g->GetGraph());
    if (!loopBodyCg.Prepare()) {
      MS_LOG(WARNING) << "LoopBodyReCapture CodeGen Prepare Failed";
      return false;
    }
    auto new_code = loopBodyCg.Build();
    if (new_code.ptr() != nullptr) {
      jcr->code()->SetPythonCode(new_code);
      jcr->set_stat(JitCompileResults::GRAPH_CALLABLE);
      return true;
    }
    MS_LOG(WARNING) << "LoopBodyReCapture CodeGen Build Failed";
  }
  return false;
}

static auto HandleBreakAtLoop(JitCompileResults *jcr, const GraphBuilderPtr &g) {
  // one stage need adapter
  if (jcr->conf()->GetBoolConfig(GraphJitConfig::kLoopUnrolling) && g->GetGraph()->IsBreakAtLoop()) {
    if (IsPiJitLogOn(LogCfg::kGraphBreak)) {
      GRAPH_JIT_LOG_F("===> graph break after loop unrolling\n%s\n", g->GetGraph()->ToString(1).c_str());
    }
    MS_LOG(INFO) << "Top graph is graph break at loop after unrolling. Disable loop unrolling and re-capture graph";
    // reset guard
    jcr->code()->SetGuard(std::make_shared<OptGuard>());
    AddConfigToGuard(*jcr->conf(), jcr->code()->GetGuard());
    // disable loop unroll
    jcr->conf()->SetBool<GraphJitConfig::kLoopUnrolling>(Py_False);
    // restart captured
    GraphCapture(jcr);
    // reset config
    jcr->conf()->SetBool<GraphJitConfig::kLoopUnrolling>(Py_True);
    return true;
  }
  return false;
}

static auto HandleUnsupportedSyntax(JitCompileResults *jcr, const GraphBuilderPtr &g) {
  int break_bci = g->GetGraph()->GetStopTraceBci();
  if (break_bci == -1) {
    return false;
  }
  int break_op = g->GetGraph()->GetCFG()->instr_pool()[break_bci]->op();
  bool unsupported =
    break_op == WITH_CLEANUP_START || break_op == WITH_CLEANUP_FINISH || break_op == END_FINALLY || break_op == RERAISE;
  if (g->StackSize() > 0 || unsupported) {
    // something happened in with syntax
    MS_LOG(DEBUG) << "Break at unsupported syntax";
    jcr->code()->SetGuard(std::make_shared<OptGuard>());
    AddConfigToGuard(*jcr->conf(), jcr->code()->GetGuard());
    jcr->conf()->SetBool<GraphJitConfig::kSkipException>(Py_True);
    GraphCapture(jcr);
    g->GetTryBlockStacks().clear();
    jcr->conf()->SetBool<GraphJitConfig::kSkipException>(Py_False);
    return true;
  }
  return false;
}

static auto TraceRun(JitCompileResults *jcr) {
  TimeRecorder recorder(__FUNCTION__, kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kLogPerf));
  GraphBuilderPtr g = std::make_shared<GraphBuilder>(jcr->origin_frame());
  (void)g->TraceRun();
  return g;
}

static auto Analyze(const GraphBuilderPtr &g) {
  TimeRecorder recorder(__FUNCTION__, kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kLogPerf));

  auto analyzer = std::make_shared<GraphAnalyzer>(g);
  analyzer->Analyze();
  return analyzer;
}

// preprocess before compile, split bytecode to sub-function
// return whether the code should be modified
static void GraphCapture(JitCompileResults *jcr) {
  TimeRecorder recorder(__FUNCTION__, kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kLogPerf));
  MS_EXCEPTION_IF_NULL(jcr->code());

  GraphBuilderPtr g = TraceRun(jcr);
  if (HandleUnsupportedSyntax(jcr, g)) {
    return;
  }
  if (g->GetGraph()->IsBreakAtLoop() && TryLoopBodyReCapture(jcr, g)) {
    return;
  }
  if (g->GetGraph()->ShouldNeverCompile()) {
    if (IsPiJitLogOn(LogCfg::kGraphBreak)) {
      GRAPH_JIT_LOG_F("===> graph break after loop unrolling\n%s\n", g->GetGraph()->ToString(1).c_str());
    }
    MS_LOG(INFO) << "Cannot capture graph, mark it as NEVER_COMPILE: " << ToString(jcr->origin_frame().GetCode());
    jcr->set_stat(JitCompileResults::NEVER_COMPILE);
    return;
  }
  GraphAnalyzerPtr analyzer = Analyze(g);
  if (HandleBreakAtLoop(jcr, g)) {
    return;
  }
  MarkBreak(g->GetGraph());

  // dump DFG
  if (IsPiJitLogOn(LogCfg::kOthers)) {
    g->DumpDFG();
    const auto &debug_str = analyzer->GetCaptureInfo().ToString();
    PY_PRINTF_WITH_FLUSH("*** Dump One Stage ByteCode Collection After CodeGen *** \n%s", debug_str.c_str());
  }

  py::object new_code = MakeCodeFromCodeGen(g, analyzer, jcr->origin_frame().Globals().ptr());
  if (new_code.ptr() != nullptr) {
    jcr->code()->SetPythonCode(new_code);
    jcr->set_stat(JitCompileResults::GRAPH_CALLABLE);
  }

  if (IsPiJitLogOn(LogCfg::kBytecode)) {
    PY_PRINTF_WITH_FLUSH("MODIFIED BYTECODE of %A", new_code.ptr());
    Utils::DisFuncObject(new_code.ptr());
  }

  // collect stop trace reason to traceback
  jcr->tbs()->PushStopTraceRes(g->GetGraph()->GetCodeName(), g->GetGraph()->GetStopTraceReason());
}

static void CollectTraceBack(JitCompileResults *c, PyCodeObject *code, bool is_graph_mode) {
  if (code == nullptr) {
    code = c->origin_frame().GetCode().ptr();
  }
  std::string name = c->origin_frame().GetCode().Name();
  std::string changed_name = Utils::GetPyName(code->co_name);
  int code_size = _PyCode_NBYTES(code);
  c->tbs()->PushTbs({name, changed_name, code_size, is_graph_mode});
}

void AddConfigToGuard(const GraphJitConfig &c, OptGuardPtr guard) {
  std::map<std::string, bool> bool_cfg;
  std::map<std::string, int> int_cfg;
  bool_cfg[kSpecializeScalar] = c.GetBoolConfig(GraphJitConfig::kGuardSpecializeScalar);
  bool_cfg[kSpecializeContainer] = c.GetBoolConfig(GraphJitConfig::kGuardSpecializeContainer);
  bool_cfg[kSpecializeTensor] = c.GetBoolConfig(GraphJitConfig::kGuardSpecializeTensor);
  int_cfg[kGuardRelaxCnt] = c.getIntConfig(GraphJitConfig::kGuardRelaxCount);
  guard->UpdateConfig(bool_cfg, int_cfg);
}

static void AddGradFlagForParam(const OptGuardPtr &guard) {
  bool grad_flag = pynative::GradState::Get().RequiresGrad();
  CustomizedTracePtr ptr = std::make_shared<CustomizedTrace>(
    grad_flag ? Py_True : Py_False,
    [](PTraceContext context) -> PyObject * {
      PyObject *ret = pynative::GradState::Get().RequiresGrad() ? Py_True : Py_False;
      Py_INCREF(ret);
      return ret;
    },
    [grad_flag](bool simple) -> std::string {
      if (simple) {
        return std::string("g\\") + std::to_string(grad_flag ? 1 : 0);
      }
      return std::string("{PyNativeExecutor::GetInstance()->RequiresGrad() == ") + std::to_string(grad_flag) +
             std::string("}(type:") + std::to_string(TraceType::Customized) + std::string(")");
    });
  guard->GuardOn(ptr, mindspore::pijit::GuardLevel::GEqual, true);
}

extern bool UnsupportedCodeTypeCheck(PyCodeObject *co);

static bool JitCompile(PyThreadState *tstate, JitCompileResults *c) {
  const auto &frame = c->origin_frame();
  PyCodeObject *code = frame.GetCode().ptr();
  if (UnsupportedCodeTypeCheck(code)) {
    return false;
  }
  ShapeContext sc(c->origin_frame(), c->enable_dynamic_dict());
  sc.ApplyEnableDynamic();
  MS_LOG(INFO) << "Start compile " << ToString(frame.GetCode());

  ParameterManager::ScopedCleaner param_auto_cleaner;
  PyObjRegistry py_obj_registry;
  // new guard code
  c->set_code(c->codehub()->AddOptTarget(OptOption::CreateOptionByPoint(c)));
  AddConfigToGuard(*c->conf(), c->code()->GetGuard());

  if (c->stat() == JitCompileResults::GRAPH_CANDIDATE) {
    TimeRecorder time_recorder("kTimeCompileCapture", kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kLogPerf));
    runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kCapture, runtime::ProfilerEvent::kCaptureProcess,
                                       "PIJitCapture");
    c->set_stat(JitCompileResults::GRAPH_BUILDING);
    auto aobject_resource = AObject::MakeResource();
    GuardForFrame(frame, c->code(), *c->conf());
    GraphCapture(c);

    // global guard is generated while graph build
    c->code()->guard_status() = nullptr;
    aobject_resource.Release();
  }

  if (c->conf()->getIntConfig(GraphJitConfig::kGuardRelaxCount) > 0) {
    auto guard = c->code()->GetGuard()->Optimize();
    if (guard != nullptr) {
      c->code()->SetGuard(guard);
    }
  }
  c->code()->GetGuard()->FilterConstItem();

  CollectTraceBack(c, c->code()->GetPythonCode(), c->code()->GetNativeFunc() != nullptr);

  PIJIT_DEBUG_LOG(LogCfg::kOthers) << std::endl << c->tbs()->Dump().c_str();
  PIJIT_DEBUG_LOG(LogCfg::kGuard) << std::endl
                                  << "generated guard at "
                                  << std::string(py::str(reinterpret_cast<PyObject *>(code))).c_str() << std::endl
                                  << c->code()->GetGuard()->ToString().c_str();
  if (c->stat() != JitCompileResults::GRAPH_CALLABLE) {
    c->set_stat(JitCompileResults::NEVER_COMPILE);
    return false;
  }
  return true;
}

static py::object CallGraph(const JitCompileResults *c, const py::object &args, const py::object &kwvargs) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kCapture, runtime::ProfilerEvent::kCaptureRunGraph,
                                     "PIJitRunGraph");

  StaticAnalysisExceptionCleaner exception_cleaner;
  CaptureContext::DisableScope compiler_disable_scope;

  RunEnvironment runEnvironment;
  runEnvironment.fetchAndSetRunEnv(c);
  PyObject *py_args = args.ptr();
  PyObject *py_kwvargs = kwvargs.ptr() == Py_None ? nullptr : kwvargs.ptr();
  PyObject *res;
  if (c->conf()->GetBoolConfig(GraphJitConfig::kPerfStatistics) &&
      c->code()->GetPerf(OptPerf::PerfKind::kPerfGraph)->GetStatistics()->GetTotalCount() <
        c->conf()->getIntConfig(GraphJitConfig::kPerfStatisticsCount)) {
    std::function<PyObject *(PyObject * py_args, PyObject * py_kwvargs)> func = [c](PyObject *py_args,
                                                                                    PyObject *py_kwvargs) {
      auto ret = c->code()->GetNativeFunc()(py_args, py_kwvargs);
      runtime::Pipeline::Get().WaitAll();
      return ret;
    };
    runtime::Pipeline::Get().WaitAll();
    res = CallFunction(c->code()->GetPerf(OptPerf::PerfKind::kPerfGraph), func, py_args, py_kwvargs);
  } else {
    res = c->code()->GetNativeFunc()(py_args, py_kwvargs);
  }
  runEnvironment.resumePreviousRunEnv();
  if (res == NULL && !PyErr_Occurred()) {
    PyErr_SetString(PyExc_RuntimeError, "compiled graph execute failed");
  }
  return py::reinterpret_steal<py::object>(res);
}

static py::object CallCompiledCallable(PyThreadState *tstate, const PyFrameWrapper &f, const JitCompileResults *c) {
  PyObject *res;
  if (c->conf()->GetBoolConfig(GraphJitConfig::kPerfStatistics) &&
      c->code()->GetPerf(OptPerf::PerfKind::kPerfPyNative)->GetStatistics()->GetTotalCount() <
        c->conf()->getIntConfig(GraphJitConfig::kPerfStatisticsCount)) {
    auto func = [&tstate, &f, &c]() {
      auto res = f.EvalNewCode(tstate, c->code()->GetPythonCode());
      runtime::Pipeline::Get().WaitAll();
      return res;
    };
    runtime::Pipeline::Get().WaitAll();
    res = CallFunction(c->code()->GetPerf(OptPerf::PerfKind::kPerfPyNative), std::function(func));
  } else {
    res = f.EvalNewCode(tstate, c->code()->GetPythonCode());
  }
  if (res == NULL && !PyErr_Occurred()) {
    PyErr_Format(PyExc_RuntimeError, "compiled function failed with unknown error");
  }
  return py::reinterpret_steal<py::object>(res);
}

static void SetExecStatus(const JitCompileResults *c, const PyFrameWrapper &f, bool graph_preferred) {
  bool enable_statistics = c->conf()->GetBoolConfig(GraphJitConfig::kPerfStatistics);
  int graph_bytecode_min = c->conf()->getIntConfig(GraphJitConfig::kStaticGraphBytecodeMin);
  if (enable_statistics || (graph_bytecode_min > 0)) {
    auto globals = f.Globals();
    auto code = reinterpret_cast<PyObject *>(f.GetCode().ptr());
    PyObject_SetItem(globals.ptr(), code, graph_preferred ? Py_True : Py_False);
  }
}

static py::object CallCompiledResults(PyThreadState *tstate, const PyFrameWrapper &f, JitCompileResults *c) {
  if (MsContext::GetInstance()->get_param<bool>(MS_CTX_PRECOMPILE_ONLY) ||
      common::GetEnv("MS_DEV_PRECOMPILE_ONLY") == "1") {
    return py::none();
  }

  ValidateCompiledResults(c);

  auto packed_args = f.PackArgs();
  if (packed_args[1].ptr() != Py_None) {
    PyList_Append(packed_args[0].ptr(), packed_args[1].ptr());
  }

  py::object args = py::reinterpret_steal<py::object>(PyList_AsTuple(packed_args[0].ptr()));
  py::object kwvargs = packed_args[2];
  bool graph_preferred = c->code()->GetNativeFunc() != nullptr;
  SetExecStatus(c, f, graph_preferred);
  py::object res;
  if (!graph_preferred) {
    res = CallCompiledCallable(tstate, f, c);
  } else if (!c->conf()->GetBoolConfig(GraphJitConfig::kCompileWithTry)) {
    res = CallGraph(c, args, kwvargs);
  } else {
    try {
      res = CallGraph(c, args, kwvargs);
    } catch (std::exception &e) {
      MS_LOG(WARNING) << "compile result has an error, de-optimization\n" << e.what();
      res = CallCompiledCallable(tstate, f, c);
      c->set_stat(JitCompileResults::NEVER_COMPILE);
    }
  }
  c->code()->Inc();

  if (!PyErr_Occurred()) {
    c->tbs()->Clear();
  }
  return res;
}

static bool CheckGuard(JitCompileResults *c, const PyFrameWrapper &f) {
  TimeRecorder time_recorder(__FUNCTION__, kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kLogPerf));

  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kCapture, runtime::ProfilerEvent::kCaptureGuard,
                                     "PIJitGuard");

  if (c->code() == nullptr) {
    return false;
  }
  CaptureContext::DisableScope compiler_disable_scope;
  GuardContext context;

  bool log_perf = c->conf()->GetBoolConfig(GraphJitConfig::kLogGuardPerf);
  if (c->code()->GetGuard()->Check(f, log_perf)) {
    return true;
  }

  OptOptionPtr opt = OptOption::CreateOptionByPoint(c);
  OptCodeSet defaults;
  const auto &set = c->codehub()->GetOptTarget(opt, defaults);
  OptCodePtr skip = c->code();
  c->set_code(nullptr);
  for (size_t i = set.size(); i != 0; i--) {
    const OptCodePtr &oc = set[i - 1];
    if (oc == skip) {
      continue;
    }
    if (oc->GetGuard()->Check(f, log_perf)) {
      c->set_code(oc);
      MS_LOG(DEBUG) << "select the compiled code due to guard is match: "
                    << (oc->GetPythonCode() != nullptr
                          ? std::string(py::str(reinterpret_cast<PyObject *>(oc->GetPythonCode())))
                          : oc->GetPhase())
                    << std::endl
                    << "generated guard:" << std::endl
                    << oc->GetGuard()->ToString();
      break;
    }
  }
  if (c->code() == nullptr) {  // recompiled
    c->CacheFailGuard(f);
  }
  return c->code() != nullptr;
}

static void InitCompileConfig() {
  static bool initialized = false;
  if (!initialized) {
    MS_LOG(INFO) << "Init compile config";
    CompileConfigManager::GetInstance().CollectCompileConfig();
    initialized = true;
  }
}

static bool JitCompileWithTry(PyThreadState *tstate, JitCompileResults *c) {
  TimeRecorder time_recorder(__FUNCTION__, kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kLogPerf));

  MS_LOG(INFO) << "Start run PIJit";
  JitSyntaxLevelScope jit_syntax_level_scope;
  StaticAnalysisExceptionCleaner exception_cleaner;
  CaptureContext::DisableScope compiler_disable_scope;
  InitCompileConfig();

  if (!c->conf()->GetBoolConfig(GraphJitConfig::kCompileWithTry)) {
    return JitCompile(tstate, c);
  }

  bool compiled = false;
  try {
    compiled = JitCompile(tstate, c);
  } catch (pijit::GraphBreakException &e) {
    throw;  // Throw the exception to the python side.
  } catch (std::exception &e) {
    MS_LOG(ERROR) << "got an unexpected c++ error [" << e.what() << "]";
  }
  if (PyErr_Occurred()) {
    MS_LOG(ERROR) << "got an unexpected python error [" << py::error_already_set().what() << "]";
    PyErr_Clear();
    compiled = false;
  }
  return compiled;
}

static py::object CodeHook(PyThreadState *tstate, JitCompileResults *c, PyFrameWrapper frame) {
  MS_EXCEPTION_IF_NULL(c);
  PyCodeObject *co = PyFrameWrapper(frame).GetCode().ptr();
  bool just_compiled = false;
  switch (c->stat()) {
    case JitCompileResults::NEVER_COMPILE:
      break;
    case JitCompileResults::GRAPH_CAPTURED:
      if (c->conf()->GetBoolConfig(GraphJitConfig::kInterpretCapturedCode)) {
        break;
      }
    /* fallthrough */
    case JitCompileResults::GRAPH_CANDIDATE:
      MS_EXCEPTION_IF_CHECK_FAIL(c->origin_frame().frame() == nullptr || c->origin_frame().frame() == frame.frame(),
                                 "check recursive call compiling function");
      c->set_origin_frame(frame);
      if (!JitCompileWithTry(tstate, c)) {
        c->set_stat(JitCompileResults::NEVER_COMPILE);
        break;
      }
      just_compiled = true;
    /* fallthrough */
    case JitCompileResults::GRAPH_CALLABLE: {
      if (CheckGuard(c, PyFrameWrapper(frame))) {
        c->set_origin_frame(PyFrameWrapper());
        return CallCompiledResults(tstate, PyFrameWrapper(frame), c);
      }
      if (c->stat() == JitCompileResults::NEVER_COMPILE) {
        break;
      }
      if (!just_compiled) {
        c->set_stat(JitCompileResults::GRAPH_CANDIDATE);
        PIJIT_DEBUG_LOG(LogCfg::kRecompiles)
          << "Recompile func: " << std::string(py::str(reinterpret_cast<PyObject *>(co)));
        return CodeHook(tstate, c, frame);
      }
      MS_LOG(INTERNAL_EXCEPTION) << "shouldn't reach here";
    }
    case JitCompileResults::GRAPH_BUILDING:
      MS_LOG(ERROR) << "recursive call, compiler call the code "
                    << std::string(py::str(reinterpret_cast<PyObject *>(co))) << " which is compiling";
      break;
    default:
      MS_LOG(INTERNAL_EXCEPTION) << "shouldn't reach here";
      break;
  }
  MS_LOG(INFO) << "Fall back to python execute: " << ToString(PyFrameWrapper(frame).GetCode());
  c->set_origin_frame(PyFrameWrapper{});
  PyObject *res = _PyEval_EvalFrameDefault(tstate, frame.frame(), 0);
  return py::reinterpret_steal<py::object>(res);
}

PyObject *CallCodeHook(PyThreadState *tstate, PyFrameWrapper f, JitCompileResults *c) {
  py::object res;
  try {
    res = CodeHook(tstate, c, f);
  } catch (pijit::GraphBreakException &e) {
    e.set_error();
  } catch (py::error_already_set &e) {
    e.restore();
  } catch (py::builtin_exception &e) {
    e.set_error();
  } catch (std::exception &e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
  }
  return res.inc_ref().ptr();
}

py::list CollectGradientArguments(PyCodeObject *co, PyObject *const *fast_locals) {
  py::list arguments;
  bool has_va;
  bool has_kw_va;
  auto argc = PyCodeWrapper(co).ArgCount(&has_va, &has_kw_va);
  argc = argc - has_va - has_kw_va;

  // Collect Positional Arguments
  for (int index = 1; index < argc; index++) {
    arguments.append(py::cast<py::object>(fast_locals[index]));
  }

  // Collect Variable Arguments
  if (has_va) {
    auto var_args = py::cast<py::tuple>(fast_locals[argc++]);
    std::for_each(var_args.begin(), var_args.end(), [&arguments](const auto &arg) { arguments.append(arg); });
  }

  // Collect Variable Arguments
  if (has_kw_va) {
    auto kw_args = py::cast<py::dict>(fast_locals[argc++]);
    std::for_each(kw_args.begin(), kw_args.end(), [&arguments](const auto &item) { arguments.append(item.second); });
  }

  return arguments;
}

PyObject *EvalFrame(PY_FRAME_EVAL_FUNCTION_SIGNATURE) {
#ifdef PY_FRAME_EVAL_FUNCTION_DECLARE_THREAD_STATE
  PY_FRAME_EVAL_FUNCTION_DECLARE_THREAD_STATE();
#endif
  // exception handler
  if (exc != 0) {
    return _PyEval_EvalFrameDefault(ts, f, exc);
  }
  return PyFrameEvalHookManager::GetInstance()->RunHook(ts, PyFrameWrapper(f));
}
}  // namespace pijit
}  // namespace mindspore

namespace mindspore {
constexpr auto kEnableDynamic = "__enable_dynamic__";

#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION >= 7) && (PY_MINOR_VERSION <= 11)

py::bool_ pi_jit_enable() {
  PyInterpreterState *inter = PyInterpreterState_Main();
  _PyFrameEvalFunction prev = _PyInterpreterState_GetEvalFrameFunc(inter);
  _PyFrameEvalFunction def = _PyEval_EvalFrameDefault;
  if (prev != def) {
    return false;
  }
  mindspore::pijit::ensureInitialize();
  _PyInterpreterState_SetEvalFrameFunc(inter, mindspore::pijit::EvalFrame);
  return true;
}

py::bool_ pi_jit_disable() {
  PyInterpreterState *inter = PyInterpreterState_Main();
  _PyFrameEvalFunction prev = _PyInterpreterState_GetEvalFrameFunc(inter);
  _PyFrameEvalFunction def = _PyEval_EvalFrameDefault;
  if (prev != mindspore::pijit::EvalFrame) {
    return false;
  }
  _PyInterpreterState_SetEvalFrameFunc(inter, def);
  return true;
}

bool pi_jit_should_compile(const py::handle &func_handle) {
  PyObject *func = func_handle.ptr();
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
  mindspore::pijit::JitCompileResults *c = mindspore::pijit::CreateJitCompileResults(code);
  MS_LOG(DEBUG) << "mark to compile " << std::string(py::str(code));

  if (c == nullptr) {
    return false;
  }
  if (PyObject_HasAttrString(func, kEnableDynamic)) {
    PyObject *enable_dynamic_dict = PyObject_GetAttrString(func, kEnableDynamic);
    MS_LOG(INFO) << "Set enable_dynamic: " << std::string(py::str(enable_dynamic_dict));
    c->set_enable_dynamic_dict(py::cast<py::object>(enable_dynamic_dict));
  }

  auto new_config = mindspore::pijit::GraphJitConfig();
  if (c->stat() != mindspore::pijit::JitCompileResults::NEVER_COMPILE) {
    *c->conf() = new_config;
    return true;
  }

  pijit::PyCodeWrapper co(code);
  py::object bytes = co.Code();

  auto raw_code_size = PyBytes_GET_SIZE(bytes.ptr());
  std::string raw_func_info_name = py::str(code).cast<std::string>();
  std::string raw_func_name = "";
  if (PyFunction_Check(func)) {
    const char *module_name = PyUnicode_AsUTF8(PyFunction_GET_MODULE(func));
    const char *s = strchr(module_name, '.');
    std::string top_module = s ? std::string(module_name, s - module_name) : module_name;
    mindspore::pijit::kPIJitConfigDefault.AddAllowedInlineModules(top_module);

    raw_func_name = mindspore::pijit::Utils::GetPyName(reinterpret_cast<PyFunctionObject *>(func)->func_qualname);
  }

  c->set_stat(mindspore::pijit::JitCompileResults::GRAPH_CANDIDATE);
  *c->conf() = new_config;
  *c->tbs() = mindspore::pijit::Traceback(raw_func_name, raw_func_info_name, raw_code_size);
  return true;
}
#else

py::bool_ pi_jit_enable() {
  MS_LOG(ERROR) << "PiJit not support this python version " << PY_MAJOR_VERSION << '.' << PY_MINOR_VERSION
                << " only support on python3.7, python3.8, python3.9, python3.10";
  return py::bool_(false);
}
py::bool_ pi_jit_disable() { return py::bool_(false); }
py::bool_ pi_jit_should_compile(const py::object &func_handle) { return py::bool_(false); }

#endif

static py::object ConvertCodeExtra(mindspore::pijit::JitCompileResults *c) {
  if (c->code() == nullptr) {
    return py::object();
  }
  PyCodeObject *compiled_code = c->code()->GetPythonCode();
  auto compiled_func = c->code()->GetNativeFunc();
  auto guard = c->code()->GetGuard();
  if (compiled_func == nullptr && compiled_code == nullptr) {
    return py::object();
  }
  py::dict code;
  if (compiled_code != nullptr) {
    PyDict_SetItemString(code.ptr(), "compiled_code_", reinterpret_cast<PyObject *>(compiled_code));
  }
  if (compiled_func != nullptr) {
    PyDict_SetItemString(code.ptr(), "phase_", py::str(c->code()->GetPhase()).ptr());
  }
  if (guard != nullptr && !guard->IsEmpty()) {
    PyDict_SetItemString(code.ptr(), "guard_", py::str(guard->ToString()).ptr());
  }
  PyDict_SetItemString(code.ptr(), "call_count_", py::int_(c->code()->Count()).ptr());
  return code;
}

py::object get_code_extra(const py::object &func) {
  py::object code = mindspore::pijit::GetPyCodeObject(func);
  if (code.ptr() == nullptr) {
    return py::none();
  }
  auto c = mindspore::pijit::GetJitCompileResults(code.ptr());
  if (c == nullptr) {
    return py::none();
  }

  constexpr const char *stat_str[] = {
    "NEVER_COMPILE", "GRAPH_CANDIDATE", "GRAPH_CAPTURED", "GRAPH_BUILDING", "GRAPH_CALLABLE",
  };

  py::dict result;
  py::object compiled_code = ConvertCodeExtra(c);
  if (compiled_code.ptr() != nullptr) {
    PyDict_SetItemString(result.ptr(), "code", compiled_code.ptr());
  }
  PyDict_SetItemString(result.ptr(), "stat", py::str(stat_str[c->stat()]).ptr());
  PyDict_SetItemString(result.ptr(), "compile_count_", py::int_(c->compile_count()).ptr());
  PyDict_SetItemString(result.ptr(), "break_count_", py::int_(c->break_count()).ptr());
  return result;
}

// op must be PyCFunctionObject
static void *GetCFunctionId(PyObject *op) {
  void *result = reinterpret_cast<void *>(PyCFunction_GET_FUNCTION(op));
  PyObject *self = PyCFunction_GET_SELF(op);
  if (self == nullptr || !PyCapsule_CheckExact(self)) {
    return result;
  }
  PyObject *m = (reinterpret_cast<PyCFunctionObject *>(op))->m_module;
  constexpr std::string_view ms_name("mindspore");
  if (m == nullptr || !PyUnicode_Check(m) ||
      ms_name.compare(0, ms_name.size(), PyUnicode_AsUTF8(m), 0, ms_name.size()) != 0) {
    MS_LOG(DEBUG) << "got capsule method which is not mindspore defined: " << py::str(op) << " addr: " << op
                  << " func_ptr: " << result << " module is: " << (m ? std::string(py::str(m)) : "<NULL>")
                  << " capsule name: " << (PyCapsule_GetName(self) ? PyCapsule_GetName(self) : "<NULL>");
    return result;
  }
  const char *doc = (reinterpret_cast<PyCFunctionObject *>(op))->m_ml->ml_doc;
  const char *end = strchr(doc, '\n');
  MS_LOG(DEBUG) << "got mindspore defined capsule method: " << py::str(op) << " addr: " << op << " func_ptr: " << result
                << " module is: " << PyUnicode_AsUTF8(m)
                << " capsule name: " << (PyCapsule_GetName(self) ? PyCapsule_GetName(self) : "<NULL>")
                << " first line doc is: " << (end ? std::string(doc, end) : "<NULL>");

  // pybind signature regex `function_name(args...) -> return\n`
  if (!end || !std::regex_match(doc, end, std::regex("^\\w+\\(.*\\) *->[\\( ]*\\w+.*"))) {
    // cpython use `function_name(args...)\n--\n\n` as `builtin_func.__text_signature__`
    MS_LOG(DEBUG) << "mindspore function without signature";  // maybe not pybind11 defined
  }
  return op;
}

size_t FunctionId(const py::object &callable) {
  PyObject *op = callable.ptr();
  // limit max reference depth
  for (int limit = 10; limit && (PyMethod_Check(op) || PyInstanceMethod_Check(op)); --limit) {
    op = PyMethod_Check(op) ? PyMethod_GET_FUNCTION(op) : op;
    op = PyInstanceMethod_Check(op) ? PyInstanceMethod_GET_FUNCTION(op) : op;
  }
  void *result = op;
  if (PyCFunction_Check(op)) {
    // types.BuiltinFunctionType = type(len) same as types.BuiltinMethodType = type(list().append)
    result = GetCFunctionId(op);
  } else if (Py_IS_TYPE(op, &PyMethodDescr_Type)) {
    // types.MethodDescriptorType = type(list.append)
    PyCFunction func = reinterpret_cast<PyMethodDescrObject *>(op)->d_method->ml_meth;
    result = reinterpret_cast<void *>(func);
  } else if (Py_IS_TYPE(op, &PyWrapperDescr_Type)) {
    // types.WrapperDescriptorType = type(object.__init__)
    result = reinterpret_cast<PyWrapperDescrObject *>(op)->d_wrapped;
  } else if (Py_IS_TYPE(op, &_PyMethodWrapper_Type)) {
    // types.WrapperDescriptorType = type(object().__str__)
    PyObject *self = PyObject_GetAttrString(op, "__self__");
    PyObject *attr = PyObject_GetAttrString(op, "__name__");
    PyObject *descr = PyObject_GetAttr(reinterpret_cast<PyObject *>(Py_TYPE(self)), attr);
    result = reinterpret_cast<PyWrapperDescrObject *>(descr)->d_wrapped;
    Py_DECREF(self);
    Py_DECREF(attr);
    Py_DECREF(descr);
  }
  return reinterpret_cast<size_t>(result);
}

void PIJitSetContext(py::args va, py::kwargs kw) {
  auto ctx = pijit::CaptureContext::GetInstance();
  ctx->SetContext(va, kw);
}

bool ClearJitCompileResults(const py::handle &func) {
  PyObject *code = func.ptr();
  code = PyMethod_Check(code) ? PyMethod_GET_FUNCTION(code) : code;
  code = PyFunction_Check(code) ? PyFunction_GET_CODE(code) : code;
  return pijit::JitCompileResults::Clear(reinterpret_cast<PyCodeObject *>(code));
}
}  // namespace mindspore
