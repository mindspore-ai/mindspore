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
#include "frontend/jit/pi/pi_jit_config.h"
#include <string>
#include <unordered_map>
#include <vector>
#include "utils/log_adapter.h"
#include "frontend/jit/pi/external.h"
#include "frontend/jit/pi/utils/utils.h"
#include "frontend/jit/pi/capture_context.h"
#include "frontend/jit/pi/python_adapter/pydef.h"

namespace mindspore {
namespace pijit {

GraphJitConfig kPIJitConfigDefault;

constexpr int kDefaultMaxTraceDepth = 16;

constexpr const char *kModuleName = "mindspore.graph._pijit.pijit_func_white_list";
constexpr const char *kFuncMapName = "_func_map";
constexpr const char *kGuardFuncMapName = "guard_func_map";

static const std::unordered_map<std::string, bool (GraphJitConfig::*)(PyObject *)> key_map = {
  {"auto_jit_func_filter", &GraphJitConfig::SetAutoJitFilter},
  {"auto_jit_cell", &GraphJitConfig::SetBool<GraphJitConfig::kAutoJitCell>},
  {"print_bb", &GraphJitConfig::SetBool<GraphJitConfig::kPrintBB>},
  {"interpret_captured_code", &GraphJitConfig::SetBool<GraphJitConfig::kInterpretCapturedCode>},
  {"compile_with_try", &GraphJitConfig::SetBool<GraphJitConfig::kCompileWithTry>},
  {"specialize_scalar", &GraphJitConfig::SetBool<GraphJitConfig::kGuardSpecializeScalar>},
  {"specialize_container", &GraphJitConfig::SetBool<GraphJitConfig::kGuardSpecializeContainer>},
  {"specialize_tensor", &GraphJitConfig::SetBool<GraphJitConfig::kGuardSpecializeTensor>},
  {"loop_unrolling", &GraphJitConfig::SetBool<GraphJitConfig::kLoopUnrolling>},
  {"infer_only", &GraphJitConfig::SetBool<GraphJitConfig::kInferOnly>},
  {"strict_trace", &GraphJitConfig::SetBool<GraphJitConfig::kStrictTrace>},
  {"perf_statistics", &GraphJitConfig::SetBool<GraphJitConfig::kPerfStatistics>},
  {"LOG_PERF", &GraphJitConfig::SetBool<GraphJitConfig::kLogPerf>},
  {"LOG_GUARD_PERF", &GraphJitConfig::SetBool<GraphJitConfig::kLogGuardPerf>},
  {"enable_dynamic_shape", &GraphJitConfig::SetBool<GraphJitConfig::kEnableDynamicShape>},
  {"expand_graph_input", &GraphJitConfig::SetBool<GraphJitConfig::kExpandGraphInput>},
  {"expand_graph_output", &GraphJitConfig::SetBool<GraphJitConfig::kExpandGraphOutput>},
  {"eliminate_redundant_args", &GraphJitConfig::SetBool<GraphJitConfig::kEliminateRedundantArgs>},
  {"subgraph_break_opt", &GraphJitConfig::SetBool<GraphJitConfig::kSubgraphBreakOpt>},
  {"fullgraph", &GraphJitConfig::SetBool<GraphJitConfig::kFullGraph>},
  {"enable_old_guard_strategy", &GraphJitConfig::SetBool<GraphJitConfig::kEnableOldGuardStrategy>},
  {"tensor_setitem_side_effect_opt", &GraphJitConfig::SetBool<GraphJitConfig::kTensorSetitemSideEffectOpt>},
  // kEnableOptimizeForAttrItem
  {"_symbolic", &GraphJitConfig::SetInt<GraphJitConfig::kSymbolic>},
  {"MAX_TRACE_DEPTH", &GraphJitConfig::SetInt<GraphJitConfig::kMaxTraceDepth>},
  {"STATIC_GRAPH_BYTECODE_MIN", &GraphJitConfig::SetInt<GraphJitConfig::kStaticGraphBytecodeMin>},
  {"PERF_STATISTICS_COUNT", &GraphJitConfig::SetInt<GraphJitConfig::kPerfStatisticsCount>},
  {"PERF_STATISTICS_SCALE_10000X", &GraphJitConfig::SetInt<GraphJitConfig::kPerfStatisticsScale10000x>},
  {"limit_graph_size", &GraphJitConfig::SetInt<GraphJitConfig::kLimitGraphSize>},
  {"limit_graph_count", &GraphJitConfig::SetInt<GraphJitConfig::kLimitGraphCount>},
  {"relax_guard_count", &GraphJitConfig::SetInt<GraphJitConfig::kGuardRelaxCount>},
  {"allowed_inline_modules", &GraphJitConfig::AddAllowedInlineModules},
  {"pijit_forbidden", &GraphJitConfig::AddJitForbidden},
  {"pijit_constexpr", &GraphJitConfig::AddJitConstexpr},
  {"relax_guard_func", &GraphJitConfig::AddJitRelaxGuard},
  {"jit_level", &GraphJitConfig::AddJitLevel},
  {"recapture_loop_body", &GraphJitConfig::SetBool<GraphJitConfig::kReCaptureLoopBody>},
};

static const std::unordered_map<std::string, LogCfg> key_to_log_map = {
  {"print_after_all", LogCfg::kAll},
  {"print_bytecode", LogCfg::kBytecode},
  {"print_guard", LogCfg::kGuard},
  {"LOG_GRAPH_BREAK", LogCfg::kGraphBreak},
};

GraphJitConfig::GraphJitConfig() : int_conf{0}, bool_conf{false} {
  bool_conf[kAutoJitCell - kBoolConf] = false;
  bool_conf[kPrintBB - kBoolConf] = false;
  bool_conf[kInterpretCapturedCode - kBoolConf] = false;
  bool_conf[kCompileWithTry - kBoolConf] = true;
  bool_conf[kGuardSpecializeScalar - kBoolConf] = true;
  bool_conf[kGuardSpecializeContainer - kBoolConf] = false;
  bool_conf[kGuardSpecializeTensor - kBoolConf] = false;
  bool_conf[kLoopUnrolling - kBoolConf] = true;
  bool_conf[kSkipException - kBoolConf] = false;
  bool_conf[kInferOnly - kBoolConf] = true;
  bool_conf[kStrictTrace - kBoolConf] = true;
  bool_conf[kPerfStatistics - kBoolConf] = false;
  bool_conf[kLogPerf - kBoolConf] = false;
  bool_conf[kLogGuardPerf - kBoolConf] = false;
  bool_conf[kEnableDynamicShape - kBoolConf] = false;
  bool_conf[kExpandGraphInput - kBoolConf] = true;
  bool_conf[kExpandGraphOutput - kBoolConf] = true;
  bool_conf[kEliminateRedundantArgs - kBoolConf] = false;
  bool_conf[kSubgraphBreakOpt - kBoolConf] = true;
  bool_conf[kReCaptureLoopBody - kBoolConf] = false;
  bool_conf[kFullGraph - kBoolConf] = false;
  bool_conf[kTensorSetitemSideEffectOpt - kBoolConf] = false;

  int_conf[kMaxTraceDepth - kIntConf] = kDefaultMaxTraceDepth;
  int_conf[kStaticGraphBytecodeMin - kIntConf] = 0;
  int_conf[kPerfStatisticsCount - kIntConf] = 1;
  int_conf[kPerfStatisticsScale10000x - kIntConf] = 1000;
  int_conf[kLimitGraphSize - kIntConf] = 0;
  int_conf[kLimitGraphCount - kIntConf] = 0;
  int_conf[kGuardRelaxCount - kIntConf] = 0;
  int_conf[kSymbolic - kIntConf] = 0;

  AddAllowedInlineModules("mindspore");

  jit_level = "O0";
}

static py::object GetObjectsMap() {
  py::str mod_name("mindspore");
  py::str key_name("<pijit.registry>");
  // can't import module while the module is deallocated
  py::object ms = py::reinterpret_steal<py::object>(PyImport_GetModule(mod_name.ptr()));
  if (ms.ptr() == nullptr || !PyModule_Check(ms.ptr())) {
    return py::object();
  }
  PyObject *registry = PyObject_GetAttr(ms.ptr(), key_name.ptr());
  if (registry != nullptr) {
    MS_EXCEPTION_IF_CHECK_FAIL(PyDict_CheckExact(registry), "got duplicate attribute for <pijit.registry>");
    return py::reinterpret_steal<py::object>(registry);
  }
  PyErr_Clear();

  // just set once, module reload will not rewrite attribute.
  static bool init = false;
  if (init) {
    return py::object();
  }
  init = true;
  registry = PyDict_New();
  PyObject_SetAttr(ms.ptr(), key_name.ptr(), registry);
  return py::reinterpret_steal<py::object>(registry);
}

static bool AddToFuncMap(PyObject *list, const std::string &map_name, const std::string &key) {
  py::object func_map = Utils::GetModuleAttr(kModuleName, map_name, true, true);
  py::object key_object = Utils::GetModuleAttr(kModuleName, key, true, true);
  for (const py::handle &i : py::iter(list)) {
    if (!PyCallable_Check(i.ptr())) {
      return false;
    }
    py::int_ id = FunctionId(py::reinterpret_borrow<py::object>(i));
    PyDict_SetItem(func_map.ptr(), id.ptr(), key_object.ptr());
  }
  return true;
}

bool GraphJitConfig::AddJitForbidden(PyObject *list) {
  return AddToFuncMap(list, kFuncMapName, "FUNC_KEY_PIJIT_FORBIDDEN");
}

bool GraphJitConfig::AddJitLevel(PyObject *str) {
  if (py::isinstance<py::str>(str)) {
    py::str jit_level_obj = py::cast<py::str>(str);
    auto jit_level_str = py::cast<std::string>(jit_level_obj);
    if (jit_level_str != "O0" && jit_level_str != "O1" && jit_level_str != "O2") {
      return false;
    }
    jit_level = jit_level_str;
    return true;
  }
  return false;
}

std::string GraphJitConfig::getJitLevel() const { return jit_level; }

bool GraphJitConfig::AddJitConstexpr(PyObject *list) {
  return AddToFuncMap(list, kFuncMapName, "FUNC_KEY_PIJIT_CONSTEXPR");
}

bool GraphJitConfig::AddJitRelaxGuard(PyObject *list) {
  return AddToFuncMap(list, kGuardFuncMapName, "GUARD_KEY_RELAX_FUNC");
}

bool GraphJitConfig::AddAllowedInlineModules(PyObject *list) {
  py::object l = py::reinterpret_borrow<py::object>(list);
  for (const auto &i : py::iter(l)) {
    const char *name = nullptr;
    if (PyUnicode_Check(i.ptr())) {
      name = PyUnicode_AsUTF8(i.ptr());
    } else if (PyModule_Check(i.ptr())) {
      name = PyModule_GetName(i.ptr());
    } else {
      continue;
    }
    if (name == nullptr) {
      PyErr_Clear();
      continue;
    }
    AddAllowedInlineModules(name);
  }
  return true;
}

void GraphJitConfig::AddAllowedInlineModules(const std::string &module_name) {
  CaptureContext::GetInstance()->AddKnownModule(module_name);
}

const std::set<std::string> &GraphJitConfig::allowed_inline_modules() const {
  return CaptureContext::GetInstance()->known_modules();
}

bool GraphJitConfig::SetAutoJitFilter(PyObject *callable) {
  if (!PyCallable_Check(callable)) {
    MS_LOG(WARNING) << "PIJit option 'auto_jit_func_filter' only accept callable, but got "
                    << std::string(py::str(callable));
    return false;
  }
  py::object map = GetObjectsMap();
  if (map.ptr() == nullptr) {
    return false;
  }
  (void)SetBool<kAutoJit>(Py_True);
  PyDict_SetItemString(map.ptr(), "<auto jit filter>", callable);
  return true;
}

bool GraphJitConfig::ShouldAutoJit(PyFrameWrapper f) {
  if (!GetBoolConfig(kAutoJit)) {
    return false;
  }
  py::object map = GetObjectsMap();
  if (map.ptr() == nullptr) {
    // mindspore module is unload
    (void)SetBool<kAutoJit>(Py_False);
    return false;
  }
  PyObject *filter = PyDict_GetItemString(map.ptr(), "<auto jit filter>");
  if (filter == nullptr) {
    (void)SetBool<kAutoJit>(Py_False);
    return false;
  }
  PyObject *arg = reinterpret_cast<PyObject *>(f.GetCode().ptr());
  PyObject *res = PyObject_Vectorcall(filter, &arg, 1, nullptr);
  if (PyErr_Occurred()) {
    MS_LOG(ERROR) << "***" << py::error_already_set().what() << "*** at " << std::string(py::str(filter)) << " ignored";
    PyErr_Clear();
    (void)SetBool<kAutoJit>(Py_False);
    return false;
  }
  Py_DECREF(res);
  return res == Py_True;
}

GraphJitConfig::GraphJitConfig(const py::object &c) {
  *this = kPIJitConfigDefault;
  this->Update(c);
}

void GraphJitConfig::Update(const py::object &c) {
  (void)c.cast<py::dict>();
  PyObject *key;
  PyObject *value;
  Py_ssize_t pos = 0;
  while (PyDict_Next(c.ptr(), &pos, &key, &value)) {
    if (PyUnicode_Check(key)) {
      const char *k = PyUnicode_AsUTF8(key);
      auto iter = key_map.find(k);
      if (iter != key_map.end() && (this->*(iter->second))(value)) {
        continue;
      }
      auto log_iter = key_to_log_map.find(k);
      if (log_iter != key_to_log_map.end()) {
        MS_LOG(WARNING) << "For 'jit_config', the parameter '" << k
                        << "' has been deprecated. Please use the "
                           "environment variable 'MS_JIT_BYTECODE_LOGS' instead. For more details, please refer to "
                           "https://www.mindspore.cn/docs/en/master/api_python/env_var_list.html.";
        g_pijit_log_conf[static_cast<int>(log_iter->second)] = value;
        continue;
      }
    }
    MS_LOG(WARNING) << "Unknown PIJit option: " << std::string(py::str(key)) << ":" << std::string(py::str(value));
  }

  // Log config
  std::stringstream jit_log(common::GetEnv("MS_JIT_BYTECODE_LOGS"));
  std::vector<std::string> tokens;
  std::string token;

  while (std::getline(jit_log, token, ',')) {
    token.erase(0, token.find_first_not_of(" "));
    token.erase(token.find_last_not_of(" ") + 1);
    if (!token.empty()) {
      tokens.push_back(token);
    }
  }

  for (const auto &t : tokens) {
    auto it = g_pijit_log_map.find(t);
    if (it != g_pijit_log_map.end()) {
      g_pijit_log_conf[static_cast<int>(it->second)] = true;
      MS_LOG(DEBUG) << it->first << "=true";
    }

    if (t == "recompiles_verbose") {
      auto iter = g_pijit_log_map.find("recompiles");
      if (iter != g_pijit_log_map.end()) {
        g_pijit_log_conf[static_cast<int>(iter->second)] = true;
        MS_LOG(DEBUG) << iter->first << "=true";
      }
    }
  }
}

static void ReplaceMethod(const py::object &cls, PyMethodDef *mdef, const char *save_name, bool enable) {
  py::object func = cls.attr(mdef->ml_name);
  bool is_hook = false;
  if (Py_IS_TYPE(func.ptr(), &PyMethodDescr_Type)) {
    is_hook = reinterpret_cast<PyMethodDescrObject *>(func.ptr())->d_method->ml_meth == mdef->ml_meth;
  }
  if (enable && !is_hook) {
    PyTypeObject *tp = reinterpret_cast<PyTypeObject *>(cls.ptr());
    py::object hook = py::reinterpret_steal<py::object>(PyDescr_NewMethod(tp, mdef));
    cls.attr(mdef->ml_name) = hook;
    cls.attr(save_name) = func;
  }
  if (!enable && is_hook) {
    cls.attr(mdef->ml_name) = cls.attr(save_name);
    py::delattr(cls, save_name);
  }
}

void GraphJitConfig::ApplyAutoJitCell() {
  static constexpr const char *name = "__call__";
  static constexpr const char *save_name = "_old__call__";
  static const PyCFunctionWithKeywords CellForward = [](PyObject *self, PyObject *vargs, PyObject *kwargs) {
    PyObject *construct = PyObject_GetAttrString(self, "construct");
    py::object handle = py::reinterpret_steal<py::object>(construct);
    if (construct != nullptr) {
      (void)pi_jit_should_compile(handle);
    } else {
      PyErr_Clear();
    }

    PyObject *func = PyObject_GetAttrString(self, save_name);
    PyObject *ret = PyObject_Call(func, vargs, kwargs);
    Py_DECREF(func);
    return ret;
  };
  static PyMethodDef mdef = {name, reinterpret_cast<PyCFunction>(CellForward), METH_VARARGS | METH_KEYWORDS, "Hook"};

  bool enable = kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kAutoJitCell);
  py::object cls = Utils::GetModuleAttr("mindspore.nn", "Cell", false, false);
  ReplaceMethod(cls, &mdef, save_name, enable);
}

}  // namespace pijit

void update_pijit_default_config(const py::kwargs &conf) {
  mindspore::pijit::kPIJitConfigDefault.Update(conf);
  mindspore::pijit::GraphJitConfig::ApplyAutoJitCell();
}

}  // namespace mindspore
