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
#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_PI_CONFIG_H
#define MINDSPORE_CCSRC_FRONTEND_JIT_PI_CONFIG_H

#include <set>
#include <string>
#include "pybind11/pybind11.h"
#include "frontend/jit/pi/python_adapter/py_frame.h"
#include "frontend/jit/pi/utils/utils.h"

namespace mindspore {
namespace pijit {
namespace py = pybind11;

class GraphJitConfig {
 public:
  enum Options {
    kBoolConf = 0,
    kAutoJitCell,
    kAutoJit,
    kPrintBB,
    kInterpretCapturedCode,
    kCompileWithTry,
    kGuardSpecializeScalar,
    kGuardSpecializeContainer,
    kGuardSpecializeTensor,
    kLoopUnrolling,
    kInferOnly,
    kStrictTrace,
    kPerfStatistics,
    kLogPerf,
    kLogGuardPerf,
    kEnableDynamicShape,
    kSkipException,
    kExpandGraphInput,
    kExpandGraphOutput,
    kEliminateRedundantArgs,
    kReCaptureLoopBody,
    kSubgraphBreakOpt,
    kFullGraph,
    kEnableOldGuardStrategy,
    kTensorSetitemSideEffectOpt,
    /* ------------------------------ */
    kIntConf,
    kSymbolic,
    kMaxTraceDepth,
    kStaticGraphBytecodeMin,
    kPerfStatisticsCount,
    kPerfStatisticsScale10000x,
    kLimitGraphSize,
    kLimitGraphCount,
    kGuardRelaxCount,
    /* ------------------------------ */
    kOptionsCount
  };
  GraphJitConfig();
  explicit GraphJitConfig(const py::object &c);
  bool GetBoolConfig(Options o) const { return o > kBoolConf && o < kIntConf ? bool_conf[o - kBoolConf] : false; }
  int getIntConfig(Options o) const { return o > kIntConf && o < kOptionsCount ? int_conf[o - kIntConf] : 0; }
  const std::set<std::string> &allowed_inline_modules() const;

  bool ShouldAutoJit(PyFrameWrapper f);

  void AddAllowedInlineModules(const std::string &module_name);

  bool SetAutoJitFilter(PyObject *callable);
  bool AddJitRelaxGuard(PyObject *list);
  bool AddJitConstexpr(PyObject *callable_list);
  bool AddJitForbidden(PyObject *callable_list);

  bool AddAllowedInlineModules(PyObject *str_list);
  std::string getJitLevel() const;
  bool AddJitLevel(PyObject *str);

  template <Options o>
  bool SetBool(PyObject *value) {
    static_assert(o > kBoolConf && o < kIntConf);
    bool_conf[o - kBoolConf] = value == Py_True;
    return true;
  }

  template <Options o>
  bool SetInt(PyObject *value) {
    static_assert(o > kIntConf && o < kOptionsCount);
    int res = PyLong_AsLong(value);
    if (PyErr_Occurred()) {
      PyErr_Clear();
      return false;
    }
    int_conf[o - kIntConf] = res;
    return true;
  }

  static void ApplyAutoJitCell();
  void Update(const py::object &c);

 private:
  int int_conf[kOptionsCount - kIntConf];
  bool bool_conf[kIntConf - kBoolConf];
  std::string jit_level;
};

extern GraphJitConfig kPIJitConfigDefault;

}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_JIT_PI_CONFIG_H
