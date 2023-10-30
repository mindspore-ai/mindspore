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
#include "pipeline/jit/pi_jit/graph_guard/strategy.h"

namespace mindspore {
namespace jit {
namespace graph {

OptStrategy::ExecKind OptStrategy::MakeExecStrategyByPerf(OptPerfPtr graph_perf, OptPerfPtr pynative_perf,
                                                          double adj_coef) {
  PerfStatisticsPtr graph_stat = graph_perf->GetStatistics();
  PerfStatisticsPtr pynative_stat = graph_perf->GetStatistics();
  if (graph_stat->GetAverageDuration() * (1 + adj_coef) > pynative_stat->GetAverageDuration()) {
    return ExecKind::kExecPyNative;
  } else {
    return ExecKind::kExecGraph;
  }
}

OptStrategy::ExecKind OptStrategy::MakeExecStrategyByComplex(PyObject *code, int threshold) {
  PyCodeObject *co = nullptr;
  if (PyCode_Check(code)) {
    co = reinterpret_cast<PyCodeObject *>(code);
  } else if (PyFunction_Check(code)) {
    co = reinterpret_cast<PyCodeObject *>(PyFunction_GET_CODE(code));
  }
  // currently just use instruction count to judge whether to use graph build
  // later it need cost model to make judgement here
  if (co != nullptr && static_cast<int>(PyBytes_GET_SIZE(co->co_code) / sizeof(_Py_CODEUNIT)) < threshold) {
    return ExecKind::kExecPyNative;
  } else {
    return ExecKind::kExecGraph;
  }
}
}  // namespace graph
}  // namespace jit
}  // namespace mindspore
