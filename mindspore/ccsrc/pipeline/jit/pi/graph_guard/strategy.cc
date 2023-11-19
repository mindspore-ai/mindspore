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
#include "pipeline/jit/pi/graph_guard/strategy.h"
#include "pipeline/jit/ps/pipeline.h"

namespace mindspore {
namespace jit {
namespace graph {

OptStrategy::ExecKind OptStrategy::MakeExecStrategyByPerf(OptPerfPtr graph_perf, OptPerfPtr pynative_perf, int count,
                                                          double adj_coef) {
  PerfStatisticsPtr graph_stat = graph_perf->GetStatistics();
  PerfStatisticsPtr pynative_stat = graph_perf->GetStatistics();
  if (graph_stat->GetTotalCount() < count) {
    return ExecKind::kExecGraph;
  } else if (pynative_stat->GetTotalCount() < count) {
    return ExecKind::kExecPyNative;
  } else {
    if (graph_stat->GetAverageDuration() * (1 + adj_coef) > pynative_stat->GetAverageDuration()) {
      return ExecKind::kExecPyNative;
    } else {
      return ExecKind::kExecGraph;
    }
  }
}

OptStrategy::ExecKind OptStrategy::MakeExecStrategyByComplex(PyCodeObject *co, int threshold) {
  // currently just use instruction count to judge whether to use graph build
  // later it need cost model to make judgement here
  if (co != nullptr && static_cast<int>(PyBytes_GET_SIZE(co->co_code) / sizeof(_Py_CODEUNIT)) < threshold) {
    return ExecKind::kExecPyNative;
  } else {
    return ExecKind::kExecGraph;
  }
}

static bool CompareOptCodeByCount(OptCodePtr a, OptCodePtr b) {
  if (a->Count() > b->Count()) {
    return false;
  } else {
    return true;
  }
}

void OptStrategy::MakeGCStrategy(OptCodeHubPtr hub, int limit_size, int limit_count) {
  if (limit_size <= 0 || limit_count <= 0) {
    return;
  }
  std::vector<OptCodeSet> vec = hub->GetAllOptTarget();
  for (auto set : vec) {
    std::sort(set.begin(), set.end(), CompareOptCodeByCount);
    if (limit_count > 0) {
      if (set.size() > (size_t)limit_count) {
        OptCodeSet toDel;
        toDel.insert(toDel.begin(), set.begin() + limit_count, set.end());
        for (auto item : toDel) {
          hub->DelOptTarget(item);
        }
      }
    }
    if (limit_size > 0) {
      auto graph_executor = mindspore::pipeline::GraphExecutorPy::GetInstance();
      OptCodeSet toDel;
      for (auto item : set) {
        if (limit_size == 0) {
          toDel.push_back(item);
        }
        std::string phase = item->GetPhase();
        if (phase.size() > 0) {
          FuncGraphPtr ms_func_graph = graph_executor->GetFuncGraph(phase);
          int node_count = (int)(ms_func_graph->nodes().size());
          for (auto fg : ms_func_graph->func_graphs_used_total()) {
            node_count += (int)(fg->nodes().size());
          }
          if (limit_size > node_count) {
            limit_size -= node_count;
          } else {
            limit_size = 0;
          }
        }
      }
      for (auto item : toDel) {
        hub->DelOptTarget(item);
      }
    }
  }
}

}  // namespace graph
}  // namespace jit
}  // namespace mindspore
