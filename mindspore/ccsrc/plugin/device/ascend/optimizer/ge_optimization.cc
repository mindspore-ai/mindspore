/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/ge_optimization.h"

#include <string>
#include <memory>

#include "include/backend/optimizer/optimizer.h"
#include "include/common/debug/anf_ir_dump.h"
#include "plugin/device/ascend/optimizer/mindir/reduce_axis_update.h"
#include "include/backend/debug/profiler/profiling.h"

namespace mindspore {
namespace opt {
void ReduceOptimization(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_LOG(INFO) << "Reduce optimization start, graph: " << func_graph->ToString() << ".";

#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_before_reduce_optimization_graph_" + func_graph->ToString() + ".ir";
    DumpIR(file_name, func_graph);
  }
#endif

  profiler::CollectHostInfo("Ascend", "Graph Optimization", "GeOptimizeGraph_ReduceOptimization", 0, 0, 0);
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>("reduce_optimization_pm");
  MS_EXCEPTION_IF_NULL(pm);
  pm->AddPass(std::make_shared<opt::ReduceAxisUpdate>());
  MS_EXCEPTION_IF_NULL(optimizer);
  optimizer->AddPassManager(pm);

  (void)optimizer->Optimize(func_graph);
  profiler::CollectHostInfo("Ascend", "Graph Optimization", "GeOptimizeGraph_ReduceOptimization", 0, 0, 1);

#ifdef ENABLE_DUMP_IR
  if (context->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_after_reduce_optimization_graph_" + func_graph->ToString() + ".ir";
    DumpIR(file_name, func_graph);
  }
#endif

  MS_LOG(INFO) << "Reduce optimization end.";
}
}  // namespace opt
}  // namespace mindspore
