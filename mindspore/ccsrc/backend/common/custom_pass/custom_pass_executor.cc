/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "include/backend/common/custom_pass/custom_pass_executor.h"

#include <memory>
#include <string>
#include "include/backend/common/custom_pass/custom_pass_plugin_manager.h"
#include "include/backend/common/pass_manager/graph_optimizer.h"
#include "include/utils/anfalgo.h"
#include "utils/log_adapter.h"
#include "include/utils/utils.h"

#ifdef ENABLE_DUMP_IR
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#endif

namespace mindspore {
namespace opt {

void CustomPassExecutor::ExecuteCustomPasses(const KernelGraphPtr &graph, const std::string &device_target) {
  MS_EXCEPTION_IF_NULL(graph);
  PROF_START(CustomOptimization);
  MS_LOG(INFO) << "start custom optimization. device: " << device_target << ", graph id: " << graph->graph_id();

#ifdef ENABLE_DUMP_IR
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name =
      "hwopt_" + device_target + "_custom_optimization_before_graph_" + std::to_string(graph->graph_id()) + ".ir";
    DumpIR(file_name, graph);
  }
#endif

  auto opt = std::make_shared<GraphOptimizer>();
  auto &plugin_manager = CustomPassPluginManager::GetInstance();

  // Register device-specific custom passes
  plugin_manager.RegisterPassesToOptimizer(opt, device_target);

  (void)opt->Optimize(graph);

#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name =
      "hwopt_" + device_target + "_custom_optimization_after_graph_" + std::to_string(graph->graph_id()) + ".ir";
    DumpIR(file_name, graph);
  }
#endif

  MS_LOG(INFO) << "end custom optimization. device: " << device_target << ", graph id: " << graph->graph_id();
  PROF_END(CustomOptimization);
}

}  // namespace opt
}  // namespace mindspore
