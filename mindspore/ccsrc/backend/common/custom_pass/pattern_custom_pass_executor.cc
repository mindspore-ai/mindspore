/**
 * Copyright 2025-2026 Huawei Technologies Co., Ltd
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

#include "include/backend/common/custom_pass/pattern_custom_pass_executor.h"

#include <memory>
#include <string>
#include "include/utils/custom_pass/custom_pass_plugin_manager.h"
#include "include/backend/common/pass_manager/graph_optimizer.h"
#include "include/utils/anfalgo.h"
#include "utils/log_adapter.h"
#include "include/utils/utils.h"

#ifdef ENABLE_DUMP_IR
#include "include/utils/ir_dump/anf_ir_dump_interface.h"
#endif

namespace mindspore {
namespace opt {
namespace {
constexpr auto kDeviceAll = "all";
}

void PatternCustomPassExecutor::ExecutePatternCustomPasses(const KernelGraphPtr &graph,
                                                           const std::string &device_target) {
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
  auto ordered_pass_manager = std::make_shared<PassManager>("custom_ordered_passes");

  // Register device-specific custom passes
  ProcessPasses(
    [device_target](const CustomPassPluginManager::PassExecution &pass_exec) {
      // Check device compatibility using the recorded device for this specific call
      if (pass_exec.device != kDeviceAll && pass_exec.device != device_target) {
        MS_LOG(DEBUG) << "Skipping pass '" << pass_exec.pass_name << "' - device '" << pass_exec.device
                      << "' incompatible with target device '" << device_target << "'";
        return false;
      }

      // Check if this is a backend custom pass
      if (pass_exec.pass_type != CustomPassType::kPattern) {
        MS_LOG(DEBUG) << "Pass '" << pass_exec.pass_name << "' has pass_type '" << pass_exec.pass_type
                      << "' Skipping backend optimizer registration.";
        return false;
      }

      return true;
    },
    [&ordered_pass_manager](const std::string &pass_name, const std::shared_ptr<CustomPassPlugin> &plugin) {
      // Create and add pass inline
      auto pass = plugin->CreatePass(pass_name);
      if (pass == nullptr) {
        MS_LOG(ERROR) << "Failed to create pass '" << pass_name << "' from plugin '" << plugin->GetPluginName() << "'";
        return;
      }

      ordered_pass_manager->AddPass(pass);
      MS_LOG(INFO) << "Added pass '" << pass_name << "' from plugin '" << plugin->GetPluginName()
                   << "' in execution order position " << ordered_pass_manager->Passes().size();
    });

  if (ordered_pass_manager->Passes().empty()) {
    MS_LOG(INFO) << "No passes registered for device: " << device_target;
  } else {
    opt->AddPassManager(ordered_pass_manager);
    MS_LOG(INFO) << "Added ordered pass manager with " << ordered_pass_manager->Passes().size()
                 << " passes for device: " << device_target;
    (void)opt->Optimize(graph);
  }

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
