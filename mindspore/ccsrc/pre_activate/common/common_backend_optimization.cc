/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "pre_activate/common/common_backend_optimization.h"
#include <memory>
#include <string>
#include "pre_activate/common/optimizer.h"
#include "pre_activate/pass/convert_const_input_to_attr.h"
#include "pre_activate/pass/convert_tuple_output_to_maketuple.h"
#include "pre_activate/pass/convert_const_input_to_tensor_input.h"
#include "pre_activate/pass/convert_tuple_input_to_dynamic_input.h"
#include "pre_activate/pass/const_to_attr_strided_slice_grad.h"
#include "utils/context/ms_context.h"
#include "debug/anf_ir_dump.h"

namespace mindspore {
namespace opt {
void BackendCommonOptimization(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  MS_LOG(INFO) << "start common opt graph:" << kernel_graph->graph_id();
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->save_graphs_flag();
  auto save_graphs_path = context_ptr->save_graphs_path();
  if (save_graphs_path.empty()) {
    save_graphs_path = ".";
  }
  if (save_graphs) {
    std::string file_path = save_graphs_path + "/" + "hwopt_common_before.ir";
    DumpIR(file_path, kernel_graph);
  }
  auto optimizer = std::make_shared<GraphOptimizer>();
  auto common_pm = std::make_shared<PassManager>("common_pm");
  common_pm->AddPass(std::make_shared<ConvertConstInputToAttr>());
  common_pm->AddPass(std::make_shared<ConstToAttrStridedSliceGradPass>());
  common_pm->AddPass(std::make_shared<ConvertConstInputToTensorInput>());
  common_pm->AddPass(std::make_shared<ConvertTupleInputToDynamicInput>());
  common_pm->AddPass(std::make_shared<ConvertTupleOutputToMaketuple>());
  optimizer->AddPassManager(common_pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
  if (save_graphs) {
    std::string file_path = save_graphs_path + "/" + "hwopt_common_after.ir";
    DumpIR(file_path, kernel_graph);
  }
}
}  // namespace opt
}  // namespace mindspore
