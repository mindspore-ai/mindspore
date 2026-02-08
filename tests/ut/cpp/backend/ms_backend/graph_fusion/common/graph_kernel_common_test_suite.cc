/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "include/utils/anfalgo.h"
#include "backend/ms_backend/graph_fusion/common/graph_kernel_common_test_suite.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel_flags.h"
#include "backend/ms_backend/graph_fusion/core/graph_kernel_pass_manager.h"
#include "ir/func_graph_flag.h"
#include "ir/graph_utils.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"

namespace mindspore::graphkernel::test {
class EmptyPass : public mindspore::opt::Pass {
 public:
  using Pass::Pass;
  bool Run(const FuncGraphPtr &) override { return false; }
};

void GraphKernelCommonTestSuite::RunPass(const FuncGraphPtr &graph, const std::vector<opt::PassPtr> &passes) {
  UT_CHECK_NULL(graph);
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<GraphKernelPassManager>(pass_stage_++, "ut");
  // when running UT, user can use the environment "export MS_DEV_SAVE_GRAPHS=2" to dump ir with PassManager.
  // add an empty pass to dump the original graph before running.
  pm->Add(std::make_shared<EmptyPass>("ir_before_running"), 0, true);
  for (const auto &pass : passes) {
    UT_CHECK_NULL(pass);
    pm->Add(pass, 0, true);
  }
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(graph);
}

AnfNodePtrList GraphKernelCommonTestSuite::GetAllNodes(const FuncGraphPtr &fg) { return TopoSort(fg->output()); }

CNodePtrList GraphKernelCommonTestSuite::GetAllCNodes(const FuncGraphPtr &fg) {
  CNodePtrList cnodes;
  for (auto &node : GetAllNodes(fg)) {
    if (node->isa<CNode>()) {
      (void)cnodes.emplace_back(node->cast<CNodePtr>());
    }
  }
  return cnodes;
}

CNodePtrList GraphKernelCommonTestSuite::GetAllGKNodes(const FuncGraphPtr &fg) {
  auto cnodes = GetAllCNodes(fg);
  CNodePtrList gk_nodes;
  std::copy_if(cnodes.begin(), cnodes.end(), std::back_inserter(gk_nodes),
               [](const CNodePtr &node) { return common::AnfAlgo::HasNodeAttr(FUNC_GRAPH_ATTR_GRAPH_KERNEL, node); });
  return gk_nodes;
}

void GraphKernelCommonTestSuite::SetGraphKernelFlags(const std::string &flags) {
  std::map<std::string, std::string> jit_config;
  jit_config["graph_kernel_flags"] = flags;
  graphkernel::GraphKernelFlags::SaveJitConfig(jit_config);
}

void GraphKernelCommonTestSuite::SetDeviceTarget(const std::string &device) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  context->set_param<std::string>(MS_CTX_DEVICE_TARGET, device);
}

void GraphKernelCommonTestSuite::CheckInputOutputType(const AnfNodePtr &node,
                                                      const std::vector<TypeId> &target_inputs_type,
                                                      TypeId target_output_type) {
  // check abstract
  auto abstract = node->abstract();
  MS_EXCEPTION_IF_NULL(abstract);
  auto tensor = abstract->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(tensor->element());
  auto type_ptr = tensor->element()->GetType();
  MS_EXCEPTION_IF_NULL(type_ptr);
  auto type = type_ptr->type_id();
  if (type != target_output_type) {
    MS_LOG(ERROR) << "abstract expect data type: " << TypeIdToString(target_output_type);
    MS_LOG(ERROR) << "abstract output data type: " << TypeIdToString(type);
    ASSERT_TRUE(false);
  }
  // check build info
  MS_EXCEPTION_IF_NULL(node->kernel_info());
  auto build_info = AnfAlgo::GetSelectKernelBuildInfo(node);
  MS_EXCEPTION_IF_NULL(build_info);
  auto inputs_type = build_info->GetAllInputDeviceTypes();
  for (size_t i = 0; i < target_inputs_type.size(); ++i) {
    if (inputs_type[i] != target_inputs_type[i]) {
      MS_LOG(ERROR) << "build_info expect input[" << i << "] data type: " << TypeIdToString(target_inputs_type[i]);
      MS_LOG(ERROR) << "build_info output input[" << i << "] data type: " << TypeIdToString(inputs_type[i]);
      ASSERT_TRUE(false);
    }
  }
  auto outputs_type = build_info->GetAllOutputDeviceTypes();
  ASSERT_TRUE(!outputs_type.empty());
  if (outputs_type[0] != target_output_type) {
    MS_LOG(ERROR) << "build_info expect data type: " << TypeIdToString(target_output_type);
    MS_LOG(ERROR) << "build_info output data type: " << TypeIdToString(outputs_type[0]);
    ASSERT_TRUE(false);
  }
}
}  // namespace mindspore::graphkernel::test
