/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_PASS_H_
#define MINDSPORE_CCSRC_FRONTEND_JIT_PASS_H_

#include <vector>
#include <functional>
#include <utility>
#include <string>
#include "frontend/jit/ps/resource.h"
#include "frontend/optimizer/irpass/view_inplace_utils.h"

namespace mindspore {
namespace opt {
namespace irpass {
class OptimizeIRPassLib;
}  // namespace irpass
}  // namespace opt

namespace pipeline {
using PassItem = std::pair<std::string, std::function<bool(ResourcePtr)>>;

extern std::vector<PassItem> kVmPasses;
extern std::vector<PassItem> kInlinePasses;
extern std::vector<PassItem> kPynativePasses;
extern std::vector<PassItem> kAddAttrWithInlinePass;

bool OptPassAGroup(const ResourcePtr &resource);
bool CconvPass(const ResourcePtr &resource);
bool DatasetRepeatReaderOptPass(const ResourcePtr &resource);
bool DetachBackward(const ResourcePtr &resource);
bool PipelineSplitPass(const ResourcePtr &resource);
bool PipelineParallelScheduler(const ResourcePtr &resource);
bool AutoParallelPass(const ResourcePtr &resource);
bool AutoParallelSymbolPassWithReNormalize(const ResourcePtr &resource);
bool ParallelVirtualDatasetPass(const ResourcePtr &resource);
bool EliminateUnusedParamsPass(const ResourcePtr &resource);
bool ProcessOffloadAnnotationPass(const ResourcePtr &resource);
void ParseOffloadAnnotation(const CNodePtr &cnode, std::string_view comments);
bool ValidatePass(const ResourcePtr &resource);
bool GradPartialTransformPass(const ResourcePtr &resource);
bool PynativeOptPass(const ResourcePtr &resource);
bool OptAfterJitGradPass(const ResourcePtr &resource);
bool AutoMonadElimOptPass(const FuncGraphPtr &func_graph);
FuncGraphPtr PrimBpOptPassStep1(const opt::irpass::OptimizeIRPassLib &irpass, const ResourcePtr &resource);
FuncGraphPtr PrimBpOptPassStep2(const opt::irpass::OptimizeIRPassLib &irpass, const ResourcePtr &resource,
                                const std::vector<bool> &need_grad_flags);
FuncGraphPtr JitBpropGraphPass(const ResourcePtr &resource, bool need_renormalize);
FuncGraphPtr FinalBpropGraphPass(const ResourcePtr &resource, bool has_control_flow);
void UpdateArgsSpec(const FuncGraphPtr &func_graph, const ResourcePtr &resource);
bool RewriterBeforeOptAPass(const ResourcePtr &resource);
bool ExpandDumpFlagPass(const ResourcePtr &resource);
bool JitOptPassAGroup(const ResourcePtr &resource);
bool JitOptPassBGroup(const ResourcePtr &resource);
bool LoopUnrollPass(const ResourcePtr &resource);
bool JitOptPassAfterCconvGroup(const ResourcePtr &resource);
bool RemoveValueNodeDuplicationsPassForJit(const ResourcePtr &resource);
bool PartialUnusedArgsEliminatePass(const ResourcePtr &resource);
bool PartialUnusedArgsEliminateAfterRecomputePass(const ResourcePtr &resource);
bool MutableEliminatePass(const ResourcePtr &resource);
bool EnvironConversionPass(const ResourcePtr &resource);
bool PyInterpretToExecutePass(const ResourcePtr &resource);
bool ConvertAfterRewriterPass(const ResourcePtr &resource);
bool OrderPyExecuteAfterRewriterPass(const ResourcePtr &resource);
bool RewriterAfterOptAPass(const ResourcePtr &resource);
bool AddRecomputationPass(const ResourcePtr &resource);
bool OptAfterRecomputeGroup(const ResourcePtr &resource);
bool SetTrainingFlagPass(const ResourcePtr &resource);
bool BackendPass(const ResourcePtr &resource);
void ViewInplaceBeforeGradProcessPass(const ResourceBasePtr &resource, const FuncGraphPtr &func_graph,
                                      opt::irpass::ViewInplacePassType type);
bool SymbolEngineOptGroup(const ResourcePtr &resource);
bool IsFrontendPassEnabledForGPTO();
}  // namespace pipeline
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_JIT_PASS_H_
