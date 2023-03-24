/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PIPELINE_JIT_PASS_H_
#define MINDSPORE_CCSRC_PIPELINE_JIT_PASS_H_

#include <vector>
#include <functional>
#include <utility>
#include <string>
#include "pipeline/jit/resource.h"
#include "include/common/visible.h"

namespace mindspore {
namespace opt {
namespace irpass {
class OptimizeIRPassLib;
}  // namespace irpass
}  // namespace opt

namespace pipeline {
using PassItem = std::pair<std::string, std::function<bool(ResourcePtr)>>;

extern std::vector<PassItem> kGePasses;
extern std::vector<PassItem> kVmPasses;
extern std::vector<PassItem> kInlinePasses;
extern std::vector<PassItem> kPynativePasses;

bool OptPassAGroup(const ResourcePtr &resource);
bool CconvPass(const ResourcePtr &resource);
bool PipelineSplitPass(const ResourcePtr &resource);
bool AutoParallelPass(const ResourcePtr &resource);
bool PynativeShardPass(const ResourcePtr &resource);
bool ValidatePass(const ResourcePtr &resource);
bool ConvertPrepareAdapt(const ResourcePtr &resource);
bool AddCacheEmbeddingPass(const ResourcePtr &resource);
bool MetaUnpackPreparePass(const ResourcePtr &resource);
void ReclaimOptimizer();
bool PynativeOptPass(const ResourcePtr &resource);
bool EliminateSpecialOpOptPass(const ResourcePtr &resource);
bool ME_EXPORT AutoMonadElimOptPass(const FuncGraphPtr &func_graph);
FuncGraphPtr PrimBpOptPassStep1(const opt::irpass::OptimizeIRPassLib &irpass, const ResourcePtr &resource);
FuncGraphPtr PrimBpOptPassStep2(const opt::irpass::OptimizeIRPassLib &irpass, const ResourcePtr &resource,
                                const std::vector<bool> &need_grad_flags);
FuncGraphPtr MsFunctionBpropGraphPass(const ResourcePtr &resource, bool need_renormalize);
FuncGraphPtr FinalBpropGraphPass(const ResourcePtr &resource);
}  // namespace pipeline
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_PASS_H_
