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

namespace mindspore {
namespace pipeline {
using PassItem = std::pair<std::string, std::function<bool(ResourcePtr)>>;

extern std::vector<PassItem> kGePasses;
extern std::vector<PassItem> kVmPasses;
extern std::vector<PassItem> kInlinePasses;
extern std::vector<PassItem> kPynativePasses;

bool CconvPass(const ResourcePtr &res);
bool PipelineSplitPass(const ResourcePtr &res);
bool ValidatePass(const ResourcePtr &res);
bool ConvertPrepareAdapt(const ResourcePtr &res);
bool AddCacheEmbeddingPass(const ResourcePtr &res);
bool InferenceOptPreparePass(const ResourcePtr &res);
void ReclaimOptimizer();
bool PynativeOptPass(const ResourcePtr &res);
}  // namespace pipeline
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_PASS_H_
