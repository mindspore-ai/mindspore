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

#ifndef MINDSPORE_CCSRC_PIPELINE_JIT_PIPELINE_SPLIT_H_
#define MINDSPORE_CCSRC_PIPELINE_JIT_PIPELINE_SPLIT_H_

#include <string>
#include <vector>
#include "pipeline/jit/resource.h"

namespace mindspore {
namespace pipeline {
constexpr size_t NODE_INPUT_NUM = 2;
bool PipelineSplit(const ResourcePtr &res);
std::string GetWorldGroup();
bool HasVirtualDataset(const std::vector<AnfNodePtr> &all_nodes);
void InsertVirtualDataset(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes);
}  // namespace pipeline
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_PIPELINE_SPLIT_H_
