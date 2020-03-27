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

#ifndef MINDSPORE_CCSRC_KERNEL_KERNELFUSION_H_
#define MINDSPORE_CCSRC_KERNEL_KERNELFUSION_H_
#include <vector>
#include <map>
#include "kernel/kernel.h"
namespace mindspore {
namespace kernel {
/*
 * @brief fuse op and return a callable mod
 */
struct FusionScopeInfo {
  int32_t scope_id;
  std::vector<AnfNodePtr> input_nodes;
  std::vector<AnfNodePtr> compute_nodes;
  std::vector<AnfNodePtr> output_nodes;
};

std::map<int32_t, KernelModPtr> KernelFusion(const std::vector<FusionScopeInfo> &fusion_scopes);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_KERNELFUSION_H_
