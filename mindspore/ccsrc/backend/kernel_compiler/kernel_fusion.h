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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_KERNELFUSION_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_KERNELFUSION_H_
#include <utility>
#include <vector>
#include <map>
#include <string>
#include "backend/kernel_compiler/kernel.h"
namespace mindspore {
namespace kernel {
/*
 * @brief fuse op and return a callable mod
 */
struct FusionScopeInfo {
  FusionScopeInfo(int64_t id, std::string f_name, std::vector<AnfNodePtr> in, std::vector<AnfNodePtr> comp,
                  std::vector<AnfNodePtr> out)
      : scope_id(id),
        full_name(f_name),
        input_nodes(std::move(in)),
        compute_nodes(std::move(comp)),
        output_nodes(std::move(out)) {}
  int64_t scope_id{};
  std::string full_name{};
  std::vector<AnfNodePtr> input_nodes;
  std::vector<AnfNodePtr> compute_nodes;
  std::vector<AnfNodePtr> output_nodes;
};

std::map<int64_t, KernelModPtr> KernelFusion(const std::vector<FusionScopeInfo> &fusion_scopes);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_KERNELFUSION_H_
