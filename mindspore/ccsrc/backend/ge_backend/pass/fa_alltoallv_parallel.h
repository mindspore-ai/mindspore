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
#ifndef MINDSPORE_CCSRC_BACKEND_GE_BACKEND_PASS_FA_ALLTOALLV_PARALLEL_H_
#define MINDSPORE_CCSRC_BACKEND_GE_BACKEND_PASS_FA_ALLTOALLV_PARALLEL_H_

#include <string>
#include <vector>
#include "include/backend/common/pass_manager/optimizer.h"

namespace mindspore {
namespace opt {
class FaAlltoAllvParallel : public Pass {
 public:
  explicit FaAlltoAllvParallel(bool multigraph = true) : Pass("fa_alltoallv_parallel") {}
  ~FaAlltoAllvParallel() override = default;
  bool Run(const FuncGraphPtr &graph) override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_GE_BACKEND_PASS_FA_ALLTOALLV_PARALLEL_H_
