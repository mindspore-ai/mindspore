/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "backend/common/graph_kernel/core/split_reshape_and_cache.h"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "include/backend/kernel_info.h"
#include "include/backend/optimizer/helper.h"
#include "mindspore/core/ops/framework_ops.h"
#include "mindspore/core/ops/nn_ops.h"
#include "mindspore/core/ops/nn_optimizer_ops.h"
namespace mindspore::graphkernel {
const BaseRef SplitReshapeAndCache::DefinePattern() const {
  VarPtr v0 = std::make_shared<Var>();
  VarPtr v1 = std::make_shared<Var>();
  VarPtr v2 = std::make_shared<Var>();
  VarPtr v3 = std::make_shared<Var>();
  VarPtr v4 = std::make_shared<Var>();
  VarPtr UMonad = std::make_shared<Var>();
  return VectorRef({prim::kPrimReshapeAndCache, v0, v1, v2, v3, v4, UMonad});
}

const bool SplitReshapeAndCache::CanSplit(const AnfNodePtr &node) const {
  return IsPrimitiveCNode(node, prim::kPrimReshapeAndCache);
}

}  // namespace mindspore::graphkernel
