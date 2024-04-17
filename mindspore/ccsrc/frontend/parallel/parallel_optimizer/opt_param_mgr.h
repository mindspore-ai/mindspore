/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_OPTPARAMMGR_H
#define MINDSPORE_OPTPARAMMGR_H

#include <string>
#include <memory>
#include "frontend/parallel/tensor_layout/tensor_layout.h"
#include "frontend/parallel/graph_util/node_info.h"
#include "base/base.h"

namespace mindspore {
namespace parallel {
class OptParamMgr {
 public:
  virtual ~OptParamMgr() = default;
  virtual std::string ShardOptGroup(const AnfNodePtr &parameter, TensorLayout *const tensor_layout,
                                    const OperatorInfoPtr &distribute_operator) const = 0;
};

std::unique_ptr<OptParamMgr> createOptParamMgr(const FuncGraphPtr &root);
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_OPTPARAMMGR_H
