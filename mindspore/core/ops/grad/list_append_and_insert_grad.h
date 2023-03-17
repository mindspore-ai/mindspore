/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_LIST_APPEND_AND_INSERT_GRAD_H_
#define MINDSPORE_CORE_OPS_LIST_APPEND_AND_INSERT_GRAD_H_

#include "ops/base_operator.h"
#include "mindspore/core/ops/core_ops.h"

namespace mindspore {
namespace ops {
/// \brief List pop operation 'input_data.pop(index)'.
class MIND_API ListAppendAndInsertGrad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ListAppendAndInsertGrad);
  /// \brief Constructor.
  ListAppendAndInsertGrad() : BaseOperator(prim::kListAppendAndInsertGrad) {
    InitIOName({"input_data", "index"}, {"output_data"});
  }
  /// \brief Init function.
  void Init() const {}
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_LIST_APPEND_AND_INSERT_GRAD_H_
