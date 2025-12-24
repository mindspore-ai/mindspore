/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_SPARSE_APPLY_ADAGRAD_DA_H_
#define MINDSPORE_CORE_OPS_SPARSE_APPLY_ADAGRAD_DA_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSparseApplyAdagradDA = "SparseApplyAdagradDA";
class OPS_API SparseApplyAdagradDA : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SparseApplyAdagradDA);
  SparseApplyAdagradDA() : BaseOperator(kNameSparseApplyAdagradDA) {
    InitIOName({"var", "grad_accum", "grad_square_accum", "grad", "indices", "lr", "l1", "l2", "global_step"}, {"var"});
  }
};

OPS_API abstract::AbstractBasePtr SparseApplyAdagradDAInfer(const abstract::AnalysisEnginePtr &,
                                                            const PrimitivePtr &primitive,
                                                            const std::vector<AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SPARSE_APPLY_ADAGRAD_DA_H_
