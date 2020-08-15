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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_VALUE_BASED_ELIMINATE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_VALUE_BASED_ELIMINATE_H_

#include <algorithm>
#include <memory>
#include <vector>

#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/irpass/prim_eliminate.h"
#include "frontend/optimizer/optimizer_caller.h"
#include "frontend/optimizer/anf_visitor.h"
#include "ir/pattern_matcher.h"

namespace mindspore {
namespace opt {
namespace irpass {
// {prim::kPrimSelect, {prim::kPrimGreater, X, 0}, Y, Z}} -> Y when X is always greater than 0
// {prim::kPrimMaximum, X, Y} -> X when Y is smaller than LOWER_FLT_LIMIT
// {prim::kPrimMinimum, X, Y} -> X when Y is greater than UPPER_FLT_LIMIT
class ValueBasedEliminate : public OptimizerCaller {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_VALUE_BASED_ELIMINATE_H_
