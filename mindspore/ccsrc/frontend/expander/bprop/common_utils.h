/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_FRONTEND_EXPANDER_BPROP_GRAD_OPS_COMMON_UTILS_H_
#define MINDSPORE_CCSRC_FRONTEND_EXPANDER_BPROP_GRAD_OPS_COMMON_UTILS_H_

#include <cmath>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "mindspore/ops/infer/dynamic_broadcast_gradient_args.h"
#include "frontend/expander/bprop/bprop_irbuilder.h"
#include "mindspore/ccsrc/include/utils/expander/node.h"

namespace mindspore::expander::bprop {
using mindspore::ops::BroadcastGradientArgsInferValue;
int64_t CheckRange(int64_t idx, int64_t dim_size);
std::vector<int64_t> GetIntList(const ValuePtr &value);
std::vector<int64_t> GetIntList(const NodePtr &node);
}  // namespace mindspore::expander::bprop
#endif  // MINDSPORE_CCSRC_FRONTEND_EXPANDER_BPROP_GRAD_OPS_COMMON_UTILS_H_
