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

#include "frontend/parallel/ops_info/comparison_function_info.h"

#include "frontend/parallel/dynamic_creator.h"

namespace mindspore {
namespace parallel {
REGISTER(EqualInfo);
REGISTER(ApproximateEqualInfo);
REGISTER(NotEqualInfo);
REGISTER(MaximumInfo);
REGISTER(MinimumInfo);
REGISTER(GreaterInfo);
REGISTER(GreaterEqualInfo);
REGISTER(LessInfo);
REGISTER(LessEqualInfo);
}  // namespace parallel
}  // namespace mindspore
