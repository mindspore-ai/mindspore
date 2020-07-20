/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#ifndef MINDSPORE_CCSRC_PIPELINE_JIT_VALIDATOR_H_
#define MINDSPORE_CCSRC_PIPELINE_JIT_VALIDATOR_H_

#include <string>
#include <iostream>
#include <memory>
#include <unordered_set>
#include "frontend/operator/ops.h"
#include "ir/anf.h"
#include "utils/misc.h"

namespace mindspore {
namespace validator {
void Validate(const FuncGraphPtr &func_graph);
void ValidateAbstract(const AnfNodePtr &node);
void ValidateOperation(const AnfNodePtr &node);
}  // namespace validator
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_VALIDATOR_H__
