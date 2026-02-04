/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_IR_GRAPH_UTILS_EXTENDS_H_
#define MINDSPORE_CORE_IR_GRAPH_UTILS_EXTENDS_H_

#include <functional>
#include <string>
#include <vector>

#include "ir/anf.h"
#include "ir/scalar.h"
#include "ir/tensor.h"

namespace mindspore {
IncludeType IncludeBelongGraph(const FuncGraphPtr &fg, const AnfNodePtr &node);
}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_GRAPH_UTILS_EXTENDS_H_
