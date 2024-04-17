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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_GET_VALUE_HELPER_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_GET_VALUE_HELPER_H_

#include <string>
#include <vector>
#include "base/base.h"

namespace mindspore {
namespace opt {
using AnfNodePtr = mindspore::AnfNodePtr;
std::string GetNodeFormatValue(const AnfNodePtr &node);

template <typename T>
T GetNodeScalarValue(const AnfNodePtr &node);
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_GET_VALUE_HELPER_H_
