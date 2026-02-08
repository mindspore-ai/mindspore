/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_EXPANDER_BASE_UTILS_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_EXPANDER_BASE_UTILS_H_

#include <vector>
#include <string>
#include <functional>
#include "backend/ms_backend/graph_fusion/expander/base/ir_builder.h"

namespace mindspore::graphkernel::expander {
bool FormatDefaultNchwSame(const std::string &f0, const std::string &f1);
bool CheckAllFormatsSame(const DefaultIrBuilder *ib,
                         const std::function<bool(const std::string &, const std::string &)> &check = nullptr);
bool CheckAllDataTypeSame(const DefaultIrBuilder *ib);
bool CheckAttrs(const DefaultIrBuilder *ib, const std::vector<std::string> &attrs);
ShapeVector ExpandDimsInferShape(const ShapeVector &shape, const std::vector<int64_t> &axis);
std::vector<int64_t> GetAxisList(const ValuePtr &value);
}  // namespace mindspore::graphkernel::expander
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_EXPANDER_BASE_UTILS_H_
