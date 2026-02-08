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

#ifndef MINDSPORE_AICPU_ATTR_AND_INPUT_CONVERT_REGIST_H
#define MINDSPORE_AICPU_ATTR_AND_INPUT_CONVERT_REGIST_H
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "include/utils/kernel_base/kernel.h"

namespace mindspore {
namespace kernel {
/**
 * Attr convert to Input
 * */
bool GetAicpuOpAttrToInputInfo(const CNodePtr &kernel_node, std::vector<std::pair<string, size_t>> *info);
/**
 * Input convert To Attr
 * */
bool GetAicpuOpInputToAttrInfo(const CNodePtr &kernel_node, std::map<size_t, std::string> *input_to_attr_info);

void ConvertAttrAndInputBeforeAicpuKernelSelect(const CNodePtr &kernel_node);
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_AICPU_ATTR_AND_INPUT_CONVERT_REGIST_H
