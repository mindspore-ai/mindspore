/**
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
#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_CALLBACKS_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_CALLBACKS_H_

#include <map>
#include <string>
#include <vector>
#include <memory>
#include "ir/tensor.h"
#include "include/utils/visible.h"

namespace mindspore {
namespace callbacks {

using mindspore::tensor::TensorPtr;
// define the python callback module and function
extern const char PYTHON_MOD_CALLBACK_MODULE[];
extern const char PYTHON_FUN_PROCESS_CHECKPOINT[];
extern const char PYTHON_FUN_PROCESS_SUMMARY[];

COMMON_EXPORT extern const char kSummary[];
COMMON_EXPORT extern const char kCheckPoint[];

const int kCallbackOk = 0;
const int kCallbackFalied = 1;

uint32_t SummarySaveCallback(uint32_t, const std::map<std::string, TensorPtr> &);
}  // namespace callbacks
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_CALLBACKS_H_
