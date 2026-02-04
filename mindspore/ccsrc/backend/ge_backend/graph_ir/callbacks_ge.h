/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_CALLBACKS_GE_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_CALLBACKS_GE_H_
#include <map>
#include <string>
#include <memory>
#include "backend/ge_backend/graph_ir/types.h"
#include "ir/tensor.h"

namespace mindspore {
namespace callbacks {
using mindspore::tensor::TensorPtr;
extern const char kSummary[];
extern const char kCheckPoint[];
uint32_t CheckpointSaveCallback(uint32_t, const std::map<std::string, ge::Tensor> &);
uint32_t SummarySaveCallback(uint32_t, const std::map<std::string, ge::Tensor> &);
}  // namespace callbacks
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_CALLBACKS_GE_H_
