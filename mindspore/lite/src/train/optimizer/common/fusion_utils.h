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

#ifndef MINDSPORE_LITE_SRC_TRAIN_OPTIMIZER_COMMON_FUSION_UTILS_H_
#define MINDSPORE_LITE_SRC_TRAIN_OPTIMIZER_COMMON_FUSION_UTILS_H_

#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include "src/common/utils.h"
#include "schema/inner/model_generated.h"
#include "tools/converter/legacy_optimizer/fusion/fusion_pattern.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::lite::STATUS;
namespace mindspore {
namespace opt {
inline constexpr int kInputIndexZero = 0;
inline constexpr int kInputIndexOne = 1;
inline constexpr int kInputIndexTwo = 2;
inline constexpr int kOutputIndexZero = 0;
inline constexpr int kOutputIndexOne = 1;
inline constexpr size_t kInputSizeTwo = 2;
inline constexpr size_t kInputSizeThree = 3;
inline constexpr size_t kOutputSizeOne = 1;
inline constexpr size_t kMatchPathLenTwo = 2;
inline constexpr size_t kMatchPathLenThree = 3;

STATUS GetMatchNodeIndex(schema::MetaGraphT *graph,
                         const std::unordered_map<std::string, std::shared_ptr<lite::Path>> &matched_path,
                         const std::string &node_name, size_t *node_index);

bool IsMultiOutputNode(schema::MetaGraphT *graph, size_t out_node_index);
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_TRAIN_OPTIMIZER_COMMON_FUSION_UTILS_H_
