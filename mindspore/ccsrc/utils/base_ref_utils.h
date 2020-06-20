/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include <vector>
#include <memory>
#include "utils/base_ref.h"
#include "include/ms_tensor.h"

#ifndef MINDSPORE_CCSRC_UTILS_BASE_REF_UTILS_H
#define MINDSPORE_CCSRC_UTILS_BASE_REF_UTILS_H
namespace mindspore {
std::vector<std::shared_ptr<inference::MSTensor>> TransformBaseRefToMSTensor(const BaseRef &base_ref);

std::vector<std::vector<std::shared_ptr<inference::MSTensor>>> TransformVectorRefToMultiTensor(
  const VectorRef &vector_ref);
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_UTILS_BASE_REF_UTILS_H
