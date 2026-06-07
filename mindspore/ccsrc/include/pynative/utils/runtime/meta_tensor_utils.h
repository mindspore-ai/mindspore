/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_RUNTIME_META_TENSOR_UTILS_H_
#define MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_RUNTIME_META_TENSOR_UTILS_H_

#include <vector>
#include "abstract/abstract_value.h"
#include "include/utils/visible.h"
#include "ir/tensor.h"

namespace mindspore::runtime {
PYNATIVE_UTILS_EXPORT tensor::TensorPtr CreateMetaOutputTensor(const TypeId &type_id,
                                                               const ShapeVector &shape_vector);
PYNATIVE_UTILS_EXPORT void CreateMetaOutputTensors(const AbstractBasePtr &abstract,
                                                   std::vector<tensor::TensorPtr> *outputs);
}  // namespace mindspore::runtime

#endif  // MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_RUNTIME_META_TENSOR_UTILS_H_
