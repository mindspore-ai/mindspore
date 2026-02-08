/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PYNATIVE_UTILS_PYBOOST_FUNCTIONS_COMPOSITE_RANDN_H_
#define MINDSPORE_MINDSPORE_CCSRC_PYNATIVE_UTILS_PYBOOST_FUNCTIONS_COMPOSITE_RANDN_H_
#include <memory>
#include <optional>
#include "ir/tensor.h"
#include "ir/value.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr PYBOOST_API randn(const ValueTuplePtr &shape, const tensor::TensorPtr &seed,
                                    const tensor::TensorPtr &offset, const std::optional<Int64ImmPtr> &dtype,
                                    const std::optional<Int64ImmPtr> &device);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PYNATIVE_UTILS_PYBOOST_FUNCTIONS_COMPOSITE_RANDN_H_
