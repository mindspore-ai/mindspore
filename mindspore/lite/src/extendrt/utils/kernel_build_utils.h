/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_UTILS_KERNEL_BUILD_UTILS_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_UTILS_KERNEL_BUILD_UTILS_H_

#include <utility>
#include <string>
#include <vector>

#include "ir/anf.h"
#include "ir/dtype/type.h"
#include "include/common/utils/utils.h"
#include "mindspore/ccsrc/kernel/kernel.h"

namespace mindspore {
namespace infer {
using DataType = std::pair<TypeId, std::string>;
void SetKernelInfo(const CNodePtr &apply_kernel_ptr);
void CopyInputWeights(const CNodePtr &kernel_node, const std::vector<kernel::KernelTensorPtr> &inputs);
}  // namespace infer
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_UTILS_KERNEL_BUILD_UTILS_H_
