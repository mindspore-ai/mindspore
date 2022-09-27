/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_MINDIR_LOADER_MINDIR_MODEL_KERNEL_MOD_UTIL_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_MINDIR_LOADER_MINDIR_MODEL_KERNEL_MOD_UTIL_H_

#include <vector>
#include <memory>

#include "extendrt/mindir_loader/mindir_model/inner_kernel.h"
#include "src/tensor.h"
#include "include/model.h"

namespace mindspore::kernel {
class KernelModUtil {
 public:
  static std::shared_ptr<mindspore::kernel::InnerKernel> GetInnerKernel(
    const std::vector<mindspore::lite::Tensor *> &in_tensors, const std::vector<mindspore::lite::Tensor *> &out_tensors,
    const mindspore::lite::LiteGraph::Node *node, lite::InnerContext *context);
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_MINDIR_LOADER_MINDIR_MODEL_KERNEL_MOD_UTIL_H_
