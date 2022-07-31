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
#ifndef MINDSPORE_LITE_SRC_TRAIN_TRAIN_UTILS_H_
#define MINDSPORE_LITE_SRC_TRAIN_TRAIN_UTILS_H_

#include <vector>
#include <string>
#include "src/tensor.h"
#include "src/litert/kernel_exec.h"

namespace mindspore {
namespace kernel {
class KernelExec;
}

namespace lite {
kernel::KernelExec *TSFindKernel(const std::vector<kernel::KernelExec *> &where, const std::string &searchParameter);
size_t TSFindTensor(const std::vector<lite::Tensor *> &where, const lite::Tensor *searchParameter);
size_t TSFindTensorByName(const std::vector<lite::Tensor *> &where, const std::string &searchParameter);
kernel::KernelExec *TSFindKernel(const std::vector<kernel::KernelExec *> &where, const std::string &searchParameter);
size_t TSFindTensor(const std::vector<lite::Tensor *> &where, const lite::Tensor *searchParameter);
float CalculateSparseClassification(lite::Tensor *input, lite::Tensor *output);
float CalculateOneHotClassification(lite::Tensor *input, lite::Tensor *output);
Tensor *CastTensor(Tensor *tensor, TypeId dst_data_type, bool support_fp16);
int ScaleTensor(Tensor *tensor, float scale);
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_TRAIN_TRAIN_UTILS_H_
