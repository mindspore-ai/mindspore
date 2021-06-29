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
#include "include/ms_tensor.h"
#include "src/tensor.h"
#include "src/lite_kernel.h"

namespace mindspore {
namespace kernel {
class LiteKernel;
}

namespace lite {
kernel::LiteKernel *TSFindKernel(const std::vector<kernel::LiteKernel *> &where, const std::string &searchParameter);
size_t TSFindTensor(const std::vector<lite::Tensor *> &where, const lite::Tensor *searchParameter);
size_t TSFindTensorByName(const std::vector<lite::Tensor *> &where, const std::string &searchParameter);
kernel::LiteKernel *TSFindKernel(const std::vector<kernel::LiteKernel *> &where, const std::string &searchParameter);
size_t TSFindTensor(const std::vector<lite::Tensor *> &where, const lite::Tensor *searchParameter);
float CalculateSparseClassification(tensor::MSTensor *input, tensor::MSTensor *output);
float CalculateOneHotClassification(tensor::MSTensor *input, tensor::MSTensor *output);
Tensor *CastTensor(Tensor *tensor, TypeId dst_data_type, bool support_fp16);
int ScaleTensor(Tensor *tensor, float scale);
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_TRAIN_TRAIN_UTILS_H_
