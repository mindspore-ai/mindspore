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

#ifndef MINDSPORE_LITE_SRC_COMMON_TENSOR_UTIL_H_
#define MINDSPORE_LITE_SRC_COMMON_TENSOR_UTIL_H_
#include <vector>
#include "src/tensor.h"
#include "src/tensorlist.h"
#include "nnacl/tensor_c.h"
#include "nnacl/infer/common_infer.h"

namespace mindspore {
namespace lite {
int InputTensor2TensorC(const std::vector<lite::Tensor *> &tensors_in, std::vector<TensorC *> *tensors_out);
int OutputTensor2TensorC(const std::vector<lite::Tensor *> &tensors_in, std::vector<TensorC *> *tensors_out);
void SetOutputTensorAttr(const std::vector<TensorC *> &tensors_in, std::vector<lite::Tensor *> *tensors_out);
void FreeAllTensorC(std::vector<TensorC *> *tensors_in);
void FreeTensorListC(TensorListC *tensorListC);
void Tensor2TensorC(Tensor *src, TensorC *dst);
void TensorC2Tensor(TensorC *src, Tensor *dst);
int TensorList2TensorListC(TensorList *src, TensorListC *dst);
void TensorListC2TensorList(TensorListC *src, TensorList *dst);
int GenerateMergeSwitchOutTensorC(const std::vector<lite::Tensor *> &inputs, std::vector<lite::Tensor *> *outputs,
                                  std::vector<TensorC *> *out_tensor_c);
int GenerateInTensorC(const OpParameter *const parameter, const std::vector<lite::Tensor *> &inputs,
                      std::vector<lite::Tensor *> *outputs, std::vector<TensorC *> *in_tensor_c);
int GenerateOutTensorC(const OpParameter *const parameter, const std::vector<lite::Tensor *> &inputs,
                       std::vector<lite::Tensor *> *outputs, std::vector<TensorC *> *out_tensor_c);
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_COMMON_TENSOR_UTIL_H_
