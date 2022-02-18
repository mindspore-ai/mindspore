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
#include <unordered_map>
#include <memory>
#include "src/tensor.h"
#include "nnacl/tensor_c.h"
#ifndef CONTROLFLOW_TENSORLIST_CLIP
#include "src/tensorlist.h"
#include "nnacl/infer/common_infer.h"
#endif
#include "src/cxx_api/tensor/tensor_impl.h"

namespace mindspore {
namespace lite {
int OutputTensor2TensorC(const std::vector<lite::Tensor *> &tensors_in, std::vector<TensorC *> *tensors_out);
void FreeAllTensorC(std::vector<TensorC *> *tensors_in);
int Tensor2TensorC(const Tensor *src, TensorC *dst);
int TensorC2Tensor(const TensorC *src, Tensor *dst);
#ifndef CONTROLFLOW_TENSORLIST_CLIP
void FreeTensorListC(TensorListC *tensorListC);
int TensorList2TensorListC(TensorList *src, TensorListC *dst);
int TensorListC2TensorList(const TensorListC *src, TensorList *dst);
#endif
int GenerateInTensorC(const OpParameter *const parameter, const std::vector<lite::Tensor *> &inputs,
                      std::vector<TensorC *> *in_tensor_c);
int GenerateOutTensorC(const OpParameter *const parameter, const std::vector<lite::Tensor *> &outputs,
                       std::vector<TensorC *> *out_tensor_c);
int CheckTensorsInvalid(const std::vector<Tensor *> &tensors);
int CheckGraphInputShapes(const std::vector<Tensor *> &inputs,
                          const std::unordered_map<Tensor *, std::vector<int>> &input_shape_map);
std::vector<mindspore::MSTensor> LiteTensorsToMSTensors(const std::vector<lite::Tensor *> &lite_tensors);
void MoveCommonTensorData(Tensor *dst_tensor, Tensor *src_tensor);
void MoveTensorData(Tensor *dst_tensor, Tensor *src_tensor);
void SetTensorData(Tensor *dst_tensor, Tensor *src_tensor);
void SetCommonTensorData(Tensor *dst_tensor, Tensor *src_tensor);
#ifndef CONTROLFLOW_TENSORLIST_CLIP
void MoveTensorListTensorData(TensorList *dst_tensor, TensorList *src_tensor);
void SetTensorListTensorData(TensorList *dst_tensor, TensorList *src_tensor);
#endif
void SetTensorShape(Tensor *dst, Tensor *src);
#ifndef CONTROLFLOW_TENSORLIST_CLIP
void SetTensorListShape(Tensor *dst, Tensor *src);
#endif
bool NeedCastData(Tensor *dst_tensor, Tensor *src_tensor);
int CastTensorData(Tensor *dst_tensor, Tensor *src_tensor, bool support_fp16);
int CastCommonTensorData(Tensor *dst_tensor, Tensor *src_tensor, bool support_fp16);
#ifndef CONTROLFLOW_TENSORLIST_CLIP
int CastTensorListTensorData(TensorList *dst_tensor, TensorList *src_tensor, bool support_fp16);
#endif
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_COMMON_TENSOR_UTIL_H_
