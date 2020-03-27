/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef DATASET_KERNELS_DATA_DATA_UTILS_H_
#define DATASET_KERNELS_DATA_DATA_UTILS_H_

#include <memory>
#include <vector>
#include "dataset/core/constants.h"
#include "dataset/core/cv_tensor.h"
#include "dataset/core/data_type.h"
#include "dataset/core/tensor.h"

namespace mindspore {
namespace dataset {
// Returns Onehot encoding of the input tensor.
//          Example: if input=2 and numClasses=3, the output is [0 0 1].
// @param input: Tensor has type DE_UINT64, the non-one hot values are stored
//               along the first dimensions or rows..
//               If the rank of input is not 1 or the type is not DE_UINT64,
//               then it will fail.
// @param output: Tensor. The shape of the output tensor is <input_shape, numClasses>
//                and the type is same as input.
// @param num_classes: Number of classes to.
Status OneHotEncoding(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> *output, dsize_t num_classes);

Status OneHotEncodingUnsigned(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output,
                              dsize_t num_classes, int64_t index);

Status OneHotEncodingSigned(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, dsize_t num_classes,
                            int64_t index);

// Returns a type changed input tensor.
//          Example: if input tensor is float64, the output will the specified dataType. See DataTypes.cpp
// @param input  Tensor
// @param output Tensor. The shape of the output tensor is same as input with the type changed.
// @param data_type: type of data to cast data to
// @note: this operation will do a memcpy and if the value is truncated then precision will be lost

template <typename T>
void CastFrom(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output);

template <typename FROM, typename TO>
void Cast(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output);

Status ToFloat16(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output);

Status TypeCast(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const DataType &data_type);
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_KERNELS_DATA_DATA_UTILS_H_
