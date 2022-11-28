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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_TEXT_DATA_UTILS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_TEXT_DATA_UTILS_H_

#include <memory>
#include <string>
#include <vector>
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/core/data_type.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/core/cv_tensor.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/core/tensor_row.h"

namespace mindspore {
namespace dataset {
/// \brief Helper method that perform sliding window on input tensor.
/// \param[in] input - Input tensor.
/// \param[in] out_shape - Output shape of output tensor.
/// \param[in] width - The axis along which sliding window is computed.
/// \param[in] axis - The width of the window.
/// \param[out] output - Output tensor
/// \return Status return code
Status SlidingWindowHelper(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, TensorShape out_shape,
                           uint32_t width, int32_t axis);

/// \brief Helper method that append offsets tensor to output TensorRow.
/// \param[in] offsets_start - Offsets start index vector.
/// \param[in] offsets_limit - Offsets length vector.
/// \param[out] output - Output TensorRow
/// \return Status return code
Status AppendOffsetsHelper(const std::vector<uint32_t> &offsets_start, const std::vector<uint32_t> &offsets_limit,
                           TensorRow *output);

/// \brief Helper method that add token on input tensor.
/// \param[in] input Input tensor.
/// \param[in] token The token to be added.
/// \param[in] begin Whether to insert token at start or end of sequence.
/// \param[out] output Output tensor.
/// \return Status return code.
Status AddToken(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const std::string &token,
                bool begin);

/// \brief Truncate the input sequence so that it does not exceed the maximum length.
/// \param[in] max_seq_len Maximum allowable length.
/// \param[out] output Output Tensor.
Status Truncate(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int max_seq_len);
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_TEXT_DATA_UTILS_H_
