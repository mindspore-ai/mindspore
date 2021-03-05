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
#include "minddata/dataset/include/constants.h"
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
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_TEXT_DATA_UTILS_H_
