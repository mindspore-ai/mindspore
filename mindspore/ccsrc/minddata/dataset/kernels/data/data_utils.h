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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_DATA_DATA_UTILS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_DATA_DATA_UTILS_H_

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include "minddata/dataset/include/constants.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/core/cv_tensor.h"
#endif
#include "minddata/dataset/core/data_type.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/core/tensor_row.h"

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

// Returns a tensor of shape input filled with the passed fill_value
// @param input  Tensor
// @param output Tensor. The shape and type of the output tensor is same as input
// @param fill_value Tensor. A scalar tensor used to fill the output tensor

Status Fill(const std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> *output, std::shared_ptr<Tensor> fill_value);

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

// Pad input tensor according pad_shape, need to have same rank.
// Based on the type of the input tensor, PadEndNumeric/String will be called.
// @param std::shared_ptr<Tensor> src - tensor to pad from
// @param std::shared_ptr<Tensor> *dst - return tensor padded
// @param std::vector<dsize_t> pad_shape - shape to pad to
// @param std::shared_ptr<Tensor> pad_val - value to pad with in Tensor format,
// @return Status The status code returned
Status PadEnd(const std::shared_ptr<Tensor> &src, std::shared_ptr<Tensor> *dst, const std::vector<dsize_t> &pad_shape,
              const std::shared_ptr<Tensor> &pad_val);

// Pad input numeric tensor according pad_shape, need to have same rank.
// @param std::shared_ptr<Tensor> src - tensor to pad from
// @param std::shared_ptr<Tensor> *dst - return tensor padded
// @param std::vector<dsize_t> pad_shape - shape to pad to
// @param float pad_val - value to pad with
// @return Status The status code returned
Status PadEndNumeric(const std::shared_ptr<Tensor> &src, std::shared_ptr<Tensor> *dst,
                     const std::vector<dsize_t> &pad_shape, float pad_val);

// recursive helper function for padding numric tensors. This function could be very expensive if called on a
// multi-dimensional tensor it is only meant to be called by PadEndNumeric.
// @tparam T - type of tensor and fill value
// @param std::shared_ptr<Tensor> src - Tensor to pad from
// @param std::shared_ptr<Tensor>* dst - Tensor to pad to, return value
// @param std::vector<dsize_t> cur_ind - recursion helper
// @param T pad_val - value to pad tensor with
// @param size_t cur_dim - recursion helper
// @return Status The status code returned
Status PadEndNumericHelper(const std::shared_ptr<Tensor> &src, std::shared_ptr<Tensor> dst,
                           std::vector<dsize_t> cur_ind, size_t cur_dim = 0);

// Pad input string tensor according pad_shape, need to have same rank.
// @param std::shared_ptr<Tensor> src - tensor to pad from
// @param std::shared_ptr<Tensor> *dst - return tensor padded
// @param std::vector<dsize_t> pad_shape - shape to pad to
// @param std::string pad_val - value to pad with
// @return Status The status code returned
Status PadEndString(const std::shared_ptr<Tensor> &src, std::shared_ptr<Tensor> *dst,
                    const std::vector<dsize_t> &pad_shape, const std::string &pad_val);

// recursive helper function for padding string tensors. This function could be very expensive if called on a
// multi-dimensional tensor it is only meant to be called by PadEndString.
// @tparam T - type of tensor and fill value
// @param std::shared_ptr<Tensor> src - Tensor to pad from
// @param std::shared_ptr<Tensor>* dst - Tensor to pad to, return value
// @param std::vector<dsize_t> cur_ind - recursion helperas text
// @param std::string pad_val - value to pad tensor with
// @param size_t cur_dim - recursion helper
// @return Status The status code returned
Status PadEndStringHelper(const std::shared_ptr<Tensor> &src, std::vector<std::string> *dst,
                          const TensorShape &dst_shape, std::vector<dsize_t> cur_ind, size_t cur_dim,
                          const std::string &pad_value);

enum class RelationalOp {
  kEqual = 0,     // ==
  kNotEqual,      // !=
  kLess,          // <
  kLessEqual,     // <=
  kGreater,       // >
  kGreaterEqual,  // >=
};

/// Helper method that masks the input tensor
/// @tparam T type of the tensor
/// @param input[in] input tensor
/// @param output[out] output tensor
/// @param value_tensor[in] scalar tensor value to compared with
/// @param op[in] RelationalOp enum
/// @return Status ok/error
template <typename T>
Status MaskHelper(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &output,
                  const std::shared_ptr<Tensor> &value_tensor, RelationalOp op);

/// Mask the input tensor
/// @param input[in] input tensor
/// @param output[out] output tensor
/// @param value[in] scalar tensor value to compared with
/// @param op[in] RelationalOp enum
/// @return Status ok/error
Status Mask(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const std::shared_ptr<Tensor> &value,
            RelationalOp op);

Status Concatenate(const TensorRow &input, TensorRow *output, int8_t axis, std::shared_ptr<Tensor> prepend,
                   std::shared_ptr<Tensor> append);

// helper for concat, always append to the input, and pass that to the output
Status ConcatenateHelper(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int8_t axis,
                         std::shared_ptr<Tensor> append);

/// Convert an n-dimensional Tensor to a vector of (n-1)-dimensional CVTensors
/// \param input[in] input tensor
/// \param output[out] output vector of CVTensors
/// \return Status ok/error
Status BatchTensorToCVTensorVector(const std::shared_ptr<Tensor> &input,
                                   std::vector<std::shared_ptr<CVTensor>> *output);

/// Convert an n-dimensional Tensor to a vector of (n-1)-dimensional Tensors
/// \param input[in] input tensor
/// \param output[out] output vector of tensors
/// \return Status ok/error
Status BatchTensorToTensorVector(const std::shared_ptr<Tensor> &input, std::vector<std::shared_ptr<Tensor>> *output);

/// Convert a vector of (n-1)-dimensional Tensors to an n-dimensional Tensor
/// \param input[in] input vector of tensors
/// \param output[out] output tensor
/// \return Status ok/error
Status TensorVectorToBatchTensor(const std::vector<std::shared_ptr<Tensor>> &input, std::shared_ptr<Tensor> *output);

/// Helper method that uniques the input tensor
/// @tparam T type of the tensor
/// \param input[in] input 1d tensor
/// \param output[out] output tensor
/// \param output[out] output tensor of item index
/// \param output[out] output tensor of item count
/// \return Status ok/error
template <typename T>
Status UniqueHelper(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output,
                    std::shared_ptr<Tensor> *output_idx, std::shared_ptr<Tensor> *output_cnt);

/// Unique the input tensor
/// @tparam T type of the tensor
/// \param input[in] input 1d tensor
/// \param output[out] output tensor
/// \param output[out] output tensor of item index
/// \param output[out] output tensor of item count
/// \return Status ok/error
Status Unique(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output,
              std::shared_ptr<Tensor> *output_idx, std::shared_ptr<Tensor> *output_cnt);
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_DATA_DATA_UTILS_H_
