/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_EXTENSION_TENSOR_UTILS_H_
#define MINDSPORE_CCSRC_EXTENSION_TENSOR_UTILS_H_
#include <vector>
#include "include/pynative/utils/pyboost/custom/tensor.h"

namespace ms {
/// \brief Creates a tensor from a single integer value.
///
/// Constructs a tensor containing a single scalar integer value with the specified data type.
///
/// \param[in] value The scalar integer value to initialize the tensor with.
/// \param[in] dtype The data type of the tensor. Defaults to `ms::TypeId::kNumberTypeInt64`.
///
/// \return A tensor containing the specified integer value.
PYBOOST_API Tensor tensor(int64_t value, TypeId dtype = TypeId::kNumberTypeInt64);

/// \brief Creates a tensor from a vector of integer values.
///
/// Constructs a tensor from a 1D vector of integers with the specified data type.
///
/// \param[in] value A vector of integers to initialize the tensor with.
/// \param[in] dtype The data type of the tensor. Defaults to `ms::TypeId::kNumberTypeInt64`.
///
/// \return A tensor containing the specified integer values.
PYBOOST_API Tensor tensor(const std::vector<int64_t> &value, TypeId dtype = TypeId::kNumberTypeInt64);

/// \brief Creates a tensor from a single floating-point value.
///
/// Constructs a tensor containing a single scalar floating-point value with the specified data type.
///
/// \param[in] value The scalar floating-point value to initialize the tensor with.
/// \param[in] dtype The data type of the tensor. Defaults to `ms::TypeId::kNumberTypeFloat64`.
///
/// \return A tensor containing the specified floating-point value.
PYBOOST_API Tensor tensor(double value, TypeId dtype = TypeId::kNumberTypeFloat64);

/// \brief Creates a tensor from a vector of floating-point values.
///
/// Constructs a tensor from a 1D vector of floating-point values with the specified data type.
///
/// \param[in] value A vector of floating-point values to initialize the tensor with.
/// \param[in] dtype The data type of the tensor. Defaults to `ms::TypeId::kNumberTypeFloat64`.
///
/// \return A tensor containing the specified floating-point values.
PYBOOST_API Tensor tensor(const std::vector<double> &value, TypeId dtype = TypeId::kNumberTypeFloat64);

/// \brief Creates a tensor filled with ones.
///
/// Constructs a tensor of the specified shape, where every element is initialized to `1`.
///
/// \param[in] shape The shape of the tensor as a vector of integers.
/// \param[in] dtype The data type of the tensor. Defaults to `TypeId::kNumberTypeFloat32`.
///
/// \return A tensor filled with ones.
PYBOOST_API Tensor ones(const ShapeVector &shape, TypeId dtype = TypeId::kNumberTypeFloat32);

/// \brief Creates a tensor filled with zeros.
///
/// Constructs a tensor of the specified shape, where every element is initialized to `0`.
///
/// \param[in] shape The shape of the tensor as a vector of integers.
/// \param[in] dtype The data type of the tensor. Defaults to `TypeId::kNumberTypeFloat32`.
///
/// \return A tensor filled with zeros.
PYBOOST_API Tensor zeros(const ShapeVector &shape, TypeId dtype = TypeId::kNumberTypeFloat32);
}  // namespace ms
#endif  // MINDSPORE_CCSRC_EXTENSION_TENSOR_UTILS_H_
