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

#ifndef MINDSPORE_LITE_INCLUDE_LITE_UTILS_H_
#define MINDSPORE_LITE_INCLUDE_LITE_UTILS_H_
#include <vector>
#include <string>
#include <memory>
#include "include/ms_tensor.h"

namespace mindspore::schema {
struct Tensor;
}  // namespace mindspore::schema

namespace mindspore::lite {
/// \brief Allocator defined a memory pool for malloc memory and free memory dynamically.
///
/// \note List public class and interface for reference.
class Allocator;

/// \brief DeviceContext defined a device context.
struct DeviceContext;

using TensorPtrVector = std::vector<mindspore::schema::Tensor *>;
using DeviceContextVector = std::vector<DeviceContext>;
using Uint32Vector = std::vector<uint32_t>;
using String = std::string;
using NodeType = int; /**< 0 : NodeType_ValueNode, 1 : NodeType_Parameter, 2 : NodeType_CNode. */
using AllocatorPtr = std::shared_ptr<Allocator>;

/// \brief Set data of MSTensor from string vector.
///
/// \param[in] input string vector.
/// \param[out] MSTensor.
///
/// \return STATUS as an error code of this interface, STATUS is defined in errorcode.h.
int MS_API StringsToMSTensor(const std::vector<std::string> &inputs, tensor::MSTensor *tensor);

/// \brief Get string vector from MSTensor.
/// \param[in] MSTensor.
/// \return string vector.
std::vector<std::string> MS_API MSTensorToStrings(const tensor::MSTensor *tensor);
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_INCLUDE_LITE_UTILS_H_
