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

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdio.h>
#include <float.h>
#include <new>
#include <vector>
#include <string>
#include <memory>
#include <functional>

#ifndef MS_API
#ifdef _WIN32
#define MS_API __declspec(dllexport)
#else
#define MS_API __attribute__((visibility("default")))
#endif
#endif

namespace mindspore {
namespace schema {
struct Tensor;
}  // namespace schema

namespace tensor {
class MSTensor;
}  // namespace tensor

namespace lite {
struct DeviceContext;
struct LiteQuantParam;
}  // namespace lite

class Allocator;
using AllocatorPtr = std::shared_ptr<Allocator>;

class Delegate;
using DelegatePtr = std::shared_ptr<Delegate>;

using TensorPtrVector = std::vector<mindspore::schema::Tensor *>;
using Uint32Vector = std::vector<uint32_t>;
template <typename T>
inline std::string to_string(T t) {
  return std::to_string(t);
}

/// \brief CallBackParam defined input arguments for callBack function.
struct CallBackParam {
  std::string node_name; /**< node name argument */
  std::string node_type; /**< node type argument */
};

struct GPUCallBackParam : CallBackParam {
  double execute_time{-1.f};
};

/// \brief KernelCallBack defined the function pointer for callBack.
using KernelCallBack = std::function<bool(std::vector<tensor::MSTensor *> inputs,
                                          std::vector<tensor::MSTensor *> outputs, const CallBackParam &opInfo)>;

namespace lite {
using DeviceContextVector = std::vector<DeviceContext>;
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
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_INCLUDE_LITE_UTILS_H_
