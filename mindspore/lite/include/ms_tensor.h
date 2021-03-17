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

#ifndef MINDSPORE_LITE_INCLUDE_MS_TENSOR_H_
#define MINDSPORE_LITE_INCLUDE_MS_TENSOR_H_

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "ir/dtype/type_id.h"

#ifndef MS_API
#ifdef _WIN32
#define MS_API __declspec(dllexport)
#else
#define MS_API __attribute__((visibility("default")))
#endif
#endif

namespace mindspore {
namespace tensor {
/// \brief MSTensor defined tensor in MindSpore Lite.
class MS_API MSTensor {
 public:
  /// \brief Constructor of MindSpore Lite MSTensor.
  ///
  /// \return Instance of MindSpore Lite MSTensor.
  MSTensor() = default;

  /// \brief Destructor of MindSpore Lite Model.
  virtual ~MSTensor() = default;

  /// \brief Get data type of the MindSpore Lite MSTensor.
  ///
  /// \note TypeId is defined in mindspore/mindspore/include/api/type_id.h. Only number types in TypeId enum are
  /// suitable for MSTensor.
  ///
  /// \return MindSpore Lite TypeId of the MindSpore Lite MSTensor.
  virtual TypeId data_type() const = 0;

  /// \brief Get shape of the MindSpore Lite MSTensor.
  ///
  /// \return A vector of int as the shape of the MindSpore Lite MSTensor.
  virtual std::vector<int> shape() const = 0;

  /// \brief Get size of the dimension of the MindSpore Lite MSTensor index by the parameter index.
  ///
  /// \param[in] index Define index of dimension returned.
  ///
  /// \return Size of dimension of the MindSpore Lite MSTensor.
  virtual int DimensionSize(size_t index) const = 0;

  /// \brief Get number of element in MSTensor.
  ///
  /// \return Number of element in MSTensor.
  virtual int ElementsNum() const = 0;

  /// \brief Get byte size of data in MSTensor.
  ///
  /// \return Byte size of data in MSTensor.
  virtual size_t Size() const = 0;

  /// \brief Get the pointer of data in MSTensor.
  ///
  /// \note The data pointer can be used to both write and read data in MSTensor.
  ///
  /// \return the pointer points to data in MSTensor.
  virtual void *MutableData() = 0;

  /// \brief Get the name of MSTensor.
  ///
  /// \return the name of MSTensor.
  virtual std::string tensor_name() const = 0;

  /// \brief Set the name of MSTensor.
  virtual void set_tensor_name(const std::string name) = 0;

  /// \brief Set the data of MSTensor.
  virtual void set_data(void *data) = 0;
};
}  // namespace tensor
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
}  // namespace mindspore
#endif  // MINDSPORE_LITE_INCLUDE_MS_TENSOR_H_
