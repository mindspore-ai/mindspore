/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_TENSOR_ARRAY_H_
#define MINDSPORE_CORE_OPS_TENSOR_ARRAY_H_
#include <vector>
#include <string>
#include "ops/primitive_c.h"

namespace mindspore {
namespace ops {

constexpr auto kNameTensorArray = "TensorArray";

/// \brief Assert defined TensorArray operator prototype of lite.
class MS_CORE_API TensorArray : public PrimitiveC {
 public:
  /// \brief Constructor.
  TensorArray() : PrimitiveC(kNameTensorArray) { InitIOName({"size"}, {"handle", "flow"}); }
  /// \brief Destructor.
  ~TensorArray() = default;
  MS_DECLARE_PARENT(TensorArray, PrimitiveC);
  /// \brief Method to init the op's attributes.
  void Init(bool dynamic_size, bool identical_element_shapes, const std::vector<int> &element_shape, int data_type);
  /// \brief Method to set dynamic_size attributes.
  ///
  /// \param[in] dynamic_size Define the dynamic_size.
  void set_dynamic_size(bool dynamic_size);
  /// \brief Method to set identical_element_shapes attributes.
  ///
  /// \param[in] identical_element_shapes Define the identical element shapes
  void set_identical_element_shapes(bool identical_element_shapes);
  /// \brief Method to set element_shape attributes.
  ///
  /// \param[in] element_shape Define the element shape.
  void set_element_shape(const std::vector<int> &element_shape);
  /// \brief Method to set data_type attributes.
  ///
  /// \param[in] data_type Define the data type.
  void set_data_type(int data_type);
  /// \brief Method to get dynamic_size attributes.
  bool get_dynamic_size() const;
  /// \brief Method to get element_shapes attributes.
  bool get_identical_element_shapes() const;
  /// \brief Method to get element_shape attributes.
  const std::vector<int> get_element_shape() const;
  /// \brief Method to get data_type attributes.
  int get_data_type() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_TENSOR_ARRAY_H_
