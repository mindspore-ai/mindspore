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

class TensorArray : public PrimitiveC {
 public:
  TensorArray() : PrimitiveC(kNameTensorArray) { InitIOName({"size"}, {"handle", "flow"}); }
  ~TensorArray() = default;
  MS_DECLARE_PARENT(TensorArray, PrimitiveC);
  void Init(bool dynamic_size, bool identical_element_shapes, const std::vector<int> &element_shape, int data_type);

  void set_dynamic_size(bool dynamic_size);
  void set_identical_element_shapes(bool identical_element_shapes);
  void set_element_shape(const std::vector<int> &element_shape);
  void set_data_type(int data_type);

  bool get_dynamic_size() const;
  bool get_identical_element_shapes() const;
  const std::vector<int> get_element_shape() const;
  int get_data_type() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_TENSOR_ARRAY_H_
