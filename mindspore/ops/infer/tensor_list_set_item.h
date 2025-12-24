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

#ifndef MINDSPORE_CORE_OPS_TENSOR_LIST_SET_ITEM_H_
#define MINDSPORE_CORE_OPS_TENSOR_LIST_SET_ITEM_H_
#include <memory>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameTensorListSetItem = "TensorListSetItem";
/// \brief TensorListSetItem defined TensorListSetItem operator prototype of lite.
class OPS_API TensorListSetItem : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(TensorListSetItem);
  /// \brief Constructor.
  TensorListSetItem() : BaseOperator(kNameTensorListSetItem) {}

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] element_dtype Define the element data type in op;
  void Init(const int64_t element_dtype);

  /// \brief Method to set the op's element_dtype attributes.
  ///
  /// \param[in] element_dtype Define the element data type in op;
  void set_element_dtype(const int64_t element_dtype);

  /// \brief Method to get the op's element_dtype attributes.
  int64_t get_element_dtype() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_TENSOR_LIST_SET_ITEM_H_
