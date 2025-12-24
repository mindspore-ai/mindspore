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

#ifndef MINDSPORE_CORE_OPS_ALL_H_
#define MINDSPORE_CORE_OPS_ALL_H_
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameAll = "All";
/// \brief All defined All operator prototype of lite.
class OPS_API All : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(All);
  /// \brief Constructor.
  All() : BaseOperator(kNameAll) {}

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] keep_dims Define the dim.
  void Init(const int64_t keep_dims);

  /// \brief Method to set keep_dims attributes.
  ///
  /// \param[in] keep_dims Define the dim.
  void set_keep_dims(const int64_t keep_dims);

  /// \brief Method to get keep_dims attributes.
  ///
  /// \return keep_dims attributes.
  int64_t get_keep_dims() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_ALL_H_
