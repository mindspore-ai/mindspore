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

#ifndef MINDSPORE_CORE_OPS_TENSOR_ARRAY_WRITE_H_
#define MINDSPORE_CORE_OPS_TENSOR_ARRAY_WRITE_H_
#include <string>
#include <vector>
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameTensorArrayWrite = "TensorArrayWrite";

/// \brief Assert defined TensorArrayWrite operator prototype of lite.
class OPS_API TensorArrayWrite : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(TensorArrayWrite);
  /// \brief Constructor.
  TensorArrayWrite() : BaseOperator(kNameTensorArrayWrite) {
    InitIOName({"handle", "index", "value", "flow_in"}, {"flow_out"});
  }
  /// \brief Method to init the op's attributes.
  void Init() const {}
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_TENSOR_ARRAY_WRITE_H_
