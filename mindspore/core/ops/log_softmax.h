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

#ifndef MINDSPORE_CORE_OPS_LOG_SOFTMAX_H_
#define MINDSPORE_CORE_OPS_LOG_SOFTMAX_H_

#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameLogSoftmax = "LogSoftmax";
/// \brief Log Softmax activation function. Refer to Python API @ref mindspore.ops.LogSoftmax for more details.
class MS_CORE_API LogSoftmax : public PrimitiveC {
 public:
  /// \brief Constructor.
  LogSoftmax() : PrimitiveC(kNameLogSoftmax) { InitIOName({"x"}, {"output"}); }
  /// \brief Destructor.
  ~LogSoftmax() = default;
  MS_DECLARE_PARENT(LogSoftmax, PrimitiveC);
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.LogSoftmax for the inputs.
  void Init(const int64_t axis = -1);
  /// \brief Set axis.
  void set_axis(const int64_t axis);
  /// \brief Get axis.
  ///
  /// \return axis.
  int64_t get_axis() const;
};
using PrimLogSoftmaxPtr = std::shared_ptr<LogSoftmax>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_LOG_SOFTMAX_H_
