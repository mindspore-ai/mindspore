/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_PRINT_H_
#define MINDSPORE_CORE_OPS_PRINT_H_
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNamePrint = "Print";
/// \brief Outputs the tensor or string to stdout.
/// Refer to Python API @ref mindspore.ops.Print for more details.
class OPS_API Print : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Print);
  /// \brief Constructor.
  Print() : BaseOperator(kNamePrint) {}

  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.Print for the inputs.
  void Init() const {}

  /// In GPU pass, the string will be translated to attr.
  void set_string_value(const std::vector<std::string> &string_value);
  void set_string_pos(const std::vector<int64_t> &string_pos);
  void set_value_type(const std::vector<int64_t> &value_type);
  void set_value_type_pos(const std::vector<int64_t> &value_type_pos);

  std::vector<std::string> get_string_value() const;
  std::vector<int64_t> get_string_pos() const;
  std::vector<int64_t> get_value_type() const;
  std::vector<int64_t> get_value_type_pos() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_PRINT_H_
