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

#ifndef MINDSPORE_CORE_OPS_SPLIT_H_
#define MINDSPORE_CORE_OPS_SPLIT_H_
#include <vector>
#include <memory>

#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSplit = "Split";
/// \brief Splits the input tensor into output_num of tensors along the given axis and output numbers.
/// Refer to Python API @ref mindspore.ops.Split for more details.
class MS_CORE_API Split : public PrimitiveC {
 public:
  /// \brief Constructor.
  Split() : PrimitiveC(kNameSplit) {}
  /// \brief Destructor.
  ~Split() = default;
  MS_DECLARE_PARENT(Split, PrimitiveC);
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.Split for the inputs.
  void Init(const int64_t axis, const int64_t output_num);
  /// \brief Set size_splits.
  void set_size_splits(const std::vector<int64_t> &size_splits);
  /// \brief Set axis.
  void set_axis(const int64_t axis);
  /// \brief Set output_num.
  void set_output_num(const int64_t output_num);
  /// \brief Get size_splits.
  ///
  /// \return size_splits.
  std::vector<int64_t> get_size_splits() const;
  /// \brief Get size_splits.
  ///
  /// \return size_splits.
  int64_t get_axis() const;
  /// \brief Get output_num.
  ///
  /// \return output_num.
  int64_t get_output_num() const;
};
AbstractBasePtr SplitInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args);
using PrimSplit = std::shared_ptr<Split>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SPLIT_H_
