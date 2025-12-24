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

#ifndef MINDSPORE_CORE_OPS_CROP_H_
#define MINDSPORE_CORE_OPS_CROP_H_
#include <memory>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameCrop = "Crop";
/// \brief Crop defined the Crop operator prototype of lite, which can be replaced by slice operator.
class OPS_API Crop : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Crop);
  /// \brief Constructor.
  Crop() : BaseOperator(kNameCrop) {}

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] axis Define a dim which is the first dimension to slice.
  /// \param[in] offsets Define a vector to indicate the start index to slice on the corresponding axis.
  void Init(const int64_t axis, const std::vector<int64_t> &offsets);

  /// \brief Method to set axis attribute.
  ///
  /// \param[in] axis Define a dim which is the first dimension to slice.
  void set_axis(const int64_t axis);

  /// \brief Method to set offsets attribute.
  ///
  /// \param[in] offsets Define a vector to indicate the start index to slice on the corresponding axis.
  void set_offsets(const std::vector<int64_t> &offsets);

  /// \brief Method to get axis attribute.
  ///
  /// \return a dim which indicates the first dimension to slice.
  int64_t get_axis() const;

  /// \brief Method to get offsets attribute.
  ///
  /// \return a vector which indicates the start index to slice on the corresponding axis.
  std::vector<int64_t> get_offsets() const;
};
OPS_API abstract::AbstractBasePtr CropInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimCrop = std::shared_ptr<Crop>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_CROP_H_
