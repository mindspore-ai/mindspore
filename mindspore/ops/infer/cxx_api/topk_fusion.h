/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_TOPK_FUSION_H_
#define MINDSPORE_CORE_OPS_TOPK_FUSION_H_

#include "mindapi/base/types.h"
#include "infer/topk.h"

namespace mindspore {
namespace ops {
constexpr auto kNameTopKFusion = "TopKFusion";
/// \brief TopKFusion defined TopK operator prototype of lite.
class OPS_API TopKFusion : public TopK {
 public:
  MIND_API_BASE_MEMBER(TopKFusion);
  /// \brief Constructor.
  TopKFusion() : TopK(kNameTopKFusion) {}

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] sorted Define a boolean value indicate whether the output must be sorted.
  /// \param[in] axis Define the operation axis, which is no use now, but the default is the last dimension.
  /// \param[in] largest Define the number of largest value along with the axis, which is not attribute now, but the
  ///            second input of this operation.
  void Init(const bool sorted, const int64_t axis, const int64_t largest);

  /// \brief Method to set axis attribute. Do not use.
  ///
  /// \param[in] axis Define the operation axis, which is no use now, but the default is the last dimension.
  void set_axis(const int64_t axis);

  /// \brief Method to set largest attribute, which is no use and needs to be converted to the second input.
  ///
  /// \param[in] largest Define the number of largest value along with the axis, which is not attribute now, but the
  ///            second input of this operation.
  void set_largest(const int64_t largest);

  /// \brief Method to get axis attribute.
  ///
  /// \return axis value.
  int64_t get_axis() const;

  /// \brief Method to get largest attribute.
  ///
  /// \return the number of largest value
  int64_t get_largest() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_TOPK_FUSION_H_
