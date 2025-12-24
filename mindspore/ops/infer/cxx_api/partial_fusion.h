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

#ifndef MINDSPORE_CORE_OPS_PARTIAL_FUSION_H_
#define MINDSPORE_CORE_OPS_PARTIAL_FUSION_H_
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNamePartialFusion = "PartialFusion";
/// \brief PartialFusion defined Partial operator prototype of lite.
class OPS_API PartialFusion : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(PartialFusion);
  /// \brief Constructor.
  PartialFusion() : BaseOperator(kNamePartialFusion) {}

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] sub_graph_index Define a index value to indicate which sub-graph.
  void Init(const int64_t sub_graph_index);

  /// \brief Method to set sub_graph_index attribute.
  ///
  /// \param[in] sub_graph_index Define a index value to indicate which sub-graph.
  void set_sub_graph_index(const int64_t sub_graph_index);

  /// \brief Method to get sub_graph_index attribute.
  ///
  /// \return sub-graph index.
  int64_t get_sub_graph_index() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_PARTIAL_FUSION_H_
