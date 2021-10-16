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

#ifndef MINDSPORE_CORE_OPS_BROADCAST_H_
#define MINDSPORE_CORE_OPS_BROADCAST_H_
#include <string>
#include <vector>
#include <memory>

#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameBroadcast = "Broadcast";
/// \brief Broadcasts the tensor to the whole group. Refer to Python API @ref mindspore.ops.Broadcast for more details.
class MS_CORE_API Broadcast : public PrimitiveC {
 public:
  /// \brief Constructor.
  Broadcast() : PrimitiveC(kNameBroadcast) {}
  /// \brief Destructor.
  ~Broadcast() = default;
  MS_DECLARE_PARENT(Broadcast, PrimitiveC);
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.Broadcast for the inputs.
  void Init(const int64_t root_rank, const std::string &group = "hccl_world_group");
  /// \brief Set root_rank.
  void set_root_rank(const int64_t root_rank);
  /// \brief Set group.
  void set_group(const std::string &group);
  /// \brief Get root_rank.
  ///
  /// \return root_rank.
  int64_t get_root_rank() const;
  /// \brief Get group.
  ///
  /// \return group.
  std::string get_group() const;
};
AbstractBasePtr BroadcastInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args);
using PrimBroadcast = std::shared_ptr<Broadcast>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_BROADCAST_H_
