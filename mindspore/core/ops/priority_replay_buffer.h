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

#ifndef MINDSPORE_CORE_OPS_PRIORITY_REPLAY_BUFFER_H_
#define MINDSPORE_CORE_OPS_PRIORITY_REPLAY_BUFFER_H_
#include <memory>
#include <vector>

#include "ir/dtype/type.h"
#include "mindapi/base/types.h"
#include "mindapi/ir/common.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNamePriorityReplayBufferCreate = "PriorityReplayBufferCreate";
constexpr auto kNamePriorityReplayBufferPush = "PriorityReplayBufferPush";
constexpr auto kNamePriorityReplayBufferSample = "PriorityReplayBufferSample";
constexpr auto kNamePriorityReplayBufferUpdate = "PriorityReplayBufferUpdate";
constexpr auto kNamePriorityReplayBufferDestroy = "PriorityReplayBufferDestroy";

class MIND_API PriorityReplayBufferCreate : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(PriorityReplayBufferCreate);
  /// \brief Constructor.
  PriorityReplayBufferCreate() : BaseOperator(kNamePriorityReplayBufferCreate) { InitIOName({}, {"handle"}); }
  /// \brief Init.
  /// Refer to the parameters of python API @ref mindspore.ops._rl_inner_ops.PriorityReplayBufferCreate for the inputs.
  void Init(const int64_t &capacity, const float &alpha, std::vector<std::vector<int64_t>> &shapes,
            const std::vector<TypePtr> &types, const int64_t &seed0, const int64_t &seed1);

  void set_capacity(const int64_t &capacity);
  void set_alpha(const float &alpha);
  void set_shapes(const std::vector<std::vector<int64_t>> &shapes);
  void set_types(const std::vector<TypePtr> &types);
  void set_schema(const std::vector<int64_t> &schema);
  void set_seed0(const int64_t &seed0);
  void set_seed1(const int64_t &seed1);

  int64_t get_capacity() const;
  float get_alpha() const;
  std::vector<std::vector<int64_t>> get_shapes() const;
  std::vector<TypePtr> get_types() const;
  std::vector<int64_t> get_schema() const;
  int64_t get_seed0() const;
  int64_t get_seed1() const;
};

class MIND_API PriorityReplayBufferPush : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(PriorityReplayBufferPush);
  /// \brief Constructor.
  PriorityReplayBufferPush() : BaseOperator(kNamePriorityReplayBufferPush) { InitIOName({"transition"}, {"handle"}); }
  /// \brief Init.
  /// Refer to the parameters of python API @ref mindspore.ops._rl_inner_ops.PriorityReplayBufferPush for the inputs.
  void Init(const int64_t &handle);

  void set_handle(const int64_t &handle);
  int64_t get_handle() const;
};

class MIND_API PriorityReplayBufferSample : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(PriorityReplayBufferSample);
  /// \brief Constructor.
  PriorityReplayBufferSample() : BaseOperator(kNamePriorityReplayBufferSample) {
    InitIOName({}, {"indices", "weights"});
  }
  /// \brief Init.
  /// Refer to the parameters of python API @ref mindspore.ops._rl_inner_ops.PriorityReplayBufferSample for the inputs.
  void Init(const int64_t &handle, const int64_t batch_size, const std::vector<std::vector<int64_t>> &shapes,
            const std::vector<TypePtr> &types);

  void set_handle(const int64_t &handle);
  void set_batch_size(const int64_t &batch_size);
  void set_shapes(const std::vector<std::vector<int64_t>> &shapes);
  void set_types(const std::vector<TypePtr> &types);
  void set_schema(const std::vector<int64_t> &schama);

  int64_t get_handle() const;
  int64_t get_batch_size() const;
  std::vector<std::vector<int64_t>> get_shapes() const;
  std::vector<TypePtr> get_types() const;
  std::vector<int64_t> get_schema() const;
};

class MIND_API PriorityReplayBufferUpdate : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(PriorityReplayBufferUpdate);
  /// \brief Constructor.
  PriorityReplayBufferUpdate() : BaseOperator(kNamePriorityReplayBufferUpdate) {
    InitIOName({"indices", "priorities"}, {"handle"});
  }
  /// \brief Init.
  /// Refer to the parameters of python API @ref mindspore.ops._rl_inner_ops.PriorityReplayBufferUpdate for the inputs.
  void Init(const int64_t &handle);

  void set_handle(const int64_t &handle);
  int64_t get_handle() const;
};

class MIND_API PriorityReplayBufferDestroy : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(PriorityReplayBufferDestroy);
  /// \brief Constructor.
  PriorityReplayBufferDestroy() : BaseOperator(kNamePriorityReplayBufferDestroy) { InitIOName({"handle"}, {"handle"}); }
  /// \brief Init.
  /// Refer to the parameters of python API @ref mindspore.ops._rl_inner_ops.PriorityReplayBufferUpdate for the inputs.
  void Init(const int64_t &handle);

  void set_handle(const int64_t &handle);
  int64_t get_handle() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_PRIORITY_REPLAY_BUFFER_H_
