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
 * limitations under the License.MINDSPORE_CORE_OPS_PRIORITY_REPLAY_BUFFER_H_
 */

#ifndef MINDSPORE_CORE_OPS_RESERVOIR_REPLAY_BUFFER_H_
#define MINDSPORE_CORE_OPS_RESERVOIR_REPLAY_BUFFER_H_
#include <memory>
#include <vector>

#include "ir/dtype/type.h"
#include "mindapi/base/types.h"
#include "mindapi/ir/common.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameReservoirReplayBufferCreate = "ReservoirReplayBufferCreate";
constexpr auto kNameReservoirReplayBufferPush = "ReservoirReplayBufferPush";
constexpr auto kNameReservoirReplayBufferSample = "ReservoirReplayBufferSample";
constexpr auto kNameReservoirReplayBufferDestroy = "ReservoirReplayBufferDestroy";

class MIND_API ReservoirReplayBufferCreate : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ReservoirReplayBufferCreate);
  /// \brief Constructor.
  ReservoirReplayBufferCreate() : BaseOperator(kNameReservoirReplayBufferCreate) { InitIOName({}, {"handle"}); }
  /// \brief Init.
  /// Refer to the parameters of python API @ref mindspore.ops._rl_inner_ops.ReservoirReplayBufferCreate for the inputs.
  void Init(const int64_t &capacity, std::vector<std::vector<int64_t>> &shapes, const std::vector<TypePtr> &types,
            const int64_t &seed0, const int64_t &seed1);

  void set_capacity(const int64_t &capacity);
  void set_shapes(const std::vector<std::vector<int64_t>> &shapes);
  void set_types(const std::vector<TypePtr> &types);
  void set_schema(const std::vector<int64_t> &schema);
  void set_seed0(const int64_t &seed0);
  void set_seed1(const int64_t &seed1);

  int64_t get_capacity() const;
  std::vector<std::vector<int64_t>> get_shapes() const;
  std::vector<TypePtr> get_types() const;
  std::vector<int64_t> get_schema() const;
  int64_t get_seed0() const;
  int64_t get_seed1() const;
};

class MIND_API ReservoirReplayBufferPush : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ReservoirReplayBufferPush);
  /// \brief Constructor.
  ReservoirReplayBufferPush() : BaseOperator(kNameReservoirReplayBufferPush) { InitIOName({"transition"}, {"handle"}); }
  /// \brief Init.
  /// Refer to the parameters of python API @ref mindspore.ops._rl_inner_ops.ReservoirReplayBufferPush for the inputs.
  void Init(const int64_t &handle);

  void set_handle(const int64_t &handle);
  int64_t get_handle() const;
};

class MIND_API ReservoirReplayBufferSample : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ReservoirReplayBufferSample);
  /// \brief Constructor.
  ReservoirReplayBufferSample() : BaseOperator(kNameReservoirReplayBufferSample) {
    InitIOName({}, {"indices", "weights"});
  }
  /// \brief Init.
  /// Refer to the parameters of python API @ref mindspore.ops._rl_inner_ops.ReservoirReplayBufferSample for the inputs.
  void Init(const int64_t &handle, const int64_t &batch_size, const std::vector<std::vector<int64_t>> &shapes,
            const std::vector<TypePtr> &types);

  void set_handle(const int64_t &handle);
  int64_t get_handle() const;

  void set_batch_size(const int64_t &batch_size);
  int64_t get_batch_size() const;

  void set_shapes(const std::vector<std::vector<int64_t>> &shapes);
  std::vector<std::vector<int64_t>> get_shapes() const;

  void set_types(const std::vector<TypePtr> &types);
  std::vector<TypePtr> get_types() const;

  void set_schema(const std::vector<int64_t> &schama);
  std::vector<int64_t> get_schema() const;
};

class MIND_API ReservoirReplayBufferDestroy : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ReservoirReplayBufferDestroy);
  /// \brief Constructor.
  ReservoirReplayBufferDestroy() : BaseOperator(kNameReservoirReplayBufferDestroy) {
    InitIOName({"handle"}, {"handle"});
  }
  /// \brief Init.
  /// Refer to the parameters of python API @ref mindspore.ops._rl_inner_ops.ReservoirReplayBufferUpdate for the inputs.
  void Init(const int64_t &handle);

  void set_handle(const int64_t &handle);
  int64_t get_handle() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_RESERVOIR_REPLAY_BUFFER_H_
