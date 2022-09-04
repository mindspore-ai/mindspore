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

#ifndef MINDSPORE_CORE_MINDRT_INCLUDE_ACTOR_OP_ACTOR_H
#define MINDSPORE_CORE_MINDRT_INCLUDE_ACTOR_OP_ACTOR_H

#include <list>
#include <vector>
#include <memory>
#include <string>
#include "utils/hash_map.h"
#include "utils/macros.h"
#include "actor/actor.h"
#include "async/uuid_base.h"
#include "async/future.h"
#include "async/async.h"
#include "mindrt/include/async/collect.h"

namespace mindspore {
// OpActor data route.
struct DataArrow {
  DataArrow(int from_output_index, const AID &to_op_id, int to_input_index)
      : from_output_index_(from_output_index), to_op_id_(to_op_id), to_input_index_(to_input_index), flag_{0} {}
  int from_output_index_;
  AID to_op_id_;
  int to_input_index_;
  // Used to indicate the attribute of data arrow.
  size_t flag_;
};
using DataArrowPtr = std::shared_ptr<DataArrow>;

// OpActor control route.
struct ControlArrow {
  explicit ControlArrow(const AID &to_op_id) : to_op_id_(to_op_id), flag_{0} {}
  AID to_op_id_;
  // Used to indicate the attribute of control arrow.
  size_t flag_;
};
using ControlArrowPtr = std::shared_ptr<ControlArrow>;

// OpActor data.
template <typename T>
struct OpData {
  OpData(const AID &op_id, T *data, int index) : op_id_(op_id), data_(data), index_(index) {}
  AID op_id_;
  T *data_;
  int index_;
};

class MS_CORE_API RandInt {
 public:
  int Get() const { return rand(); }
  static RandInt &Instance();

 private:
  RandInt() { srand(static_cast<unsigned int>(time(nullptr))); }
};

template <typename T>
using OpDataPtr = std::shared_ptr<OpData<T>>;

template <typename T>
using OpDataUniquePtr = std::unique_ptr<OpData<T>>;

// The context of opActor running.
template <typename T>
struct OpContext {
  int sequential_num_;
  std::vector<OpDataPtr<T>> *output_data_;
  std::vector<Promise<int>> *results_;
  // Record the error info for print.
  std::string error_info_{""};
  const void *kernel_call_back_before_;
  const void *kernel_call_back_after_;

  void SetFailed(int32_t code) const {
    if (code == MindrtStatus::KINIT) {
      code = MindrtStatus::KERROR;
    }
    results_->front().SetFailed(code);
  }

  void SetSuccess(int32_t code) const {
    for (auto promise : *results_) {
      promise.SetValue(code);
    }
  }

  void SetResult(size_t index, int value) const { results_->at(index).SetValue(value); }
};

template <typename T>
class OpActor : public ActorBase {
 public:
  explicit OpActor(const std::string &op_name) : ActorBase(op_name) {}
  ~OpActor() override = default;

  // The op actor run when receive the input data.
  virtual void RunOpData(OpData<T> *input_data, OpContext<T> *context = nullptr) {}

  // The op actor run when receive the input control.
  virtual void RunOpControl(AID *input_control, OpContext<T> *context = nullptr) {}

  const std::vector<DataArrowPtr> &output_data_arrows() const { return output_data_arrows_; }
  const std::vector<ControlArrowPtr> &output_control_arrows() const { return output_control_arrows_; }

 protected:
  // The op data.
  mindspore::HashMap<int, std::vector<OpData<T> *>> input_op_datas_;
  std::vector<DataArrowPtr> output_data_arrows_;

  // The op controls.
  mindspore::HashMap<int, std::vector<AID *>> input_op_controls_;
  std::vector<ControlArrowPtr> output_control_arrows_;
};
}  // namespace mindspore

#endif  // MINDSPORE_CORE_MINDRT_INCLUDE_ACTOR_OP_ACTOR_H
