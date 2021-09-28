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
#include <unordered_map>
#include "actor/actor.h"
#include "async/uuid_base.h"
#include "async/future.h"
#include "async/async.h"
#include "mindrt/include/async/collect.h"

namespace mindspore {
// OpActor data route.
struct DataArrow {
  DataArrow(int from_output_index, const AID &to_op_id, int to_input_index)
      : from_output_index_(from_output_index), to_op_id_(to_op_id), to_input_index_(to_input_index) {}
  int from_output_index_;
  AID to_op_id_;
  int to_input_index_;
};
using DataArrowPtr = std::shared_ptr<DataArrow>;

// OpActor data.
template <typename T>
struct OpData {
  OpData(const AID &op_id, T *data, int index) : op_id_(op_id), data_(data), index_(index) {}
  AID op_id_;
  T *data_;
  int index_;
};

class RandInt {
 public:
  int Get() { return rand(); }
  static RandInt &Instance() {
    static RandInt instance;
    return instance;
  }

 private:
  RandInt() { srand(time(NULL)); }
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
  const void *kernel_call_back_before_;
  const void *kernel_call_back_after_;

  void SetFailed(int32_t code) {
    if (code == MindrtStatus::KINIT) {
      code = MindrtStatus::KERROR;
    }
    for (auto promise : *results_) {
      promise.SetFailed(code);
    }
  }

  void SetSuccess(int32_t code) {
    for (auto promise : *results_) {
      promise.SetValue(code);
    }
  }

  void SetResult(size_t index, int value) { results_->at(index).SetValue(value); }
};

template <typename T>
class OpActor : public ActorBase {
 public:
  explicit OpActor(std::string op_name) : ActorBase(op_name) {}
  virtual ~OpActor() = default;

  // The op actor run when receive the input data.
  virtual void RunOpData(OpData<T> *input_data, OpContext<T> *context = nullptr) {}

  // The op actor run when receive the input control.
  virtual void RunOpControl(AID *input_control, OpContext<T> *context = nullptr) {}

  std::vector<DataArrowPtr> output_data_arrows() const { return output_data_arrows_; }
  std::vector<AID> output_control_arrows() const { return output_control_arrows_; }

 protected:
  // The op data.
  std::unordered_map<int, std::vector<OpData<T> *>> input_op_datas_;
  std::vector<DataArrowPtr> output_data_arrows_;

  // The op controls.
  std::unordered_map<int, std::vector<AID *>> input_op_controls_;
  std::vector<AID> output_control_arrows_;
};

template <typename T>
Future<std::list<int>> MindrtAsyncRun(const std::vector<OpDataPtr<T>> &input_data, OpContext<T> *context) {
  std::list<Future<int>> futures;
  for (auto promise : *(context->results_)) {
    futures.push_back(promise.GetFuture());
  }
  Future<std::list<int>> collect = mindspore::Collect<int>(futures);

  for (auto data : input_data) {
    Async(data->op_id_, &mindspore::OpActor<T>::RunOpData, data.get(), context);
  }

  return collect;
}

template <typename T>
int MindrtRun(const std::vector<OpDataPtr<T>> &input_data, std::vector<OpDataPtr<T>> *output_data,
              const void *kernel_call_back_before, const void *kernel_call_back_after) {
  OpContext<T> context;
  std::vector<Promise<int>> promises(output_data->size());
  context.sequential_num_ = RandInt::Instance().Get();
  context.results_ = &promises;
  context.output_data_ = output_data;
  context.kernel_call_back_before_ = kernel_call_back_before;
  context.kernel_call_back_after_ = kernel_call_back_after;

  auto collect = MindrtAsyncRun<T>(input_data, &context);
  collect.Wait();
  if (!collect.IsOK()) {
    return -1;
  }

  return 0;
}

}  // namespace mindspore

#endif  // MINDSPORE_CORE_MINDRT_INCLUDE_ACTOR_OP_ACTOR_H
