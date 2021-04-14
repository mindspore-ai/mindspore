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
struct OpArrow {
  OpArrow(int from_output_index, AID to_op_id, int to_input_index)
      : from_output_index_(from_output_index), to_op_id_(to_op_id), to_input_index_(to_input_index) {}
  int from_output_index_;
  AID to_op_id_;
  int to_input_index_;
};

// OpActor data.
template <typename T>
struct OpData {
  OpData(const AID &op_id, T *data, int index) : op_id_(op_id), data_(data), index_(index) {}
  AID op_id_;
  T *data_;
  int index_;
};

using OpArrowPtr = std::shared_ptr<OpArrow>;
template <typename T>
using OpDataPtr = std::shared_ptr<OpData<T>>;
// The context of opActor running.
template <typename T>
struct OpContext {
  uuids::uuid *sequential_num_;
  std::vector<OpDataPtr<T>> *outputData_;
  std::vector<Promise<int>> *results_;
  const void *kernel_call_back_before_;
  const void *kernel_call_back_after_;
  void SetFailed(int32_t code) {
    for (auto promise : *results_) {
      promise.SetFailed(code);
    }
  }
  void SetResult(size_t index, int value) { results_->at(index).SetValue(value); }
};

template <typename T>
class OpActor : public ActorBase {
 public:
  explicit OpActor(std::string op_name) : ActorBase(op_name) {}
  virtual ~OpActor() = default;
  virtual void OpRun(OpDataPtr<T> inputs, OpContext<T> *context = nullptr) {}

 protected:
  std::unordered_map<uuids::uuid *, std::vector<OpDataPtr<T>>> input_op_datas_;
  std::vector<OpArrowPtr> output_op_arrow_;
};

template <typename T>
Future<std::list<int>> MindrtAsyncRun(const std::vector<OpDataPtr<T>> &inputData, OpContext<T> *context) {
  std::list<Future<int>> futures;
  for (auto promise : *(context->results_)) {
    futures.push_back(promise.GetFuture());
  }
  Future<std::list<int>> collect = mindspore::Collect<int>(futures);

  for (auto data : inputData) {
    Async(data->op_id_, &mindspore::OpActor<T>::OpRun, data, context);
  }

  return collect;
}

template <typename T>
int MindrtRun(const std::vector<OpDataPtr<T>> &inputData, std::vector<OpDataPtr<T>> *outputData,
              const void *kernel_call_back_before, const void *kernel_call_back_after) {
  OpContext<T> context;
  std::vector<Promise<int>> promises(outputData->size());
  uuids::uuid uid;
  context.sequential_num_ = &uid;
  context.results_ = &promises;
  context.outputData_ = outputData;
  context.kernel_call_back_before_ = kernel_call_back_before;
  context.kernel_call_back_after_ = kernel_call_back_after;

  auto collect = MindrtAsyncRun<T>(inputData, &context);
  collect.Wait();
  if (!collect.IsOK()) {
    return -1;
  }
  return 0;
}
}  // namespace mindspore
