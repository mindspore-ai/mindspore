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

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include "actor/actor.h"
#include "async/uuid_base.h"
#include "async/future.h"

namespace mindspore {
// OpActor data route.
struct OpArrow {
  OpArrow(int from_output_index, AID *to_op_id, int to_input_index)
      : from_output_index_(from_output_index), to_op_id_(to_op_id), to_input_index_(to_input_index) {}
  int from_output_index_;
  AID *to_op_id_;
  int to_input_index_;
};

// OpActor data.
template <typename T>
struct OpData {
  OpData(T *data, int to_input_index) : data_(data), to_input_index_(to_input_index) {}
  T *data_;
  int to_input_index_;
};

// The context of opActor running.
template <typename T>
struct OpContext {
  uuids::uuid *sequential_num_;
  std::vector<Promise<T *>> *results_;
};

using OpArrowPtr = std::shared_ptr<OpArrow>;
template <typename T>
using OpDataPtr = std::shared_ptr<OpData<T>>;

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
}  // namespace mindspore
