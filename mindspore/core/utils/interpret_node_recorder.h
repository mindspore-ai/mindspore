/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_UTILS_InterpretNodeRecorder_H_
#define MINDSPORE_CORE_UTILS_InterpretNodeRecorder_H_

#include "ir/anf.h"
#include "utils/hash_set.h"
#include "mindapi/base/macros.h"

namespace mindspore {
class MS_CORE_API InterpretNodeRecorder {
 public:
  explicit InterpretNodeRecorder(InterpretNodeRecorder &&) = delete;
  explicit InterpretNodeRecorder(const InterpretNodeRecorder &) = delete;
  InterpretNodeRecorder &operator=(const InterpretNodeRecorder &) = delete;
  InterpretNodeRecorder &operator=(InterpretNodeRecorder &&) = delete;
  static InterpretNodeRecorder &GetInstance();

  void PushPyInterpretNode(const AnfNodePtr &node) { (void)py_interpret_nodes_.emplace(node); }
  const mindspore::HashSet<AnfNodePtr> &PyInterpretNodes() const { return py_interpret_nodes_; }

  void PushPyExecuteNode(const AnfNodePtr &node) { (void)py_execute_nodes_.emplace(node); }
  const mindspore::HashSet<AnfNodePtr> &PyExecuteNodes() const { return py_execute_nodes_; }

  void Clear() {
    py_interpret_nodes_.clear();
    py_execute_nodes_.clear();
  }

 protected:
  InterpretNodeRecorder() = default;
  virtual ~InterpretNodeRecorder() = default;

 private:
  mindspore::HashSet<AnfNodePtr> py_interpret_nodes_;
  mindspore::HashSet<AnfNodePtr> py_execute_nodes_;
};
}  // namespace mindspore
#endif  // MINDSPORE_CORE_UTILS_InterpretNodeRecorder_H_
