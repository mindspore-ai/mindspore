/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_GRAPH_EXECUTOR_DELEGATE_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_GRAPH_EXECUTOR_DELEGATE_H_

#include <memory>

#include "include/api/delegate.h"
#include "extendrt/session/lite_graph_executor.h"

namespace mindspore {
class GraphExecutorDelegate : public Delegate {
 public:
  GraphExecutorDelegate() = default;
  explicit GraphExecutorDelegate(std::shared_ptr<mindspore::LiteGraphExecutor> graph_executor)
      : graph_executor_(graph_executor) {}
  virtual ~GraphExecutorDelegate() = default;

  virtual Status Init();

  virtual Status Build(DelegateModel<schema::Primitive> *model);

  std::shared_ptr<mindspore::LiteGraphExecutor> GetGraphExecutor() { return graph_executor_; }

  void SetGraphExecutor(std::shared_ptr<mindspore::LiteGraphExecutor> graph_executor) {
    graph_executor_ = graph_executor;
  }

 private:
  std::shared_ptr<mindspore::LiteGraphExecutor> graph_executor_;
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_GRAPH_EXECUTOR_DELEGATE_H_
