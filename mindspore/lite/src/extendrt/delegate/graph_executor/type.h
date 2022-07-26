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
#ifndef MINDSPORE_LITE_EXTENDRT_DELEGATE_GRAPH_EXECUTOR_TYPE_H_
#define MINDSPORE_LITE_EXTENDRT_DELEGATE_GRAPH_EXECUTOR_TYPE_H_

#include <memory>
#include <vector>

#include "extendrt/delegate/type.h"

namespace mindspore {
class GraphExecutorConfig : public DelegateConfig {
 public:
  GraphExecutorConfig() = default;
  explicit GraphExecutorConfig(const std::shared_ptr<Context> &context) : DelegateConfig(context) {}
  ~GraphExecutorConfig() = default;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_EXTENDRT_DELEGATE_GRAPH_EXECUTOR_TYPE_H_
