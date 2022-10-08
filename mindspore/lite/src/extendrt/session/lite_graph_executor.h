/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_EXTENDRT_SESSION_LITE_GRAPH_EXECUTOR_H_
#define MINDSPORE_LITE_EXTENDRT_SESSION_LITE_GRAPH_EXECUTOR_H_

#include <vector>
#include <string>
#include <memory>

#include "include/api/types.h"
#include "runtime/hardware/device_context.h"

namespace mindspore {
class LiteGraphExecutor : public device::GraphExecutor {
 public:
  LiteGraphExecutor() = default;
  ~LiteGraphExecutor() = default;

  virtual bool Resize(const FuncGraphPtr &, const std::vector<tensor::Tensor> &inputs,
                      const std::vector<std::vector<int64_t>> &new_shapes) {
    (void)inputs;
    (void)new_shapes;
    return true;
  }
  virtual std::vector<tensor::Tensor> GetInputInfos(const FuncGraphPtr &) { return {}; }
  virtual std::vector<tensor::Tensor> GetOutputInfos(const FuncGraphPtr &) { return {}; }

  void SetBefore(const MSKernelCallBack &before) { before_ = before; }

  void SetAfter(const MSKernelCallBack &after) { after_ = after; }

 protected:
  MSKernelCallBack before_;
  MSKernelCallBack after_;
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_EXTENDRT_SESSION_LITE_GRAPH_EXECUTOR_H_
