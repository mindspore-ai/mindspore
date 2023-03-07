/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_LABEL_ASSIGN_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_LABEL_ASSIGN_H_

#include <memory>
#include <map>
#include "include/backend/kernel_graph.h"
#include "include/common/utils/contract.h"

namespace mindspore {
namespace device {
namespace ascend {
class AscendLabelAssign {
 public:
  static AscendLabelAssign &GetInstance() {
    static AscendLabelAssign instance;  // Guaranteed to be destroyed.
    return instance;
  }

  AscendLabelAssign(const AscendLabelAssign &) = delete;
  AscendLabelAssign &operator=(const AscendLabelAssign &) = delete;

  void AssignLabel(NotNull<std::shared_ptr<session::KernelGraph>> graph);
  uint32_t GetLabelNum(NotNull<const session::KernelGraph *> graph);
  uint32_t GetLabelNum(NotNull<std::shared_ptr<session::KernelGraph>> graph);

 private:
  AscendLabelAssign() = default;
  ~AscendLabelAssign() = default;

  std::map<const session::KernelGraph *, uint32_t> label_num_;
  std::mutex label_num_mutex_;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_LABEL_ASSIGN_H_
