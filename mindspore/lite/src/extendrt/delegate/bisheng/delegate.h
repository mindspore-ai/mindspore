/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_BISHENG_DELEGATE_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_BISHENG_DELEGATE_H_

#include <memory>

#include "extendrt/delegate/type.h"

namespace mindspore {
class BishengDelegate : public ExtendDelegate {
 public:
  BishengDelegate() = default;
  virtual ~BishengDelegate() = default;

  void ReplaceNodes(const std::shared_ptr<FuncGraph> &graph) override;

  bool IsDelegateNode(const std::shared_ptr<AnfNode> &node) override;

  std::shared_ptr<kernel::BaseKernel> CreateKernel(const std::shared_ptr<AnfNode> &node) override;
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_BISHENG_DELEGATE_H_
