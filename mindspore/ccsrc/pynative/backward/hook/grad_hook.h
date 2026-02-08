/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PYNATIVE_GRAD_FUNCTION_GRAD_HOOK_H_
#define MINDSPORE_CCSRC_PYNATIVE_GRAD_FUNCTION_GRAD_HOOK_H_

#include "include/utils/pynative/variable.h"

namespace mindspore::pynative::autograd {
class GradHook : public GradHookInterface {
  [[nodiscard]] bool requires_grad(const TensorPtr &self) const override;
  void set_requires_grad(const TensorPtr &self, bool requires_grad) override;
  [[nodiscard]] bool retains_grad(const TensorPtr &self) const override;
  void retain_grad(const TensorPtr &self) override;
  [[nodiscard]] TensorPtr grad(const TensorPtr &self) const override;
  void set_grad(const TensorPtr &self, const TensorPtr &grad) override;
  [[nodiscard]] BackwardNodePtr grad_node(const TensorPtr &self) const override;
  [[nodiscard]] bool is_leaf(const TensorPtr &self) const override;
  [[nodiscard]] size_t output_index(const TensorPtr &self) const override;
  // [[nodiscard]] size_t version(const TensorPtr &self) const override;
};
}  // namespace mindspore::pynative::autograd

#endif  // MINDSPORE_CCSRC_PYNATIVE_GRAD_FUNCTION_GRAD_HOOK_H_
