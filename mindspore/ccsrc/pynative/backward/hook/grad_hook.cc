/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "pynative/backward/hook/grad_hook.h"
#include <memory>
#include <string>
#include <utility>
#include "pynative/backward/op_grad/func_grad.h"
#include "pynative/backward/grad_utils.h"
#include "mindspore/core/include/ir/tensor.h"
#include "include/utils/pynative/hook.h"

namespace mindspore::pynative::autograd {
namespace {
struct GradHookRegister {
 public:
  GradHookRegister() noexcept { tensor::Tensor::InitializeGradImpl(std::make_unique<GradHook>()); }
};
[[maybe_unused]] GradHookRegister grad_hook_register;
}  // namespace

BackwardNodePtr GradHook::grad_node(const TensorPtr &self) const {
  runtime::Pipeline::Get().WaitBpropStage();
  return SafeGetGradNodeImpl(self);
}

TensorPtr GradHook::grad(const TensorPtr &self) const {
  auto grad_meta = self->auto_grad_meta_data();
  if (grad_meta == nullptr) {
    return nullptr;
  }
  return grad_meta->grad();
}

void GradHook::set_grad(const TensorPtr &self, const TensorPtr &grad) {
  auto grad_meta = self->auto_grad_meta_data();
  if (grad_meta == nullptr) {
    grad_meta = std::make_shared<AutoGradMetaData>();
    self->set_auto_grad_meta_data(grad_meta);
  }
  grad_meta->set_grad(grad);
}

bool GradHook::is_leaf(const TensorPtr &self) const {
  runtime::Pipeline::Get().WaitBpropStage();
  const auto &grad_node = impl::GetUnsafeGradNodeImpl(self);
  if (grad_node == nullptr || grad_node->IsLeaf()) {
    return true;
  }
  return false;
}

size_t GradHook::output_index(const TensorPtr &self) const {
  runtime::Pipeline::Get().WaitBpropStage();
  const auto &auto_grad_meta_data = self->auto_grad_meta_data();
  if (auto_grad_meta_data != nullptr) {
    return auto_grad_meta_data->output_index();
  }
  return 0;
}

bool GradHook::requires_grad(const TensorPtr &self) const {
  if (self->param_info() != nullptr && self->param_info()->requires_grad()) {
    return true;
  }
  const auto &grad_meta = self->auto_grad_meta_data();
  if (grad_meta == nullptr) {
    return false;
  }
  runtime::Pipeline::Get().WaitBpropStage();
  return self->auto_grad_meta_data()->requires_grad();
}

void GradHook::set_requires_grad(const TensorPtr &self, bool requires_grad) {
  runtime::Pipeline::Get().WaitBpropStage();
  if (self->param_info() != nullptr) {
    self->param_info()->set_requires_grad(requires_grad);
  }
  if (!is_leaf(self) && !requires_grad) {
    MS_LOG(EXCEPTION) << "You can only set requires_grad false to a leaf tensor! You can use tensor.detach() to "
                         "set output tensor to calculate grad.";
  }
  auto grad_meta = self->auto_grad_meta_data();
  if (grad_meta == nullptr) {
    if (!requires_grad) {
      return;
    }
    grad_meta = std::make_shared<AutoGradMetaData>();
    self->set_auto_grad_meta_data(grad_meta);
  }
  grad_meta->set_requires_grad(requires_grad);
  if (impl::GetUnsafeGradNodeImpl(self) == nullptr && requires_grad) {
    MS_LOG(DEBUG) << "create leaf tensor requires grad true" << self->ToString();
    grad_meta->set_grad_node(std::make_shared<autograd::LeafNode>(
      self->param_info() != nullptr ? self->param_info()->name() : "input_" + std::to_string(self->id()), self,
      self->shape(), self->Dtype(), self->is_parameter()));
    grad_meta->set_input_type(self->is_parameter() ? InputType::kParameter : InputType::kInput);
  }
}

bool GradHook::retains_grad(const TensorPtr &self) const {
  const auto auto_grad_meta_data = impl::GetAutogradMetaImpl(self);
  if (auto_grad_meta_data != nullptr) {
    return auto_grad_meta_data->retains_grad();
  }
  return false;
}

void GradHook::retain_grad(const TensorPtr &self) {
  runtime::Pipeline::Get().WaitBpropStage();
  MS_EXCEPTION_IF_CHECK_FAIL(self->requires_grad(), "Can't retain grad, if tensor has requires=False");
  if (self->is_leaf()) {
    return;
  }
  std::weak_ptr<Tensor> weak_tensor(self);
  const auto retain_grad_fn = [weak_tensor](const TensorPtr &grad) {
    MS_LOG(DEBUG) << "Begin execute retain_grad hook";
    if (!weak_tensor.expired() && grad != nullptr) {
      auto tensor = weak_tensor.lock();
      if (tensor->grad() == nullptr) {
        tensor->set_grad(AutoGradUtil::Clone(grad));
      } else {
        tensor->set_grad(AutoGradUtil::Add(tensor->grad(), grad));
      }
    }
  };
  auto auto_grad_meta_data = impl::GetAutogradMetaImpl(self);
  MS_EXCEPTION_IF_NULL(auto_grad_meta_data);
  auto grad_node = auto_grad_meta_data->UnsafeGetGradNodeImpl();
  MS_EXCEPTION_IF_NULL(grad_node);
  auto retain_grad_hook = std::make_unique<RetainGradHook>(retain_grad_fn);
  grad_node->AddRetainGradHook(auto_grad_meta_data->output_index(), std::move(retain_grad_hook));
  auto_grad_meta_data->set_retains_grad(true);
}
}  // namespace mindspore::pynative::autograd
