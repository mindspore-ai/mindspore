/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "pynative/backward/hook/hook_py.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <map>
#include <unordered_map>
#include "include/utils/tensor_py.h"
#include "include/utils/pynative/adapter.h"
#include "include/utils/pynative/dispatch/py_convert.h"
#include "include/utils/pynative/hook.h"
#include "include/runtime/pipeline/pipeline.h"
#include "pynative/backward/grad_utils.h"
#include "pynative/backward/op_grad/func_grad.h"

namespace mindspore::pynative::autograd {
namespace {
class TensorBackwardHookRegistry {
 public:
  uint64_t NewHandleId() { return handle_id_.fetch_add(1, std::memory_order_relaxed) + 1; }

  void Register(uint64_t handle_id, const BackwardNodePtr &grad_node) {
    std::lock_guard<std::mutex> lock(mutex_);
    hook_id_node_map_.emplace(handle_id, grad_node);
  }

  BackwardNodePtr Pop(uint64_t handle_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto iter = hook_id_node_map_.find(handle_id);
    if (iter == hook_id_node_map_.end()) {
      return nullptr;
    }
    auto grad_node = iter->second.lock();
    hook_id_node_map_.erase(iter);
    return grad_node;
  }

  void Erase(uint64_t handle_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    hook_id_node_map_.erase(handle_id);
  }

  void Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    hook_id_node_map_.clear();
  }

  size_t Size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return hook_id_node_map_.size();
  }

 private:
  std::atomic<uint64_t> handle_id_{0};
  mutable std::mutex mutex_;
  std::unordered_map<uint64_t, std::weak_ptr<BackwardNode>> hook_id_node_map_;
};

const std::shared_ptr<TensorBackwardHookRegistry> &GetTensorBackwardHookRegistry() {
  static const auto registry = std::make_shared<TensorBackwardHookRegistry>();
  return registry;
}

class RegisteredPyTensorBackwardNodePreHook final : public PyTensorBackwardNodePreHook {
 public:
  RegisteredPyTensorBackwardNodePreHook(const py::function &hook_fn, size_t output_idx, uint64_t handle_id,
                                        const std::shared_ptr<TensorBackwardHookRegistry> &registry)
      : PyTensorBackwardNodePreHook(hook_fn, output_idx), handle_id_(handle_id), registry_(registry) {}

  ~RegisteredPyTensorBackwardNodePreHook() override {
    if (auto registry = registry_.lock(); registry != nullptr) {
      registry->Erase(handle_id_);
    }
  }

 private:
  uint64_t handle_id_;
  std::weak_ptr<TensorBackwardHookRegistry> registry_;
};

BackwardNodePtr BuildAutoGradMeta(const tensor::TensorPtr &tensor) {
  runtime::Pipeline::Get().WaitFrontend();
  runtime::Pipeline::Get().WaitBpropStage();
  auto auto_grad_meta_data = impl::GetAutogradMetaImpl(tensor);
  if (auto_grad_meta_data == nullptr) {
    if (tensor->param_info() != nullptr && !tensor->param_info()->requires_grad()) {
      MS_LOG(EXCEPTION) << "The tensor requires grad is false, which can not register tensor hook";
    }
    MS_LOG(DEBUG) << "Create leaf node for: " << tensor->ToString();
    auto_grad_meta_data = std::make_shared<AutoGradMetaData>();
    auto fn = std::make_shared<autograd::LeafNode>(tensor->param_info() != nullptr
                                                     ? tensor->param_info()->name()
                                                     : "register_hook_input_" + std::to_string(tensor->id()),
                                                   tensor, tensor->shape(), tensor->Dtype(), tensor->is_parameter());
    auto_grad_meta_data->set_requires_grad(true);
    auto_grad_meta_data->set_grad_node(fn);
    tensor->set_auto_grad_meta_data(auto_grad_meta_data);
    return fn;
  }
  auto grad_node = auto_grad_meta_data->UnsafeGetGradNodeImpl();
  if (grad_node == nullptr) {
    MS_LOG(EXCEPTION) << "The tensor requires grad is false, which can not register tensor hook";
  }
  return grad_node;
}
}  // namespace

void PyTensorBackwardNodePreHook::operator()(ValuePtrList *grad) {
  if (output_idx_ >= grad->size()) {
    MS_LOG(EXCEPTION) << "PyTensor hook output_idx out of range";
  }

  py::gil_scoped_acquire gil;
  const auto py_grad = CValueToPybindObj((*grad)[output_idx_]);
  const auto ret = hook_fn_(py_grad);
  if (!ret.is_none()) {
    if (tensor::IsTensorPy(ret)) {
      (*grad)[output_idx_] = ConvertPyObjectToTensorWithSourcePyObj(ret);
    } else {
      MS_LOG(EXCEPTION) << "Tensor hook should be return Tensor, but get type: "
                        << py::str(ret.get_type().attr("__name__")).cast<std::string>() << ".";
    }
  }
}

uint64_t RegisterHook::RegisterTensorBackwardHook(const tensor::TensorPtr &tensor, const py::function &hook) {
  const auto &registry = GetTensorBackwardHookRegistry();
  const auto handle_id = registry->NewHandleId();
  MS_LOG(DEBUG) << "Register hook " << py::str(py::cast<py::object>(hook)).cast<std::string>() << " for tensor "
                << tensor->id() << " with handle " << handle_id;

  auto grad_node = BuildAutoGradMeta(tensor);
  grad_node->AddPyTensorHook(handle_id, std::make_unique<RegisteredPyTensorBackwardNodePreHook>(
                                          hook, tensor->auto_grad_meta_data()->output_index(), handle_id, registry));
  registry->Register(handle_id, grad_node);
  return handle_id;
}

void RegisterHook::RemoveTensorBackwardHook(uint64_t handle_id) {
  MS_LOG(DEBUG) << "Remove hook by id " << handle_id;

  runtime::Pipeline::Get().WaitFrontend();
  runtime::Pipeline::Get().WaitBpropStage();

  if (auto grad_node = GetTensorBackwardHookRegistry()->Pop(handle_id); grad_node != nullptr) {
    grad_node->RemovePyTensorHook(handle_id);
  }
}

py::list RegisterHook::GetHooks(const tensor::TensorPtr &tensor) {
  py::list hooks;

  runtime::Pipeline::Get().WaitFrontend();
  runtime::Pipeline::Get().WaitBpropStage();

  if (const auto auto_grad_meta_data = impl::GetAutogradMetaImpl(tensor)) {
    const auto output_idx = auto_grad_meta_data->output_index();
    if (const auto grad_node = auto_grad_meta_data->UnsafeGetGradNodeImpl()) {
      if (const auto &py_tensor_pre_hooks = grad_node->py_tensor_pre_hooks()) {
        for (const auto &item : *py_tensor_pre_hooks) {
          const auto &py_hooks = item.second;
          if (py_hooks->output_idx_ == output_idx) {
            hooks.append(py_hooks->hook_fn_);
          }
        }
      }
    }
  }

  return hooks;
}

unsigned RegisterHook::RegisterCppTensorBackwardHook(const tensor::TensorPtr &tensor, const CppHookFn &hook) {
  auto grad_node = BuildAutoGradMeta(tensor);
  auto cpp_hook = std::make_unique<CppTensorBackwardNodePreHook>(hook, tensor->auto_grad_meta_data()->output_index());
  return grad_node->AddCppTensorHook(std::move(cpp_hook));
}

void RegisterHook::RemoveCppTensorBackwardHook(const tensor::TensorPtr &tensor, unsigned hook_id) {
  if (const auto grad_node = impl::GetUnsafeGradNodeImpl(tensor)) {
    grad_node->RemoveCppTensorHook(hook_id);
  }
}

void RegisterHook::ClearHookMap() { GetTensorBackwardHookRegistry()->Clear(); }

size_t RegisterHook::GetTensorBackwardHookMapSize() { return GetTensorBackwardHookRegistry()->Size(); }

struct HookAdapterRegister {
  HookAdapterRegister() {
    MS_LOG(DEBUG) << "Register hook adapter";
    HookAdapter::SetRegisterTensorBackwardHookHandler(
      [](const tensor::TensorPtr &tensor, const py::function &hook) -> uint64_t {
        return RegisterHook::RegisterTensorBackwardHook(tensor, hook);
      });

    HookAdapter::SetRemoveTensorBackwardHookHandler(
      [](uint64_t id) -> void { RegisterHook::RemoveTensorBackwardHook(id); });

    HookAdapter::SetGetHooksHandler(
      [](const tensor::TensorPtr &tensor) -> py::list { return RegisterHook::GetHooks(tensor); });
  }
} hook_adapter_register;
}  // namespace mindspore::pynative::autograd
