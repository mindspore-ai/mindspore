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
#include <memory>
#include <string>
#include <utility>
#include <map>
#include <unordered_map>
#include "include/utils/tensor_py.h"
#include "include/utils/pynative/adapter.h"
#include "include/utils/pynative/hook.h"
#include "include/runtime/pipeline/pipeline.h"
#include "pynative/backward/grad_utils.h"
#include "pynative/backward/op_grad/func_grad.h"

namespace mindspore::pynative::autograd {
namespace {
// For store hook
uint64_t unique_id_ = 0;
std::unordered_map<uint64_t, std::weak_ptr<BackwardNode>> hook_id_node_map_;

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
      (*grad)[output_idx_] = tensor::ConvertToTensor(ret);
    } else {
      MS_LOG(EXCEPTION) << "Tensor hook should be return Tensor, but get type: "
                        << py::str(ret.get_type().attr("__name__")).cast<std::string>() << ".";
    }
  }
}

uint64_t RegisterHook::RegisterTensorBackwardHook(const tensor::TensorPtr &tensor, const py::function &hook) {
  ++unique_id_;
  MS_LOG(DEBUG) << "Register hook " << py::str(py::cast<py::object>(hook)).cast<std::string>() << " for tensor "
                << tensor->id() << " with handle " << unique_id_;

  auto grad_node = BuildAutoGradMeta(tensor);
  grad_node->AddPyTensorHook(
    unique_id_, std::make_unique<PyTensorBackwardNodePreHook>(hook, tensor->auto_grad_meta_data()->output_index()));
  hook_id_node_map_.emplace(unique_id_, grad_node);
  return unique_id_;
}

void RegisterHook::RemoveTensorBackwardHook(uint64_t handle_id) {
  MS_LOG(DEBUG) << "Remove hook by id " << handle_id;

  runtime::Pipeline::Get().WaitFrontend();
  runtime::Pipeline::Get().WaitBpropStage();

  if (const auto iter = hook_id_node_map_.find(handle_id); iter != hook_id_node_map_.end()) {
    if (auto grad_node = iter->second.lock(); grad_node != nullptr) {
      grad_node->RemovePyTensorHook(handle_id);
    }
    hook_id_node_map_.erase(iter);
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

void RegisterHook::ClearHookMap() { hook_id_node_map_.clear(); }

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
