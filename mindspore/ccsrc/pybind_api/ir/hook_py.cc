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

#include "pybind_api/ir/hook_py.h"
#include <memory>
#include <string>
#include "include/common/utils/hook.h"

namespace mindspore {
namespace tensor {

namespace {
AutoGradMetaDataWeakPtr BuildAutoGradMeta(const tensor::Tensor &tensor) {
  auto auto_grad_meta_data = tensor.auto_grad_meta_data();
  if (auto_grad_meta_data == nullptr) {
    auto_grad_meta_data = std::make_shared<AutoGradMetaData>();
    const_cast<Tensor &>(tensor).set_auto_grad_meta_data(auto_grad_meta_data);
    MS_LOG(DEBUG) << "Tensor has no auto_grad_meta_data, build it";
  }
  return {auto_grad_meta_data};
}

inline uint64_t GetTensorNumId(const std::string &id) { return std::stoull(id.substr(1)); }
}  // namespace

std::map<uint64_t, std::pair<AutoGradMetaDataWeakPtr, TensorBackwardHookPtr>> RegisterHook::hook_meta_fn_map_ = {};

uint64_t RegisterHook::RegisterTensorBackwardHook(const Tensor &tensor, const py::function &hook) {
  // Delete char 'T'
  const auto &tensor_id = GetTensorNumId(tensor.id());
  MS_LOG(DEBUG) << "Register hook " << py::str(py::cast<py::object>(hook)).cast<std::string>() << " for tensor "
                << tensor.ToString() << " with id " << tensor_id;
  auto meta = BuildAutoGradMeta(tensor);
  MS_EXCEPTION_IF_NULL(meta.lock());
  meta.lock()->ClearBackwardHooks();
  auto tensor_backward_hook = std::make_shared<TensorBackwardHook>(tensor_id, hook);
  meta.lock()->AddBackwardHook(tensor_id, tensor_backward_hook);
  // Just keep last hook
  hook_meta_fn_map_[tensor_id] = {meta, tensor_backward_hook};
  return tensor_id;
}

void RegisterHook::RemoveTensorBackwardHook(uint64_t id) {
  const auto it = hook_meta_fn_map_.find(id);
  if (it == hook_meta_fn_map_.end()) {
    return;
  }
  auto meta = it->second.first.lock();
  if (meta == nullptr) {
    return;
  }
  MS_LOG(DEBUG) << "Remove hook by id " << id;
  meta->RemoveBackwardHook(id);
}

void RegisterHook::UpdateTensorBackwardHook(const AutoGradMetaDataPtr &auto_grad_meta_data, const std::string &id) {
  MS_EXCEPTION_IF_NULL(auto_grad_meta_data);
  const auto &tensor_id = GetTensorNumId(id);
  auto it = hook_meta_fn_map_.find(tensor_id);
  if (it != hook_meta_fn_map_.end()) {
    MS_LOG(DEBUG) << "Update tensor backward hook for tensor id " << id;
    auto_grad_meta_data->AddBackwardHook(tensor_id, it->second.second);
    // Update remove handle
    hook_meta_fn_map_[tensor_id].first = std::weak_ptr<AutoGradMetaData>(auto_grad_meta_data);
  }
}
}  // namespace tensor
}  // namespace mindspore
