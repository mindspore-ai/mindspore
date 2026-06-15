/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "mindspore/ccsrc/utils/sliding_window/source_group_manager.h"

#include <algorithm>
#include <utility>

namespace mindspore {
namespace utils {
namespace sliding_window {

SourceGroup *SourceGroupManager::GetOrCreateGroup(const TensorPtr &tensor, uint64_t timestamp) {
  if (tensor == nullptr) {
    return nullptr;
  }

  TensorId source_id = tensor->id();
  auto iter = source_groups_.find(source_id);
  if (iter == source_groups_.end()) {
    SourceGroup group;
    group.source_id = source_id;
    group.source = tensor;
    group.last_access_timestamp = timestamp;
    iter = source_groups_.emplace(source_id, std::move(group)).first;
  }

  tensor_to_source_[tensor->id()] = source_id;
  return &(iter->second);
}

TensorId SourceGroupManager::ResolveSourceIdByAddress(const device::DeviceAddressPtr &addr) const {
  if (addr == nullptr) {
    return 0;
  }
  auto addr_iter = addr_to_source_.find(addr.get());
  if (addr_iter == addr_to_source_.end()) {
    return 0;
  }
  auto group_iter = source_groups_.find(addr_iter->second);
  if (group_iter == source_groups_.end()) {
    return 0;
  }
  auto shared_addr = group_iter->second.shared_addr.lock();
  if (shared_addr == nullptr || shared_addr.get() != addr.get()) {
    return 0;
  }
  return addr_iter->second;
}

SourceGroup *SourceGroupManager::FindGroupForTensor(const TensorPtr &tensor) {
  if (tensor == nullptr) {
    return nullptr;
  }
  auto tensor_iter = tensor_to_source_.find(tensor->id());
  if (tensor_iter != tensor_to_source_.end()) {
    return GetGroup(tensor_iter->second);
  }
  auto source_id = ResolveSourceIdByAddress(tensor->device_address());
  return source_id == 0 ? nullptr : GetGroup(source_id);
}

const SourceGroup *SourceGroupManager::FindGroupForTensor(const TensorPtr &tensor) const {
  if (tensor == nullptr) {
    return nullptr;
  }
  auto tensor_iter = tensor_to_source_.find(tensor->id());
  if (tensor_iter != tensor_to_source_.end()) {
    return GetGroup(tensor_iter->second);
  }
  auto source_id = ResolveSourceIdByAddress(tensor->device_address());
  return source_id == 0 ? nullptr : GetGroup(source_id);
}

void SourceGroupManager::UpdateAddressMapping(SourceGroup *group, const device::DeviceAddressPtr &new_addr) {
  if (group == nullptr) {
    return;
  }
  for (auto iter = addr_to_source_.begin(); iter != addr_to_source_.end();) {
    if (iter->second == group->source_id) {
      iter = addr_to_source_.erase(iter);
    } else {
      ++iter;
    }
  }
  group->shared_addr = new_addr;
  if (new_addr != nullptr) {
    addr_to_source_[new_addr.get()] = group->source_id;
  }
}

void SourceGroupManager::AddViewToGroup(SourceGroup *group, const TensorPtr &view_output) {
  if (group == nullptr || view_output == nullptr) {
    return;
  }
  TensorId view_id = view_output->id();
  tensor_to_source_[view_id] = group->source_id;

  auto view_iter = std::find_if(group->views.begin(), group->views.end(),
                                [view_id](const auto &view) { return view.tensor_id == view_id; });
  if (view_iter == group->views.end()) {
    group->views.push_back({view_id, view_output});
  } else {
    view_iter->tensor = view_output;
  }
}

void SourceGroupManager::RegisterAlias(const TensorPtr &tensor, SourceGroup *group) {
  if (tensor == nullptr || group == nullptr) {
    return;
  }
  if (tensor->id() == group->source_id) {
    tensor_to_source_[tensor->id()] = group->source_id;
    return;
  }
  AddViewToGroup(group, tensor);
}

SourceGroup *SourceGroupManager::RegisterSource(const TensorPtr &tensor, uint64_t timestamp) {
  if (tensor == nullptr || !CanOffloadTensor(tensor)) {
    return nullptr;
  }

  if (auto *existing_group = FindGroupForTensor(tensor); existing_group != nullptr) {
    if (existing_group->source_id != tensor->id()) {
      AddViewToGroup(existing_group, tensor);
      existing_group->last_access_timestamp = timestamp;
      return existing_group;
    }
  }

  TensorId source_id = tensor->id();
  auto iter = source_groups_.find(source_id);
  if (iter == source_groups_.end()) {
    SourceGroup group;
    group.source_id = source_id;
    iter = source_groups_.emplace(source_id, std::move(group)).first;
  }

  auto &group = iter->second;
  group.source = tensor;
  group.aligned_size = AlignToPoolSize(tensor->Size());
  group.offloadable = true;
  group.source_registered = true;
  group.last_access_timestamp = timestamp;
  UpdateAddressMapping(&group, tensor->device_address());
  if (tensor->device_address() != nullptr && tensor->device_address()->GetDevicePtr() != nullptr) {
    group.status = ResidencyStatus::kOnDevice;
  }
  tensor_to_source_[tensor->id()] = source_id;
  return &group;
}

SourceGroup *SourceGroupManager::RegisterView(const TensorPtr &input, const TensorPtr &view_output,
                                              uint64_t timestamp) {
  if (input == nullptr || view_output == nullptr) {
    return nullptr;
  }

  auto *group = FindGroupForTensor(input);
  if (group == nullptr) {
    group = FindGroupForTensor(view_output);
  }
  if (group == nullptr && CanOffloadTensor(input)) {
    group = RegisterSource(input, timestamp);
  }
  if (group == nullptr) {
    group = GetOrCreateGroup(input, timestamp);
    group->offloadable = false;
    group->source_registered = false;
    group->aligned_size = AlignToPoolSize(input->Size());
  }
  if (group == nullptr) {
    return nullptr;
  }

  RegisterAlias(input, group);
  AddViewToGroup(group, view_output);

  if (auto source = group->source.lock(); source != nullptr) {
    if (group->source_registered) {
      group->offloadable = CanOffloadTensor(source);
    }
    group->aligned_size = AlignToPoolSize(source->Size());
    if (source->device_address() != nullptr) {
      UpdateAddressMapping(group, source->device_address());
    }
  } else {
    if (input->id() == group->source_id) {
      group->source = input;
      group->aligned_size = AlignToPoolSize(input->Size());
      if (group->source_registered) {
        group->offloadable = CanOffloadTensor(input);
      }
    }
    if (input->device_address() != nullptr) {
      UpdateAddressMapping(group, input->device_address());
    } else if (view_output->device_address() != nullptr) {
      UpdateAddressMapping(group, view_output->device_address());
    }
  }

  group->last_access_timestamp = timestamp;
  return group;
}

TensorId SourceGroupManager::ResolveSourceId(TensorId tensor_id) const {
  auto iter = tensor_to_source_.find(tensor_id);
  if (iter != tensor_to_source_.end()) {
    return iter->second;
  }
  return tensor_id;
}

TensorId SourceGroupManager::ResolveSourceId(const TensorPtr &tensor) const {
  if (tensor == nullptr) {
    return 0;
  }
  auto tensor_iter = tensor_to_source_.find(tensor->id());
  if (tensor_iter != tensor_to_source_.end()) {
    return tensor_iter->second;
  }
  auto source_id = ResolveSourceIdByAddress(tensor->device_address());
  if (source_id != 0) {
    return source_id;
  }
  return tensor->id();
}

SourceGroup *SourceGroupManager::GetGroup(TensorId source_id) {
  auto iter = source_groups_.find(source_id);
  if (iter == source_groups_.end()) {
    return nullptr;
  }
  return &(iter->second);
}

const SourceGroup *SourceGroupManager::GetGroup(TensorId source_id) const {
  auto iter = source_groups_.find(source_id);
  if (iter == source_groups_.end()) {
    return nullptr;
  }
  return &(iter->second);
}

std::unordered_set<TensorId> SourceGroupManager::BuildProtectedSourceIdSet(
    const std::vector<TensorPtr> &tensors) const {
  std::unordered_set<TensorId> ids;
  ids.reserve(tensors.size());
  for (const auto &tensor : tensors) {
    if (tensor == nullptr) {
      continue;
    }
    ids.insert(ResolveSourceId(tensor));
  }
  return ids;
}

void SourceGroupManager::UpdateAccess(TensorId source_id, uint64_t timestamp) {
  auto *group = GetGroup(source_id);
  if (group == nullptr) {
    return;
  }
  group->last_access_timestamp = timestamp;
}

std::vector<TensorId> SourceGroupManager::GetOffloadCandidatesLRU() const {
  std::vector<TensorId> ids;
  ids.reserve(source_groups_.size());
  for (const auto &[source_id, group] : source_groups_) {
    if (group.status == ResidencyStatus::kOnDevice && group.offloadable) {
      ids.push_back(source_id);
    }
  }
  std::sort(ids.begin(), ids.end(), [this](TensorId lhs, TensorId rhs) {
    const auto *lhs_group = GetGroup(lhs);
    const auto *rhs_group = GetGroup(rhs);
    if (lhs_group == nullptr || rhs_group == nullptr) {
      return lhs < rhs;
    }
    if (lhs_group->last_access_timestamp == rhs_group->last_access_timestamp) {
      return lhs < rhs;
    }
    return lhs_group->last_access_timestamp < rhs_group->last_access_timestamp;
  });
  return ids;
}

std::vector<TensorId> SourceGroupManager::GetAllSourceIds() const {
  std::vector<TensorId> ids;
  ids.reserve(source_groups_.size());
  for (const auto &item : source_groups_) {
    ids.push_back(item.first);
  }
  return ids;
}

void SourceGroupManager::PruneExpiredViews(SourceGroup *group) {
  if (group == nullptr) {
    return;
  }
  for (auto iter = group->views.begin(); iter != group->views.end();) {
    if (iter->tensor.expired()) {
      tensor_to_source_.erase(iter->tensor_id);
      iter = group->views.erase(iter);
    } else {
      ++iter;
    }
  }
}

bool SourceGroupManager::IsGroupExpired(const SourceGroup &group) const {
  if (!group.source.expired()) {
    return false;
  }
  if (!group.views.empty()) {
    return false;
  }
  return group.shared_addr.expired();
}

void SourceGroupManager::RemoveMappingsForSource(TensorId source_id) {
  for (auto iter = tensor_to_source_.begin(); iter != tensor_to_source_.end();) {
    if (iter->second == source_id) {
      iter = tensor_to_source_.erase(iter);
    } else {
      ++iter;
    }
  }
  for (auto iter = addr_to_source_.begin(); iter != addr_to_source_.end();) {
    if (iter->second == source_id) {
      iter = addr_to_source_.erase(iter);
    } else {
      ++iter;
    }
  }
}

std::vector<TensorId> SourceGroupManager::PurgeExpired() {
  std::vector<TensorId> purged;
  for (auto iter = source_groups_.begin(); iter != source_groups_.end();) {
    PruneExpiredViews(&(iter->second));
    if (IsGroupExpired(iter->second)) {
      purged.push_back(iter->first);
      RemoveMappingsForSource(iter->first);
      iter = source_groups_.erase(iter);
    } else {
      ++iter;
    }
  }
  return purged;
}

}  // namespace sliding_window
}  // namespace utils
}  // namespace mindspore
