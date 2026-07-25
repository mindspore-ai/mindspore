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

#ifndef MINDSPORE_CCSRC_UTILS_SLIDING_WINDOW_SOURCE_GROUP_MANAGER_H_
#define MINDSPORE_CCSRC_UTILS_SLIDING_WINDOW_SOURCE_GROUP_MANAGER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "device_address/device_address.h"
#include "ir/tensor.h"
#include "pynative/hyper_offload/sliding_window/sliding_window_internal.h"

namespace mindspore {
namespace utils {
namespace sliding_window {

enum class ResidencyStatus {
  kOnDevice,
  kOnHost,
};

struct ViewTensorRef {
  TensorId tensor_id{0};
  std::weak_ptr<tensor::Tensor> tensor;
};

struct SourceGroup {
  TensorId source_id{0};
  std::weak_ptr<tensor::Tensor> source;
  std::vector<ViewTensorRef> views;
  std::weak_ptr<device::DeviceAddress> shared_addr;
  TensorPtr host_copy{nullptr};
  ResidencyStatus status{ResidencyStatus::kOnDevice};
  size_t aligned_size{0};
  uint64_t last_access_timestamp{0};
  bool offloadable{false};
  bool source_registered{false};
};

class SourceGroupManager {
 public:
  SourceGroupManager() = default;
  ~SourceGroupManager() = default;

  SourceGroup *RegisterSource(const TensorPtr &tensor, uint64_t timestamp);
  SourceGroup *RegisterView(const TensorPtr &input, const TensorPtr &view_output, uint64_t timestamp);
  void RegisterAlias(const TensorPtr &tensor, SourceGroup *group);

  TensorId ResolveSourceId(TensorId tensor_id) const;
  TensorId ResolveSourceId(const TensorPtr &tensor) const;

  SourceGroup *GetGroup(TensorId source_id);
  const SourceGroup *GetGroup(TensorId source_id) const;
  void UpdateAddressMapping(SourceGroup *group, const device::DeviceAddressPtr &new_addr);

  std::unordered_set<TensorId> BuildProtectedSourceIdSet(const std::vector<TensorPtr> &tensors) const;

  void UpdateAccess(TensorId source_id, uint64_t timestamp);
  std::vector<TensorId> GetOffloadCandidatesLRU() const;
  std::vector<TensorId> GetAllSourceIds() const;

  std::vector<TensorId> PurgeExpired();
  size_t Size() const { return source_groups_.size(); }

 private:
  SourceGroup *GetOrCreateGroup(const TensorPtr &tensor, uint64_t timestamp);
  SourceGroup *FindGroupForTensor(const TensorPtr &tensor);
  const SourceGroup *FindGroupForTensor(const TensorPtr &tensor) const;
  TensorId ResolveSourceIdByAddress(const device::DeviceAddressPtr &addr) const;
  void AddViewToGroup(SourceGroup *group, const TensorPtr &view_output);
  void PruneExpiredViews(SourceGroup *group);
  bool IsGroupExpired(const SourceGroup &group) const;
  void RemoveMappingsForSource(TensorId source_id);

  std::unordered_map<TensorId, TensorId> tensor_to_source_;
  std::unordered_map<device::DeviceAddress *, TensorId> addr_to_source_;
  std::unordered_map<TensorId, SourceGroup> source_groups_;
};

}  // namespace sliding_window
}  // namespace utils
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_UTILS_SLIDING_WINDOW_SOURCE_GROUP_MANAGER_H_
