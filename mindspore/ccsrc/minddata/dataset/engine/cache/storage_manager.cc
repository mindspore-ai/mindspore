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
#include "minddata/dataset/engine/cache/storage_manager.h"

#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>
#include "utils/ms_utils.h"
#include "minddata/dataset/util/log_adapter.h"
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/services.h"

namespace mindspore {
namespace dataset {
std::string StorageManager::GetBaseName(const std::string &prefix, int32_t file_id) {
  std::ostringstream oss;
  oss << prefix << std::setfill('0') << std::setw(5) << file_id;
  return oss.str();
}

std::string StorageManager::ConstructFileName(const std::string &prefix, int32_t file_id, const std::string &suffix) {
  std::string base_name = GetBaseName(prefix, file_id);
  return (base_name + "." + suffix);
}

Status StorageManager::AddOneContainer() {
  const std::string kPrefix = "IMG";
  const std::string kSuffix = "LB";
  Path container_name = root_ / ConstructFileName(kPrefix, file_id_, kSuffix);
  std::shared_ptr<StorageContainer> sc;
  RETURN_IF_NOT_OK(StorageContainer::CreateStorageContainer(&sc, container_name.toString()));
  containers_.push_back(sc);
  file_id_++;
  return Status::OK();
}

Status StorageManager::DoServiceStart() {
  containers_.reserve(1000);
  if (root_.IsDirectory()) {
    RETURN_IF_NOT_OK(AddOneContainer());
  } else {
    RETURN_STATUS_UNEXPECTED("Not a directory");
  }
  return Status::OK();
}

Status StorageManager::Write(key_type *key, const std::vector<ReadableSlice> &buf) {
  RETURN_UNEXPECTED_IF_NULL(key);
  size_t sz = 0;
  for (auto &v : buf) {
    sz += v.GetSize();
  }
  if (sz == 0) {
    RETURN_STATUS_UNEXPECTED("Unexpected 0 length");
  }
  std::shared_ptr<StorageContainer> cont;
  key_type out_key;
  value_type out_value;
  bool create_new_container = false;
  size_t last_num_container = -1;
  do {
    SharedLock lock_s(&rw_lock_);
    size_t num_containers = containers_.size();
    if (create_new_container && (num_containers == last_num_container)) {
      // Upgrade to exclusvie lock.
      lock_s.Upgrade();
      create_new_container = false;
      // Check again if someone has already added a
      // new container after we got the x lock
      if (containers_.size() == num_containers) {
        RETURN_IF_NOT_OK(AddOneContainer());
      }
      // Refresh how many containers there are.
      num_containers = containers_.size();
      // Downgrade back to shared lock
      lock_s.Downgrade();
    }
    if (num_containers == 0) {
      RETURN_STATUS_UNEXPECTED("num_containers is zero");
    }
    // Go to the last container to insert.
    cont = containers_.at(num_containers - 1);
    off64_t offset;
    Status rc = cont->Insert(buf, &offset);
    if (rc.StatusCode() == StatusCode::kMDBuddySpaceFull) {
      create_new_container = true;
      // Remember how many containers we saw. In the next iteration we will do a comparison to see
      // if someone has already created it.
      last_num_container = num_containers;
    } else if (rc.IsOk()) {
      out_value = std::make_pair(num_containers - 1, std::make_pair(offset, sz));
      RETURN_IF_NOT_OK(index_.insert(out_value, &out_key));
      *key = out_key;
      break;
    } else {
      return rc;
    }
  } while (true);
  return Status::OK();
}

Status StorageManager::Read(StorageManager::key_type key, WritableSlice *dest, size_t *bytesRead) const {
  RETURN_UNEXPECTED_IF_NULL(dest);
  auto r = index_.Search(key);
  if (r.second) {
    auto &it = r.first;
    value_type v = *it;
    int container_inx = v.first;
    off_t offset = v.second.first;
    size_t sz = v.second.second;
    if (dest->GetSize() < sz) {
      std::string errMsg = "Destination buffer too small. Expect at least " + std::to_string(sz) +
                           " but length = " + std::to_string(dest->GetSize());
      RETURN_STATUS_UNEXPECTED(errMsg);
    }
    if (bytesRead != nullptr) {
      *bytesRead = sz;
    }
    auto cont = containers_.at(container_inx);
    RETURN_IF_NOT_OK(cont->Read(dest, offset));
  } else {
    RETURN_STATUS_UNEXPECTED("Key not found");
  }
  return Status::OK();
}

Status StorageManager::DoServiceStop() noexcept {
  Status rc;
  Status rc1;
  for (auto const &p : containers_) {
    // The destructor of StorageContainer is not called automatically until the use
    // count drops to 0. But it is not always the case. We will do it ourselves.
    rc = p.get()->Truncate();
    if (rc.IsError()) {
      rc1 = rc;
    }
  }
  containers_.clear();
  file_id_ = 0;
  return rc1;
}

StorageManager::StorageManager(const Path &root) : root_(root), file_id_(0), index_() {}

StorageManager::~StorageManager() { (void)StorageManager::DoServiceStop(); }

std::ostream &operator<<(std::ostream &os, const StorageManager &s) {
  os << "Dumping all containers ..."
     << "\n";
  for (auto const &p : s.containers_) {
    os << *(p.get());
  }
  return os;
}
}  // namespace dataset
}  // namespace mindspore
