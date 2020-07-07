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
#include <algorithm>
#include "common/utils.h"
#include "dataset/util/cache_pool.h"
#include "dataset/util/services.h"

namespace mindspore {
namespace dataset {
CachePool::CachePool(const value_allocator &alloc, const std::string &root)
    : alloc_(alloc), root_(root), subfolder_(Services::GetUniqueID()), sm_(nullptr), tree_(nullptr) {}

Status CachePool::DoServiceStart() {
  tree_ = std::make_shared<data_index>();
  // If we are given a disk path, set up the StorageManager
  if (!root_.toString().empty()) {
    Path spill = GetSpillPath();
    RETURN_IF_NOT_OK(spill.CreateDirectories());
    sm_ = std::make_shared<StorageManager>(spill);
    RETURN_IF_NOT_OK(sm_->ServiceStart());
    MS_LOG(INFO) << "CachePool will use disk folder: " << common::SafeCStr(spill.toString());
  }
  return Status::OK();
}
Status CachePool::DoServiceStop() {
  Status rc;
  Status rc2;
  if (sm_ != nullptr) {
    rc = sm_->ServiceStop();
    if (rc.IsError()) {
      rc2 = rc;
    }
  }
  sm_.reset();
  for (auto &bl : *tree_) {
    if (bl.ptr != nullptr) {
      alloc_.deallocate(bl.ptr, bl.sz);
    }
  }
  tree_.reset();
  if (!root_.toString().empty()) {
    Path spill = GetSpillPath();
    auto it = Path::DirIterator::OpenDirectory(&spill);
    while (it->hasNext()) {
      rc = it->next().Remove();
      if (rc.IsError() && rc2.IsOk()) {
        rc2 = rc;
      }
    }
    rc = spill.Remove();
    if (rc.IsError() && rc2.IsOk()) {
      rc2 = rc;
    }
  }
  return rc2;
}
CachePool::~CachePool() noexcept { (void)ServiceStop(); }
Status CachePool::Insert(const std::vector<ReadableSlice> &buf, CachePool::key_type *key) {
  DataLocator bl;
  Status rc;
  size_t sz = 0;
  // We will consolidate all the slices into one piece.
  for (auto &v : buf) {
    sz += v.GetSize();
  }
  bl.sz = sz;
  try {
    bl.ptr = alloc_.allocate(sz);
    // We will do a piecewise copy.
    WritableSlice dest(bl.ptr, bl.sz);
    size_t pos = 0;
    for (auto &v : buf) {
      WritableSlice out(dest, pos);
      rc = WritableSlice::Copy(&out, v);
      if (rc.IsError()) {
        break;
      }
      pos += v.GetSize();
    }
    if (rc.IsError()) {
      alloc_.deallocate(bl.ptr, sz);
      bl.ptr = nullptr;
      return rc;
    }
  } catch (std::bad_alloc &e) {
    if (sm_ != nullptr) {
      RETURN_IF_NOT_OK(sm_->Write(&bl.storage_key, buf));
      // We have an assumption 0 is not a valid key from the design of AutoIndexObj.
      // Make sure it is not 0.
      if (bl.storage_key == 0) {
        RETURN_STATUS_UNEXPECTED("Key 0 is returned which is unexpected");
      }
    } else {
      return Status(StatusCode::kOutOfMemory, __LINE__, __FILE__);
    }
  }
  rc = tree_->insert(bl, key);
  if (rc.IsError() && bl.ptr != nullptr) {
    alloc_.deallocate(bl.ptr, sz);
  }
  return rc;
}
Status CachePool::Read(CachePool::key_type key, WritableSlice *dest, size_t *bytesRead) const {
  RETURN_UNEXPECTED_IF_NULL(dest);
  auto r = tree_->Search(key);
  if (r.second) {
    auto &it = r.first;
    if (it->ptr != nullptr) {
      ReadableSlice src(it->ptr, it->sz);
      RETURN_IF_NOT_OK(WritableSlice::Copy(dest, src));
    } else if (sm_ != nullptr) {
      size_t expectedLength = 0;
      RETURN_IF_NOT_OK(sm_->Read(it->storage_key, dest, &expectedLength));
      if (expectedLength != it->sz) {
        MS_LOG(ERROR) << "Unexpected length. Read " << expectedLength << ". Expected " << it->sz << "."
                      << " Internal key: " << key << "\n";
        RETURN_STATUS_UNEXPECTED("Length mismatch. See log file for details.");
      }
    }
    if (bytesRead != nullptr) {
      *bytesRead = it->sz;
    }
  } else {
    RETURN_STATUS_UNEXPECTED("Key not found");
  }
  return Status::OK();
}
const CachePool::value_allocator &CachePool::get_allocator() const { return alloc_; }
Path CachePool::GetSpillPath() const {
  auto spill = Path(root_) / subfolder_;
  return spill;
}
CachePool::CacheStat CachePool::GetStat() const {
  CacheStat cs{0};
  for (auto &it : *tree_) {
    if (it.ptr != nullptr) {
      ++cs.num_mem_cached;
    } else {
      ++cs.num_disk_cached;
    }
  }
  return cs;
}
Status CachePool::Spill(CachePool::DataLocator *dl) {
  if (sm_ == nullptr) {
    RETURN_STATUS_UNEXPECTED("No disk storage to spill");
  }
  RETURN_UNEXPECTED_IF_NULL(dl);
  RETURN_UNEXPECTED_IF_NULL(dl->ptr);
  if (dl->storage_key == 0) {
    ReadableSlice data(dl->ptr, dl->sz);
    RETURN_IF_NOT_OK(sm_->Write(&dl->storage_key, {data}));
  }
  alloc_.deallocate(dl->ptr, dl->sz);
  dl->ptr = nullptr;
  return Status::OK();
}
Status CachePool::Locate(CachePool::DataLocator *dl) {
  RETURN_UNEXPECTED_IF_NULL(dl);
  if (dl->ptr == nullptr) {
    if (sm_ == nullptr) {
      RETURN_STATUS_UNEXPECTED("No disk storage to locate the data");
    }
    try {
      dl->ptr = alloc_.allocate(dl->sz);
      WritableSlice dest(dl->ptr, dl->sz);
      Status rc = Read(dl->storage_key, &dest);
      if (rc.IsError()) {
        alloc_.deallocate(dl->ptr, dl->sz);
        dl->ptr = nullptr;
        return rc;
      }
    } catch (const std::bad_alloc &e) {
      return Status(StatusCode::kOutOfMemory, __LINE__, __FILE__);
    }
  }
  return Status::OK();
}
size_t CachePool::GetSize(CachePool::key_type key) const {
  auto r = tree_->Search(key);
  if (r.second) {
    auto &it = r.first;
    return it->sz;
  } else {
    return 0;
  }
}
}  // namespace dataset
}  // namespace mindspore
