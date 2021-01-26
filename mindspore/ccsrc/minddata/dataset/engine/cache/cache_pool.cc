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
#include "utils/ms_utils.h"
#include "minddata/dataset/engine/cache/cache_pool.h"
#include "minddata/dataset/engine/cache/cache_server.h"
#include "minddata/dataset/util/services.h"

namespace mindspore {
namespace dataset {
CachePool::CachePool(std::shared_ptr<NumaMemoryPool> mp, const std::string &root)
    : mp_(std::move(mp)), root_(root), subfolder_(Services::GetUniqueID()), sm_(nullptr), tree_(nullptr) {}

Status CachePool::DoServiceStart() {
  tree_ = std::make_shared<data_index>();
  // If we are given a disk path, set up the StorageManager
  if (!root_.toString().empty()) {
    Path spill = GetSpillPath();
    RETURN_IF_NOT_OK(spill.CreateDirectories());
    sm_ = std::make_shared<StorageManager>(spill);
    RETURN_IF_NOT_OK(sm_->ServiceStart());
    MS_LOG(INFO) << "CachePool will use disk folder: " << spill.toString();
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

  // We used to free the memory allocated from each DataLocator but
  // since all of them are coming from NumaMemoryPool and we will
  // skip this and release the whole NumaMemoryPool instead. Otherwise
  // release each buffer in the DataLocator one by one.

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

Status CachePool::Insert(CachePool::key_type key, const std::vector<ReadableSlice> &buf) {
  DataLocator bl;
  Status rc;
  size_t sz = 0;
  // We will consolidate all the slices into one piece.
  for (auto &v : buf) {
    sz += v.GetSize();
  }
  bl.sz = sz;
  rc = mp_->Allocate(sz, reinterpret_cast<void **>(&bl.ptr));
  if (rc.IsOk()) {
    // Write down which numa node where we allocate from. It only make sense if the policy is kOnNode.
    if (CacheServerHW::numa_enabled()) {
      auto &cs = CacheServer::GetInstance();
      auto node_id = cs.GetHWControl()->GetMyNode();
      bl.node_id = mp_->FindNode(bl.ptr);
      CHECK_FAIL_RETURN_UNEXPECTED(bl.node_id != -1, "Allocator is not from numa memory pool");
      bl.node_hit = (bl.node_id == node_id);
    }
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
      mp_->Deallocate(bl.ptr);
      bl.ptr = nullptr;
      return rc;
    }
  } else if (rc == StatusCode::kMDOutOfMemory) {
    // If no memory, write to disk.
    if (sm_ != nullptr) {
      MS_LOG(DEBUG) << "Spill to disk directly ... " << bl.sz << " bytes.";
      RETURN_IF_NOT_OK(sm_->Write(&bl.storage_key, buf));
    } else {
      // If asked to spill to disk instead but there is no storage set up, simply return no memory
      // instead.
      return Status(StatusCode::kMDOutOfMemory, __LINE__, __FILE__, "No enough storage for cache server to cache data");
    }
  } else {
    return rc;
  }
  // Insert into the B+ tree. We may still get out of memory error. So need to catch it.
  try {
    rc = tree_->DoInsert(key, bl);
  } catch (const std::bad_alloc &e) {
    rc = Status(StatusCode::kMDOutOfMemory, __LINE__, __FILE__);
  }
  // Duplicate key is treated as error and we will also free the memory.
  if (rc.IsError() && bl.ptr != nullptr) {
    mp_->Deallocate(bl.ptr);
    bl.ptr = nullptr;
    return rc;
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

Path CachePool::GetSpillPath() const {
  auto spill = Path(root_) / subfolder_;
  return spill;
}

CachePool::CacheStat CachePool::GetStat(bool GetMissingKeys) const {
  tree_->LockShared();  // Prevent any node split while we search.
  CacheStat cs{-1, -1, 0, 0, 0, 0};
  int64_t total_sz = 0;
  if (tree_->begin() != tree_->end()) {
    cs.min_key = tree_->begin().key();
    cs.max_key = cs.min_key;  // will adjust later.
    for (auto it = tree_->begin(); it != tree_->end(); ++it) {
      it.LockShared();
      total_sz += it.value().sz;
      if (it.value().ptr != nullptr) {
        ++cs.num_mem_cached;
      } else {
        ++cs.num_disk_cached;
      }
      if (it.value().node_hit) {
        ++cs.num_numa_hit;
      }
      auto cur_key = it.key();
      if (GetMissingKeys) {
        for (auto i = cs.max_key + 1; i < cur_key; ++i) {
          cs.gap.push_back((i));
        }
      }
      cs.max_key = cur_key;
      it.Unlock();
    }
  }
  if (total_sz > 0) {
    // integer arithmetic. NO need to cast to float or double.
    cs.average_cache_sz = total_sz / (cs.num_disk_cached + cs.num_mem_cached);
    if (cs.average_cache_sz == 0) {
      cs.average_cache_sz = 1;
    }
  }
  tree_->Unlock();
  return cs;
}

Status CachePool::GetDataLocator(key_type key, const std::shared_ptr<flatbuffers::FlatBufferBuilder> &fbb,
                                 flatbuffers::Offset<DataLocatorMsg> *out) const {
  RETURN_UNEXPECTED_IF_NULL(out);
  auto r = tree_->Search(key);
  if (r.second) {
    auto &it = r.first;
    DataLocatorMsgBuilder bld(*fbb);
    bld.add_key(key);
    bld.add_size(it->sz);
    bld.add_node_id(it->node_id);
    bld.add_addr(reinterpret_cast<int64_t>(it->ptr));
    auto offset = bld.Finish();
    *out = offset;
  } else {
    // Key not in the cache.
    auto offset = CreateDataLocatorMsg(*fbb, key, 0, 0, 0);
    *out = offset;
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
