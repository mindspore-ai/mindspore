/**
 * Copyright 2020 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#include "minddata/dataset/engine/cache/cache_service.h"
#include "minddata/dataset/engine/cache/cache_server.h"
#include "minddata/dataset/util/slice.h"

namespace mindspore {
namespace dataset {
CacheService::CacheService(uint64_t mem_sz, const std::string &root, bool generate_id)
    : root_(root),
      cache_mem_sz_(mem_sz),
      cp_(nullptr),
      next_id_(0),
      generate_id_(generate_id),
      st_(generate_id ? State::kBuildPhase : State::kNone),
      cur_mem_usage_(0),
      cur_disk_usage_(0) {}

CacheService::~CacheService() { (void)ServiceStop(); }

bool CacheService::UseArena() {
  // If fixed size, use Arena instead of the pool from global context.
  return (cache_mem_sz_ > 0);
}

Status CacheService::DoServiceStart() {
  std::shared_ptr<MemoryPool> mp_;
  CacheServer &cs = CacheServer::GetInstance();
  if (UseArena()) {
    auto avail_mem = cs.GetAvailableSystemMemory() / 1048576L;
    if (cache_mem_sz_ > avail_mem) {
      // Output a warning that we use more than recommended. If we fail to allocate, we will fail anyway.
      MS_LOG(WARNING) << "Requesting cache size " << cache_mem_sz_ << " MB while available system memory " << avail_mem
                      << " MB";
    }
    // Create a fixed size arena based on the parameter.
    std::shared_ptr<Arena> arena;
    RETURN_IF_NOT_OK(Arena::CreateArena(&arena, cache_mem_sz_));
    mp_ = std::move(arena);
    // update the global usage only.
    cs.UpdateMemoryUsage(cache_mem_sz_ * 1048576L, CacheServer::MemUsageOp::kAllocate);
  } else {
    // Unlimited size. Simply use a system pool. Another choice is CircularPool.
    mp_ = std::make_shared<SystemPool>();
  }
  // Put together a CachePool for backing up the Tensor
  cp_ = std::make_shared<CachePool>(CachePool::value_allocator(mp_), UseArena(), root_);
  RETURN_IF_NOT_OK(cp_->ServiceStart());
  // Assign a name to this cache. Used for exclusive connection. But we can just use CachePool's name.
  cookie_ = cp_->MyName();
  return Status::OK();
}

Status CacheService::DoServiceStop() {
  if (cp_ != nullptr) {
    RETURN_IF_NOT_OK(cp_->ServiceStop());
  }
  CacheServer &cs = CacheServer::GetInstance();
  if (UseArena()) {
    cs.UpdateMemoryUsage(cache_mem_sz_ * 1048576L, CacheServer::MemUsageOp::kFree);
  } else {
    MS_LOG(INFO) << "Memory/disk usage for the current service: " << GetMemoryUsage() << " bytes and " << GetDiskUsage()
                 << " bytes.";
    cs.UpdateMemoryUsage(GetMemoryUsage(), CacheServer::MemUsageOp::kFree);
  }
  return Status::OK();
}

Status CacheService::CacheRow(const std::vector<const void *> &buf, row_id_type *row_id_generated) {
  SharedLock rw(&rw_lock_);
  RETURN_UNEXPECTED_IF_NULL(row_id_generated);
  if (st_ == State::kFetchPhase) {
    // For this kind of cache service, once we are done with the build phase into fetch phase, we can't
    // allow other to cache more rows.
    RETURN_STATUS_UNEXPECTED("Can't accept cache request in fetch phase");
  }
  if (st_ == State::kNoLocking) {
    // We ignore write this request once we turn off locking on the B+ tree. So we will just
    // return out of memory from now on.
    return Status(StatusCode::kOutOfMemory);
  }
  try {
    // The first buffer is a flatbuffer which describes the rest of the buffers follow
    auto fb = buf.front();
    RETURN_UNEXPECTED_IF_NULL(fb);
    auto msg = GetTensorRowHeaderMsg(fb);
    // If the server side is designed to ignore incoming row id, we generate row id.
    if (generate_id_) {
      *row_id_generated = GetNextRowId();
      // Some debug information on how many rows we have generated so far.
      if ((*row_id_generated) % 1000 == 0) {
        MS_LOG(DEBUG) << "Number of rows cached: " << (*row_id_generated) + 1;
      }
    } else {
      if (msg->row_id() < 0) {
        std::string errMsg = "Expect positive row id: " + std::to_string(msg->row_id());
        RETURN_STATUS_UNEXPECTED(errMsg);
      }
      *row_id_generated = msg->row_id();
    }
    auto size_of_this = msg->size_of_this();
    size_t total_sz = size_of_this;
    auto column_hdr = msg->column();
    // Number of tensor buffer should match the number of columns plus one.
    if (buf.size() != column_hdr->size() + 1) {
      std::string errMsg = "Column count does not match. Expect " + std::to_string(column_hdr->size() + 1) +
                           " but get " + std::to_string(buf.size());
      RETURN_STATUS_UNEXPECTED(errMsg);
    }
    // Next we store in either memory or on disk. Low level code will consolidate everything in one piece.
    std::vector<ReadableSlice> all_data;
    all_data.reserve(column_hdr->size() + 1);
    all_data.emplace_back(fb, size_of_this);
    for (auto i = 0; i < column_hdr->size(); ++i) {
      all_data.emplace_back(buf.at(i + 1), msg->data_sz()->Get(i));
      total_sz += msg->data_sz()->Get(i);
    }
    // Now we cache the buffer. If we are using Arena which has a fixed cap, then just do it.
    // Otherwise, we check how much (globally) how much we use and may simply spill to disk
    // directly.
    CacheServer &cs = CacheServer::GetInstance();
    bool write_to_disk_directly = UseArena() ? false : (total_sz + cs.GetMemoryUsage()) > cs.GetAvailableSystemMemory();
    Status rc = cp_->Insert(*row_id_generated, all_data, write_to_disk_directly);
    if (rc == Status(StatusCode::kDuplicateKey)) {
      MS_LOG(DEBUG) << "Ignoring duplicate key.";
    } else {
      RETURN_IF_NOT_OK(rc);
    }
    // All good, then update the memory usage local and global (if not using arena)
    if (write_to_disk_directly) {
      cur_disk_usage_ += total_sz;
    } else {
      cur_mem_usage_ += total_sz;
      if (!UseArena()) {
        cs.UpdateMemoryUsage(total_sz, CacheServer::MemUsageOp::kAllocate);
      }
    }
    return Status::OK();
  } catch (const std::exception &e) {
    RETURN_STATUS_UNEXPECTED(e.what());
  }
}

Status CacheService::FastCacheRow(const ReadableSlice &src, row_id_type *row_id_generated) {
  SharedLock rw(&rw_lock_);
  RETURN_UNEXPECTED_IF_NULL(row_id_generated);
  if (st_ == State::kFetchPhase) {
    // For this kind of cache service, once we are done with the build phase into fetch phase, we can't
    // allow other to cache more rows.
    RETURN_STATUS_UNEXPECTED("Can't accept cache request in fetch phase");
  }
  if (st_ == State::kNoLocking) {
    // We ignore write this request once we turn off locking on the B+ tree. So we will just
    // return out of memory from now on.
    return Status(StatusCode::kOutOfMemory);
  }
  try {
    // If we don't need to generate id, we need to find it from the buffer.
    if (generate_id_) {
      *row_id_generated = GetNextRowId();
      // Some debug information on how many rows we have generated so far.
      if ((*row_id_generated) % 1000 == 0) {
        MS_LOG(DEBUG) << "Number of rows cached: " << (*row_id_generated) + 1;
      }
    } else {
      auto msg = GetTensorRowHeaderMsg(src.GetPointer());
      if (msg->row_id() < 0) {
        std::string errMsg = "Expect positive row id: " + std::to_string(msg->row_id());
        RETURN_STATUS_UNEXPECTED(errMsg);
      }
      *row_id_generated = msg->row_id();
    }
    // Now we cache the buffer. If we are using Arena which has a fixed cap, then just do it.
    // Otherwise, we check how much (globally) how much we use and may simply spill to disk
    // directly.
    auto total_sz = src.GetSize();
    CacheServer &cs = CacheServer::GetInstance();
    bool write_to_disk_directly = UseArena() ? false : (total_sz + cs.GetMemoryUsage()) > cs.GetAvailableSystemMemory();
    Status rc = cp_->Insert(*row_id_generated, {src}, write_to_disk_directly);
    if (rc == Status(StatusCode::kDuplicateKey)) {
      MS_LOG(DEBUG) << "Ignoring duplicate key.";
    } else {
      RETURN_IF_NOT_OK(rc);
    }
    // All good, then update the memory usage local and global (if not using arena)
    if (write_to_disk_directly) {
      cur_disk_usage_ += total_sz;
    } else {
      cur_mem_usage_ += total_sz;
      if (!UseArena()) {
        cs.UpdateMemoryUsage(total_sz, CacheServer::MemUsageOp::kAllocate);
      }
    }
    return Status::OK();
  } catch (const std::exception &e) {
    RETURN_STATUS_UNEXPECTED(e.what());
  }
}

std::ostream &operator<<(std::ostream &out, const CacheService &cs) {
  // Then show any custom derived-internal stuff
  out << "\nCache memory size: " << cs.cache_mem_sz_;
  out << "\nSpill path: ";
  if (cs.root_.empty()) {
    out << "None";
  } else {
    out << cs.GetSpillPath();
  }
  return out;
}

Path CacheService::GetSpillPath() const { return cp_->GetSpillPath(); }

Status CacheService::FindKeysMiss(std::vector<row_id_type> *out) {
  RETURN_UNEXPECTED_IF_NULL(out);
  std::unique_lock<std::mutex> lock(get_key_miss_mux_);
  if (key_miss_results_ == nullptr) {
    // Just do it once.
    key_miss_results_ = std::make_shared<std::vector<row_id_type>>();
    auto stat = cp_->GetStat(true);
    key_miss_results_->push_back(stat.min_key);
    key_miss_results_->push_back(stat.max_key);
    key_miss_results_->insert(key_miss_results_->end(), stat.gap.begin(), stat.gap.end());
  }
  out->insert(out->end(), key_miss_results_->begin(), key_miss_results_->end());
  return Status::OK();
}

Status CacheService::GetStat(CacheService::ServiceStat *out) {
  SharedLock rw(&rw_lock_);
  RETURN_UNEXPECTED_IF_NULL(out);
  out->stat_ = cp_->GetStat();
  out->state_ = static_cast<ServiceStat::state_type>(st_);
  return Status::OK();
}

Status CacheService::PreBatchFetch(const std::vector<row_id_type> &v, std::vector<key_size_pair> *out,
                                   int64_t *mem_sz) {
  SharedLock rw(&rw_lock_);
  RETURN_UNEXPECTED_IF_NULL(out);
  RETURN_UNEXPECTED_IF_NULL(mem_sz);
  const auto num_elements = v.size();
  *mem_sz = (num_elements + 1) * sizeof(int64_t);
  (*out).reserve(num_elements);
  for (auto row_id : v) {
    auto sz = cp_->GetSize(row_id);
    if (sz > 0) {
      (*out).emplace_back(row_id, sz);
      (*mem_sz) += sz;
    } else {
      // key not found
      (*out).emplace_back(-1, 0);
    }
  }
  return Status::OK();
}

Status CacheService::BatchFetch(const std::vector<row_id_type> &v, const std::vector<key_size_pair> &info,
                                WritableSlice *out) const {
  RETURN_UNEXPECTED_IF_NULL(out);
  SharedLock rw(&rw_lock_);
  if (st_ == State::kBuildPhase) {
    // For this kind of cache service, we can't fetch yet until we are done with caching all the rows.
    RETURN_STATUS_UNEXPECTED("Can't accept cache request in fetch phase");
  }
  const auto num_elements = v.size();
  int64_t data_offset = (num_elements + 1) * sizeof(int64_t);
  auto *offset_array = reinterpret_cast<int64_t *>(out->GetMutablePointer());
  offset_array[0] = data_offset;
  for (auto i = 0; i < num_elements; ++i) {
    auto sz = info.at(i).second;
    offset_array[i + 1] = offset_array[i] + sz;
    if (sz > 0) {
      WritableSlice row_data(*out, offset_array[i], sz);
      auto key = info.at(i).first;
      size_t bytesRead = 0;
      RETURN_IF_NOT_OK(cp_->Read(key, &row_data, &bytesRead));
      if (bytesRead != sz) {
        MS_LOG(ERROR) << "Unexpected length. Read " << bytesRead << ". Expected " << sz << "."
                      << " Internal key: " << key << "\n";
        RETURN_STATUS_UNEXPECTED("Length mismatch. See log file for details.");
      }
    }
  }
  return Status::OK();
}

Status CacheService::CacheSchema(const void *buf, int64_t len) {
  UniqueLock rw(&rw_lock_);
  // In case we are calling the same function from multiple threads, only
  // the first one is considered. Rest is ignored.
  if (schema_.empty()) {
    schema_.assign(static_cast<const char *>(buf), len);
  } else {
    MS_LOG(DEBUG) << "Caching Schema already done";
  }
  return Status::OK();
}

Status CacheService::FetchSchema(std::string *out) const {
  SharedLock rw(&rw_lock_);
  if (st_ == State::kBuildPhase) {
    // For this kind of cache service, we can't fetch yet until we are done with caching all the rows.
    RETURN_STATUS_UNEXPECTED("Can't accept cache request in fetch phase");
  }
  RETURN_UNEXPECTED_IF_NULL(out);
  // We are going to use std::string to allocate and hold the result which will be eventually
  // 'moved' to the protobuf message (which underneath is also a std::string) for the purpose
  // to minimize memory copy.
  std::string mem(schema_);
  if (!mem.empty()) {
    *out = std::move(mem);
  } else {
    return Status(StatusCode::kFileNotExist, __LINE__, __FILE__, "No schema has been cached");
  }
  return Status::OK();
}

Status CacheService::BuildPhaseDone() {
  if (HasBuildPhase()) {
    // Exclusive lock to switch phase
    UniqueLock rw(&rw_lock_);
    st_ = State::kFetchPhase;
    cp_->SetLocking(false);
    return Status::OK();
  } else {
    RETURN_STATUS_UNEXPECTED("Not a cache that has a build phase");
  }
}

Status CacheService::ToggleWriteMode(bool on_off) {
  UniqueLock rw(&rw_lock_);
  if (HasBuildPhase()) {
    RETURN_STATUS_UNEXPECTED("Not applicable to non-mappable dataset");
  } else {
    // If we stop accepting write request, we turn off locking for the
    // underlying B+ tree. All future write request we will return kOutOfMemory.
    if (st_ == State::kNone && !on_off) {
      st_ = State::kNoLocking;
      cp_->SetLocking(on_off);
      MS_LOG(WARNING) << "Locking mode is switched off.";
    } else if (st_ == State::kNoLocking && on_off) {
      st_ = State::kNone;
      cp_->SetLocking(on_off);
    }
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
