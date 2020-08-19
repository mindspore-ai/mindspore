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
#include "minddata/dataset/util/slice.h"

namespace mindspore {
namespace dataset {
CacheService::CacheService(uint64_t mem_sz, const std::string &root, bool generate_id)
    : root_(root),
      cache_mem_sz_(mem_sz),
      cp_(nullptr),
      map_(nullptr),
      next_id_(0),
      generate_id_(generate_id),
      schema_key_(-1),
      st_(generate_id ? State::kBuildPhase : State::kNone) {}
CacheService::~CacheService() { (void)ServiceStop(); }
bool CacheService::UseArena() {
  // If fixed size, use Arena instead of the pool from global context.
  return (cache_mem_sz_ > 0);
}
Status CacheService::DoServiceStart() {
  std::shared_ptr<MemoryPool> mp_;
  if (UseArena()) {
    // Create a fixed size arena based on the parameter.
    std::shared_ptr<Arena> arena;
    RETURN_IF_NOT_OK(Arena::CreateArena(&arena, cache_mem_sz_));
    mp_ = std::move(arena);
  } else {
    // Unlimited size. Simply use a system pool. Another choice is CircularPool.
    mp_ = std::make_shared<SystemPool>();
  }
  // Put together a CachePool for backing up the Tensor
  cp_ = std::make_shared<CachePool>(CachePool::value_allocator(mp_), root_);
  RETURN_IF_NOT_OK(cp_->ServiceStart());
  // Set up the B+ tree as well. But use the system pool instead.
  map_ = std::make_shared<row_map>();
  // Assign a name to this cache. Used for exclusive connection. But we can just use CachePool's name.
  cookie_ = cp_->MyName();
  return Status::OK();
}
Status CacheService::DoServiceStop() {
  if (cp_ != nullptr) {
    RETURN_IF_NOT_OK(cp_->ServiceStop());
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
    }
    // Now we cache the flat buffer.
    CachePool::key_type key;
    RETURN_IF_NOT_OK(cp_->Insert(all_data, &key));
    Status rc = map_->DoInsert(*row_id_generated, key);
    if (rc == Status(StatusCode::kDuplicateKey)) {
      MS_LOG(DEBUG) << "Ignoring duplicate key.";
    } else {
      RETURN_IF_NOT_OK(rc);
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
    // Now we cache the flat buffer.
    CachePool::key_type key;
    RETURN_IF_NOT_OK(cp_->Insert({src}, &key));
    Status rc = map_->DoInsert(*row_id_generated, key);
    if (rc == Status(StatusCode::kDuplicateKey)) {
      MS_LOG(DEBUG) << "Ignoring duplicate key.";
    } else {
      RETURN_IF_NOT_OK(rc);
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
Status CacheService::Purge() {
  // First we must lock exclusively. No one else can cache/restore anything.
  UniqueLock rw(&rw_lock_);
  RETURN_IF_NOT_OK(cp_->ServiceStop());
  auto new_map = std::make_shared<row_map>();
  map_.reset();
  map_ = std::move(new_map);
  next_id_ = 0;
  RETURN_IF_NOT_OK(cp_->ServiceStart());
  return Status::OK();
}
Status CacheService::GetStat(CacheService::ServiceStat *out) {
  SharedLock rw(&rw_lock_);
  RETURN_UNEXPECTED_IF_NULL(out);
  if (st_ == State::kNone || st_ == State::kFetchPhase) {
    out->stat_ = cp_->GetStat();
    out->state_ = static_cast<ServiceStat::state_type>(st_);
    auto it = map_->begin();
    if (it != map_->end()) {
      out->min_ = it.key();
      auto end_it = map_->end();
      --end_it;
      out->max_ = end_it.key();
    }
  } else {
    out->state_ = static_cast<ServiceStat::state_type>(st_);
  }
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
    auto r = map_->Search(row_id);
    if (r.second) {
      auto &it = r.first;
      CachePool::key_type key = it.value();
      auto sz = cp_->GetSize(key);
      if (sz == 0) {
        std::string errMsg = "Key not found: ";
        errMsg += std::to_string(key);
        RETURN_STATUS_UNEXPECTED(errMsg);
      }
      (*out).emplace_back(key, sz);
      (*mem_sz) += sz;
    } else {
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
  SharedLock rw(&rw_lock_);
  if (st_ == State::kFetchPhase) {
    // For this kind of cache service, once we are done with the build phase into fetch phase, we can't
    // allow other to cache more rows.
    RETURN_STATUS_UNEXPECTED("Can't accept cache request in fetch phase");
  }
  // This is a special request and we need to remember where we store it.
  // In case we are calling the same function from multiple threads, only
  // the first one is considered. Rest is ignored.
  CachePool::key_type cur_key = schema_key_;
  CachePool::key_type key;
  if (cur_key < 0) {
    RETURN_IF_NOT_OK(cp_->Insert({ReadableSlice(buf, len)}, &key));
    auto result = std::atomic_compare_exchange_strong(&schema_key_, &cur_key, key);
    MS_LOG(DEBUG) << "Caching Schema. Result = " << result;
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
  std::string mem;
  if (schema_key_ >= 0) {
    auto len = cp_->GetSize(schema_key_);
    try {
      mem.resize(len);
      CHECK_FAIL_RETURN_UNEXPECTED(mem.capacity() >= len, "Programming error");
    } catch (const std::bad_alloc &e) {
      return Status(StatusCode::kOutOfMemory);
    }
    auto slice = WritableSlice(mem.data(), len);
    RETURN_IF_NOT_OK(cp_->Read(schema_key_, &slice));
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
    return Status::OK();
  } else {
    RETURN_STATUS_UNEXPECTED("Not a cache that has a build phase");
  }
}
}  // namespace dataset
}  // namespace mindspore
