/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd

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
#include "minddata/dataset/engine/cache/cache_request.h"
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
#include <sched.h>
#include <sys/types.h>
#include <unistd.h>
#endif
#include <cstdlib>
#include <cstring>
#include <thread>
#include "minddata/dataset/include/constants.h"
#include "minddata/dataset/engine/cache/cache_client.h"
#include "minddata/dataset/engine/cache/cache_fbb.h"
namespace mindspore {
namespace dataset {
Status BaseRequest::Wait() {
  RETURN_IF_NOT_OK(wp_.Wait());
  Status remote_rc(static_cast<StatusCode>(reply_.rc()), reply_.msg());
  RETURN_IF_NOT_OK(remote_rc);
  // Any extra work to do before we return back to the client.
  RETURN_IF_NOT_OK(PostReply());
  return Status::OK();
}
Status CacheRowRequest::SerializeCacheRowRequest(const CacheClient *cc, const TensorRow &row) {
  CHECK_FAIL_RETURN_UNEXPECTED(row.size() > 0, "Empty tensor row");
  CHECK_FAIL_RETURN_UNEXPECTED(cc->SupportLocalClient() == support_local_bypass_, "Local bypass mismatch");
  // Calculate how many bytes (not counting the cookie) we are sending to the server. We only
  // use shared memory (if supported) if we exceed certain amount
  std::shared_ptr<flatbuffers::FlatBufferBuilder> fbb;
  RETURN_IF_NOT_OK(::mindspore::dataset::SerializeTensorRowHeader(row, &fbb));
  sz_ += fbb->GetSize();
  for (const auto &ts : row) {
    sz_ += ts->SizeInBytes();
  }
  bool sent_using_local_bypass = support_local_bypass_ ? (sz_ >= kLocalByPassThreshold) : false;
  uint32_t flag = 0;
  if (support_local_bypass_) {
    BitSet(&flag, kLocalClientSupport);
  }
  if (sent_using_local_bypass) {
    BitSet(&flag, kDataIsInSharedMemory);
  }
  rq_.set_flag(flag);
  if (sent_using_local_bypass) {
    MS_LOG(DEBUG) << "Requesting " << sz_ << " bytes of shared memory data";
    // Allocate shared memory from the server
    auto mem_rq = std::make_shared<AllocateSharedBlockRequest>(rq_.connection_id(), cc->GetClientId(), sz_);
    RETURN_IF_NOT_OK(cc->PushRequest(mem_rq));
    RETURN_IF_NOT_OK(mem_rq->Wait());
    addr_ = mem_rq->GetAddr();
    // Now we need to add that to the base address of where we attach.
    auto base = cc->SharedMemoryBaseAddr();
    auto p = reinterpret_cast<void *>(reinterpret_cast<int64_t>(base) + addr_);
    // Now we copy the data onto shared memory.
    WritableSlice all(p, sz_);
    auto offset = fbb->GetSize();
    ReadableSlice header(fbb->GetBufferPointer(), fbb->GetSize());
    Status copy_rc;
    copy_rc = WritableSlice::Copy(&all, header);
    if (copy_rc.IsOk()) {
      for (const auto &ts : row) {
        WritableSlice row_data(all, offset, ts->SizeInBytes());
        ReadableSlice src(ts->GetBuffer(), ts->SizeInBytes());
        copy_rc = WritableSlice::Copy(&row_data, src);
        if (copy_rc.IsError()) {
          break;
        }
        offset += ts->SizeInBytes();
      }
      // Fill in where to find the data
      AddDataLocation();
    }
    if (copy_rc.IsError()) {
      // We need to return the memory back to the server
      auto mfree_req = GenerateFreeBlockRequest();
      Status rc = cc->PushRequest(mfree_req);
      // But we won't wait for the result for the sake of performance.
      if (rc.IsError()) {
        MS_LOG(ERROR) << "Push request for free memory failed.";
      }
      return copy_rc;
    }
  } else {
    // We have already filled the first buffer which is the cookie.
    sz_ += rq_.buf_data(0).size();
    rq_.add_buf_data(fbb->GetBufferPointer(), fbb->GetSize());
    for (const auto &ts : row) {
      rq_.add_buf_data(ts->GetBuffer(), ts->SizeInBytes());
    }
    MS_LOG(DEBUG) << "Sending " << sz_ << " bytes of tensor data in " << rq_.buf_data_size() << " segments";
  }
  return Status::OK();
}

Status CacheRowRequest::PostReply() {
  if (!reply_.result().empty()) {
    row_id_from_server_ = strtoll(reply_.result().data(), nullptr, 10);
  }
  return Status::OK();
}

Status CacheRowRequest::Prepare() {
  if (BitTest(rq_.flag(), kDataIsInSharedMemory)) {
    // First one is cookie, followed by address and then size.
    CHECK_FAIL_RETURN_UNEXPECTED(rq_.buf_data_size() == 3, "Incomplete rpc data");
  } else {
    // First one is cookie. 2nd one is the google flat buffers followed by a number of buffers.
    // But we are not going to decode them to verify.
    CHECK_FAIL_RETURN_UNEXPECTED(rq_.buf_data_size() >= 3, "Incomplete rpc data");
  }
  return Status::OK();
}

CacheRowRequest::CacheRowRequest(const CacheClient *cc)
    : BaseRequest(RequestType::kCacheRow),
      support_local_bypass_(cc->local_bypass_),
      addr_(-1),
      sz_(0),
      row_id_from_server_(-1) {
  rq_.set_connection_id(cc->server_connection_id_);
  rq_.set_client_id(cc->client_id_);
  rq_.add_buf_data(cc->cookie_);
}

BatchFetchRequest::BatchFetchRequest(const CacheClient *cc, const std::vector<row_id_type> &row_id)
    : BaseRequest(RequestType::kBatchFetchRows), support_local_bypass_(cc->local_bypass_), row_id_(row_id) {
  rq_.set_connection_id(cc->server_connection_id_);
  rq_.set_client_id(cc->client_id_);
  rq_.set_flag(support_local_bypass_ ? kLocalClientSupport : 0);
  // Convert the row id into a flatbuffer
  flatbuffers::FlatBufferBuilder fbb;
  auto off_t = fbb.CreateVector(row_id);
  TensorRowIdsBuilder bld(fbb);
  bld.add_row_id(off_t);
  auto off = bld.Finish();
  fbb.Finish(off);
  rq_.add_buf_data(fbb.GetBufferPointer(), fbb.GetSize());
}

Status BatchFetchRequest::RestoreRows(TensorTable *out, const void *baseAddr, int64_t *out_addr) {
  RETURN_UNEXPECTED_IF_NULL(out);
  auto num_elements = row_id_.size();
  const char *ptr = nullptr;
  int64_t sz = 0;
  // Tap into the reply flag to see where we can find the data. Server may decide the amount is
  // so small that it doesn't use shared memory method.
  auto flag = reply_.flag();
  bool dataOnSharedMemory = support_local_bypass_ ? (BitTest(flag, kDataIsInSharedMemory)) : false;
  if (dataOnSharedMemory) {
    auto addr = strtoll(reply_.result().data(), nullptr, 10);
    ptr = reinterpret_cast<const char *>(reinterpret_cast<int64_t>(baseAddr) + addr);
    RETURN_UNEXPECTED_IF_NULL(out);
    *out_addr = addr;
  } else {
    ptr = reply_.result().data();
    *out_addr = -1;
  }
  auto *offset_array = reinterpret_cast<const int64_t *>(ptr);
  sz = offset_array[num_elements];
  CHECK_FAIL_RETURN_UNEXPECTED(support_local_bypass_ || sz == reply_.result().length(), "Length mismatch");
  TensorTable tbl;
  tbl.reserve(num_elements);
  ReadableSlice all(ptr, sz);
  for (auto i = 0; i < num_elements; ++i) {
    auto len = offset_array[i + 1] - offset_array[i];
    TensorRow row;
    row.setId(row_id_.at(i));
    if (len > 0) {
      ReadableSlice row_data(all, offset_array[i], len);
      // Next we de-serialize flat buffer to get back each column
      auto msg = GetTensorRowHeaderMsg(row_data.GetPointer());
      auto msg_sz = msg->size_of_this();
      // Start of the tensor data
      auto ts_offset = msg_sz;
      row.reserve(msg->column()->size());
      for (auto k = 0; k < msg->column()->size(); ++k) {
        auto col_ts = msg->column()->Get(k);
        std::shared_ptr<Tensor> ts;
        ReadableSlice data(row_data, ts_offset, msg->data_sz()->Get(k));
        RETURN_IF_NOT_OK(mindspore::dataset::RestoreOneTensor(col_ts, data, &ts));
        row.push_back(ts);
        ts_offset += data.GetSize();
      }
    } else {
      CHECK_FAIL_RETURN_UNEXPECTED(len == 0, "Data corruption detected.");
    }
    tbl.push_back(std::move(row));
  }
  *out = std::move(tbl);
  return Status::OK();
}

CreateCacheRequest::CreateCacheRequest(CacheClient *cc, const CacheClientInfo &cinfo, uint64_t cache_mem_sz,
                                       CreateCacheRequest::CreateCacheFlag flag)
    : BaseRequest(RequestType::kCreateCache), cache_mem_sz_(cache_mem_sz), flag_(flag), cc_(cc) {
  // Type has been set already in the base constructor. So we need to fill in the connection info.
  // On successful return, we will get the connection id
  rq_.mutable_connection_info()->operator=(cinfo);
}

Status CreateCacheRequest::Prepare() {
  try {
    flatbuffers::FlatBufferBuilder fbb;
    CreateCacheRequestMsgBuilder bld(fbb);
    bld.add_cache_mem_sz(cache_mem_sz_);
    bld.add_flag(static_cast<uint32_t>(flag_));
    auto off = bld.Finish();
    fbb.Finish(off);
    rq_.add_buf_data(fbb.GetBufferPointer(), fbb.GetSize());
    return Status::OK();
  } catch (const std::bad_alloc &e) {
    return Status(StatusCode::kMDOutOfMemory, __LINE__, __FILE__);
  }
}

Status CreateCacheRequest::PostReply() {
  auto p = flatbuffers::GetRoot<CreateCacheReplyMsg>(reply_.result().data());
  cc_->server_connection_id_ = p->connection_id();
  cc_->cookie_ = p->cookie()->str();
  cc_->client_id_ = p->client_id();
  // Next is a set of cpu id that we should re-adjust ourselves for better affinity.
  auto sz = p->cpu_id()->size();
  cc_->cpu_list_.reserve(sz);
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
  std::string c_list;
  cpu_set_t cpu_set;
  CPU_ZERO(&cpu_set);
#endif
  for (auto i = 0; i < sz; ++i) {
    auto cpu_id = p->cpu_id()->Get(i);
    cc_->cpu_list_.push_back(cpu_id);
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
    c_list += std::to_string(cpu_id) + " ";
    CPU_SET(cpu_id, &cpu_set);
#endif
  }

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
  if (sz > 0) {
    auto err = sched_setaffinity(getpid(), sizeof(cpu_set), &cpu_set);
    if (err == -1) {
      RETURN_STATUS_UNEXPECTED("Unable to set affinity. Errno = " + std::to_string(errno));
    }
    MS_LOG(INFO) << "Changing cpu affinity to the following list of cpu id: " + c_list;
  }
#endif

  return Status::OK();
}

Status CacheSchemaRequest::SerializeCacheSchemaRequest(const std::unordered_map<std::string, int32_t> &map) {
  try {
    flatbuffers::FlatBufferBuilder fbb;
    std::vector<flatbuffers::Offset<ColumnNameMsg>> v;
    v.reserve(map.size());
    for (auto &column : map) {
      auto c = CreateColumnNameMsg(fbb, fbb.CreateString(column.first), column.second);
      v.push_back(c);
    }
    auto v_off = fbb.CreateVector(v);
    auto final_off = CreateSchemaMsg(fbb, v_off);
    fbb.Finish(final_off);
    rq_.add_buf_data(fbb.GetBufferPointer(), fbb.GetSize());
    return Status::OK();
  } catch (const std::bad_alloc &e) {
    return Status(StatusCode::kMDOutOfMemory, __LINE__, __FILE__);
  }
}

Status FetchSchemaRequest::PostReply() {
  auto *map_msg = flatbuffers::GetRoot<SchemaMsg>(reply_.result().data());
  auto v = map_msg->column();
  for (auto i = 0; i < v->size(); ++i) {
    auto col = map_msg->column()->Get(i);
    column_name_id_map_.emplace(col->name()->str(), col->id());
  }
  return Status::OK();
}

std::unordered_map<std::string, int32_t> FetchSchemaRequest::GetColumnMap() { return column_name_id_map_; }

Status GetStatRequest::PostReply() {
  auto *msg = flatbuffers::GetRoot<ServiceStatMsg>(reply_.result().data());
  stat_.num_disk_cached = msg->num_disk_cached();
  stat_.num_mem_cached = msg->num_mem_cached();
  stat_.avg_cache_sz = msg->avg_cache_sz();
  stat_.num_numa_hit = msg->num_numa_hit();
  stat_.max_row_id = msg->max_row_id();
  stat_.min_row_id = msg->min_row_id();
  stat_.cache_service_state = msg->state();
  return Status::OK();
}

Status GetCacheStateRequest::PostReply() {
  try {
    cache_service_state_ = std::stoi(reply_.result());
  } catch (const std::exception &e) {
    RETURN_STATUS_UNEXPECTED(e.what());
  }
  return Status::OK();
}

Status ListSessionsRequest::PostReply() {
  auto *msg = flatbuffers::GetRoot<ListSessionsMsg>(reply_.result().data());
  auto session_vector = msg->sessions();
  for (auto i = 0; i < session_vector->size(); ++i) {
    SessionCacheInfo current_info{};
    CacheServiceStat stats{};
    auto current_session_info = session_vector->Get(i);
    current_info.session_id = current_session_info->session_id();
    current_info.connection_id = current_session_info->connection_id();
    stats.num_mem_cached = current_session_info->stats()->num_mem_cached();
    stats.num_disk_cached = current_session_info->stats()->num_disk_cached();
    stats.avg_cache_sz = current_session_info->stats()->avg_cache_sz();
    stats.num_numa_hit = current_session_info->stats()->num_numa_hit();
    stats.min_row_id = current_session_info->stats()->min_row_id();
    stats.max_row_id = current_session_info->stats()->max_row_id();
    stats.cache_service_state = current_session_info->stats()->state();
    current_info.stats = stats;  // fixed length struct.  = operator is safe
    session_info_list_.push_back(current_info);
  }
  server_cfg_.num_workers = msg->num_workers();
  server_cfg_.log_level = msg->log_level();
  server_cfg_.spill_dir = msg->spill_dir()->str();
  return Status::OK();
}

Status ServerStopRequest::PostReply() {
  CHECK_FAIL_RETURN_UNEXPECTED(strcmp(reply_.result().data(), "OK") == 0, "Not the right response");
  return Status::OK();
}

BatchCacheRowsRequest::BatchCacheRowsRequest(const CacheClient *cc, int64_t addr, int32_t num_ele)
    : BaseRequest(RequestType::kBatchCacheRows) {
  rq_.set_connection_id(cc->server_connection_id_);
  rq_.set_client_id(cc->client_id_);
  rq_.add_buf_data(cc->cookie());
  rq_.add_buf_data(std::to_string(addr));
  rq_.add_buf_data(std::to_string(num_ele));
}
}  // namespace dataset
}  // namespace mindspore
