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
#include "minddata/dataset/engine/cache/storage_container.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include "utils/ms_utils.h"
#include "minddata/dataset/util/log_adapter.h"
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
Status StorageContainer::Create() {
  RETURN_IF_NOT_OK(BuddySpace::CreateBuddySpace(&bs_));
  RETURN_IF_NOT_OK(cont_.CreateFile(&fd_));
  is_open_ = true;
  MS_LOG(INFO) << "Container " << cont_ << " created";
  return Status::OK();
}

Status StorageContainer::Open() noexcept {
  std::lock_guard<std::mutex> lck(mutex_);
  // Check again
  if (!is_open_) {
    RETURN_IF_NOT_OK(cont_.OpenFile(&fd_));
    is_open_ = true;
  }
  return Status::OK();
}

Status StorageContainer::Close() noexcept {
  if (is_open_) {
    std::lock_guard<std::mutex> lck(mutex_);
    // Check again
    if (is_open_) {
      RETURN_IF_NOT_OK(cont_.CloseFile(fd_));
      is_open_ = false;
      fd_ = -1;
    }
  }
  return Status::OK();
}

Status StorageContainer::Read(WritableSlice *dest, off64_t offset) const noexcept {
  MS_ASSERT(is_open_);
  RETURN_UNEXPECTED_IF_NULL(dest);
  auto sz = dest->GetSize();
#if defined(_WIN32) || defined(_WIN64)
  // Doesn't seem there is any pread64 on mingw.
  // So we will do a seek and then a read under
  // a protection of mutex.
  std::lock_guard<std::mutex> lck(mutex_);
  auto seek_err = lseek(fd_, offset, SEEK_SET);
  if (seek_err < 0) {
    RETURN_STATUS_UNEXPECTED(strerror(errno));
  }
  auto r_sz = read(fd_, dest->GetMutablePointer(), sz);
#elif defined(__APPLE__)
  auto r_sz = pread(fd_, dest->GetMutablePointer(), sz, offset);
#else
  auto r_sz = pread64(fd_, dest->GetMutablePointer(), sz, offset);
#endif
  if (r_sz != sz) {
    errno_t err = (r_sz == 0) ? EOF : errno;
    RETURN_STATUS_UNEXPECTED(strerror(err));
  }
  return Status::OK();
}

Status StorageContainer::Write(const ReadableSlice &dest, off64_t offset) const noexcept {
  MS_ASSERT(is_open_);
  auto sz = dest.GetSize();
#if defined(_WIN32) || defined(_WIN64)
  // Doesn't seem there is any pwrite64 on mingw.
  // So we will do a seek and then a read under
  // a protection of mutex.
  std::lock_guard<std::mutex> lck(mutex_);
  auto seek_err = lseek(fd_, offset, SEEK_SET);
  if (seek_err < 0) {
    RETURN_STATUS_UNEXPECTED(strerror(errno));
  }
  auto r_sz = write(fd_, dest.GetPointer(), sz);
#elif defined(__APPLE__)
  auto r_sz = pwrite(fd_, dest.GetPointer(), sz, offset);
#else
  auto r_sz = pwrite64(fd_, dest.GetPointer(), sz, offset);
#endif
  if (r_sz != sz) {
    errno_t err = (r_sz == 0) ? EOF : errno;
    if (errno == ENOSPC) {
      return Status(StatusCode::kMDNoSpace, __LINE__, __FILE__);
    } else {
      RETURN_STATUS_UNEXPECTED(strerror(err));
    }
  }
  return Status::OK();
}

Status StorageContainer::Insert(const std::vector<ReadableSlice> &buf, off64_t *offset) noexcept {
  size_t sz = 0;
  for (auto &v : buf) {
    sz += v.GetSize();
  }
  if (sz == 0) {
    RETURN_STATUS_UNEXPECTED("Unexpected 0 length");
  }
  if (sz > bs_->GetMaxSize()) {
    RETURN_STATUS_UNEXPECTED("Request size too big");
  }
  BSpaceDescriptor bspd{0};
  addr_t addr = 0;
  RETURN_IF_NOT_OK(bs_->Alloc(sz, &bspd, &addr));
  *offset = static_cast<off64_t>(addr);
  // We will do piecewise copy of the data to disk.
  for (auto &v : buf) {
    RETURN_IF_NOT_OK(Write(v, addr));
    addr += v.GetSize();
  }
  return Status::OK();
}

Status StorageContainer::Truncate() const noexcept {
  if (is_open_) {
    RETURN_IF_NOT_OK(cont_.TruncateFile(fd_));
    MS_LOG(INFO) << "Container " << cont_ << " truncated";
  }
  return Status::OK();
}

StorageContainer::~StorageContainer() noexcept {
  (void)Truncate();
  (void)Close();
}

std::ostream &operator<<(std::ostream &os, const StorageContainer &s) {
  os << "File path : " << s.cont_ << "\n" << *(s.bs_.get());
  return os;
}

Status StorageContainer::CreateStorageContainer(std::shared_ptr<StorageContainer> *out_sc, const std::string &path) {
  Status rc;
  auto sc = new (std::nothrow) StorageContainer(path);
  if (sc == nullptr) {
    return Status(StatusCode::kMDOutOfMemory);
  }
  rc = sc->Create();
  if (rc.IsOk()) {
    (*out_sc).reset(sc);
  } else {
    delete sc;
  }
  return rc;
}
}  // namespace dataset
}  // namespace mindspore
