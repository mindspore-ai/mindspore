/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_TRAIN_STATIC_ALLOCATOR_H_
#define MINDSPORE_LITE_SRC_TRAIN_STATIC_ALLOCATOR_H_

namespace mindspore {
class StaticAllocator : public Allocator {
 public:
  void SetContex(void *buf, size_t size) {
    start_buf_ = buf;
    size_ = size;
  }
  int SetRefCount(void *ptr, int ref_count) override { return 0; }
  int DecRefCount(void *ptr, int ref_count) override { return 0; }
  int IncRefCount(void *ptr, int ref_count) override { return 0; }
  size_t total_size() { return total_size_; }
  void Clear() {}
  void *Malloc(size_t size) override {
    total_size_ += size;
    return malloc(size);
  }
  void Free(void *ptr) override {
    if (RefCount(ptr) != 0) free(ptr);
  }

  int RefCount(void *ptr) override {
    if (ptr == nullptr) return STATIC_ALLOCATION;
    char *ptrc = reinterpret_cast<char *>(ptr);
    char *bufc = reinterpret_cast<char *>(start_buf_);
    return ((ptrc < bufc) || (ptrc - bufc >= static_cast<ptrdiff_t>(size_)) ? 1 : 0);
  }

 private:
  void *start_buf_;
  size_t size_;
  size_t total_size_ = 0;
};
};      // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_TRAIN_STATIC_ALLOCATOR_H_
