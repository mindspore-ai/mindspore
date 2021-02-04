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

#ifndef MINDSPORE_LITE_MICRO_CODER_MEMORY_MANAGER_H_
#define MINDSPORE_LITE_MICRO_CODER_MEMORY_MANAGER_H_

#include <map>
#include <vector>
#include <memory>
#include <utility>
#include "src/tensor.h"

namespace mindspore::lite::micro {
class OperatorCoder;

enum Status { kUnused = 0, kReused = 1 };

class Membuf {
 public:
  Membuf(Status status, size_t size, size_t offset) : status_(status), size_(size), offset_(offset) {}
  Membuf(Tensor *key, Status status, size_t size, size_t offset)
      : key_(key), status_(status), size_(size), offset_(offset) {}
  ~Membuf() = default;
  Tensor *key_ = nullptr;
  Status status_;
  size_t size_;
  size_t offset_;
};
using MembufPtr = std::shared_ptr<Membuf>;

class MemoryManager {
 public:
  MemoryManager() = default;
  ~MemoryManager() = default;

  int AssignMemory(const std::vector<std::unique_ptr<OperatorCoder>> &nodes);
  size_t GetAllocatedSize() const;
  std::map<Tensor *, size_t> variables_offset() { return variables_offset_; }

 private:
  void AssignOutputs(const std::unique_ptr<OperatorCoder> &node);
  void ReleaseInputs(const std::unique_ptr<OperatorCoder> &node);

  void SplitMembuf(size_t index, size_t size);
  void MergeMembuf();
  void UpdataMembufInfo(const MembufPtr &membuf, Tensor *key);
  void AssignNewMembuf(Tensor *key, size_t size);
  void ReuseExistedMembuf(size_t index, Tensor *key, size_t size);

  std::map<size_t, size_t> GetReusableMembufMap(size_t size);

  void StoreMembufListInfo(const std::unique_ptr<OperatorCoder> &node);

 private:
  std::vector<MembufPtr> membuf_list_;
  std::vector<std::pair<size_t, std::vector<MembufPtr>>> all_membuf_list_info_;
  std::map<Tensor *, size_t> variables_offset_;
};

}  // namespace mindspore::lite::micro
#endif  // MINDSPORE_LITE_MICRO_CODER_MEMORY_MANAGER_H_
