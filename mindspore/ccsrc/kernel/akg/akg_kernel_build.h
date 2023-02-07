/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_AKG_KERNEL_BUILD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_AKG_KERNEL_BUILD_H_

#include <string>
#include <utility>
#include <vector>
#include <map>
#include <set>
#include "nlohmann/json.hpp"
#include "ir/anf.h"
#include "kernel/kernel.h"
#include "backend/common/session/kernel_build_client.h"
#include "kernel/akg/akg_kernel_json_generator.h"

namespace mindspore {
namespace kernel {
using graphkernel::AkgKernelJsonGenerator;
using JsonNodePair = std::pair<AkgKernelJsonGenerator, AnfNodePtr>;

class BACKEND_EXPORT AkgKernelBuilder {
 public:
  AkgKernelBuilder() = default;
  virtual ~AkgKernelBuilder() = default;

  virtual KernelBuildClient *GetClient() = 0;
  virtual KernelPackPtr AkgSearchCache(const std::string &kernel_name);
  virtual KernelPackPtr AkgInsertCache(const std::string &kernel_name);
  virtual void AkgSetKernelMod(const KernelPackPtr &kernel_pack, const AkgKernelJsonGenerator &json_generator,
                               const AnfNodePtr &anf_node) = 0;
  virtual void AkgSaveJsonInfo(const string &kernel_name, const string &kernel_json) = 0;
  virtual void LoadCache();
  bool AkgKernelParallelBuild(const std::vector<AnfNodePtr> &anf_nodes);

  virtual bool ParallelBuild(const std::vector<JsonNodePair> &build_args);

 private:
  std::vector<JsonNodePair> GetNotCachedKernels(const std::vector<JsonNodePair> &build_args);
  std::vector<std::string> GetKernelJsonsByHashId(const std::vector<JsonNodePair> &build_args,
                                                  const std::set<size_t> &fetched_ids);
  bool InsertToCache(const std::vector<JsonNodePair> &build_args);
  bool HandleRepeatNodes();
  bool AkgOpParallelBuild(const std::vector<JsonNodePair> &build_args);

  std::vector<JsonNodePair> repeat_nodes_;
  nlohmann::json build_attrs_;
  std::string CollectBuildAttrs();
};

class AkgKernelPool {
 public:
  class LockMng {
   public:
    explicit LockMng(const int32_t fd, const char *function, const uint32_t line) {
      fd_ = fd;
      calling_position_ = std::string(function) + ":" + std::to_string(line);
      locked_ = TryLock();
    }

    virtual ~LockMng() {
      if (locked_) {
        Unlock();
      }
    }

    bool locked_{false};

   private:
    bool TryLock() const;
    void Unlock() const noexcept;

    int32_t fd_{-1};
    std::string calling_position_;
  };

  AkgKernelPool() = default;
  virtual ~AkgKernelPool() {
    // Close key file
    if (fd_ != -1) {
      (void)close(fd_);
    }
  }

  int32_t Init(const std::vector<JsonNodePair> &build_args);
  int32_t Release() const;
  int32_t FetchKernels(std::set<size_t> *out);
  int32_t UpdateAndWait(const std::set<size_t> &ids);

  constexpr inline static size_t kMaxKernelNum_{1000};

  // allocate memory for todo_list, doing_list, done_list
  constexpr inline static size_t kListNum_{3};

  constexpr inline static auto kKeyName_ = "./akg_build_tmp.key";

  constexpr inline static int32_t kToDoIdx_ = 0;
  constexpr inline static int32_t kDoingIdx_ = 1;
  constexpr inline static int32_t kDoneIdx_ = 2;

 private:
  inline size_t *ListBegin(int32_t list_idx) { return kernel_lists_[list_idx]; }
  inline const size_t *ListBegin(int32_t list_idx) const { return kernel_lists_[list_idx]; }

  inline size_t *ListEnd(int32_t list_idx) { return kernel_lists_[list_idx] + kernel_lists_[list_idx][kMaxKernelNum_]; }
  inline const size_t *ListEnd(int32_t list_idx) const {
    return kernel_lists_[list_idx] + kernel_lists_[list_idx][kMaxKernelNum_];
  }

  inline void ResetListSize(int32_t list_idx, size_t val) { kernel_lists_[list_idx][kMaxKernelNum_] = val; }

  inline void IncListSize(int32_t list_idx, size_t val) { kernel_lists_[list_idx][kMaxKernelNum_] += val; }

  void *CreateSharedMem(const std::string &path);
  std::string GetCurrentPath() const;

  inline void InitKernelLists(void *addr) {
    kernel_lists_[kToDoIdx_] = static_cast<size_t *>(addr);
    kernel_lists_[kDoingIdx_] = kernel_lists_[kToDoIdx_] + kMaxKernelNum_ + 1;
    kernel_lists_[kDoneIdx_] = kernel_lists_[kDoingIdx_] + kMaxKernelNum_ + 1;
  }

  int32_t AddKernels(const std::vector<JsonNodePair> &build_args);
  int32_t Wait() const;

  int32_t shm_id_{-1};
  bool is_creator_{false};
  int32_t fd_{-1};

  // includes 3 lists: todo_list, doing_list, done_list.
  // each list has kMaxKernelNum_ + 1 elements and, the count of elements in each list
  // is stored in kernel_lists_[xx][kMaxKernelNum_]
  size_t *kernel_lists_[kListNum_]{nullptr, nullptr, nullptr};

  std::set<size_t> self_kernel_ids_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_AKG_KERNEL_BUILD_H_
