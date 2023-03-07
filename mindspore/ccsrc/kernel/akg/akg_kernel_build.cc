/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "kernel/akg/akg_kernel_build.h"

#include <sys/shm.h>
#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <iostream>
#include "ir/dtype.h"
#include "ir/func_graph.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "kernel/common_utils.h"
#include "kernel/akg/akg_kernel_json_generator.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace kernel {
constexpr int32_t MAX_ERROR_LEN = 1024;
constexpr int32_t PROCESS_NUM = 16;
constexpr int32_t TIME_OUT = 300;
constexpr auto kLogLevel = "log_level";

#define ACQUIRE_LOCK LockMng lock(fd_, __func__, __LINE__)

inline std::string GetErrorInfo() {
  char buf[MAX_ERROR_LEN + 1] = {0};
  auto ret = strerror_r(errno, buf, MAX_ERROR_LEN);
#if (_POSIX_C_SOURCE >= 200112L) && !_GNU_SOURCE
  if (ret != 0 || strlen(buf) == 0) {
    return "Call strerror_r failed";
  }

  return std::string(buf);
#else
  return ret != nullptr ? std::string(ret) : "Failed to get error info";
#endif
}

bool AkgKernelPool::LockMng::TryLock() const {
  // Try to lock trial times. Return errno if lock unsuccessfully
  uint32_t trial = 2000;
  const uint32_t sleep_time_us = 5000;

  int32_t ret;
  while (trial > 0) {
    ret = lockf(fd_, F_TLOCK, 0);
    if (ret == 0 || (errno != EACCES && errno != EAGAIN)) {
      break;
    }

    trial--;
    (void)usleep(sleep_time_us);
  }

  if (ret == -1) {
    MS_LOG(ERROR) << "Failed to acquire the lock, error msg:" << GetErrorInfo() << ", left trying times: " << trial;
    return false;
  }

  MS_LOG(INFO) << "AkgKernelBuild successfully acquire lock called at " << calling_position_;
  return true;
}

void AkgKernelPool::LockMng::Unlock() const noexcept {
  auto ret = lockf(fd_, F_ULOCK, 0);
  if (ret == -1) {
    MS_LOG(ERROR) << "Failed to release the lock, error msg:" << GetErrorInfo();
  }
  MS_LOG(INFO) << "AkgKernelBuild successfully release lock called at " << calling_position_;
}

std::string AkgKernelPool::GetCurrentPath() const {
  char cwd[PATH_MAX];
  char *ret = getcwd(cwd, sizeof(cwd));
  if (ret == nullptr) {
    MS_LOG(ERROR) << "Get current work directory failed, error msg:" << GetErrorInfo();
    return "";
  }

  char abspath[PATH_MAX];
  char *res = realpath(cwd, abspath);
  if (res == nullptr) {
    MS_LOG(ERROR) << "Change to realpath failed, error msg:" << GetErrorInfo();
    return "";
  }

  return std::string(abspath);
}

void *AkgKernelPool::CreateSharedMem(const std::string &path) {
  is_creator_ = false;

  auto hash_id = std::hash<std::string>()(path);
  auto key_id = static_cast<key_t>(hash_id);
  const size_t min_mem_size = 512;
  auto mem_size = sizeof(size_t) * kListNum_ * (kMaxKernelNum_ + 1) + min_mem_size;

  {
    ACQUIRE_LOCK;
    if (!lock.locked_) {
      MS_LOG(ERROR) << "Failed to acquire lock.";
      return nullptr;
    }

    // check if the shared memory exists or not.
    // remove shared memory if exists and the nattach is 0
    struct shmid_ds buf;
    auto id = shmget(key_id, mem_size, 0);
    if (id != -1) {
      auto ret = shmctl(id, IPC_STAT, &buf);
      if (ret == -1) {
        MS_LOG(ERROR) << "Failed to get the info of shared memory, error msg:" << GetErrorInfo();
        return nullptr;
      }

      if (buf.shm_nattch == 0) {
        ret = shmctl(id, IPC_RMID, nullptr);
        if (ret < 0) {
          MS_LOG(EXCEPTION) << "Release shared_mem failed, error msg:" << GetErrorInfo();
        }
      }
    }
  }

  ACQUIRE_LOCK;
  if (!lock.locked_) {
    MS_LOG(ERROR) << "Failed to acquire lock.";
    return nullptr;
  }

  shm_id_ = shmget(key_id, mem_size, IPC_CREAT | IPC_EXCL | 0600);
  if (shm_id_ == -1) {
    if (errno == EEXIST) {
      shm_id_ = shmget(key_id, mem_size, 0);
    }

    if (shm_id_ == -1) {
      MS_LOG(ERROR) << "Create shared_mem failed, error msg:" << GetErrorInfo();
      return nullptr;
    }
  } else {
    is_creator_ = true;
  }

  auto local_addr = shmat(shm_id_, nullptr, 0);
  if (local_addr == reinterpret_cast<void *>(-1)) {
    MS_LOG(ERROR) << "Attach to shared_mem failed, error msg:" << GetErrorInfo();
    return nullptr;
  }

  if (is_creator_) {
    (void)memset_s(local_addr, mem_size, 0, mem_size);
  }

  return local_addr;
}

int32_t AkgKernelPool::Init(const std::vector<JsonNodePair> &build_args) {
  auto cp = GetCurrentPath();
  if (cp.empty()) {
    return -1;
  }

  fd_ = open(kKeyName_, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
  if (fd_ == -1) {
    MS_LOG(ERROR) << "open file <" << kKeyName_ << "> failed, error msg:" << GetErrorInfo();
    return -1;
  }

  auto addr = CreateSharedMem(cp);
  if (addr == nullptr) {
    return -1;
  }

  InitKernelLists(addr);

  auto ret = AddKernels(build_args);
  if (ret != 0) {
    MS_LOG(ERROR) << "AkgKernelPool AddKernels failed.";
    return -1;
  }

  return 0;
}

int32_t AkgKernelPool::Release() const {
  {
    ACQUIRE_LOCK;
    if (!lock.locked_) {
      MS_LOG(ERROR) << "Failed to acquire lock.";
      return -1;
    }

    struct shmid_ds buf;
    auto ret = shmctl(shm_id_, IPC_STAT, &buf);
    if (ret == -1) {
      MS_LOG(ERROR) << "Failed to get the info of shared memory, error msg:" << GetErrorInfo();
      return -1;
    }

    bool need_delete_by_last = false;

    // if the creator exits unexpectedly and fails to delete the shm, the last process will try to delete the shm
    if (((buf.shm_perm.mode & SHM_DEST) == 0) && (buf.shm_nattch == 1)) {
      need_delete_by_last = true;
    }

    // Detach shared memory
    ret = shmdt(static_cast<void *>(kernel_lists_[0]));
    if (ret < 0) {
      MS_LOG(ERROR) << "Shared_mem detach failed, error msg:" << GetErrorInfo();
      return -1;
    }

    // Release shared memory
    if (is_creator_ || need_delete_by_last) {
      ret = shmctl(shm_id_, IPC_RMID, nullptr);
      if (ret < 0) {
        MS_LOG(ERROR) << "Release shared_mem failed, error msg:" << GetErrorInfo();
        return -1;
      }
    }
  }

  return 0;
}

int32_t AkgKernelPool::AddKernels(const std::vector<JsonNodePair> &build_args) {
  ACQUIRE_LOCK;
  if (!lock.locked_) {
    MS_LOG(ERROR) << "Failed to acquire lock.";
    return -1;
  }

  std::set<size_t> todo_list(ListBegin(kToDoIdx_), ListEnd(kToDoIdx_));
  std::set<size_t> doing_list(ListBegin(kDoingIdx_), ListEnd(kDoingIdx_));
  std::set<size_t> done_list(ListBegin(kDoneIdx_), ListEnd(kDoneIdx_));

  for (const auto &[json_generator, anf_node] : build_args) {
    MS_EXCEPTION_IF_NULL(anf_node);
    auto kernel_name = json_generator.kernel_name();

    auto hash_id = std::hash<std::string>()(kernel_name);
    if (self_kernel_ids_.count(hash_id) != 0) {
      MS_LOG(ERROR) << "Duplicated kernel found in the kernel compiling list. kernel_name[" << kernel_name << "]";
      return -1;
    }

    (void)self_kernel_ids_.emplace(hash_id);
  }

  std::set<size_t> diff_from_todo;
  std::set<size_t> diff_from_doing;
  std::set<size_t> diff_from_done;

  // add the unique kernel only once, so need to check if it exists in todo_list, doing_list, or done_list
  (void)std::set_difference(self_kernel_ids_.begin(), self_kernel_ids_.end(), todo_list.begin(), todo_list.end(),
                            std::inserter(diff_from_todo, diff_from_todo.begin()));
  (void)std::set_difference(diff_from_todo.begin(), diff_from_todo.end(), doing_list.begin(), doing_list.end(),
                            std::inserter(diff_from_doing, diff_from_doing.begin()));
  (void)std::set_difference(diff_from_doing.begin(), diff_from_doing.end(), done_list.begin(), done_list.end(),
                            std::inserter(diff_from_done, diff_from_done.begin()));

  auto new_kernel_size = diff_from_done.size();
  if (new_kernel_size + todo_list.size() > static_cast<size_t>(kMaxKernelNum_)) {
    MS_LOG(ERROR) << "The size of kernels is " << new_kernel_size << ", while the left space of the pool is "
                  << kMaxKernelNum_ - todo_list.size();
    return -1;
  }

  (void)std::copy(diff_from_done.begin(), diff_from_done.end(), ListEnd(kToDoIdx_));
  IncListSize(kToDoIdx_, new_kernel_size);

  return 0;
}

int32_t AkgKernelPool::FetchKernels(std::set<size_t> *out) {
  ACQUIRE_LOCK;
  if (!lock.locked_) {
    MS_LOG(ERROR) << "Failed to acquire lock.";
    return -1;
  }

  std::set<size_t> left_in_todo_list;

  // filter out kernels which does not belongs to this process
  auto FilterBySelfList = [&left_in_todo_list, &out, this](size_t id) {
    if (this->self_kernel_ids_.count(id) != 0) {
      (void)out->emplace(id);
    } else {
      (void)left_in_todo_list.emplace(id);
    }
  };

  (void)std::for_each(ListBegin(kToDoIdx_), ListEnd(kToDoIdx_), FilterBySelfList);

  (void)std::copy(out->begin(), out->end(), ListEnd(kDoingIdx_));
  IncListSize(kDoingIdx_, out->size());

  (void)std::copy(left_in_todo_list.begin(), left_in_todo_list.end(), ListBegin(kToDoIdx_));
  ResetListSize(kToDoIdx_, left_in_todo_list.size());

  return 0;
}

int32_t AkgKernelPool::UpdateAndWait(const std::set<size_t> &ids) {
  if (!ids.empty()) {
    ACQUIRE_LOCK;
    if (!lock.locked_) {
      MS_LOG(ERROR) << "Failed to acquire lock.";
      return -1;
    }

    // update the state of finished kernels to `done`
    (void)std::copy(ids.begin(), ids.end(), ListEnd(kDoneIdx_));
    IncListSize(kDoneIdx_, ids.size());

    // delete the finished kernels from doing_list
    std::vector<size_t> left_in_doing_list;
    std::set<size_t> doing_list(ListBegin(kDoingIdx_), ListEnd(kDoingIdx_));
    (void)std::set_difference(doing_list.begin(), doing_list.end(), ids.begin(), ids.end(),
                              std::inserter(left_in_doing_list, left_in_doing_list.begin()));

    (void)std::copy(left_in_doing_list.begin(), left_in_doing_list.end(), ListBegin(kDoingIdx_));
    ResetListSize(kDoingIdx_, left_in_doing_list.size());
  }

  auto ret = Wait();
  if (ret != 0) {
    MS_LOG(ERROR) << "AkgKernelPool Wait failed.";
    return -1;
  }

  return 0;
}

int32_t AkgKernelPool::Wait() const {
  // wait until all the kernels which belong to this process finish compiling
  uint32_t trials = 1000;
  const uint32_t sleep_time_us = 1000000;

  while (trials > 0) {
    {
      ACQUIRE_LOCK;
      if (!lock.locked_) {
        MS_LOG(ERROR) << "Failed to acquire lock.";
        return -1;
      }

      std::set<size_t> done_list(ListBegin(kDoneIdx_), ListEnd(kDoneIdx_));

      if (std::all_of(self_kernel_ids_.begin(), self_kernel_ids_.end(),
                      [&done_list](size_t id) { return done_list.count(id) != 0; })) {
        return 0;
      }
    }

    (void)usleep(sleep_time_us);
    trials--;
  }

  MS_LOG(ERROR) << "Time out while wait kernel compiling";
  return -1;
}

KernelPackPtr AkgKernelBuilder::AkgSearchCache(const std::string &kernel_name) {
  auto processor = GetStrProcessorFromContext();
  return SearchCache(kernel_name, processor);
}

KernelPackPtr AkgKernelBuilder::AkgInsertCache(const std::string &kernel_name) {
  auto processor = GetStrProcessorFromContext();
  return InsertCache(kernel_name, processor);
}

std::vector<JsonNodePair> AkgKernelBuilder::GetNotCachedKernels(const std::vector<JsonNodePair> &build_args) {
  LoadCache();
  std::unordered_set<std::string> kernel_name_set;
  std::vector<JsonNodePair> new_build_args;
  for (const auto &[json_generator, anf_node] : build_args) {
    MS_EXCEPTION_IF_NULL(anf_node);
    auto kernel_name = json_generator.kernel_name();

    auto cached_kernel_pack = AkgSearchCache(kernel_name);
    if (cached_kernel_pack != nullptr) {
      MS_LOG(DEBUG) << "Use cached kernel, kernel_name[" << kernel_name << "], fullname_with_scope["
                    << anf_node->fullname_with_scope() << "].";
      AkgSetKernelMod(cached_kernel_pack, json_generator, anf_node);
      continue;
    }

    if (kernel_name_set.count(kernel_name) != 0) {
      (void)repeat_nodes_.emplace_back(json_generator, anf_node);
      continue;
    }
    (void)kernel_name_set.insert(kernel_name);
    (void)new_build_args.emplace_back(json_generator, anf_node);
  }
  return new_build_args;
}

bool AkgKernelBuilder::InsertToCache(const std::vector<JsonNodePair> &build_args) {
  for (const auto &[json_generator, anf_node] : build_args) {
    auto kernel_name = json_generator.kernel_name();
    auto new_kernel_pack = AkgInsertCache(kernel_name);
    if (new_kernel_pack == nullptr) {
      MS_LOG(ERROR) << "Insert to cache failed, kernel_name[" << kernel_name << "], fullname_with_scope["
                    << anf_node->fullname_with_scope() << "].";
      return false;
    }
    AkgSetKernelMod(new_kernel_pack, json_generator, anf_node);
    MS_LOG(DEBUG) << "Akg compile " << kernel_name << " kernel and insert cache successfully!";
  }
  return true;
}

bool AkgKernelBuilder::HandleRepeatNodes() {
  for (const auto &[json_generator, anf_node] : repeat_nodes_) {
    auto kernel_name = json_generator.kernel_name();
    auto cached_kernel_pack = AkgSearchCache(kernel_name);
    if (cached_kernel_pack == nullptr) {
      MS_LOG(ERROR) << "Kernel is not found in cache, kernel_name[" << kernel_name << "], fullname_with_scope["
                    << anf_node->fullname_with_scope() << "].";
      return false;
    }
    MS_LOG(DEBUG) << "Use the cached kernel found in cache, kernel_name[" << kernel_name << "], fullname_with_scope["
                  << anf_node->fullname_with_scope() << "].";
    AkgSetKernelMod(cached_kernel_pack, json_generator, anf_node);
  }
  return true;
}

std::vector<std::string> AkgKernelBuilder::GetKernelJsonsByHashId(const std::vector<JsonNodePair> &build_args,
                                                                  const std::set<size_t> &fetched_ids) {
  std::vector<std::string> jsons;
  for (const auto &[json_generator, anf_node] : build_args) {
    MS_EXCEPTION_IF_NULL(anf_node);
    auto kernel_name = json_generator.kernel_name();
    auto hash_id = std::hash<std::string>()(kernel_name);
    if (fetched_ids.count(hash_id) == 0) {
      continue;
    }
    auto kernel_json = json_generator.kernel_json_str();
    AkgSaveJsonInfo(kernel_name, kernel_json);
    jsons.push_back(kernel_json);
  }
  return jsons;
}

bool AkgKernelBuilder::ParallelBuild(const std::vector<JsonNodePair> &build_args) {
  struct timeval start_time, end_time;
  (void)gettimeofday(&start_time, nullptr);
  MS_LOG(INFO) << "Akg start parallel build. kernel count: " << build_args.size();

  AkgKernelPool kp;
  auto ret = kp.Init(build_args);
  if (ret != 0) {
    MS_LOG(ERROR) << "AkgKernelPool init failed.";
    return false;
  }

  std::set<size_t> fetched_ids;
  ret = kp.FetchKernels(&fetched_ids);
  if (ret != 0) {
    MS_LOG(ERROR) << "AkgKernelPool FetchKernels failed.";
    return false;
  }

  if (!fetched_ids.empty()) {
    auto jsons = GetKernelJsonsByHashId(build_args, fetched_ids);

    auto client = GetClient();
    MS_EXCEPTION_IF_NULL(client);
    if (!client->AkgStart(PROCESS_NUM, TIME_OUT)) {
      MS_LOG(ERROR) << "Akg start failed.";
      return false;
    }
    auto attrs = CollectBuildAttrs();
    if (!attrs.empty() && !client->AkgSendAttr(attrs)) {
      MS_LOG(ERROR) << "Akg send attr failed.";
      return false;
    }
    if (!client->AkgSendData(jsons)) {
      MS_LOG(ERROR) << "Akg send data failed.";
      return false;
    }
    if (!client->AkgWait()) {
      MS_LOG(ERROR) << "Akg compile failed.";
      return false;
    }
  }

  ret = kp.UpdateAndWait(fetched_ids);
  if (ret != 0) {
    MS_LOG(ERROR) << "AkgKernelPool UpdateAndWait failed.";
    return false;
  }

  if (kp.Release() != 0) {
    MS_LOG(ERROR) << "AkgKernelPool release failed.";
    return false;
  }

  (void)gettimeofday(&end_time, nullptr);
  const uint64_t kUSecondInSecond = 1000000;
  uint64_t cost = kUSecondInSecond * static_cast<uint64_t>(end_time.tv_sec - start_time.tv_sec);
  cost += static_cast<uint64_t>(end_time.tv_usec - start_time.tv_usec);
  MS_LOG(INFO) << "Akg kernel build time: " << cost << " us.";

  return true;
}

bool AkgKernelBuilder::AkgOpParallelBuild(const std::vector<JsonNodePair> &build_args) {
  repeat_nodes_.clear();
  auto new_build_args = GetNotCachedKernels(build_args);
  if (new_build_args.empty()) {
    return true;
  }

  build_attrs_[kLogLevel] = "ERROR";
  if (!ParallelBuild(new_build_args)) {
    return false;
  }

  // All unique done here, cache them and set kernel.
  if (!InsertToCache(build_args)) {
    MS_LOG(ERROR) << "Insert cache failed.";
    return false;
  }

  if (!HandleRepeatNodes()) {
    MS_LOG(ERROR) << "Handle repeat nodes failed.";
    return false;
  }

  return true;
}

void AkgKernelBuilder::LoadCache() {
  static bool has_load = false;
  if (has_load) {
    return;
  }
  auto bin_map = KernelMeta::GetInstance();
  auto kernel_dir = bin_map->kernel_meta_path();
  DIR *dir = opendir(kernel_dir.c_str());
  if (dir == nullptr) {
    MS_LOG(DEBUG) << "kernel dir [" << kernel_dir << "] not exist";
    return;
  }
  struct dirent *entry;
  constexpr size_t SUFFIX_LENS = 5;
  while ((entry = readdir(dir)) != nullptr) {
    std::string kernel_json = entry->d_name;
    if (kernel_json.length() <= SUFFIX_LENS) {
      continue;
    }
    auto suffix = kernel_json.substr(kernel_json.length() - SUFFIX_LENS);
    if (suffix != kJsonSuffix) {
      continue;
    }
    auto sp = kernel_json.rfind('/');
    if (sp != std::string::npos) {
      continue;
    }
    auto kernel_name = kernel_json.substr(0, kernel_json.length() - SUFFIX_LENS);
    (void)bin_map->Insert(kernel_name, kernel_dir + kernel_json);
  }
  has_load = true;
  (void)closedir(dir);
  return;
}

bool AkgKernelBuilder::AkgKernelParallelBuild(const std::vector<AnfNodePtr> &anf_nodes) {
  std::vector<JsonNodePair> json_and_node;
  for (const auto &anf_node : anf_nodes) {
    MS_EXCEPTION_IF_NULL(anf_node);
    // Node already has kernel mod, no need to process it.
    if (AnfAlgo::GetKernelMod(anf_node) != nullptr) {
      continue;
    }
    graphkernel::DumpOption option;
    option.get_target_info = true;
    AkgKernelJsonGenerator akg_kernel_json_generator(option);
    auto cnode = anf_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    bool is_custom_node = IsPrimitiveCNode(cnode, prim::kPrimCustom) || IsAKGSparseOP(cnode);
    // Graph kernel node and Custom node need to generate composite json
    if (common::AnfAlgo::IsGraphKernel(cnode) || is_custom_node) {
      FuncGraphPtr func_graph = is_custom_node ? cnode->func_graph() : common::AnfAlgo::GetCNodeFuncGraphPtr(cnode);
      MS_EXCEPTION_IF_NULL(func_graph);
      auto mng = func_graph->manager();
      if (mng == nullptr) {
        mng = Manage(func_graph, true);
        func_graph->set_manager(mng);
      }
      if (is_custom_node) {
        // in this case, the cnode is a CustomOp (no matter whether graph kernel mode is enabled or not)
        // generate the fused json for the single kernel cnode
        if (!akg_kernel_json_generator.CollectFusedJsonWithSingleKernel(cnode)) {
          MS_EXCEPTION(UnknownError) << "Collect op info failed. op[" << anf_node->fullname_with_scope() << "].";
        }
      } else {
        // in this case, the cnode is a IsGraphKernel when graph kernel mode is enabled
        // generate the fused json for the graph kernel subgraph
        std::vector<AnfNodePtr> node_list, input_list, output_list;
        GetValidKernelNodes(func_graph, &node_list, &input_list, &output_list);
        if (!akg_kernel_json_generator.CollectFusedJson(node_list, input_list, output_list)) {
          MS_EXCEPTION(UnknownError) << "Collect op info failed. op[" << anf_node->fullname_with_scope() << "].";
        }
      }
    } else {
      if (!akg_kernel_json_generator.CollectJson(anf_node)) {
        MS_EXCEPTION(UnknownError) << "Collect op info failed. op[" << anf_node->fullname_with_scope() << "].";
      }
    }
    (void)json_and_node.emplace_back(std::move(akg_kernel_json_generator), anf_node);
  }

  if (json_and_node.empty()) {
    MS_LOG(INFO) << "There is no akg kernel to be compiled.";
    return true;
  }

  bool res = AkgOpParallelBuild(json_and_node);
  if (!res) {
    MS_LOG(ERROR) << "Akg build kernel failed.";
  }
  return true;
}

std::string AkgKernelBuilder::CollectBuildAttrs() {
  auto &flags = graphkernel::GraphKernelFlags::GetInstance();
  if (!flags.enable_vectorization) {
    build_attrs_["enable_vectorization"] = flags.enable_vectorization;
  }
  if (flags.online_tuning > 0) {
    build_attrs_["online_tuning"] = flags.online_tuning;
  }
  if (!flags.repository_path.empty()) {
    build_attrs_["repository_path"] = flags.repository_path;
  }
  return build_attrs_.empty() ? "" : build_attrs_.dump();
}
}  // namespace kernel
}  // namespace mindspore
