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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TBE_TBE_UTILS_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TBE_TBE_UTILS_H_
#include <string>
#include <set>
#include <memory>
#include <vector>
#include <utility>
#include <map>
#include <tuple>
#include <unordered_map>
#include <nlohmann/json.hpp>

#include "backend/common/session/kernel_graph.h"
#include "ir/anf.h"
#include "kernel/kernel.h"

namespace mindspore {
namespace kernel {
namespace tbe {
using std::string;
using std::vector;

constexpr size_t OP_DEBUG_LEVEL_0 = 0;  // 0: turn off op debug, remove kernel_meta
constexpr size_t OP_DEBUG_LEVEL_1 = 1;  // 1: turn on op debug, gen cce file
constexpr size_t OP_DEBUG_LEVEL_2 = 2;  // 2: turn on op debug, gen cce file, turn off op compile optimization
constexpr size_t OP_DEBUG_LEVEL_3 = 3;  // 3: turn off op debug, keep kernel_meta
constexpr size_t OP_DEBUG_LEVEL_4 = 4;  // 4: turn off op debug, gen _compute.json file

typedef enum {
  CCE_KERNEL = 0,
  TBE_PREBUILD = 1,
} saveType;

class TbeUtils {
 public:
  TbeUtils() = default;

  ~TbeUtils() = default;

  static void SaveJsonInfo(const std::string &json_name, const std::string &info,
                           saveType save_type = saveType::CCE_KERNEL);

  static void LoadCache();

  static void GenLicInfo(nlohmann::json *lic_info_json);

  static std::string GetOpDebugConfig();

  static std::string GetOpDebugLevel();

  static std::vector<std::string> SplitAndRemoveSpace(const std::string &s, char delim);

  static nlohmann::json GenSocInfo();

  static std::string GetOpDebugPath();

  static std::string GetKernelMetaTempDir();

  static std::string GetBankPath();

  static std::string GetTuneDumpPath();

  static void SaveCompileInfo(const std::string &json_name, const std::string &build_res, bool *save_flag);

  static void GetCompileInfo(const AnfNodePtr &node, std::string *compile_info, bool *get_flag);

  static bool CheckOfflineTune();

  static KernelPackPtr SearchCache(const std::string &kernel_name, const bool is_akg = false);

  static KernelPackPtr InsertCache(const std::string &kernel_name, const std::string &processor,
                                   const bool is_akg = false);
  static void UpdateCache(const std::string &kernel_name);

  // check target value is one of the candidates
  template <typename T>
  static bool IsOneOf(const std::set<T> &candidates, const T &val) {
    if (candidates.empty()) {
      return false;
    }
    auto iter = candidates.find(val);
    return iter != candidates.end();
  }
};

struct KernelMetaInfo {
  uintptr_t result_;
  uint32_t block_dim_;
  void *handle_;
};
using KernelMetaPtr = std::shared_ptr<KernelMetaInfo>;

class KernelManager {
 public:
  static uintptr_t GenFuncStub(const KernelPack &kernel_pack, bool force_reload, uint32_t *block_dim, void **handle);
  static std::string GetStubFuncName(const KernelPackPtr &kernel_pack);

 private:
  KernelManager() = default;
  ~KernelManager() = default;
  static int BinaryRegister(const FlexArray &kernel_buffer, void **module, const string &magic,
                            const std::string &func_name, bool has_kernel_list);
  static std::unordered_map<string, KernelMetaPtr> info_table_;
  static std::atomic<uintptr_t> kernel_stub_gen_;
  static std::mutex info_table_mutex_;
};

class KernelMeta {
 public:
  static KernelMeta *GetInstance();
  bool ReadIndex(const std::string &bin_dir);
  KernelPackPtr GetKernelPack(const std::string &kernel_name, const bool is_akg = false);
  void UpdateCache(const std::string &kernel_name);
  KernelPackPtr SearchInFile(const std::string &kernel_name);

 private:
  KernelMeta() = default;
  ~KernelMeta() = default;
  KernelPackPtr LoadFromFile(const std::string &kernel_name) const;
  std::unordered_map<std::string, std::string> kernel_index_map_{};
  std::unordered_map<std::string, KernelPackPtr> kernel_pack_map_{};
};
}  // namespace tbe
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TBE_TBE_UTILS_H_
