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
#include <memory>
#include <vector>
#include <utility>
#include <map>
#include <tuple>
#include <unordered_map>
#include <nlohmann/json.hpp>

#include "backend/session/kernel_graph.h"
#include "ir/anf.h"
#include "backend/kernel_compiler/kernel.h"

namespace mindspore {
namespace kernel {
namespace tbe {
using std::string;
using std::vector;
class TbeUtils {
 public:
  TbeUtils() = default;

  ~TbeUtils() = default;

  static void SaveJsonInfo(const std::string &json_name, const std::string &info);

  static void LoadCache();

  static void GenLicInfo(nlohmann::json *lic_info_json);

  static nlohmann::json GenSocInfo();

  static std::string GetSocVersion();

  static std::string GetOpDebugPath();

  static std::string GetBankPath();

  static std::string GetTuneDumpPath();

  static void SaveCompileInfo(const std::string &json_name, const std::string &build_res, bool *save_flag);

  static void GetCompileInfo(const AnfNodePtr &node, std::string *compile_info, bool *get_flag);

  static bool CheckOfflineTune();

  static KernelPackPtr SearchCache(const std::string &kernel_name, const bool is_akg = false);

  static KernelPackPtr InsertCache(const std::string &kernel_name, const std::string &processor,
                                   const bool is_akg = false);
};

struct KernelMetaInfo {
  uintptr_t func_stub_;
  uint32_t block_dim_;
};
using KernelMetaPtr = std::shared_ptr<KernelMetaInfo>;

class KernelManager {
 public:
  static uintptr_t GenFuncStub(const KernelPack &kernel_pack, bool force_reload, uint32_t *block_dim,
                               const bool dynamic_flag = false, void **handle = nullptr,
                               std::string *origin_key = nullptr);
  static std::string GetStubFuncName(const KernelPackPtr &kernel_pack);

 private:
  KernelManager() = default;
  ~KernelManager() = default;
  static int BinaryRegister(const FlexArray &kernel_buffer, void **module, const string &magic,
                            const bool dynamic_flag);
  static std::unordered_map<string, KernelMetaPtr> info_table_;
  static uintptr_t kernel_stub_gen_;
};

class KernelMeta {
 public:
  static KernelMeta *GetInstance();
  bool ReadIndex(const std::string &bin_dir);
  KernelPackPtr GetKernelPack(const std::string &kernel_name, const bool is_akg = false);

 private:
  KernelMeta() = default;
  ~KernelMeta() = default;
  std::unordered_map<std::string, std::string> kernel_index_map_{};
  std::unordered_map<std::string, KernelPackPtr> kernel_pack_map_{};
};
}  // namespace tbe
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TBE_TBE_UTILS_H_
