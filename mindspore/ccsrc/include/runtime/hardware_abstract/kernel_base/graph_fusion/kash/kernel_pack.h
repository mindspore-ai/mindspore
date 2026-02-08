/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_KERNEL_BASE_GRAPH_FUSION_KASH_KERNEL_PACK_H_
#define MINDSPORE_CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_KERNEL_BASE_GRAPH_FUSION_KASH_KERNEL_PACK_H_
#include <string>
#include <vector>
#include <memory>
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"

#ifdef _MSC_VER
#undef OPAQUE
#endif

#ifdef OPAQUE
#undef OPAQUE
#endif

namespace mindspore {
namespace kernel {
struct FlexArray {
  size_t len;
  char contents[];
};

struct GlobalWorkspace {
  size_t size{0};
  size_t type{0};
  bool is_overflow = false;
};

struct NodeBaseInfo {
  size_t workspace_num;
  size_t input_num;
  size_t output_num;
  size_t offset_index;
};

struct KernelJsonInfo {
  std::string bin_file_name;
  std::string bin_file_suffix;
  uint32_t block_dim{0};
  std::string kernel_name;
  std::string magic;
  std::vector<size_t> parameters;
  std::string sha256;
  std::vector<size_t> workspaces_type;
  std::vector<size_t> workspaces;
  GlobalWorkspace global_workspace;
  bool has_kernel_list{false};
  uint32_t op_para_size{0};
  int32_t KBHit{0};
  uint32_t mode_in_args_first_field{0};
  uint32_t batch_bind_only{0};
  uint32_t task_ration{0};
  std::string core_type;
  std::vector<std::vector<size_t>> args_remap;
  AtomicInitInfo atomic_init_info;
  NodeBaseInfo node_base_info;
  KernelJsonInfo() = default;
};

class RUNTIME_HARDWARE_EXPORT KernelPack {
 public:
  KernelPack() : json_(nullptr), kernel_(nullptr) {}
  KernelPack(const KernelPack &) = default;
  KernelPack &operator=(const KernelPack &) = default;
  KernelJsonInfo kernel_json_info() const;
  bool LoadKernelMeta(const std::string &json_f);
  bool ReadFromJsonFile(const std::string &json_f, const std::string &processor);
  const FlexArray *GetJson() const { return json_; }
  const FlexArray *GetKernel() const { return kernel_; }
  ~KernelPack() {
    if (json_ != nullptr) {
      delete[] json_;
      json_ = nullptr;
    }
    if (kernel_ != nullptr) {
      delete[] kernel_;
      kernel_ = nullptr;
    }
  }

 private:
  bool ReadFromJsonFileHelper(std::ifstream &kernel_bin);
  void ParseKernelJson(const nlohmann::json &js);
  static void ParseKernelName(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info);
  static void ParseBinFileName(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info);
  static void ParseBinFileSuffix(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info);
  static void ParseMagic(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info);
  static void ParseBlockDim(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info);
  static void ParseTaskRatio(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info);
  static void ParseCoreType(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info);
  static void ParseParameters(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info);
  static void ParseWorkSpace(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info);
  static void ParseOpParaSize(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info);
  static void ParseSHA256(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info);
  static void ParseKBHit(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info);
  static void ParseBatchBindOnly(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info);
  static void ParseKernelList(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info);
  static void ParseModeInArgsFirstField(const std::string &key, const nlohmann::json &js,
                                        KernelJsonInfo *kernel_json_info);
  static void ParseArgsRemap(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info);
  static void ParseGlogbleWorkSpace(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info);

  KernelJsonInfo kernel_json_info_;
  FlexArray *json_;
  FlexArray *kernel_;
};
using KernelPackPtr = std::shared_ptr<KernelPack>;
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_KERNEL_BASE_GRAPH_FUSION_KASH_KERNEL_PACK_H_
