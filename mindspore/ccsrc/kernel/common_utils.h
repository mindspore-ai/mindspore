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

#ifndef MINDSPORE_CCSRC_KERNEL_COMMON_UTILS_H_
#define MINDSPORE_CCSRC_KERNEL_COMMON_UTILS_H_

#include <dirent.h>
#include <memory>
#include <unordered_map>
#include <map>
#include <string>
#include <vector>
#include "kernel/kernel.h"
#include "kernel/oplib/opinfo.h"
#include "kernel/kernel_build_info.h"

namespace mindspore {
namespace kernel {
constexpr auto kCceKernelMeta = "./kernel_meta/";
constexpr auto kGpuKernelMeta = "./cuda_meta";
constexpr auto kProcessorAiCore = "aicore";
constexpr auto kProcessorAiCpu = "aicpu";
constexpr auto kProcessorCuda = "cuda";
constexpr auto kJsonSuffix = ".json";
constexpr auto kInfoSuffix = ".info";
constexpr unsigned int AUTODIFF_COMPILE_OVERTIME = 600;
constexpr auto kAkgModule = "_akg";
constexpr auto kArgDataformat = "data_format";

const std::vector<std::string> support_devices = {"aicore", "aicpu", "cuda"};

struct KernelMetaInfo {
  uintptr_t func_stub_;
  uint32_t block_dim_;
};
using KernelMetaPtr = std::shared_ptr<KernelMetaInfo>;

class KernelMeta {
 public:
  KernelMeta() = default;
  void Initialize();
  void RemoveKernelCache();
  std::string Search(const std::string &kernel_name) const;
  bool Insert(const std::string &kernel_name, const std::string &kernel_json);
  std::string GetKernelMetaPath() { return kernel_meta_path_; }

  static KernelMeta *GetInstance() {
    static KernelMeta kernel_meta;
    return &kernel_meta;
  }
  ~KernelMeta() = default;

 private:
  bool initialized_ = false;
  std::string kernel_meta_path_;
  std::unordered_map<std::string, std::string> kernel_meta_map_;
};

bool CheckCache(const std::string &kernel_name);
KernelPackPtr SearchCache(const std::string &kernel_name, const std::string &processor);
KernelPackPtr InsertCache(const std::string &kernel_name, const std::string &processor);
TypeId DtypeToTypeId(const std::string &dtypes);
std::string Dtype2String(const std::string &dtypes);
std::string Dtype2ShortType(const std::string &dtypes);
std::string TypeId2String(TypeId type_id);
size_t GetDtypeNbyte(const std::string &dtypes);
bool ParseMetadata(const CNodePtr &kernel_node, const std::shared_ptr<const OpInfo> &op_info_ptr, Processor processor,
                   std::vector<std::shared_ptr<KernelBuildInfo>> *const kernel_info_list);
bool IsAtomicNode(const CNodePtr &kernel_node);
void SaveJsonInfo(const std::string &json_name, const std::string &info);
std::string GetProcessor(const AnfNodePtr &anf_node);
bool IsSameShape(const std::vector<size_t> &shape_a, const std::vector<size_t> &shape_b);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_COMMON_UTILS_H_
