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

#include "kernel/tbe/tbe_utils.h"

#include <sys/types.h>
#include <dirent.h>
#include <vector>
#include <string>
#include <utility>
#include <map>
#include <functional>
#include <iostream>
#include <fstream>

#include "kernel/oplib/oplib.h"
#include "utils/utils.h"
#include "session/anf_runtime_algorithm.h"
#include "common/utils.h"
#include "device/kernel_info.h"
#include "ir/dtype/type.h"
#include "kernel/tbe/tbe_convert_utils.h"
#include "securec/include/securec.h"
#include "operator/ops.h"

namespace mindspore {
namespace kernel {
namespace tbe {
constexpr auto kCceKernelMeta = "./kernel_meta/";
constexpr auto kJsonSuffix = ".json";
constexpr auto kInfoSuffix = ".info";

uintptr_t KernelManager::kernel_stub_gen_ = 0;
std::unordered_map<string, KernelMetaPtr> KernelManager::info_table_ = {};

void TbeUtils::SaveJsonInfo(const std::string &json_name, const std::string &info) {
  char real_path[PATH_MAX] = {0};
  std::string path = kCceKernelMeta + json_name + kInfoSuffix;
  if (path.size() > PATH_MAX) {
    MS_LOG(ERROR) << "file path: " << path << "is too long.";
    return;
  }
  std::ifstream fin(path);
  if (fin) {
    MS_LOG(INFO) << "json file exist, no need to create.";
    return;
  }
  std::ofstream filewrite;
  filewrite.open(path);
  if (!filewrite.is_open()) {
    return;
  }
  filewrite << info << std::endl;
  filewrite.close();
  if (nullptr == realpath(path.c_str(), real_path)) {
    MS_LOG(DEBUG) << "dir: " << path << "does not exit.";
    return;
  }
  MS_LOG(INFO) << "real path is: " << real_path;
  if (chmod(real_path, S_IRUSR) == -1) {
    MS_LOG(DEBUG) << "modify file: " << real_path << "to read only fail.";
  }
}

void TbeUtils::LoadCache() {
  static bool has_load = false;
  if (!has_load) {
    KernelMeta *bin_map = KernelMeta::GetInstance();
    if (bin_map != nullptr && !bin_map->ReadIndex(kCceKernelMeta)) {
      MS_LOG(INFO) << "Cache initialize failed[" << kCceKernelMeta << "]";
    } else {
      MS_LOG(INFO) << "Cache initialize to " << kCceKernelMeta;
    }
    has_load = true;
  }
}

KernelPackPtr TbeUtils::SearchCache(const std::string &kernel_name, const std::string &processor) {
  // search cache.
  KernelMeta *bin_map = KernelMeta::GetInstance();
  if (bin_map == nullptr) {
    MS_LOG(DEBUG) << "kernel cache is invalid.";
    return nullptr;
  }
  return bin_map->GetKernelPack(kernel_name, processor);
}

KernelPackPtr TbeUtils::InsertCache(const std::string &kernel_name, const std::string &processor) {
  MS_LOG(INFO) << "kernel name:  " << kernel_name << ", processr:" << processor;
  if (processor != kProcessorAiCore) {
    MS_LOG(EXCEPTION) << "process type should be aicore, actually is: " << processor;
  }
  return SearchCache(kernel_name, processor);
}

int KernelManager::BinaryRegister(const mindspore::kernel::FlexArray &kernel_buffer, void **module,
                                  const string &magic) {
  static std::map<string, uint32_t> magic_maps = {{"RT_DEV_BINARY_MAGIC_ELF", RT_DEV_BINARY_MAGIC_ELF},
                                                  {"RT_DEV_BINARY_MAGIC_PLAIN", RT_DEV_BINARY_MAGIC_PLAIN},
                                                  {"RT_DEV_BINARY_MAGIC_PLAIN_AICPU", RT_DEV_BINARY_MAGIC_PLAIN_AICPU},
                                                  {"RT_DEV_BINARY_MAGIC_ELF_AICPU", RT_DEV_BINARY_MAGIC_ELF_AICPU}};
  // object for device register.
  rtDevBinary_t dev_bin;
  dev_bin.data = kernel_buffer.contents;
  auto iter = magic_maps.find(magic);
  if (iter == magic_maps.end()) {
    MS_LOG(DEBUG) << "Invalid magic number: " << magic;
    return -1;
  }
  dev_bin.magic = iter->second;
  dev_bin.length = kernel_buffer.len;
  dev_bin.version = 2;
  if (RT_ERROR_NONE != rtDevBinaryRegister(&dev_bin, module)) {
    MS_LOG(DEBUG) << "Call runtime rtDevBinaryRegister error.";
    return -1;
  }
  return 0;
}

uintptr_t KernelManager::GenFuncStub(const mindspore::kernel::KernelPack &kernel_pack, bool force_reload,
                                     uint32_t *block_dim) {
  auto kernel = kernel_pack.GetKernel();
  if (kernel == nullptr) {
    MS_LOG(EXCEPTION) << "Invalid kernel pack, json or kernel is nullptr.";
  }
  auto kernel_contents = kernel->contents;
  if (kernel_contents == nullptr) {
    MS_LOG(EXCEPTION) << "Invalid kernel context, json or kernel is nullptr.";
  }
  auto kernel_json_info = kernel_pack.kernel_json_info();

  *block_dim = kernel_json_info.block_dim;
  string funcname = kernel_json_info.kernel_name;
  string magic = kernel_json_info.magic;

  if (!force_reload) {
    // use the cached object.
    auto iter = info_table_.find(funcname);
    if (iter != info_table_.end()) {
      auto kernelmeta = iter->second;
      *block_dim = kernelmeta->block_dim_;
      return kernelmeta->func_stub_;
    }
  }
  void *module = nullptr;
  if (0 != BinaryRegister((*kernel_pack.GetKernel()), &module, magic)) {
    MS_LOG(DEBUG) << "Call runtime BinaryRegister error.";
    return 0;
  }
  // to diff different funcs.
  uintptr_t funcstub = ++kernel_stub_gen_;
  if (RT_ERROR_NONE !=
      rtFunctionRegister(module, reinterpret_cast<void *>(funcstub), funcname.c_str(), funcname.c_str(), 0)) {
    MS_LOG(DEBUG) << "Call runtime rtFunctionRegister error.";
    return 0;
  }
  // cache the registered kernelmeta.
  info_table_[funcname] = std::make_shared<KernelMetaInfo>(KernelMetaInfo{funcstub, *block_dim});
  return funcstub;
}

std::string KernelManager::GetStubFuncName(const KernelPackPtr &kernel_pack) {
  auto kernel_json_info = kernel_pack->kernel_json_info();
  return kernel_json_info.kernel_name;
}

KernelMeta *KernelMeta::GetInstance() {
  static KernelMeta inst;
  return &inst;
}

bool KernelMeta::ReadIndex(const std::string &bin_dir) {
  DIR *dir = opendir(bin_dir.c_str());
  if (dir == nullptr) {
    auto ret = mkdir(bin_dir.c_str(), S_IRWXG | S_IRWXU);
    if (ret != 0) {
      MS_LOG(INFO) << "kernel dir: " << bin_dir << "not exist";
      return false;
    }
    dir = opendir(bin_dir.c_str());
  }
  struct dirent *entry;
  while ((entry = readdir(dir)) != nullptr) {
    string bin_dir_tmp = bin_dir;
    std::string cce_json = entry->d_name;
    if (cce_json.length() <= 5) {
      continue;
    }
    std::string suffix = cce_json.substr(cce_json.length() - 5);
    if (suffix != kJsonSuffix) {
      continue;
    }
    auto sp = cce_json.rfind('/');
    if (sp != std::string::npos) {
      continue;
    }
    sp = cce_json.rfind('.');
    if (sp == std::string::npos) {
      continue;
    }
    auto kernel_name = cce_json.substr(0, sp);
    (void)bin_dir_tmp.append("/");
    (void)bin_dir_tmp.append(cce_json);
    kernel_index_map_[kernel_name] = bin_dir_tmp;
  }
  (void)closedir(dir);

  MS_LOG(INFO) << "Cache kernel initialized, kernel size: " << kernel_index_map_.size();
  return true;
}

KernelPackPtr KernelMeta::GetKernelPack(const std::string &kernel_name, const std::string &processor) {
  KernelPackPtr ret = nullptr;
  // 1. pack has been created
  auto kernel_pack_iter = kernel_pack_map_.find(kernel_name);
  if (kernel_pack_iter != kernel_pack_map_.end()) {
    MS_LOG(INFO) << "kernel pack [" << kernel_name << "]has been created.";
    ret = kernel_pack_iter->second;
  } else {
    // 2. kernel file has been create, but pack does not been created.
    std::string cce_json = kCceKernelMeta;
    (void)cce_json.append(kernel_name).append(kJsonSuffix);
    ret = std::make_shared<KernelPack>();
    if (!ret->LoadKernelMeta(cce_json, processor)) {
      MS_LOG(DEBUG) << "Read cache json and bin file failed[" << cce_json << "]";
      return nullptr;
    }
    kernel_pack_map_[kernel_name] = ret;
    auto iter = kernel_index_map_.find(kernel_name);
    if (iter == kernel_index_map_.end()) {
      MS_LOG(INFO) << "kernel name [" << kernel_name << "] has been ceated first.";
      kernel_index_map_[kernel_name] = cce_json;
    }
  }
  return ret;
}
}  // namespace tbe
}  // namespace kernel
}  // namespace mindspore
