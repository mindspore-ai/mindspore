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

#include "backend/kernel_compiler/tbe/tbe_utils.h"

#include <dirent.h>
#include <string>
#include <map>
#include <set>
#include <list>
#include <functional>
#include <iostream>
#include <fstream>

#include "runtime/kernel.h"
#include "utils/utils.h"
#include "utils/ms_utils.h"
#include "utils/ms_context.h"
#include "ir/dtype/type.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/kernel_compiler/tbe/tbe_convert_utils.h"
#include "securec/include/securec.h"

namespace mindspore {
namespace kernel {
namespace tbe {
constexpr auto kCceKernelMeta = "./kernel_meta/";
constexpr auto kJsonSuffix = ".json";
constexpr auto kInfoSuffix = ".info";

uintptr_t KernelManager::kernel_stub_gen_ = 0;
std::unordered_map<string, KernelMetaPtr> KernelManager::info_table_ = {};

void TbeUtils::GenSocInfo(nlohmann::json *soc_info_json) {
  MS_EXCEPTION_IF_NULL(soc_info_json);
  std::list<int64_t> list;
  (*soc_info_json)["coreNum"] = "";
  (*soc_info_json)["coreType"] = "";
  (*soc_info_json)["l1Fusion"] = "false";
  (*soc_info_json)["l2Fusion"] = "false";
  (*soc_info_json)["l2Mode"] = "2";
  (*soc_info_json)["op_debug_level"] = "";
  (*soc_info_json)["op_impl_mode"] = "";
  (*soc_info_json)["op_impl_mode_list"] = list;
}

void TbeUtils::SaveJsonInfo(const std::string &json_name, const std::string &info) {
  char real_path[PATH_MAX] = {0};
  std::string path = kCceKernelMeta + json_name + kInfoSuffix;
  if (path.size() > PATH_MAX) {
    MS_LOG(ERROR) << "File path: " << path << "is too long.";
    return;
  }
  std::ifstream fin(path);
  if (fin) {
    MS_LOG(INFO) << "Json file exist(" << path << "), no need to create.";
    return;
  }
  std::ofstream file_write;
  file_write.open(path);
  if (!file_write.is_open()) {
    MS_LOG(WARNING) << "Create info file failed(" << path << ").";
    return;
  }
  file_write << info << std::endl;
  file_write.close();
  if (realpath(path.c_str(), real_path) == nullptr) {
    MS_LOG(WARNING) << "Get realpath failed(" << path << ").";
    return;
  }
  MS_LOG(INFO) << "real path is: " << real_path;
  if (chmod(real_path, S_IRUSR) == -1) {
    MS_LOG(INFO) << "modify file: " << real_path << "to read only fail.";
  }
}

void TbeUtils::LoadCache() {
  static bool has_load = false;
  if (!has_load) {
    KernelMeta *bin_map = KernelMeta::GetInstance();
    if (bin_map != nullptr && !bin_map->ReadIndex(kCceKernelMeta)) {
      MS_LOG(INFO) << "Cache initialize failed[" << kCceKernelMeta << "]";
    }
    has_load = true;
  }
}

KernelPackPtr TbeUtils::SearchCache(const std::string &kernel_name, const std::string &processor) {
  // search cache.
  KernelMeta *bin_map = KernelMeta::GetInstance();
  if (bin_map == nullptr) {
    MS_LOG(INFO) << "kernel cache is invalid.";
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

int KernelManager::BinaryRegister(const mindspore::kernel::FlexArray &kernel_buffer, void **module, const string &magic,
                                  const bool dynamic_flag) {
  static std::map<string, uint32_t> magic_maps = {{"RT_DEV_BINARY_MAGIC_ELF", RT_DEV_BINARY_MAGIC_ELF},
                                                  {"RT_DEV_BINARY_MAGIC_PLAIN", RT_DEV_BINARY_MAGIC_PLAIN},
                                                  {"RT_DEV_BINARY_MAGIC_PLAIN_AICPU", RT_DEV_BINARY_MAGIC_PLAIN_AICPU},
                                                  {"RT_DEV_BINARY_MAGIC_ELF_AICPU", RT_DEV_BINARY_MAGIC_ELF_AICPU}};
  // object for device register.
  rtDevBinary_t dev_bin;
  dev_bin.data = kernel_buffer.contents;
  auto iter = magic_maps.find(magic);
  if (iter == magic_maps.end()) {
    MS_LOG(INFO) << "Invalid magic number: " << magic;
    return -1;
  }
  dev_bin.magic = iter->second;
  dev_bin.length = kernel_buffer.len;
  dev_bin.version = 0;
  auto ret = dynamic_flag ? rtRegisterAllKernel(&dev_bin, module) : rtDevBinaryRegister(&dev_bin, module);
  if (RT_ERROR_NONE != ret) {
    MS_LOG(INFO) << "Call runtime rtDevBinaryRegister error.";
    return -1;
  }
  return 0;
}

uintptr_t KernelManager::GenFuncStub(const mindspore::kernel::KernelPack &kernel_pack, bool force_reload,
                                     uint32_t *block_dim, const bool dynamic_flag, void **handle,
                                     std::string *origin_key) {
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
  string func_name = kernel_json_info.kernel_name;
  string magic = kernel_json_info.magic;

  if (!force_reload) {
    // use the cached object.
    auto iter = info_table_.find(func_name);
    if (iter != info_table_.end()) {
      auto kernelmeta = iter->second;
      *block_dim = kernelmeta->block_dim_;
      if (!dynamic_flag) {
        return kernelmeta->func_stub_;
      }
    }
  }
  void *module = nullptr;
  if (BinaryRegister((*kernel_pack.GetKernel()), &module, magic, dynamic_flag) != 0) {
    MS_LOG(INFO) << "Call runtime BinaryRegister error.";
    if (module != nullptr) {
      (void)rtDevBinaryUnRegister(module);
    }
    return 0;
  }
  if (dynamic_flag) {
    *handle = module;
    *origin_key = func_name;
    return 1;
  }
  // to diff different funcs.
  uintptr_t func_stub = ++kernel_stub_gen_;
  if (RT_ERROR_NONE !=
      rtFunctionRegister(module, reinterpret_cast<void *>(func_stub), func_name.c_str(), func_name.c_str(), 0)) {
    MS_LOG(INFO) << "Call runtime rtFunctionRegister error.";
    return 0;
  }
  // cache the registered kernelmeta.
  info_table_[func_name] = std::make_shared<KernelMetaInfo>(KernelMetaInfo{func_stub, *block_dim});
  return func_stub;
}

std::string KernelManager::GetStubFuncName(const KernelPackPtr &kernel_pack) {
  MS_EXCEPTION_IF_NULL(kernel_pack);
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

  return true;
}

KernelPackPtr KernelMeta::GetKernelPack(const std::string &kernel_name, const std::string &processor) {
  KernelPackPtr ret = nullptr;
  // 1. pack has been created
  auto kernel_pack_iter = kernel_pack_map_.find(kernel_name);
  if (kernel_pack_iter != kernel_pack_map_.end()) {
    ret = kernel_pack_iter->second;
  } else {
    // 2. kernel file has been create, but pack does not been created.
    std::string cce_json = kCceKernelMeta;
    (void)cce_json.append(kernel_name).append(kJsonSuffix);
    ret = std::make_shared<KernelPack>();
    if (!ret->LoadKernelMeta(cce_json, processor)) {
      MS_LOG(INFO) << "Read cache json and bin file failed[" << cce_json << "]";
      return nullptr;
    }
    kernel_pack_map_[kernel_name] = ret;
    auto iter = kernel_index_map_.find(kernel_name);
    if (iter == kernel_index_map_.end()) {
      MS_LOG(INFO) << "kernel name [" << kernel_name << "] has been created first.";
      kernel_index_map_[kernel_name] = cce_json;
    }
  }
  return ret;
}
}  // namespace tbe
}  // namespace kernel
}  // namespace mindspore
