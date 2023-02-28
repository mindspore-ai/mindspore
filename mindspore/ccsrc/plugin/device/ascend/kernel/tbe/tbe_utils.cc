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

#include "plugin/device/ascend/kernel/tbe/tbe_utils.h"

#include <dirent.h>
#include <string>
#include <map>
#include <set>
#include <list>
#include <functional>
#include <iostream>
#include <fstream>

#include "runtime/kernel.h"
#include "include/common/utils/utils.h"
#include "utils/ms_utils.h"
#include "utils/ms_context.h"
#include "ir/dtype/type.h"
#include "runtime/dev.h"
#include "plugin/device/ascend/hal/device/lic_manager.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/device/ascend/kernel/tbe/tbe_convert_utils.h"
#include "plugin/device/ascend/kernel/tbe/tbe_version.h"
#include "plugin/device/ascend/kernel/tbe/tbe_json/tbe_json_creator.h"
#include "plugin/device/ascend/kernel/tbe/tbe_json/single_tbe_json_creator.h"
#include "include/common/utils/json_operation_utils.h"
#include "mindspore/ccsrc/include/common/debug/common.h"
#include "kernel/common_utils.h"
#include "mindspore/core/utils/file_utils.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"

namespace mindspore {
namespace kernel {
namespace tbe {
constexpr auto kCceKernelMeta = "kernel_meta/";
constexpr auto kTbePrebuildRes = "kernel_meta/tbe_prebuild_res/";
constexpr auto kJsonSuffix = ".json";
constexpr auto kInfoSuffix = ".info";
constexpr auto kMemCheck = "oom";
constexpr auto kBuildRes = "build_result";
constexpr auto kTUNE_BANK_PATH = "TUNE_BANK_PATH";
constexpr auto kTUNE_DUMP_PATH = "TUNE_DUMP_PATH";
constexpr auto kJRlTuneSwitch = "rl_tune_switch";
constexpr auto kJRlTuneList = "rl_tune_list";
constexpr auto kJOpTuneSwitch = "op_tune_switch";
constexpr auto kJOpTuneList = "op_tune_list";
constexpr auto kJPassList = "pass_list";
constexpr auto kCOMPILER_OP_LEVEL = "MS_COMPILER_OP_LEVEL";
constexpr auto kCOMPILER_OP_DEBUG_CONFIG = "MS_COMPILER_OP_DEBUG_CONFIG";

std::atomic<uintptr_t> KernelManager::kernel_stub_gen_ = 0;
std::unordered_map<string, KernelMetaPtr> KernelManager::info_table_ = {};
std::mutex KernelManager::info_table_mutex_;

void TbeUtils::GenLicInfo(nlohmann::json *lic_info_json) {
  MS_EXCEPTION_IF_NULL(lic_info_json);
  (*lic_info_json)[kJRlTuneSwitch] = LicManager::GetInstance().GetRlTuneSwitch();
  (*lic_info_json)[kJRlTuneList] = LicManager::GetInstance().GetRlTuneList();
  (*lic_info_json)[kJOpTuneSwitch] = LicManager::GetInstance().GetOpTuneSwitch();
  (*lic_info_json)[kJOpTuneList] = LicManager::GetInstance().GetOpTuneList();
  (*lic_info_json)[kJPassList] = LicManager::GetInstance().GetPassSwitch();
}

std::string TbeUtils::GetBankPath() {
  // tune bank path
  auto save_path = common::GetEnv(kTUNE_BANK_PATH);
  char real_path[PATH_MAX] = {0};
  if (!save_path.empty()) {
    if (realpath(save_path.c_str(), real_path)) {
      save_path = real_path;
      return save_path;
    }
    MS_LOG(EXCEPTION) << "Invalid environment variable '" << kTUNE_BANK_PATH << "', the path is " << save_path
                      << ". Please check (1) whether the path exists, (2) whether the path has the access "
                         "permission, (3) whether the path is too long. ";
  }
  return "";
}

std::string TbeUtils::GetTuneDumpPath() {
  // tune dump path
  auto save_path = common::GetEnv(kTUNE_DUMP_PATH);
  char real_path[PATH_MAX] = {0};
  if (!save_path.empty()) {
    if (realpath(save_path.c_str(), real_path)) {
      save_path = real_path;
      return save_path;
    }
    MS_LOG(EXCEPTION) << "Invalid environment variable '" << kTUNE_DUMP_PATH << "', the path is " << save_path
                      << ". Please check (1) whether the path exists, (2) whether the path has the access "
                         "permission, (3) whether the path is too long. ";
  }
  return "";
}

std::string TbeUtils::GetOpDebugPath() {
  static std::string debug_path = "";
  if (debug_path != "") {
    return debug_path;
  }
  debug_path = Common::GetCompilerCachePath();
  return debug_path;
}

std::string TbeUtils::GetKernelMetaTempDir() {
  static std::string debug_path;
  if (!debug_path.empty()) {
    return debug_path;
  }
  debug_path = Common::GetKernelMetaTempDir();
  return debug_path;
}

std::string TbeUtils::GetOpDebugLevel() {
  static const std::set<size_t> value_ranges = {OP_DEBUG_LEVEL_0, OP_DEBUG_LEVEL_1, OP_DEBUG_LEVEL_2, OP_DEBUG_LEVEL_3,
                                                OP_DEBUG_LEVEL_4};
  std::string op_debug_level = std::to_string(OP_DEBUG_LEVEL_3);
  auto env_level = common::GetEnv(kCOMPILER_OP_LEVEL);
  if (!env_level.empty()) {
    if (!TbeUtils::IsOneOf(value_ranges, std::stoul(env_level.c_str()))) {
      MS_LOG(WARNING)
        << "Invalid environment variable '" << kCOMPILER_OP_LEVEL << "': " << env_level
        << ", the value should be in [0, 1, 2, 3, 4], now using the default value 3."
           "Get more detail info at https://www.mindspore.cn/tutorials/experts/zh-CN/master/env/env_var_list.html";
    } else {
      op_debug_level = env_level;
    }
  }
  return op_debug_level;
}

std::vector<std::string> TbeUtils::SplitAndRemoveSpace(const std::string &s, char delim) {
  std::string item;
  std::istringstream is(s);
  std::vector<std::string> ret;
  while (std::getline(is, item, delim)) {
    auto end_pos = std::remove(item.begin(), item.end(), ' ');
    item.erase(end_pos, item.end());
    ret.push_back(item);
  }
  return ret;
}

std::string TbeUtils::GetOpDebugConfig() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  bool is_sink = ms_context->get_param<bool>(MS_CTX_ENABLE_TASK_SINK) &&
                 (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode);
  auto op_debug_config = common::GetEnv(kCOMPILER_OP_DEBUG_CONFIG);
  std::string ret_op_debug_config;
  auto val_vec = TbeUtils::SplitAndRemoveSpace(op_debug_config, ',');
  bool is_first = true;
  for (auto &it : val_vec) {
    if (it == kMemCheck && is_sink) {
      if (is_first) {
        is_first = false;
      } else {
        ret_op_debug_config += ", ";
      }
      ret_op_debug_config += it;
    }
  }
  return ret_op_debug_config;
}

std::string GetTeVersion() {
  static auto result = GetPyTeVersion();
  return result;
}

nlohmann::json TbeUtils::GenSocInfo() {
  static nlohmann::json soc_info_json = nlohmann::json();
  if (!soc_info_json.empty()) {
    return soc_info_json;
  }
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  std::list<int64_t> list;
  soc_info_json["coreNum"] = device::ascend::GetAICoreNumber();
  soc_info_json["coreType"] = "";
  soc_info_json["op_impl_mode"] = "";
  soc_info_json["vector_fp_ceiling"] = "";
  soc_info_json["op_impl_mode_list"] = list;
  soc_info_json["l2Mode"] = "2";
  soc_info_json["l1Fusion"] = "false";
  soc_info_json["l2Fusion"] = "false";
  soc_info_json["op_bank_update"] = false;
  soc_info_json["socVersion"] = device::ascend::GetSocVersion();
  soc_info_json["offlineTune"] = CheckOfflineTune();
  soc_info_json["op_debug_dir"] = GetOpDebugPath();
  soc_info_json["kernel_meta_temp_dir"] = GetKernelMetaTempDir();
  soc_info_json["op_debug_level"] = GetOpDebugLevel();
  soc_info_json["op_debug_config"] = GetOpDebugConfig();
  soc_info_json["autoTilingMode"] = context_ptr->get_param<std::string>(MS_CTX_TUNE_MODE);
  soc_info_json["deviceId"] = std::to_string(context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID));
  std::string config_path;
  if (!Common::CommonFuncForConfigPath("", common::GetEnv("OP_BANK_PATH"), &config_path)) {
    MS_LOG(EXCEPTION) << "Invalid environment variable 'OP_BANK_PATH', the path is " << common::GetEnv("OP_BANK_PATH")
                      << ". Please check (1) whether the path exists, (2) whether the path has the access "
                         "permission, (3) whether the path is too long. ";
  }
  soc_info_json["op_bank_path"] = config_path;
  if (!Common::CommonFuncForConfigPath("", common::GetEnv("MDL_BANK_PATH"), &config_path)) {
    MS_LOG(EXCEPTION) << "Invalid environment variable 'MDL_BANK_PATH', the path is " << common::GetEnv("MDL_BANK_PATH")
                      << ". Please check (1) whether the path exists, (2) whether the path has the access "
                         "permission, (3) whether the path is too long. ";
  }
  soc_info_json["mdl_bank_path"] = config_path;
  soc_info_json["deterministic"] = context_ptr->get_param<std::string>(MS_CTX_DETERMINISTIC) == "ON";
  soc_info_json["te_version"] = GetTeVersion();
  return soc_info_json;
}

void TbeUtils::SaveJsonInfo(const std::string &json_name, const std::string &info, saveType save_type) {
  auto config_path = TbeUtils::GetOpDebugPath();
  std::string path;
  if (save_type == saveType::CCE_KERNEL) {
    path = config_path + kCceKernelMeta + json_name + kInfoSuffix;
  } else if (save_type == saveType::TBE_PREBUILD) {
    path = config_path + kTbePrebuildRes + json_name + kJsonSuffix;
  } else {
    MS_LOG(WARNING) << "Save type not supported.";
    return;
  }
  auto realpath = Common::CreatePrefixPath(path);
  if (!realpath.has_value()) {
    MS_LOG(WARNING) << "Invalid environment variable '" << kCOMPILER_CACHE_PATH
                    << "', the path is: " << realpath.value() << ". Please check (1) whether the path exists, "
                    << "(2) whether the path has the access permission, (3) whether the path is too long. ";
    return;
  }
  ChangeFileMode(realpath.value(), S_IWUSR);
  std::ofstream file_write(realpath.value());
  if (!file_write.is_open()) {
    MS_LOG(WARNING) << "Create json info file failed(" << realpath.value() << ").";
    return;
  }
  file_write << info << std::endl;
  file_write.close();
  file_write.clear();
  ChangeFileMode(realpath.value(), S_IRUSR);
}

void TbeUtils::LoadCache() {
  static bool has_load = false;
  if (!has_load) {
    auto bin_map = KernelMeta::GetInstance();
    auto config_path = TbeUtils::GetOpDebugPath();
    auto path = config_path + kCceKernelMeta;
    if (!bin_map->ReadIndex(path)) {
      MS_LOG(INFO) << "Tbe Cache initialize failed[" << path << "]";
    }
    auto akg_config_path = GetCompilerCachePath();
    auto akg_path = akg_config_path + kAkgKernelMeta;
    if (access(akg_path.c_str(), F_OK) != -1 && !bin_map->ReadIndex(akg_path)) {
      MS_LOG(INFO) << "Akg Cache initialize failed[" << akg_path << "]";
    }
    has_load = true;
  }
}

void TbeUtils::UpdateCache(const std::string &kernel_name) {
  KernelMeta *bin_map = KernelMeta::GetInstance();
  if (bin_map == nullptr) {
    MS_LOG(INFO) << "kernel cache is invalid.";
    return;
  }
  return bin_map->UpdateCache(kernel_name);
}

KernelPackPtr TbeUtils::SearchCache(const std::string &kernel_name, const bool is_akg) {
  // search cache.
  KernelMeta *bin_map = KernelMeta::GetInstance();
  MS_EXCEPTION_IF_NULL(bin_map);
  return bin_map->GetKernelPack(kernel_name, is_akg);
}

KernelPackPtr TbeUtils::InsertCache(const std::string &kernel_name, const std::string &processor, const bool is_akg) {
  MS_LOG(INFO) << "kernel name:  " << kernel_name << ", processr:" << processor;
  if (processor != kProcessorAiCore) {
    MS_LOG(EXCEPTION) << "process type should be aicore, actually is: " << processor;
  }
  return SearchCache(kernel_name, is_akg);
}

int KernelManager::BinaryRegister(const mindspore::kernel::FlexArray &kernel_buffer, void **module, const string &magic,
                                  const std::string &func_name, bool has_kernel_list) {
  static std::map<string, uint32_t> magic_maps = {{"RT_DEV_BINARY_MAGIC_PLAIN", RT_DEV_BINARY_MAGIC_PLAIN},
                                                  {"RT_DEV_BINARY_MAGIC_PLAIN_AICPU", RT_DEV_BINARY_MAGIC_PLAIN_AICPU},
                                                  {"RT_DEV_BINARY_MAGIC_PLAIN_AIVEC", RT_DEV_BINARY_MAGIC_PLAIN_AIVEC},
                                                  {"RT_DEV_BINARY_MAGIC_ELF", RT_DEV_BINARY_MAGIC_ELF},
                                                  {"RT_DEV_BINARY_MAGIC_ELF_AICPU", RT_DEV_BINARY_MAGIC_ELF_AICPU},
                                                  {"RT_DEV_BINARY_MAGIC_ELF_AIVEC", RT_DEV_BINARY_MAGIC_ELF_AIVEC},
                                                  {"RT_DEV_BINARY_MAGIC_ELF_AICUBE", RT_DEV_BINARY_MAGIC_ELF_AICUBE}};
  // object for device register.
  rtDevBinary_t dev_bin;
  dev_bin.data = kernel_buffer.contents;
  auto iter = magic_maps.find(magic);
  if (iter == magic_maps.end()) {
    MS_LOG(INFO) << "Invalid magic number: " << magic << ", kernel: " << func_name;
    return -1;
  }
  dev_bin.magic = iter->second;
  dev_bin.length = kernel_buffer.len;
  dev_bin.version = 0;
  auto ret = has_kernel_list ? rtRegisterAllKernel(&dev_bin, module) : rtDevBinaryRegister(&dev_bin, module);
  if (RT_ERROR_NONE != ret) {
    MS_LOG(INFO) << "Call runtime rtDevBinaryRegister error, ret: [" << ret
                 << "], error message: " << device::ascend::ErrorManagerAdapter::GetErrorMessage(true)
                 << ". Try to delete kernel compile cache files, and restart you project again.(These cache files in "
                    "the custom directory if you used the environment variable 'MS_COMPILER_CACHE_PATH', otherwise in "
                    "the current directory). Kernel: "
                 << func_name;
    return -1;
  }
  return 0;
}

uintptr_t KernelManager::GenFuncStub(const mindspore::kernel::KernelPack &kernel_pack, bool force_reload,
                                     uint32_t *block_dim, void **handle) {
  MS_EXCEPTION_IF_NULL(block_dim);
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
    std::lock_guard<std::mutex> lock(info_table_mutex_);
    auto iter = info_table_.find(func_name);
    if (iter != info_table_.end()) {
      auto kernelmeta = iter->second;
      *block_dim = kernelmeta->block_dim_;
      if (handle != nullptr) {
        *handle = kernelmeta->handle_;
      }
      return kernelmeta->result_;
    }
  }
  void *module = nullptr;
  if (BinaryRegister((*kernel_pack.GetKernel()), &module, magic, func_name, kernel_json_info.has_kernel_list) != 0) {
    MS_LOG(INFO) << "Call runtime BinaryRegister error. Register for : " << func_name;
    if (module != nullptr) {
      (void)rtDevBinaryUnRegister(module);
    }
    return 0;
  }
  if (kernel_json_info.has_kernel_list) {
    MS_EXCEPTION_IF_NULL(handle);
    *handle = module;
    info_table_[func_name] = std::make_shared<KernelMetaInfo>(KernelMetaInfo{1, *block_dim, module});
    return 1;
  }
  // to diff different funcs.
  uintptr_t func_stub = ++kernel_stub_gen_;
  if (RT_ERROR_NONE !=
      rtFunctionRegister(module, reinterpret_cast<void *>(func_stub), func_name.c_str(), func_name.c_str(), 0)) {
    MS_LOG(INFO) << "Call runtime rtFunctionRegister error, message:"
                 << device::ascend::ErrorManagerAdapter::GetErrorMessage(true)
                 << ". Try to delete kernel compile cache files, and restart you project again.(These cache files in "
                    "the custom directory if you used the environment variable 'MS_COMPILER_CACHE_PATH', otherwise in "
                    "the current directory). "
                 << func_name;
    return 0;
  }
  // cache the registered kernelmeta.
  std::lock_guard<std::mutex> lock(info_table_mutex_);
  info_table_[func_name] = std::make_shared<KernelMetaInfo>(KernelMetaInfo{func_stub, *block_dim, module});
  return func_stub;
}

std::string KernelManager::GetStubFuncName(const KernelPackPtr &kernel_pack) {
  MS_EXCEPTION_IF_NULL(kernel_pack);
  auto kernel_json_info = kernel_pack->kernel_json_info();
  return kernel_json_info.kernel_name;
}

KernelMeta *KernelMeta::GetInstance() {
  static KernelMeta inst{};
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
    if (dir == nullptr) {
      MS_LOG(INFO) << "Open dir failed. Dir:" << bin_dir;
      return false;
    }
  }
  struct dirent *entry;
  constexpr size_t SUFFIX_LENS = 5;
  while ((entry = readdir(dir)) != nullptr) {
    string bin_dir_tmp = bin_dir;
    std::string cce_json = entry->d_name;
    if (cce_json.length() <= SUFFIX_LENS) {
      continue;
    }
    std::string suffix = cce_json.substr(cce_json.length() - SUFFIX_LENS);
    if (suffix != kJsonSuffix) {
      continue;
    }
    if (cce_json.rfind("_loc") != std::string::npos || cce_json.rfind("_compute") != std::string::npos) {
      // op debug file no need load into cache
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
    UpdateCache(kernel_name);
  }
  (void)closedir(dir);

  return true;
}

void TbeUtils::GetCompileInfo(const AnfNodePtr &node, std::string *compile_info, bool *get_flag) {
  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(DEBUG) << "Get compile info from json file start. [" << node->fullname_with_scope() << "]";
  std::string json_name;
  if (common::AnfAlgo::HasNodeAttr(kAttrJsonFileName, node->cast<CNodePtr>())) {
    json_name = common::AnfAlgo::GetNodeAttr<std::string>(node, kAttrJsonFileName);
  } else {
    auto json_creator = std::make_shared<kernel::BuildTbeJsonCreator>();
    MS_EXCEPTION_IF_NULL(json_creator);
    nlohmann::json kernel_json;
    if (!json_creator->GenJson(node, &kernel_json)) {
      MS_LOG(WARNING) << "Gen kernel json failed [" << node->fullname_with_scope() << "]";
      *get_flag = false;
      return;
    }
    json_name = json_creator->GetJsonName();
  }
  auto config_path = TbeUtils::GetOpDebugPath();
  std::string path = config_path + kCceKernelMeta + json_name + kJsonSuffix;
  if (path.size() > PATH_MAX) {
    MS_LOG(WARNING) << "File path length should be smaller than " << PATH_MAX << ", but got " << path;
    *get_flag = false;
    return;
  }
  nlohmann::json read_new_json;
  std::ifstream file(path.c_str());
  if (!file.is_open()) {
    MS_LOG(EXCEPTION) << "File is not open. File: " << path;
  }
  std::string ori_file = std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  if (!ParseJson(ori_file, &read_new_json)) {
    MS_LOG(EXCEPTION) << "Parse compile info error :" << ori_file;
  }
  auto build_res_str = GetJsonValue<std::string>(read_new_json, kBuildRes);
  nlohmann::json build_res_json;
  if (!ParseJson(build_res_str, &build_res_json)) {
    MS_LOG(EXCEPTION) << "Parse build result for " << node->fullname_with_scope() << " error :" << build_res_str;
  }
  *compile_info = read_new_json.at("compileInfo").dump();
  file.close();
  file.clear();
  MS_LOG(DEBUG) << "Get compile info from json file success.";
}

void TbeUtils::SaveCompileInfo(const std::string &json_name, const std::string &build_res, bool *save_flag) {
  MS_LOG(DEBUG) << "Save compile info to json file start, op: [" << json_name << "].";
  auto config_path = TbeUtils::GetOpDebugPath();
  std::string path = config_path + kCceKernelMeta + json_name + kJsonSuffix;
  if (path.size() > PATH_MAX) {
    MS_LOG(WARNING) << "File path length should be smaller than " << PATH_MAX << ", but got " << path;
    *save_flag = false;
    return;
  }
  nlohmann::json save_new_json;
  std::ifstream file(path.c_str());
  if (!file.is_open()) {
    MS_LOG(EXCEPTION) << "File is not open. File: " << path;
  }
  std::string ori_file = std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  if (!ParseJson(ori_file, &save_new_json)) {
    MS_LOG(EXCEPTION) << "Parse compile info error.";
  }
  file.close();
  file.clear();
  save_new_json[kBuildRes] = build_res;
  std::ofstream file_write;
  file_write.open(path);
  if (!file_write.is_open()) {
    MS_LOG(WARNING) << "Create info file failed. [" << path << "]";
    *save_flag = false;
    return;
  }
  const int indent = 4;
  auto info = save_new_json.dump(indent);
  file_write << info << std::endl;
  file_write.close();
  file_write.clear();
  MS_LOG(DEBUG) << "Save compile info to json file success.";
}

bool TbeUtils::CheckOfflineTune() {
  bool offline = false;
  std::string offline_tune = common::GetEnv("ENABLE_TUNE_DUMP");
  if (!offline_tune.empty()) {
    for (size_t j = 0; j < offline_tune.length(); j++) {
      offline_tune[j] = tolower(offline_tune[j]);
    }
    if (!(offline_tune == "true" || offline_tune == "false")) {
      MS_LOG(ERROR) << "Invalid environment variable 'ENABLE_TUNE_DUMP', it should be 'true' or 'false', but got "
                    << offline_tune;
    }
    offline = (offline_tune == "true");
  }
  return offline;
}

KernelPackPtr KernelMeta::LoadFromFile(const std::string &kernel_name) const {
  auto config_path = TbeUtils::GetOpDebugPath();
  std::string cce_json = config_path + kCceKernelMeta + kernel_name + kJsonSuffix;
  auto ret = std::make_shared<KernelPack>();
  if (!ret->LoadKernelMeta(cce_json)) {
    MS_LOG(INFO) << "Read cache json and bin file failed[" << cce_json << "]";
    return nullptr;
  }
  return ret;
}

KernelPackPtr KernelMeta::SearchInFile(const std::string &kernel_name) {
  auto ret = LoadFromFile(kernel_name);
  if (ret != nullptr) {
    kernel_pack_map_[kernel_name] = ret;
  }
  return ret;
}

void KernelMeta::UpdateCache(const std::string &kernel_name) {
  auto kernel_pack_iter = kernel_pack_map_.find(kernel_name);
  if (kernel_pack_iter != kernel_pack_map_.end()) {
    // cache exists, skip
    MS_LOG(DEBUG) << "Kernel pack already exist, skip. Kernel name:" << kernel_name;
    return;
  }
  auto ret = LoadFromFile(kernel_name);
  if (ret != nullptr) {
    kernel_pack_map_[kernel_name] = ret;
  }
}

KernelPackPtr KernelMeta::GetKernelPack(const std::string &kernel_name, const bool is_akg) {
  KernelPackPtr ret = nullptr;
  // 1. pack has been created
  auto kernel_pack_iter = kernel_pack_map_.find(kernel_name);
  if (kernel_pack_iter != kernel_pack_map_.end()) {
    ret = kernel_pack_iter->second;
  }
  if (is_akg) {
    // 2. kernel file has been create, but pack does not been created.
    std::string cce_json = GetCompilerCachePath() + kAkgKernelMeta + kernel_name + kJsonSuffix;
    ret = std::make_shared<KernelPack>();
    if (!ret->LoadKernelMeta(cce_json)) {
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
