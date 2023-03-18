/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include <fstream>
#include <thread>
#include "nlohmann/json.hpp"
#include "securec/include/securec.h"
#include "utils/log_adapter.h"
#include "include/common/utils/convert_utils.h"
#include "utils/system/sha256.h"
#include "kernel/common_utils.h"
namespace mindspore {
namespace kernel {
constexpr size_t kWorkspaceSize = 32;
constexpr size_t kJsonSuffixLength = 5;
constexpr char kMagic[] = "magic";
constexpr char kBlockDim[] = "blockDim";
constexpr char kKernelName[] = "kernelName";
constexpr char kBinFileName[] = "binFileName";
constexpr char kBinFileSuffix[] = "binFileSuffix";
constexpr char kCoreType[] = "core_type";
constexpr char kTaskRation[] = "taskRation";
constexpr char kWorkspace[] = "workspace";
constexpr char kParameters[] = "parameters";
constexpr char kOpParaSize[] = "opParaSize";
constexpr char kSHA256[] = "sha256";
constexpr char kKBHit[] = "KBHit";
constexpr char kKernelList[] = "kernelList";
constexpr char kModeInArgsFirstField[] = "modeInArgsFirstField";
constexpr char kBatchBindOnly[] = "batchBindOnly";
constexpr char kArgsRemap[] = "args_remap";
constexpr char kSize[] = "size";
constexpr char kGlobalWorkspaceSpecWorkspace[] = "globalworkspace_spec_workspace";
namespace {
bool CheckHash(const std::string &json_file, const std::string &bin_file, const nlohmann::json &js) {
  if (js.find(kSHA256) == js.end()) {
    return false;
  }
  std::string sha256_cal = system::sha256::GetHashFromFile(bin_file);
  std::string sha256_str = js[kSHA256];
  if (sha256_cal.empty() || sha256_cal != sha256_str) {
    MS_LOG(WARNING) << "Check sha256 for [" << bin_file << "] failed, it will try to rebuild the op.";
    return false;
  }
  return true;
}
const int indent = 4;  // for dump json
constexpr uint32_t kInvalidTaskRatio = 0xFFFFFFFFU;
constexpr auto kTbeMixCubePrefix = "_mix_aic";
constexpr auto kTbeMixVectorPrefix = "_mix_aiv";
constexpr auto kCoreTypeMixAIV = "MIX_AIV";
constexpr auto kCoreTypeMixAIC = "MIX_AIC";
const std::vector<std::string> kBinaryMagicTypes = {
  "RT_DEV_BINARY_MAGIC_PLAIN",     "RT_DEV_BINARY_MAGIC_PLAIN_AICPU", "RT_DEV_BINARY_MAGIC_PLAIN_AIVEC",
  "RT_DEV_BINARY_MAGIC_ELF",       "RT_DEV_BINARY_MAGIC_ELF_AICPU",   "RT_DEV_BINARY_MAGIC_ELF_AIVEC",
  "RT_DEV_BINARY_MAGIC_ELF_AICUBE"};

template <typename T>
bool ParseJsonValue(const std::string &key, const nlohmann::json &json, T *res) {
  MS_EXCEPTION_IF_NULL(res);
  auto obj_json = json.find(key);
  if (obj_json != json.end()) {
    try {
      *res = obj_json.value();
      return true;
    } catch (std::exception &e) {
      MS_LOG(DEBUG) << "Parse json value failed, detail: " << e.what();
    }
  } else {
    MS_LOG(DEBUG) << "Can not find key [" << key << "] in json file, json info: " << json.dump(indent);
  }
  return false;
}
}  // namespace

bool KernelPack::ReadFromJsonFileHelper(std::ifstream &kernel_bin) {
  size_t bin_size = LongToSize(kernel_bin.seekg(0, std::ios::end).tellg());
  // free old data
  if (kernel_ != nullptr) {
    delete[] kernel_;
    kernel_ = nullptr;
  }

  void *ptr = static_cast<void *>(new (std::nothrow) uint8_t[sizeof(KernelPack) + bin_size]);
  if (ptr != nullptr) {
    kernel_ = static_cast<FlexArray *>(ptr);
  }
  if (kernel_ == nullptr) {
    MS_LOG(ERROR) << "Memory malloc failed.";
    kernel_bin.close();
    return false;
  }
  if (memset_s(kernel_, sizeof(KernelPack) + bin_size, 0, sizeof(KernelPack) + bin_size) != EOK) {
    MS_LOG(ERROR) << "Memset kernel_ failed.";
    delete[] kernel_;
    kernel_ = nullptr;
    kernel_bin.close();
    return false;
  }
  kernel_->len = bin_size;
  (void)kernel_bin.seekg(0, std::ios::beg);
  (void)kernel_bin.read(kernel_->contents, SizeToLong(kernel_->len));
  return true;
}

bool KernelPack::ReadFromJsonFile(const std::string &json_f, const std::string &processor) {
  if (json_f.length() <= strlen(kJsonSuffix)) {
    MS_LOG(ERROR) << "Please check json path, file name: " << json_f;
    return false;
  }

  std::ifstream kernel_json(json_f);
  if (!kernel_json.is_open()) {
    MS_LOG(DEBUG) << "Read json file(" << json_f << ") error, please check kernel_meta.";
    return false;
  }
  nlohmann::json js;
  kernel_json >> js;

  size_t bin_size = LongToSize(kernel_json.seekg(0, std::ios::end).tellg());
  void *ptr = static_cast<void *>(new (std::nothrow) uint8_t[sizeof(KernelPack) + bin_size]);
  if (ptr != nullptr) {
    json_ = static_cast<FlexArray *>(ptr);
  }
  if (json_ == nullptr) {
    MS_LOG(ERROR) << "memory malloc failed.";
    kernel_json.close();
    return false;
  }
  json_->len = bin_size;
  (void)kernel_json.seekg(0, std::ios::beg);
  (void)kernel_json.read(json_->contents, SizeToLong(json_->len));

  if (processor == kProcessorCpu) {
    std::string bin_f = json_f.substr(0, json_f.length() - kJsonSuffixLength) + ".so";
    if (!CheckHash(json_f, bin_f, js)) {
      return false;
    }
    return true;
  }

  if (processor == kProcessorCuda) {
    std::string bin_f = json_f.substr(0, json_f.length() - kJsonSuffixLength) + ".ptx";
    std::ifstream kernelbin(bin_f);
    if (!kernelbin.is_open()) {
      MS_LOG(ERROR) << "read kernel ptx file error, please check kernelmeta.";
      kernel_json.close();
      return false;
    }

    if (!ReadFromJsonFileHelper(kernelbin)) {
      delete[] json_;
      json_ = nullptr;
      kernel_json.close();
      return false;
    }
    kernel_json.close();
    if (!CheckHash(json_f, bin_f, js)) {
      return false;
    }

    // cuda json file may have workspace information
    if (js.find(kWorkspace) != js.end()) {
      auto workspace = js.at(kWorkspace);
      std::vector<size_t> sizes = workspace.at(kSize);
      for (auto size : sizes) {
        kernel_json_info_.workspaces.push_back(size);
      }
    }

    return true;
  }

  std::string binfile_suffix = js[kBinFileSuffix];
  std::string bin_f = json_f.substr(0, json_f.length() - kJsonSuffixLength) + binfile_suffix;
  if (binfile_suffix == ".so") {
    // change "xx/xx.so" -> "xx/libxx.so"
    auto sp = bin_f.rfind('/');
    if (sp == std::string::npos) {
      MS_LOG(ERROR) << "illegal bin file path " << bin_f;
      kernel_json.close();
      return false;
    }
    bin_f = bin_f.substr(0, sp + 1) + "lib" + bin_f.substr(sp + 1, bin_f.length() - sp - 1);
  }

  std::ifstream kernelbin(bin_f, std::ios::binary);
  if (!kernelbin.is_open()) {
    MS_LOG(ERROR) << "read kernel binary file error, please check kernelmeta.";
    kernel_json.close();
    delete[] json_;
    json_ = nullptr;
    return false;
  }

  MS_LOG(INFO) << "kernelbin_name:" << bin_f;
  if (!ReadFromJsonFileHelper(kernelbin)) {
    delete[] json_;
    json_ = nullptr;
    kernel_json.close();
    return false;
  }
  kernel_json.close();

  if (!CheckHash(json_f, bin_f, js)) {
    return false;
  }

  return true;
}

void KernelPack::ParseKernelName(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info) {
  MS_EXCEPTION_IF_NULL(kernel_json_info);
  std::string name;
  if (!ParseJsonValue(key, js, &name)) {
    MS_LOG(DEBUG) << "Get value failed for key: " << key << ". Src json: " << js.dump(indent);
  }

  kernel_json_info->kernel_name = name;
}

void KernelPack::ParseBinFileName(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info) {
  MS_EXCEPTION_IF_NULL(kernel_json_info);
  std::string name;
  if (!ParseJsonValue(key, js, &name)) {
    MS_LOG(DEBUG) << "Get value failed for key: " << key << ". Src json: " << js.dump(indent);
  }
  kernel_json_info->bin_file_name = name;
}

void KernelPack::ParseBinFileSuffix(const std::string &key, const nlohmann::json &js,
                                    KernelJsonInfo *kernel_json_info) {
  MS_EXCEPTION_IF_NULL(kernel_json_info);
  std::string name;
  if (!ParseJsonValue(key, js, &name)) {
    MS_LOG(DEBUG) << "Get value failed for key: " << key << ". Src json: " << js.dump(indent);
  }
  kernel_json_info->bin_file_suffix = name;
}

void KernelPack::ParseMagic(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info) {
  MS_EXCEPTION_IF_NULL(kernel_json_info);
  std::string magic;
  if (!ParseJsonValue(key, js, &magic)) {
    MS_LOG(DEBUG) << "Get value failed for key: " << key << ". Src json: " << js.dump(indent);
  }
  if (std::count(kBinaryMagicTypes.begin(), kBinaryMagicTypes.end(), magic) == 0) {
    MS_LOG(ERROR) << "The value of magic [" << magic << "] is not one of BinaryMagicTypes:" << kBinaryMagicTypes;
    return;
  }
  kernel_json_info->magic = magic;
}

void KernelPack::ParseBlockDim(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info) {
  MS_EXCEPTION_IF_NULL(kernel_json_info);
  uint32_t block_dim;
  if (!ParseJsonValue(key, js, &block_dim)) {
    MS_LOG(DEBUG) << "Get value failed for key: " << key << ". Src json: " << js.dump(indent);
  }
  kernel_json_info->block_dim = block_dim;
}

void KernelPack::ParseCoreType(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info) {
  MS_EXCEPTION_IF_NULL(kernel_json_info);
  std::string core_type;
  if (!ParseJsonValue(key, js, &core_type)) {
    MS_LOG(DEBUG) << "Get value failed for key: " << key << ". Src json: " << js.dump(indent);
  }
  kernel_json_info->core_type = core_type;
}

void KernelPack::ParseTaskRatio(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info) {
  MS_EXCEPTION_IF_NULL(kernel_json_info);
  uint32_t ratio = kInvalidTaskRatio;
  if (!ParseJsonValue(key, js, &ratio)) {
    MS_LOG(DEBUG) << "Get value failed for key: " << key << ". Src json: " << js.dump(indent);
  }
  if (ratio == kInvalidTaskRatio) {
    MS_LOG(DEBUG) << "Task ratio empty, src json, " << js.dump(indent);
  }
  kernel_json_info->task_ration = ratio;
}

void KernelPack::ParseWorkSpace(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info) {
  MS_EXCEPTION_IF_NULL(kernel_json_info);
  if (js.find(key) == js.end()) {
    return;
  }
  try {
    auto workspace = js.at(key);
    if (workspace.find("num") == workspace.end() || workspace.find(kSize) == workspace.end()) {
      MS_LOG(WARNING) << "'num' and 'size' ars necessary in workspace, but not found. " << js.dump(indent);
      return;
    }
    size_t num = workspace.at("num");
    std::vector<int64_t> sizes = workspace.at(kSize);
    if (num != sizes.size()) {
      MS_LOG(WARNING) << "'num' and length of 'size' must be same. " << js.dump(indent);
      return;
    }
    if (workspace.find(kType) != workspace.end()) {
      std::vector<size_t> type = workspace.at(kType);
      if (num != type.size()) {
        MS_LOG(WARNING) << "'num' and length of 'type' must be same. " << js.dump(indent);
        return;
      }
      for (size_t i = 0; i < type.size(); i++) {
        (void)kernel_json_info->workspaces_type.emplace_back(type[i]);
      }
    }

    for (size_t i = 0; i < sizes.size(); i++) {
      auto t = sizes[i] < 0 ? kWorkspaceSize : LongToSize(sizes[i]);
      (void)kernel_json_info->workspaces.emplace_back(t);
    }
  } catch (std::exception &e) {
    MS_LOG(ERROR) << "Parse json value failed, error info: " << e.what();
  }
}

void KernelPack::ParseParameters(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info) {
  MS_EXCEPTION_IF_NULL(kernel_json_info);
  if (js.find(key) == js.end()) {
    return;
  }
  try {
    std::vector<size_t> parameters = js.at(key);
    for (size_t i = 0; i < parameters.size(); i++) {
      (void)kernel_json_info->parameters.emplace_back(parameters[i]);
    }
  } catch (std::exception &e) {
    MS_LOG(ERROR) << "Parse json value failed, error info: " << e.what();
  }
}

void KernelPack::ParseOpParaSize(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info) {
  MS_EXCEPTION_IF_NULL(kernel_json_info);
  kernel_json_info->op_para_size = (js.find(key) == js.end()) ? 0 : static_cast<uint32_t>(js.at(key));
}

void KernelPack::ParseSHA256(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info) {
  MS_EXCEPTION_IF_NULL(kernel_json_info);
  std::string sha;
  if (!ParseJsonValue(key, js, &sha)) {
    MS_LOG(DEBUG) << "Get value failed for key: " << key << ". Src json: " << js.dump(indent);
  }
  kernel_json_info->sha256 = sha;
}

void KernelPack::ParseKBHit(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info) {
  MS_EXCEPTION_IF_NULL(kernel_json_info);
  int32_t KBHit = 0;
  if (!ParseJsonValue(key, js, &KBHit)) {
    MS_LOG(DEBUG) << "Get value failed for key: " << key << ". Src json: " << js.dump(indent);
  }
  kernel_json_info->KBHit = KBHit;
}

void KernelPack::ParseModeInArgsFirstField(const std::string &key, const nlohmann::json &js,
                                           KernelJsonInfo *kernel_json_info) {
  MS_EXCEPTION_IF_NULL(kernel_json_info);
  kernel_json_info->mode_in_args_first_field = (js.find(key) == js.end()) ? 0 : static_cast<uint32_t>(js.at(key));
}

void KernelPack::ParseBatchBindOnly(const std::string &key, const nlohmann::json &js,
                                    KernelJsonInfo *kernel_json_info) {
  MS_EXCEPTION_IF_NULL(kernel_json_info);
  kernel_json_info->batch_bind_only = (js.find(key) == js.end()) ? 0 : static_cast<uint32_t>(js.at(key));
}

void KernelPack::ParseKernelList(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info) {
  MS_EXCEPTION_IF_NULL(kernel_json_info);
  kernel_json_info->has_kernel_list = (js.find(key) != js.end());
}

void KernelPack::ParseArgsRemap(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info) {
  // Parse json["args_remap"], the value is a list of list, e.g. [[0, 1], [2]]
  MS_EXCEPTION_IF_NULL(kernel_json_info);
  if (js.find(key) != js.end()) {
    try {
      auto args_remap = js.at(key);
      for (const auto &item : args_remap) {
        std::vector<size_t> indices;
        (void)std::copy(item.begin(), item.end(), std::back_inserter(indices));
        kernel_json_info->args_remap.push_back(indices);
      }
    } catch (std::exception &e) {
      MS_LOG(ERROR) << "Parse json['" << key << "'] failed, error info: " << e.what();
    }
  }
}

void KernelPack::ParseGlogbleWorkSpace(const std::string &key, const nlohmann::json &js,
                                       KernelJsonInfo *kernel_json_info) {
  MS_EXCEPTION_IF_NULL(kernel_json_info);
  if (js.find(key) == js.end()) {
    return;
  }
  try {
    auto globalWorkspace = js.at(key);
    if (globalWorkspace.find(kSize) != globalWorkspace.end()) {
      kernel_json_info->global_workspace.size = globalWorkspace.at(kSize);
      kernel_json_info->global_workspace.is_overflow = true;
    }
    if (globalWorkspace.find(kType) != globalWorkspace.end()) {
      kernel_json_info->global_workspace.type = globalWorkspace.at(kType);
      kernel_json_info->global_workspace.is_overflow = true;
    }
  } catch (std::exception &e) {
    MS_LOG(ERROR) << "Parse json value failed, jsong is:" + js.dump() + ", error info: " << e.what();
  }
}

void KernelPack::ParseKernelJson(const nlohmann::json &js) {
  using KernelJsonParser = std::function<void(const std::string &, const nlohmann::json &, KernelJsonInfo *)>;
  const std::map<std::string, KernelJsonParser> kernel_json_map = {
    {kMagic, ParseMagic},
    {kBlockDim, ParseBlockDim},
    {kKernelName, ParseKernelName},
    {kBinFileName, ParseBinFileName},
    {kBinFileSuffix, ParseBinFileSuffix},
    {kCoreType, ParseCoreType},
    {kTaskRation, ParseTaskRatio},
    {kWorkspace, ParseWorkSpace},
    {kParameters, ParseParameters},
    {kOpParaSize, ParseOpParaSize},
    {kSHA256, ParseSHA256},
    {kKBHit, ParseKBHit},
    {kKernelList, ParseKernelList},
    {kModeInArgsFirstField, ParseModeInArgsFirstField},
    {kBatchBindOnly, ParseBatchBindOnly},
    {kArgsRemap, ParseArgsRemap},
    {kGlobalWorkspaceSpecWorkspace, ParseGlogbleWorkSpace}};
  auto iter = kernel_json_map.begin();
  while (iter != kernel_json_map.end()) {
    iter->second(iter->first, js, &kernel_json_info_);
    iter++;
  }
}

bool KernelPack::LoadKernelMeta(const std::string &json_f) {
  if (json_f.length() <= strlen(kJsonSuffix)) {
    MS_LOG(ERROR) << "please check json path.";
    return false;
  }
  std::ifstream kernel_json(json_f);
  if (!kernel_json.is_open()) {
    MS_LOG(INFO) << "Open json file: " << json_f << " error, please check kernel_meta.";
    return false;
  }
  nlohmann::json js;
  try {
    kernel_json >> js;
    kernel_json.close();
  } catch (std::exception &e) {
    MS_LOG(WARNING) << "Parse json file error: " << json_f << ", sleep 500ms and retry again. error ms: " << e.what();
    kernel_json.close();
    std::this_thread::sleep_for(std::chrono::microseconds(500000));
    std::ifstream retry_tmp(json_f);
    if (!retry_tmp.is_open()) {
      MS_LOG(INFO) << "Open json file: " << json_f << " error, please check kernel_meta.";
      return false;
    }
    retry_tmp >> js;
    retry_tmp.close();
  }
  ParseKernelJson(js);

  std::string bin_f = json_f.substr(0, json_f.length() - kJsonSuffixLength) + kernel_json_info_.bin_file_suffix;
  if (kernel_json_info_.bin_file_suffix == ".so") {
    // change "xx/xx.so" -> "xx/libxx.so"
    auto sp = bin_f.rfind('/');
    if (sp == std::string::npos) {
      MS_LOG(ERROR) << "illegal bin file path " << bin_f;
      return false;
    }
    bin_f = bin_f.substr(0, sp + 1) + "lib" + bin_f.substr(sp + 1, bin_f.length() - sp - 1);
  }

  std::ifstream kernelbin(bin_f, std::ios::binary);
  if (!kernelbin.is_open()) {
    MS_LOG(ERROR) << "read kernel binary file error, please check kernelmeta.";
    return false;
  }

  if (!ReadFromJsonFileHelper(kernelbin)) {
    return false;
  }

  return CheckHash(json_f, bin_f, js);
}

KernelJsonInfo KernelPack::kernel_json_info() const { return kernel_json_info_; }
}  // namespace kernel
}  // namespace mindspore
