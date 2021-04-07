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

#include <unistd.h>
#include <fstream>
#include <thread>
#include "nlohmann/json.hpp"
#include "securec/include/securec.h"
#include "utils/log_adapter.h"
#include "utils/convert_utils.h"
#include "utils/system/sha256.h"
#include "backend/kernel_compiler/common_utils.h"
namespace mindspore {
namespace kernel {
namespace {
bool CheckHash(const std::string &json_file, const std::string &bin_file, const nlohmann::json &js) {
  if (js.find("sha256") == js.end()) {
    MS_LOG(ERROR) << "No sha256 found in " << json_file;
    return false;
  }
  std::string sha256_cal = system::sha256::GetHashFromFile(bin_file);
  std::string sha256_str = js["sha256"];
  if (sha256_cal.empty() || sha256_cal != sha256_str) {
    MS_LOG(ERROR) << "Cal sha256 of " << bin_file << " failed.";
    return false;
  }
  return true;
}
}  // namespace

const std::string KernelPack::Serialize() const {
  MS_EXCEPTION_IF_NULL(json_);
  MS_EXCEPTION_IF_NULL(kernel_);
  std::string buffer;
  (void)buffer.append((const char *)json_, json_->len + sizeof(json_->len));
  (void)buffer.append((const char *)kernel_, kernel_->len + sizeof(kernel_->len));
  return buffer;
}

bool KernelPack::ReadFromJsonFileHelper(std::ifstream &kernelbin) {
  size_t binsize = LongToSize(kernelbin.seekg(0, std::ios::end).tellg());
  // free old data
  if (kernel_ != nullptr) {
    delete[] kernel_;
    kernel_ = nullptr;
  }

  void *ptr = static_cast<void *>(new (std::nothrow) uint8_t[sizeof(KernelPack) + binsize]);
  if (ptr != nullptr) {
    kernel_ = static_cast<FlexArray *>(ptr);
  }
  if (kernel_ == nullptr) {
    MS_LOG(ERROR) << "memory malloc failed.";
    kernelbin.close();
    return false;
  }
  if (memset_s(kernel_, sizeof(KernelPack) + binsize, 0, sizeof(KernelPack) + binsize) != EOK) {
    MS_LOG(ERROR) << "memset kernel_ failed.";
    delete[] kernel_;
    kernel_ = nullptr;
    kernelbin.close();
    return false;
  }
  kernel_->len = binsize;
  (void)kernelbin.seekg(0, std::ios::beg);
  (void)kernelbin.read(kernel_->contents, SizeToLong(kernel_->len));
  return true;
}

bool KernelPack::ReadFromJsonFile(const std::string &json_f, const std::string &processor) {
  if (json_f.length() <= strlen(kJsonSuffix)) {
    MS_LOG(ERROR) << "please check json path.";
    return false;
  }

  std::ifstream kerneljson(json_f);
  if (!kerneljson.is_open()) {
    MS_LOG(DEBUG) << "read json file error, please check kernelmeta.";
    return false;
  }
  nlohmann::json js;
  kerneljson >> js;

  size_t binsize = LongToSize(kerneljson.seekg(0, std::ios::end).tellg());
  void *ptr = static_cast<void *>(new (std::nothrow) uint8_t[sizeof(KernelPack) + binsize]);
  if (ptr != nullptr) {
    json_ = static_cast<FlexArray *>(ptr);
  }
  if (json_ == nullptr) {
    MS_LOG(ERROR) << "memory malloc failed.";
    kerneljson.close();
    return false;
  }
  json_->len = binsize;
  (void)kerneljson.seekg(0, std::ios::beg);
  (void)kerneljson.read(json_->contents, SizeToLong(json_->len));

  if (processor == kProcessorCuda) {
    std::string bin_f = json_f.substr(0, json_f.length() - 5) + ".ptx";
    std::ifstream kernelbin(bin_f);
    if (!kernelbin.is_open()) {
      MS_LOG(ERROR) << "read kernel ptx file error, please check kernelmeta.";
      kerneljson.close();
      return false;
    }

    if (ReadFromJsonFileHelper(kernelbin) == false) {
      delete[] json_;
      json_ = nullptr;
      kerneljson.close();
      return false;
    }
    kerneljson.close();
    if (!CheckHash(json_f, bin_f, js)) {
      return false;
    }
    return true;
  }

  std::string binfilesuffix = js["binFileSuffix"];
  std::string bin_f = json_f.substr(0, json_f.length() - 5) + binfilesuffix;
  if (binfilesuffix.compare(".so") == 0) {
    // change "xx/xx.so" -> "xx/libxx.so"
    auto sp = bin_f.rfind('/');
    if (sp == std::string::npos) {
      MS_LOG(ERROR) << "illegal bin file path " << bin_f;
      kerneljson.close();
      return false;
    }
    bin_f = bin_f.substr(0, sp + 1) + "lib" + bin_f.substr(sp + 1, bin_f.length() - sp - 1);
  }

  std::ifstream kernelbin(bin_f, std::ios::binary);
  if (!kernelbin.is_open()) {
    MS_LOG(ERROR) << "read kernel binary file error, please check kernelmeta.";
    kerneljson.close();
    delete[] json_;
    json_ = nullptr;
    return false;
  }

  MS_LOG(INFO) << "kernelbin_name:" << bin_f;
  if (ReadFromJsonFileHelper(kernelbin) == false) {
    delete[] json_;
    json_ = nullptr;
    kerneljson.close();
    return false;
  }
  kerneljson.close();

  if (!CheckHash(json_f, bin_f, js)) {
    return false;
  }

  return true;
}

void KernelPack::ParseKernelJson(const nlohmann::json &js) {
  kernel_json_info_.bin_file_name = js["binFileName"];
  kernel_json_info_.bin_file_suffix = js["binFileSuffix"];
  kernel_json_info_.block_dim = js["blockDim"];
  kernel_json_info_.kernel_name = js["kernelName"];
  kernel_json_info_.magic = js["magic"];
  if (js.contains("opParaSize")) {
    kernel_json_info_.op_para_size = js["opParaSize"];
  }
  if (js.find("parameters") != js.end()) {
    if (!js.at("parameters").is_array()) {
      MS_LOG(DEBUG) << "Format error!,parameters should be array.";
    }
    std::vector<size_t> sizes = js.at("parameters");
    for (auto size : sizes) {
      kernel_json_info_.parameters.push_back(size);
    }
  }
  if (js.find("workspace") != js.end()) {
    auto workspace = js.at("workspace");
    std::vector<size_t> sizes = workspace.at("size");
    for (auto size : sizes) {
      kernel_json_info_.workspaces.push_back(size);
    }
  }
  kernel_json_info_.sha256 = js["sha256"];
}

bool KernelPack::LoadKernelMeta(const std::string &json_f, const std::string &processor) {
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
    MS_LOG(WARNING) << "Parse json file error: " << json_f << ", sleep 500ms and retry again.";
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

  std::string bin_f = json_f.substr(0, json_f.length() - 5) + kernel_json_info_.bin_file_suffix;
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
