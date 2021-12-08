/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "src/common_utils.h"
#include <sys/stat.h>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "include/svp_acl_rt.h"
#include "include/svp_acl.h"
#include "include/svp_acl_ext.h"

namespace mindspore {
namespace lite {
namespace {
constexpr int32_t kDeviceId = 0;
constexpr size_t kMaxSize = 1024;
bool kThreadRunning = false;

bool IsValidDoubleNum(const std::string &num_str) {
  if (num_str.empty()) {
    return false;
  }
  std::istringstream iss(num_str);
  double d;
  iss >> std::noskipws >> d;
  return iss.eof() && !iss.fail();
}

void AicpuThread() {
  MS_LOG(INFO) << "create aicpu thread success";
  while (kThreadRunning) {
    svp_acl_error ret = svp_acl_ext_process_aicpu_task(1000);  // 1000 ms
    if (ret != SVP_ACL_SUCCESS && ret != SVP_ACL_ERROR_RT_REPORT_TIMEOUT) {
      MS_LOG(ERROR) << "create aicpu thread failed!";
      break;
    }
  }
  MS_LOG(INFO) << "end to destroy aicpu thread";
  return;
}
}  // namespace

bool InferDone(const std::vector<mindspore::MSTensor> &tensors) {
  for (auto &tensor : tensors) {
    auto shape = tensor.Shape();
    if (std::find(shape.begin(), shape.end(), -1) != shape.end()) {
      return false;
    }
  }
  return true;
}

void ExtractAttrsFromPrimitive(const mindspore::schema::Primitive *primitive,
                               std::map<std::string, std::string> *attrs) {
  if (primitive == nullptr || attrs == nullptr) {
    return;
  }
  auto custom_holder = primitive->value_as_Custom();
  if (custom_holder == nullptr) {
    return;
  }
  auto attrs_holder = custom_holder->attr();
  if (attrs_holder == nullptr) {
    return;
  }

  for (size_t i = 0; i < attrs_holder->size(); i++) {
    if (attrs_holder->Get(i) == nullptr || attrs_holder->Get(i)->name() == nullptr) {
      continue;
    }
    auto attr_name = attrs_holder->Get(i)->name()->str();
    std::string attr;
    auto attr_data = attrs_holder->Get(i)->data();
    if (attr_data != nullptr) {
      if (attr_data->size() >= kMaxSize) {
        MS_LOG(WARNING) << "attr size too big, which is out of 1024 character. Obtain " << attr_name.c_str()
                        << " failed.";
      } else {
        for (size_t j = 0; j < attr_data->size(); j++) {
          attr.push_back(static_cast<char>(attr_data->Get(j)));
        }
      }
    }
    attrs->emplace(attr_name, attr);
  }
}

void *ReadBinFile(const std::string &fileName, uint32_t *fileSize) {
  if (fileSize == nullptr) {
    return nullptr;
  }
  struct stat sBuf;
  int fileStatus = stat(fileName.data(), &sBuf);
  if (fileStatus == -1) {
    MS_LOG(ERROR) << "failed to get file " << fileName.c_str();
    return nullptr;
  }
  if (S_ISREG(sBuf.st_mode) == 0) {
    MS_LOG(ERROR) << fileName.c_str() << " is not a file, please enter a file";
    return nullptr;
  }
  std::ifstream binFile(fileName, std::ifstream::binary);
  if (!binFile.is_open()) {
    MS_LOG(ERROR) << "open file " << fileName.c_str() << " failed";
    return nullptr;
  }
  binFile.seekg(0, binFile.end);
  uint32_t binFileBufferLen = binFile.tellg();
  if (binFileBufferLen == 0) {
    MS_LOG(ERROR) << "binfile is empty, filename is " << fileName.c_str();
    binFile.close();
    return nullptr;
  }
  binFile.seekg(0, binFile.beg);
  void *binFileBufferData = nullptr;
  svp_acl_error ret = SVP_ACL_SUCCESS;
  ret = svp_acl_rt_malloc(&binFileBufferData, binFileBufferLen, SVP_ACL_MEM_MALLOC_NORMAL_ONLY);
  if (ret != SVP_ACL_SUCCESS) {
    MS_LOG(ERROR) << "malloc device buffer failed. size is " << binFileBufferLen;
    binFile.close();
    return nullptr;
  }
  binFile.read(static_cast<char *>(binFileBufferData), binFileBufferLen);
  binFile.close();
  *fileSize = binFileBufferLen;
  return binFileBufferData;
}

Result JudgeOmNetType(const schema::Primitive &primitive, OmNetType *net_type) {
  auto op = primitive.value_as_Custom();
  if (op == nullptr) {
    return FAILED;
  }
  if (op->attr() == nullptr) {
    MS_LOG(ERROR) << "op attr is nullptr.";
    return FAILED;
  }
  if (op->attr()->size() < 1) {
    MS_LOG(ERROR) << "There are at least 1 attribute of Custom";
    return FAILED;
  }
  std::string net_type_str = "";
  for (size_t i = 0; i < op->attr()->size(); i++) {
    if (op->attr()->Get(i) == nullptr || op->attr()->Get(i)->name() == nullptr) {
      return FAILED;
    }
    if (op->attr()->Get(i)->name()->str() == kNetType) {
      auto output_info = op->attr()->Get(i)->data();
      if (output_info == nullptr) {
        return FAILED;
      }
      int attr_size = static_cast<int>(output_info->size());
      for (int j = 0; j < attr_size; j++) {
        net_type_str.push_back(static_cast<char>(output_info->Get(j)));
      }
      break;
    }
  }
  if (net_type_str.empty()) {
    *net_type = OmNetType_CNN;
    return SUCCESS;
  }
  if (!IsValidUnsignedNum(net_type_str)) {
    MS_LOG(ERROR) << "net_type attr data is invalid.";
    return FAILED;
  }
  int net_type_val = stoi(net_type_str);
  if (net_type_val == OmNetType_ROI) {
    *net_type = OmNetType_ROI;
  } else if (net_type_val == OmNetType_RECURRENT) {
    *net_type = OmNetType_RECURRENT;
  }
  return SUCCESS;
}

void DpicoConfigParamExtractor::InitDpicoConfigParam(const kernel::Kernel &kernel) {
  if (has_init_) {
    return;
  }
  has_init_ = true;
  UpdateDpicoConfigParam(kernel);
}

void DpicoConfigParamExtractor::UpdateDpicoConfigParam(const kernel::Kernel &kernel) {
  auto dpico_arg = kernel.GetConfig("dpico");
  if (dpico_arg.find("MaxRoiNum") != dpico_arg.end()) {
    if (IsValidUnsignedNum(dpico_arg.at("MaxRoiNum"))) {
      max_roi_num_ = stoi(dpico_arg.at("MaxRoiNum"));
    }
  }

  if (dpico_arg.find("NmsThreshold") != dpico_arg.end()) {
    if (IsValidDoubleNum(dpico_arg.at("NmsThreshold"))) {
      nms_threshold_ = stof(dpico_arg.at("NmsThreshold"));
    }
  }

  if (dpico_arg.find("ScoreThreshold") != dpico_arg.end()) {
    if (IsValidDoubleNum(dpico_arg.at("ScoreThreshold"))) {
      score_threshold_ = stof(dpico_arg.at("ScoreThreshold"));
    }
  }

  if (dpico_arg.find("MinHeight") != dpico_arg.end()) {
    if (IsValidDoubleNum(dpico_arg.at("MinHeight"))) {
      min_height_ = stof(dpico_arg.at("MinHeight"));
    }
  }

  if (dpico_arg.find("MinWidth") != dpico_arg.end()) {
    if (IsValidDoubleNum(dpico_arg.at("MinWidth"))) {
      min_width_ = stof(dpico_arg.at("MinWidth"));
    }
  }

  if (dpico_arg.find("GTotalT") != dpico_arg.end()) {
    if (IsValidUnsignedNum(dpico_arg.at("GTotalT"))) {
      g_total_t_ = stoi(dpico_arg.at("GTotalT"));
    }
  }

  if (dpico_arg.find("DetectionPostProcess") != dpico_arg.end()) {
    if (dpico_arg.at("DetectionPostProcess") == "on") {
      dpico_detection_post_process_ = 1;
    }
  }
  if (dpico_arg.find("ConfigPath") != dpico_arg.end()) {
    dpico_dump_config_file_ = dpico_arg.at("ConfigPath");
  }
}

Result DpicoContextManager::InitContext(std::string dpico_dump_config_file) {
  if (svp_context_ != nullptr) {
    return SUCCESS;
  }
  int ret = SUCCESS;
  if (dpico_dump_config_file == "") {
    ret = svp_acl_init(NULL);
  } else {
    MS_LOG(INFO)
      << "dump according to dump config file " << dpico_dump_config_file.c_str()
      << ", if not dump data, please check weather the path exists, or whether add [online_model_type] 4, or model "
         "name is not the same with instruction_name in converter cfg, default inst, muti seg custom_i";
    ret = svp_acl_init(dpico_dump_config_file.c_str());
  }
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "acl init failed";
    return FAILED;
  }
  MS_LOG(INFO) << "acl init success";
  // open device
  ret = svp_acl_rt_set_device(kDeviceId);
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "acl open device " << kDeviceId << " failed";
    return FAILED;
  }
  MS_LOG(INFO) << "open device " << kDeviceId << " success";

  // create context (set current)
  ret = svp_acl_rt_create_context(&svp_context_, kDeviceId);
  if (ret != SUCCESS || svp_context_ == nullptr) {
    MS_LOG(ERROR) << "acl create context failed";
    return FAILED;
  }
  MS_LOG(INFO) << "create context success";
  return SUCCESS;
}

void DpicoContextManager::DestroyContext() {
  if (svp_context_ != nullptr) {
    auto ret = svp_acl_rt_destroy_context(svp_context_);
    if (ret != SVP_ACL_SUCCESS) {
      MS_LOG(ERROR) << "destroy context failed";
    }
    svp_context_ = nullptr;
  }

  MS_LOG(INFO) << "end to destroy context";
  auto ret = svp_acl_rt_reset_device(kDeviceId);
  if (ret != SVP_ACL_SUCCESS) {
    MS_LOG(ERROR) << "reset device failed";
  }
  MS_LOG(INFO) << "end to reset device is " << kDeviceId;

  ret = svp_acl_finalize();
  if (ret != SVP_ACL_SUCCESS) {
    MS_LOG(ERROR) << "finalize acl failed";
  }
  MS_LOG(INFO) << "end to finalize acl";
}

void DpicoAicpuThreadManager::CreateAicpuThread(uint32_t model_id) {
  uint32_t aicpu_task_num = 0;
  svp_acl_ext_get_mdl_aicpu_task_num(model_id, &aicpu_task_num);
  all_aicpu_task_num_ += aicpu_task_num;
  if (all_aicpu_task_num_ > 0 && !is_aicpu_thread_activity_) {
    kThreadRunning = true;
    aicpu_thread_ = std::thread(AicpuThread);
    is_aicpu_thread_activity_ = true;
  }
}

void DpicoAicpuThreadManager::DestroyAicpuThread() {
  if (all_aicpu_task_num_ > 0 && is_aicpu_thread_activity_) {
    kThreadRunning = false;
    aicpu_thread_.join();
    all_aicpu_task_num_ = 0;
    is_aicpu_thread_activity_ = false;
  }
}
}  // namespace lite
}  // namespace mindspore
