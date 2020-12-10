/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "src/runtime/agent/npu/npu_manager.h"
#include <sys/fcntl.h>
#include <unistd.h>
#include "include/hiai_ir_build.h"
#include "include/HiAiModelManagerService.h"
#include "include/errorcode.h"
#include "include/graph/op/all_ops.h"
#include "src/common/file_utils.h"

namespace mindspore::lite {

bool NPUManager::IsSupportNPU() {
  if (!is_npu_check_executor) {
    CheckSupportNPU();
  }
  if (is_support_npu) {
    MS_LOG(INFO) << "The current device support NPU.";
    return true;
  } else {
    MS_LOG(INFO) << "The current device NOT SUPPORT NPU.";
    return false;
  }
}

std::string NPUManager::GetExecutorPath() {
  std::string executor_path;
  char cmdline[1024] = {0};
  int fd = open("/proc/self/cmdline", O_RDONLY);
  if (fd >= 0) {
    char ch;
    int i = 0;
    while (read(fd, &ch, sizeof(ch)) > 0 && !isspace(ch)) {
      if (':' == ch) {
        break;
      }
      cmdline[i] = ch;
      i++;
    }
    close(fd);
  }
  executor_path = std::string(cmdline);
  if (executor_path.empty()) {
    executor_path = "./";
  }
  // android
  if (executor_path.substr(0, 11) == "/data/data/") {
    executor_path = executor_path + '/';
  } else {
    // Linux
    executor_path = executor_path.substr(0, executor_path.rfind('/')) + "/";
  }
  return executor_path;
}

bool NPUManager::IsKirinChip() {
  std::ifstream cpu_info("/proc/cpuinfo");
  if (!(cpu_info.good() && cpu_info.is_open())) {
    return false;
  }
  std::string line;
  while (!cpu_info.eof()) {
    getline(cpu_info, line);
    if (line.find("Hardware") == string::npos) {
      continue;
    }
    auto index = line.find("Kirin");
    if (index == string::npos) {
      continue;
    }
    auto kirin_number_str = line.substr(index + 5);
    auto kirin_number = atoi(kirin_number_str.c_str());
    if (kirin_number >= 985 || kirin_number == 810 || kirin_number == 820) {
      cpu_info.close();
      return true;
    } else {
      cpu_info.close();
      return false;
    }
  }
  return false;
}

bool WriteToOMFile(domi::ModelBufferData om_model_buff, const std::string &om_file_path) {
  FILE *fp;
  fp = fopen(om_file_path.c_str(), "wb");
  if (fp == nullptr) {
    MS_LOG(ERROR) << om_file_path.c_str() << " open failed.";
    return false;
  }

  auto write_size = (uint32_t)fwrite(om_model_buff.data, 1, om_model_buff.length, fp);
  if (write_size != om_model_buff.length) {
    fclose(fp);
    MS_LOG(ERROR) << "Write om file failed.";
    return false;
  }
  fclose(fp);
  return true;
}

bool NPUManager::CheckOmBuildIr(const std::string &path) {
  // build test om model
  std::shared_ptr<hiai::op::Add> add_op(new (std::nothrow) hiai::op::Add("add"));
  if (add_op == nullptr) {
    MS_LOG(ERROR) << "new add_op failed.";
    return false;
  }
  ge::TensorDesc desc(ge::Shape({1}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  std::shared_ptr<hiai::op::Data> data = std::make_shared<hiai::op::Data>("data");
  data->update_input_desc_x(desc);
  add_op->set_input_x1(*data);
  add_op->set_input_x2(*data);
  domi::HiaiIrBuild ir_build;
  ge::Graph ir_graph("graph");
  std::vector<ge::Operator> inputs{*data, *data};
  std::vector<ge::Operator> outputs{*add_op};
  ir_graph.SetInputs(inputs).SetOutputs(outputs);
  ge::Model om_model("test_model", "test_version");
  om_model.SetGraph(ir_graph);

  domi::ModelBufferData om_model_buff;
  if (!ir_build.CreateModelBuff(om_model, om_model_buff)) {
    MS_LOG(ERROR) << "Create model buffer failed.";
    return false;
  }
  if (!ir_build.BuildIRModel(om_model, om_model_buff)) {
    MS_LOG(ERROR) << "Build IR model failed.";
    return false;
  }

  // save test om model
  remove(path.c_str());
  bool ret = WriteToOMFile(om_model_buff, path);
  ir_build.ReleaseModelBuff(om_model_buff);
  return ret;
}

void NPUManager::CheckSupportNPU() {
  is_npu_check_executor = true;
  std::string path_string = GetExecutorPath();

  std::string test_model_path = path_string + "/mindspore_lite_test_npu.om";
  std::ifstream ifs(test_model_path);
  if (ifs.good() && ifs.is_open()) {
    ifs.close();
    is_support_npu = true;
    return;
  }
  if (!IsKirinChip()) {
    MS_LOG(ERROR) << "The current device chip NOT SUPPORT NPU";
    is_support_npu = false;
    return;
  }

  if (!CheckOmBuildIr(test_model_path)) {
    MS_LOG(ERROR) << "Build OM IR error.";
    is_support_npu = false;
    return;
  }
  is_support_npu = true;
}

int NPUManager::AddModel(void *model_buf, uint32_t size, const std::string &model_name, int frequency) {
  hiai::MemBuffer *buffer = mc_builder_->InputMemBufferCreate(model_buf, size);
  if (buffer == nullptr) {
    MS_LOG(ERROR) << "MemBuffer is null.";
    return RET_ERROR;
  }

  auto desc = std::make_shared<hiai::AiModelDescription>(model_name, frequency, 0, 0, 0);
  desc->SetModelBuffer(buffer->GetMemBufferData(), buffer->GetMemBufferSize());
  model_desc_.push_back(desc);
  mc_builder_->MemBufferDestroy(buffer);

  index_++;
  return RET_OK;
}

int NPUManager::InitClient() {
  this->client_ = std::make_shared<hiai::AiModelMngerClient>();
  if (this->client_ == nullptr) {
    return RET_ERROR;
  }
  int ret = this->client_->Init(nullptr);
  if (ret != hiai::AI_SUCCESS) {
    return RET_ERROR;
  }
  mc_builder_ = std::make_shared<hiai::AiModelBuilder>(this->client_);
  return RET_OK;
}

int NPUManager::LoadOMModel() {
  int ret = this->client_->Load(model_desc_);
  if (ret != hiai::AI_SUCCESS) {
    MS_LOG(ERROR) << "Client load model failed." << ret;
    return RET_ERROR;
  }
  return RET_OK;
}

std::shared_ptr<hiai::AiModelMngerClient> NPUManager::GetClient() { return client_; }

int NPUManager::index() { return index_; }
}  // namespace mindspore::lite
