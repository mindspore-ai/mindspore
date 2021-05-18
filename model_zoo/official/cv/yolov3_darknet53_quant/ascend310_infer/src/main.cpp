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

#include <gflags/gflags.h>
#include <iostream>
#include <fstream>
#include <memory>
#include "../inc/SampleProcess.h"
#include "../inc/utils.h"

bool g_isDevice = false;

DEFINE_string(om_path, "", "om path");
DEFINE_string(dataset_path, "", "dataset path");
DEFINE_string(acljson_path, "", "acl json path");
DEFINE_int32(device_id, 0, "device id");

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::string om_path = FLAGS_om_path;
  std::string dataset_path = FLAGS_dataset_path;
  std::string acljson_path = FLAGS_acljson_path;
  int32_t device_id = FLAGS_device_id;
  std::ifstream fin(om_path);
  if (!fin) {
    std::cout << "Invalid om path." << std::endl;
    return FAILED;
  }
  SampleProcess processSample(device_id);
  // acl.json is deployed for dump data.
  Result ret = processSample.InitResource(acljson_path.c_str());
  if (ret != SUCCESS) {
    ERROR_LOG("sample init resource failed");
    return FAILED;
  }

  ret = processSample.Process(om_path.c_str(), dataset_path.c_str());
  if (ret != SUCCESS) {
    ERROR_LOG("sample process failed");
    return FAILED;
  }

  INFO_LOG("execute sample success");
  return SUCCESS;
}
