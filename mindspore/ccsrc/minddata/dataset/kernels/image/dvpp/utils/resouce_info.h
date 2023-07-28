/**
* Copyright 2022-2023 Huawei Technologies Co., Ltd
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at

* http://www.apache.org/licenses/LICENSE-2.0

* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_UTILS_RESOURCE_INFO_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_UTILS_RESOURCE_INFO_H_

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

// Description of data in device
struct RawData {
  size_t lenOfByte;  // Size of memory, bytes
  void *data;        // Pointer of data
};

enum ModelLoadMethod {
  LOAD_FROM_FILE = 0,       // Loading from file, memory of model and weights are managed by ACL
  LOAD_FROM_MEM,            // Loading from memory, memory of model and weights are managed by ACL
  LOAD_FROM_FILE_WITH_MEM,  // Loading from file, memory of model and weight are managed by user
  LOAD_FROM_MEM_WITH_MEM    // Loading from memory, memory of model and weight are managed by user
};

struct ModelInfo {
  std::string modelName;
  std::string modelPath;               // Path of om model file
  size_t modelFileSize;                // Size of om model file
  std::shared_ptr<void> modelFilePtr;  // Smart pointer of model file data
  uint32_t modelWidth;                 // Input width of model
  uint32_t modelHeight;                // Input height of model
  ModelLoadMethod method;              // Loading method of model
};

// Device resource info, such as model infos, etc
struct DeviceResInfo {
  std::vector<ModelInfo> modelInfos;
};

struct ResourceInfo {
  std::set<int> deviceIds;
  std::string singleOpFolderPath;
  std::unordered_map<int, DeviceResInfo> deviceResInfos;  // map <deviceId, deviceResourceInfo>
};
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_UTILS_RESOURCE_INFO_H_
