/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License"){}
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

#include <cuda.h>
#include <cuda_runtime.h>
#include "include/utils/callback.h"

namespace mindspore::device::gpu {
// Register callback functions for GPU backend.
int GetGPUCapabilityMajor() {
  // Check device computing capacity major.
  int major_version = -1;
  auto ret = cuDeviceGetAttribute(&major_version, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0);
  if (ret != CUDA_SUCCESS) {
    const char *msg = nullptr;
    cuGetErrorName(ret, &msg);
    MS_LOG(ERROR) << "Get CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR fail, error message: " << msg;
    return -1;
  }
  return major_version;
}
REGISTER_COMMON_CALLBACK(GetGPUCapabilityMajor);

int GetGPUCapabilityMinor() {
  // Check device computing capacity minor.
  int minor_version = -1;
  auto ret = cuDeviceGetAttribute(&minor_version, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 0);
  if (ret != CUDA_SUCCESS) {
    const char *msg = nullptr;
    cuGetErrorName(ret, &msg);
    MS_LOG(ERROR) << "Get CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR fail, error message: " << msg;
    return -1;
  }
  return minor_version;
}
REGISTER_COMMON_CALLBACK(GetGPUCapabilityMinor);

int GetGPUMultiProcessorCount() {
  // Check device sm_count.
  int sm_count = -1;
  auto ret = cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0);
  if (ret != CUDA_SUCCESS) {
    const char *msg = nullptr;
    cuGetErrorName(ret, &msg);
    MS_LOG(ERROR) << "Get CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT fail, error message: " << msg;
    return -1;
  }
  return sm_count;
}
REGISTER_COMMON_CALLBACK(GetGPUMultiProcessorCount);
}  // namespace mindspore::device::gpu
