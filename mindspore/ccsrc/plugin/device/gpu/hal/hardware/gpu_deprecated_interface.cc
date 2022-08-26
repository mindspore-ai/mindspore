/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/hal/hardware/gpu_deprecated_interface.h"
#include <cuda.h>
#include <vector>
#include <string>
#include "kernel/akg/akg_kernel_json_generator.h"

namespace mindspore {
namespace device {
namespace gpu {
constexpr int MINIMUM_MAJOR_VERSION = 7;
using mindspore::graphkernel::kJsonKeyComputeCapability;
using mindspore::graphkernel::kJsonKeySmCount;

void GPUDeprecatedInterface::FilterExcludedOps(const std::vector<PrimitivePtr> &src_ops,
                                               std::vector<PrimitivePtr> *dst_ops) {
  MS_EXCEPTION_IF_NULL(dst_ops);
  // Check device computing capacity.
  int major_version = 0;
  auto ret = cuDeviceGetAttribute(&major_version, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0);
  if (ret != CUDA_SUCCESS) {
    const char *msg = nullptr;
    cuGetErrorName(ret, &msg);
    MS_LOG(ERROR) << "Get CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR fail, error message: " << msg;
  }
  if (major_version >= MINIMUM_MAJOR_VERSION) {
    *dst_ops = src_ops;
    return;
  }
  // Filter excluded src_ops.
  std::vector<std::string> excluded_ops{prim::kPrimConv2D->name(), prim::kPrimMatMul->name(),
                                        prim::kPrimBatchMatMul->name()};
  std::vector<PrimitivePtr> res;
  (void)std::copy_if(src_ops.begin(), src_ops.end(), std::back_inserter(res), [&excluded_ops](const PrimitivePtr &p) {
    return std::find(excluded_ops.begin(), excluded_ops.end(), p->name()) == excluded_ops.end();
  });

  // Give hint for excluded src_ops.
  static bool give_hint = false;
  if (!give_hint && res.size() != src_ops.size()) {
    give_hint = true;
    std::stringstream ss;
    ss << "(";
    for (size_t i = 0; i < excluded_ops.size(); ++i) {
      if (i > 0) {
        ss << ", ";
      }
      ss << excluded_ops[i];
    }
    ss << ")";
    MS_LOG(WARNING) << "Some operators" << ss.str()
                    << " can not be enabled in GraphKernel because the current device's computing capacity is "
                    << major_version << ", which is < " << MINIMUM_MAJOR_VERSION
                    << ". For better performance, it is recommended to use devices with a computing capacity >= "
                    << MINIMUM_MAJOR_VERSION;
  }
  *dst_ops = res;
}

bool GPUDeprecatedInterface::GetGPUInfo(nlohmann::json *target_info) {
  MS_EXCEPTION_IF_NULL(target_info);
  int a, b, sm_count;
  auto ret = cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0);
  if (ret != CUDA_SUCCESS) {
    const char *msg = nullptr;
    cuGetErrorName(ret, &msg);
    MS_LOG(WARNING) << "Get CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR fail, cuda message: " << msg;
    return false;
  }
  ret = cuDeviceGetAttribute(&b, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 0);
  if (ret != CUDA_SUCCESS) {
    const char *msg = nullptr;
    cuGetErrorName(ret, &msg);
    MS_LOG(WARNING) << "Get CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR fail, cuda message: " << msg;
    return false;
  }
  (*target_info)[kJsonKeyComputeCapability] = std::to_string(a) + "." + std::to_string(b);

  ret = cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0);
  if (ret != CUDA_SUCCESS) {
    const char *msg = nullptr;
    cuGetErrorName(ret, &msg);
    MS_LOG(WARNING) << "Get CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT fail, cuda message: " << msg;
    return false;
  }
  (*target_info)[kJsonKeySmCount] = sm_count;
  return true;
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
