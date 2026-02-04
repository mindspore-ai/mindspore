/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/simu/kernel_mod_impl/simu_receive.h"
#include <vector>
#include "include/backend/common/ms_device_shape_transfer.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace mindspore {
namespace kernel {
bool SimuReceiveKernel::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                               const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (outputs.empty()) {
    MS_LOG(ERROR) << "Invalid simu receive input size (" << inputs.size() << ").";
    return false;
  }

  auto output_size = outputs[0]->size();
  auto output_type = outputs[0]->dtype_id();
  static const float kInitValue = 0.1f;
  static const size_t kFp32TypeSize = abstract::TypeIdSize(kNumberTypeFloat32);
  init_value_.resize(output_size, kInitValue);
  host_data_.resize(output_size, 0);
  void *host_ptr = init_value_.data();
  if (output_type != kNumberTypeFloat32) {
    auto elem_num = output_size / abstract::TypeIdSize(output_type);
    const trans::TypeIdArgs type_args{init_value_.data(), SizeToLong(elem_num), kNumberTypeFloat32,
                                      outputs[0]->dtype_id(), elem_num * kFp32TypeSize};
    auto sync_ok = trans::TransDataType(type_args, host_data_.data());
    if (!sync_ok) {
      MS_LOG(ERROR) << "simu receive trans data type failed.";
      return false;
    }
    host_ptr = host_data_.data();
  }

  auto cp_ret = CALL_ASCEND_API(aclrtMemcpyAsync, outputs[0]->device_ptr(), output_size, host_ptr, output_size,
                                ACL_MEMCPY_HOST_TO_DEVICE, stream_ptr);
  if (cp_ret != EOK) {
    MS_LOG(ERROR) << "Simu receive memset 0 failed.";
    return false;
  }

  return true;
}
}  // namespace kernel
}  // namespace mindspore
