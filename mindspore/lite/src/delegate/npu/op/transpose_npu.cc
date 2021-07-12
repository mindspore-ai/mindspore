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

#include "src/delegate/npu/op/transpose_npu.h"
namespace mindspore {
int TransposeNPUOp::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                              const std::vector<mindspore::MSTensor> &out_tensors) {
  if (in_tensors.size() < 2) {
    MS_LOG(ERROR) << "Npu transpose must get fixed values of transpose axis.";
    return RET_ERROR;
  }
  auto perm_num = in_tensors.at(1).ElementNum();
  if (in_tensors.at(1).Data() == nullptr) {
    MS_LOG(ERROR) << "Npu transpose must get fixed values of transpose axis.";
    return RET_ERROR;
  }
  auto perm_data = reinterpret_cast<const int *>(in_tensors.at(1).Data().get());
  for (int i = 0; i < perm_num; i++) {
    perm_.push_back(perm_data[i]);
  }
  std::vector<int> nh2nc_perm = {0, 3, 1, 2};
  std::vector<int> nc2nh_perm = {0, 2, 3, 1};
  if (perm_ != nh2nc_perm && perm_ != nc2nh_perm) {
    MS_LOG(WARNING) << "NPU transpose op only supports nhwc->nchw or nchw->nhwc.";
    return RET_NOT_SUPPORT;
  }
  return RET_OK;
}
}  // namespace mindspore
