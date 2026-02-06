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

#include "include/kernel/ascend/custom/pyboost_impl/atb_common.h"
#include <map>
#include "include/pynative/utils/pyboost/custom/tensor.h"
#include "ir/tensor.h"
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"

namespace ms::pynative {
atb::Tensor AtbTensor(const ms::Tensor &t) {
  static std::map<mindspore::TypeId, aclDataType> dtypeMap = {
    {mindspore::kNumberTypeBool, ACL_BOOL},     {mindspore::kNumberTypeFloat16, ACL_FLOAT16},
    {mindspore::kNumberTypeFloat32, ACL_FLOAT}, {mindspore::kNumberTypeInt32, ACL_INT32},
    {mindspore::kNumberTypeInt64, ACL_INT64},   {mindspore::kNumberTypeBFloat16, ACL_BF16},
  };
  atb::Tensor a;
  a.desc.dtype = dtypeMap[t.data_type()];
  a.desc.format = ACL_FORMAT_ND;
  auto shape = t.shape();
  a.desc.shape.dimNum = shape.size();
  for (size_t i = 0; i < shape.size(); i++) {
    a.desc.shape.dims[i] = shape[i];
  }
  a.dataSize = atb::Utils::GetTensorSize(a);
  const auto &ms_tensor = t.tensor();
  MS_EXCEPTION_IF_NULL(ms_tensor);
  const auto &address = ms_tensor->device_address();
  MS_EXCEPTION_IF_NULL(address);
  if (address->GetDeviceType() == mindspore::device::DeviceType::kCPU) {
    a.hostData = t.GetDataPtr();
  } else {
    a.deviceData = t.GetDataPtr();
  }
  return a;
}

size_t AtbOpRunner::CalcWorkspace() {
  for (size_t i = 0; i < _inputs_.size(); i++) {
    variant_pack_.inTensors.push_back(_inputs_[i].is_defined() ? AtbTensor(_inputs_[i]) : atb::Tensor());
  }
  for (size_t i = 0; i < _outputs_.size(); i++) {
    variant_pack_.outTensors.push_back(AtbTensor(_outputs_[i]));
  }
  context_ = atb::AtbContextManager::GetInstance().GetContext(stream());
  MS_EXCEPTION_IF_NULL(context_);

  workspace_size_ = 0;
  auto st = op_->get()->Setup(variant_pack_, workspace_size_, context_);
  CHECK_ATB_RET(op_name(), st, Setup);
  return static_cast<size_t>(workspace_size_);
}

void AtbOpRunner::LaunchKernel() {
  auto st = op_->get()->Execute(variant_pack_, static_cast<uint8_t *>(workspace_ptr()), workspace_size_, context_);
  CHECK_ATB_RET(op_name(), st, Execute);
  op_->Free();
}

void AtbOpRunner::_Run() {
  for (auto &inp : _inputs_) {
    if (inp.is_defined() && !inp.is_contiguous()) {
      inp = inp.contiguous();
    }
  }
  PyboostRunner::_Run();
}
}  // namespace ms::pynative
