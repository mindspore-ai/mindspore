/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include <memory>
#include "plugin/device/ascend/kernel/internal/concat.h"
namespace mindspore {
namespace kernel {
internal::OpParamPtr InternalConcat::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                   const std::vector<KernelTensor *> &outputs) {
  internal::OpParamPtr param_ptr = std::make_shared<internal::OpParam>();
  internal::ConcatParam concat_param;

  if (inputs.size() != 3) {
    MS_LOG(EXCEPTION) << "InternalConcat only support concat 2 tensor now";
  }

  auto axis_tensor = inputs.at(2);  // axis
  if (axis_tensor->dtype_id() == TypeId::kNumberTypeInt64) {
    concat_param.concatDim = axis_tensor->GetValue<int64_t>().value();
  } else {
    MS_LOG(EXCEPTION) << "InternalTranspose input[-1] dtype is not kNumberTypeInt64";
  }
  concatDim_ = concat_param.concatDim;  // for GenTilingCacheKey

  param_ptr->specificParam = concat_param;
  param_ptr->opId = internal::OpId::Concat;
  return param_ptr;
}
void InternalConcat::SetInOutIdx() {
  // only support concat 2 tensor now
  inputsIdxMap_[0] = 0;
  inputsIdxMap_[1] = 1;
  outputsIdxMap_[0] = 0;
}

uint64_t InternalConcat::GenTilingCacheKey(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &outputs) {
  // User defined CacheKey, the inputs should include all the factors which will affect tiling result.
  return TilingCacheMgr::GetInstance().GenTilingCacheKey(
    kernel_name_, inputs[0]->GetShapeVector(), inputs[0]->dtype_id(), inputs[1]->GetShapeVector(),
    inputs[1]->dtype_id(), concatDim_, outputs[0]->GetShapeVector(), outputs[0]->dtype_id());
}

MS_INTERNAL_KERNEL_FACTORY_REG(Concat, InternalConcat);
}  // namespace kernel
}  // namespace mindspore
