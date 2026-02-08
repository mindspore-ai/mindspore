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

#include "ms_extension/all.h"

namespace custom {

ms::Tensor npu_cast(const ms::Tensor &x, int64_t dst_type_id) {
  auto type_id = static_cast<ms::TypeId>(dst_type_id);
  auto result = ms::Tensor(type_id, x.shape());
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("aclnnCast");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnCast, x, type_id, result));
  runner->Run({x}, {result});
  return result;
}

}  // namespace custom

PYBIND11_MODULE(MS_EXTENSION_NAME, m) { m.def("npu_cast", PYBOOST_CALLER(1, custom::npu_cast)); }