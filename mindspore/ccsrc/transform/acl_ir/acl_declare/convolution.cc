/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "transform/acl_ir/acl_adapter_info.h"

namespace mindspore {
namespace transform {
REGISTER_ACL_OP(Conv2D)
  .Input(0, {"NCHW"}, {"NCHW", "NC1HWC0"})
  .Input(1, {"NCHW"}, {"DefaultFormat", "NCHW", "FRACTAL_Z"})
  .Input(2, {"NCHW"}, {"NCHW"})
  .Output(0, {"NCHW"}, {"NCHW"});

REGISTER_ACL_OP(Conv3D)
  .Input(0, {"NCHW"}, {"NDC1HWC0"})
  .Input(1, {"NCHW"}, {"FRACTAL_Z_3D"})
  .Input(2, {"NCHW"}, {"ND"})
  .Output(0, {"NCHW"}, {"NDC1HWC0"});

REGISTER_ACL_OP(Conv2DBackpropInput)
  .Input(0, {"NCHW"}, {"NCHW"})
  .Input(1, {"NCHW"}, {"DefaultFormat", "NCHW", "FRACTAL_Z"})
  .Input(2, {"NCHW"}, {"NCHW", "NC1HWC0"})
  .Output(0, {"NCHW"}, {"NCHW"});

REGISTER_ACL_OP(Conv3DBackpropInput)
  .Input(0, {"NCHW"}, {"ND"})
  .Input(1, {"NCHW"}, {"ND", "FRACTAL_Z_3D"})
  .Input(2, {"NCHW"}, {"NDC1HWC0"})
  .Output(0, {"NCHW"}, {"NDC1HWC0"});

REGISTER_ACL_OP(Conv2DBackpropFilter)
  .Input(0, {"NCHW"}, {"NCHW", "NC1HWC0"})
  .Input(1, {"NCHW"}, {"NCHW", "NC1HWC0"})
  .Input(2, {"NCHW"}, {"NCHW", "NC1HWC0"})
  .Output(0, {"NCHW"}, {"NCHW"});
}  // namespace transform
}  // namespace mindspore
