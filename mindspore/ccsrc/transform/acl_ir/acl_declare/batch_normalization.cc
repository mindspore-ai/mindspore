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
REGISTER_ACL_OP(BNTrainingReduce)
  .Input(0, {"NCHW"}, {"NCHW"}, "NCH")
  .Output(0, {"NCHW"}, {"NCHW"})
  .Output(1, {"NCHW"}, {"NCHW"});

REGISTER_ACL_OP(BNTrainingUpdate)
  .Input(0, {"NCHW"}, {"NCHW"}, "NCH")
  .Input(1, {"NCHW"}, {"NCHW"})
  .Input(2, {"NCHW"}, {"NCHW"})
  .Input(3, {"NCHW"}, {"NCHW", "NC1HWC0"})
  .Input(4, {"NCHW"}, {"NCHW", "NC1HWC0"})
  .Input(5, {"NCHW"}, {"NCHW", "NC1HWC0"})
  .Input(6, {"NCHW"}, {"NCHW", "NC1HWC0"})
  .Output(0, {"NCHW"}, {"NCHW"}, "NCH")
  .Output(1, {"NCHW"}, {"NCHW"})
  .Output(2, {"NCHW"}, {"NCHW"})
  .Output(3, {"NCHW"}, {"NCHW"})
  .Output(4, {"NCHW"}, {"NCHW"});

REGISTER_ACL_OP(BNTrainingUpdateGrad)
  .Input(0, {"NCHW"}, {"NCHW"}, "NCH")
  .Input(1, {"NCHW"}, {"NCHW"}, "NCH")
  .Input(2, {"NCHW"}, {"NCHW"})
  .Input(3, {"NCHW"}, {"NCHW"})
  .Output(0, {"NCHW"}, {"NCHW"})
  .Output(1, {"NCHW"}, {"NCHW"});

REGISTER_ACL_OP(BNTrainingReduceGrad)
  .Input(0, {"NCHW"}, {"NCHW"}, "NCH")
  .Input(1, {"NCHW"}, {"NCHW"}, "NCH")
  .Input(2, {"NCHW"}, {"NCHW"})
  .Input(3, {"NCHW"}, {"NCHW"})
  .Input(4, {"NCHW"}, {"NCHW", "NC1HWC0"})
  .Input(5, {"NCHW"}, {"NCHW", "NC1HWC0"})
  .Input(6, {"NCHW"}, {"NCHW", "NC1HWC0"})
  .Output(0, {"NCHW"}, {"NCHW"}, "NCH");

REGISTER_ACL_OP(BNInfer)
  .Input(0, {"NCHW"}, {"NCHW", "NC1HWC0"}, "NCH")
  .Input(1, {"NCHW"}, {"NCHW", "NC1HWC0"})
  .Input(2, {"NCHW"}, {"NCHW", "NC1HWC0"})
  .Input(3, {"NCHW"}, {"NCHW", "NC1HWC0"})
  .Input(4, {"NCHW"}, {"NCHW", "NC1HWC0"})
  .Output(0, {"NCHW"}, {"NCHW"}, "NCH");

REGISTER_ACL_OP(BNInferGrad)
  .Input(0, {"NCHW"}, {"NCHW", "NC1HWC0"}, "NCH")
  .Input(1, {"NCHW"}, {"NCHW", "NC1HWC0"})
  .Input(2, {"NCHW"}, {"NCHW", "NC1HWC0"})
  .Output(0, {"NCHW"}, {"NCHW"}, "NCH");
}  // namespace transform
}  // namespace mindspore
