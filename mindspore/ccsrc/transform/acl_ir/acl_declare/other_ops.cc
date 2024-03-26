/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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
 * limitations under the License.s
 */

#include "transform/acl_ir/acl_adapter_info.h"

namespace mindspore {
namespace transform {
REGISTER_ACL_OP(KVCacheMgr).set_run_mode(false);

REGISTER_ACL_OP(RightShift).set_run_mode(false);

REGISTER_ACL_OP(LeftShift).set_run_mode(false);

REGISTER_ACL_OP(UpsampleTrilinear3d).set_run_mode(false);
REGISTER_ACL_OP(UpsampleNearest3d).set_run_mode(false);

REGISTER_ACL_OP(CheckValid).set_run_mode(false);

REGISTER_ACL_OP(HistogramFixedWidth).set_extra_supported_datatype({ge::DT_DOUBLE});

REGISTER_ACL_OP(ResizeBilinearV2Grad).set_extra_supported_datatype({ge::DT_FLOAT16});
}  // namespace transform
}  // namespace mindspore
