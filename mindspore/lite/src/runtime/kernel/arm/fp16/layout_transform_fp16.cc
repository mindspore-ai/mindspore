/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "src/runtime/kernel/arm/fp16/layout_transform_fp16.h"
#include "src/runtime/kernel/arm/nnacl/fp16/pack_fp16.h"
#include "schema/ops_generated.h"
#include "mindspore/core/utils/log_adapter.h"

namespace mindspore::kernel {
LayoutConvertor LayoutTransformFp16(schema::Format src_format, schema::Format dst_format) {
  if (src_format == schema::Format_NHWC && dst_format == schema::Format_NC4HW4) {
    return PackNHWCToNC4HW4Fp16;
  } else if (src_format == schema::Format_NHWC && dst_format == schema::Format_NHWC4) {
    return PackNHWCToNHWC4Fp16;
  } else if (src_format == schema::Format_NC4HW4 && dst_format == schema::Format_NHWC4) {
    return PackNC4HW4ToNHWC4Fp16;
  } else if (src_format == schema::Format_NCHW && dst_format == schema::Format_NC4HW4) {
    return PackNCHWToNC4HW4Fp16;
  } else if (src_format == schema::Format_NC4HW4 && dst_format == schema::Format_NHWC) {
    return PackNC4HW4ToNHWCFp16;
  } else {
    MS_LOG(ERROR) << "Unsupported transform from " << schema::EnumNameFormat(src_format) << " to "
                  << schema::EnumNameFormat(dst_format);
    return nullptr;
  }
}
}  // namespace mindspore::kernel
