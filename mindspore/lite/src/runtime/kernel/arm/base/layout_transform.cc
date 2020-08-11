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

#include "src/runtime/kernel/arm/base/layout_transform.h"
#include "mindspore/core/utils/log_adapter.h"

using mindspore::schema::Format;
namespace mindspore::kernel {
LayoutConvertor LayoutTransformFp32(schema::Format src_format, schema::Format dst_format) {
  // todo
  if (src_format == schema::Format_NHWC && dst_format == schema::Format_NC4HW4) {
    return PackNHWCToNC4HW4Fp32;
  } else if (src_format == schema::Format_NHWC && dst_format == schema::Format_NHWC4) {
    return PackNHWCToNHWC4Fp32;
  } else if (src_format == schema::Format_NC4HW4 && dst_format == schema::Format_NHWC4) {
    return PackNC4HW4ToNHWC4Fp32;
  } else if (src_format == schema::Format_NCHW && dst_format == schema::Format_NC4HW4) {
    return PackNCHWToNC4HW4Fp32;
  } else if (src_format == schema::Format_NC4HW4 && dst_format == schema::Format_NHWC) {
    return PackNC4HW4ToNHWCFp32;
  } else {
    MS_LOG(ERROR) << "Unsupported transform from " << schema::EnumNameFormat(src_format) << " to "
                  << schema::EnumNameFormat(dst_format);
    return nullptr;
  }
}

LayoutConvertor LayoutTransformInt8(schema::Format src_format, schema::Format dst_format) {
  // todo
  if (src_format == schema::Format_NHWC && dst_format == schema::Format_NHWC4) {
    return PackNHWCToNHWC4Int8;
  } else {
    return nullptr;
  }
}

LayoutConvertor LayoutTransform(TypeId data_type, schema::Format src_format, schema::Format dst_format) {
  // todo
  switch (data_type) {
    case kNumberTypeInt8:
      return LayoutTransformInt8(src_format, dst_format);
    case kNumberTypeFloat32:
      return LayoutTransformFp32(src_format, dst_format);
    default:
      return nullptr;
  }
}
}  // namespace mindspore::kernel
