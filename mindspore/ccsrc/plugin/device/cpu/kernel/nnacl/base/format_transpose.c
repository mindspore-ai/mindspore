/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "nnacl/base/format_transpose.h"
#include "nnacl/errorcode.h"
#include "nnacl/fp32/pack_fp32.h"
#ifdef ENABLE_FP16
#include "nnacl/fp16/pack_fp16.h"
#endif

int TransposeFp32Data(void *src_data, void *dst_data, const FormatC src_format, const FormatC dst_format,
                      const int batch, const int channel, const int plane) {
  if (src_format == Format_NCHW && dst_format == Format_NC4HW4) {
    PackNCHWToNC4HW4Fp32(src_data, dst_data, batch, plane, channel);
  } else if (src_format == Format_NHWC && dst_format == Format_NC4HW4) {
    PackNHWCToNC4HW4Fp32(src_data, dst_data, batch, plane, channel);
  } else if (src_format == Format_NC4HW4 && dst_format == Format_NCHW) {
    PackNC4HW4ToNCHWFp32(src_data, dst_data, batch, plane, channel);
  } else if (src_format == Format_NC4HW4 && dst_format == Format_NHWC) {
    PackNC4HW4ToNHWCFp32(src_data, dst_data, batch, plane, channel);
  } else if (src_format == Format_NHWC && dst_format == Format_NCHW) {
    PackNHWCToNCHWFp32(src_data, dst_data, batch, plane, channel, 0, 1);
  } else if (src_format == Format_NCHW && dst_format == Format_NHWC) {
    PackNCHWToNHWCFp32(src_data, dst_data, batch, plane, channel, 0, 1);
  } else {
    return NNACL_ERR;
  }
  return NNACL_OK;
}

#ifdef ENABLE_FP16
int TransposeFp16Data(void *src_data, void *dst_data, const FormatC src_format, const FormatC dst_format, int batch,
                      int channel, int plane) {
  if (src_format == Format_NCHW && dst_format == Format_NC8HW8) {
    PackNCHWFp16ToNC8HW8Fp16(src_data, dst_data, batch, plane, channel);
  } else if (src_format == Format_NHWC && dst_format == Format_NC8HW8) {
    return NNACL_ERR;
  } else if (src_format == Format_NC8HW8 && dst_format == Format_NCHW) {
    PackNC8HW8ToNCHWFp16(src_data, dst_data, batch, plane, channel);
  } else if (src_format == Format_NC8HW8 && dst_format == Format_NHWC) {
    PackNC8HW8ToNHWCFp16((float16_t *)src_data, (float16_t *)dst_data, batch, plane, channel);
  } else {
    return NNACL_ERR;
  }
  return NNACL_OK;
}
#endif

int TransData(void *src_data, void *dst_data, const FormatC src_format, const FormatC dst_format, TypeIdC data_type,
              const int batch, const int channel, const int plane) {
  switch (data_type) {
    case kNumberTypeFloat:
    case kNumberTypeFloat32:
      return TransposeFp32Data(src_data, dst_data, src_format, dst_format, batch, channel, plane);
#ifdef ENABLE_FP16
    case kNumberTypeFloat16:
      return TransposeFp16Data(src_data, dst_data, src_format, dst_format, batch, channel, plane);
#endif
    default:
      return NNACL_ERR;
  }
}
