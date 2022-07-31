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

#include "src/litert/delegate/npu/transpose_kernel.h"
#include "src/litert/delegate/npu/npu_converter_utils.h"
#include "src/litert/delegate/npu/op/npu_op.h"
#include "src/litert/delegate/delegate_utils.h"
#include "nnacl/fp32/pack_fp32.h"
namespace mindspore::lite {
int TransposeNPUKernel::Execute() {
  if (perm_ != NHWC2NCHW_PERM && perm_ != NCHW2NHWC_PERM) {
    MS_LOG(ERROR) << "NPU transpose op only supports nhwc->nchw or nchw->nhwc.";
    return RET_ERROR;
  }
  auto shape = inputs()[0].Shape();
  if (shape.size() != NPU_SHAPE_SIZE) {
    MS_LOG(ERROR) << "NPU transpose op only supports input of 4 dims.";
    return RET_ERROR;
  }
  mindspore::MSTensor in_tensor = inputs()[0];
  mindspore::MSTensor out_tensor = outputs()[0];
  auto input = in_tensor.Data().get();
  MS_ASSERT(input);
  auto output = out_tensor.MutableData();
  MS_ASSERT(output);
  if (perm_ == NHWC2NCHW_PERM) {
    PackNHWCToNCHWFp32(input, output, shape[NHWC_N], shape[NHWC_H] * shape[NHWC_W], shape[NHWC_C]);
  } else if (perm_ == NCHW2NHWC_PERM) {
    PackNCHWToNHWCFp32(input, output, shape[NCHW_N], shape[NCHW_H] * shape[NCHW_W], shape[NCHW_C]);
  } else {
    MS_LOG(ERROR) << "NPU transpose op only supports nhwc->nchw or nchw->nhwc.";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::lite
