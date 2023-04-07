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

#include "coder/opcoders/nnacl/fp16/reduce_fp16_coder.h"
#include "coder/opcoders/file_collector.h"

using mindspore::schema::PrimitiveType_ReduceFusion;
namespace mindspore::lite::micro::nnacl {
int ReduceFP16Coder::Prepare(CoderContext *const context) {
  MS_CHECK_RET_CODE(ReduceBaseCoder::Init(), "init failed");
  if (input_tensors_.at(0)->data_type() != kNumberTypeFloat16) {
    MS_LOG(ERROR) << "Reduce fp16 coder only supports fp16 input.";
    return RET_ERROR;
  }
  data_type_ = ::kNumberTypeFloat16;
  MS_CHECK_RET_CODE(ReduceBaseCoder::ReSize(), "resize failed");
  MS_CHECK_RET_CODE(ReduceFP32Coder::MallocTmpBuffer(kNumberTypeFloat16), "malloc buffer failed");
  return RET_OK;
}

int ReduceFP16Coder::DoCode(CoderContext *const context) {
  Collect(context,
          {
            "nnacl/fp16/reduce_fp16.h",
          },
          {
            "reduce_fp16.c",
          });

  // call the op function
  switch (mode_) {
    case static_cast<int>(schema::ReduceMode_ReduceSum): {
      reduce_ = "ReduceSumFp16";
      break;
    }
    case static_cast<int>(schema::ReduceMode_ReduceMean): {
      reduce_ = "ReduceMeanFp16";
      break;
    }
    case static_cast<int>(schema::ReduceMode_ReduceMax): {
      reduce_ = "ReduceMaxFp16";
      break;
    }
    case static_cast<int>(schema::ReduceMode_ReduceMin): {
      reduce_ = "ReduceMinFp16";
      break;
    }
    case static_cast<int>(schema::ReduceMode_ReduceProd): {
      reduce_ = "ReduceProdFp16";
      break;
    }
    case static_cast<int>(schema::ReduceMode_ReduceL2): {
      reduce_ = "ReduceL2NormFp16";
      break;
    }
    default:
      MS_LOG(ERROR) << "Reduce unsupported reduce_ mode: " << mode_;
      return RET_ERROR;
  }
  GenerateCode(context);
  return RET_OK;
}

REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_ReduceFusion, CPUOpCoderCreator<ReduceFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_ReduceFusion, CPUOpCoderCreator<ReduceFP16Coder>)
}  // namespace mindspore::lite::micro::nnacl
