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

#include "coder/opcoders/nnacl/fp16/convolution_fp16_coder.h"
#include <memory>
#include <string>
#include <vector>
#include "coder/opcoders/nnacl/fp32/convolution_winograd_fp32_coder.h"
#include "nnacl/fp32/winograd_utils.h"
#include "coder/opcoders/file_collector.h"
#include "coder/log.h"
#include "coder/opcoders/parallel.h"
#include "src/common/version_manager.h"
#include "base/float16.h"

using mindspore::schema::PrimitiveType_Conv2DFusion;
namespace mindspore::lite::micro::nnacl {
int ConvolutionFP16Coder::InitTmpBuffer() {
  int in_channel = conv_param_->input_channel_;
  int row_tile = C12NUM;
  if (target_ == kARM64) {
    row_tile = C16NUM;
  } else if (target_ == kARM32) {
    row_tile = C12NUM;
  } else {
    return RET_NOT_SUPPORT;
  }
  int uint_size = conv_param_->kernel_h_ * conv_param_->kernel_w_ * in_channel * row_tile * thread_num_;
  packed_input_size_ = uint_size * DataTypeSize(data_type_);
  packed_input_ = allocator_->Malloc(data_type_, packed_input_size_, kWorkspace);
  MS_CHECK_PTR(packed_input_);
  col_major_input_size_ = uint_size * DataTypeSize(data_type_);
  col_major_input_ = allocator_->Malloc(data_type_, col_major_input_size_, kWorkspace);
  MS_CHECK_PTR(col_major_input_);
  return RET_OK;
}

int ConvolutionFP16Coder::InitWeightBias(CoderContext *const context) {
  int kernel_h = filter_tensor_->Height();
  int kernel_w = filter_tensor_->Width();
  int in_channel = filter_tensor_->Channel();
  int out_channel = filter_tensor_->Batch();
  conv_param_->input_channel_ = in_channel;
  conv_param_->output_channel_ = out_channel;
  int kernel_plane = kernel_h * kernel_w;
  int oc_block_num = UP_ROUND(out_channel, C8NUM);
  int pack_weight_size = oc_block_num * in_channel * kernel_plane;
  pack_weight_size_ = pack_weight_size * DataTypeSize(data_type_);
  packed_weight_ = allocator_->Malloc(data_type_, kOnlineSize, kOnlinePackWeight);
  MS_CHECK_PTR(packed_weight_);

  NNaclFp32Serializer init_code;

  std::string ori_weight_addr = allocator_->GetRuntimeAddr(filter_tensor_);
  size_t w_buf_size = 0;
  w_buf_size += pack_weight_size_;
  auto packed_weight_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float16 *>(packed_weight_));
  init_code.CodeBufferOffsetExpression(packed_weight_, context->weight_name(), context->weight_offset_name(),
                                       context->weight_size_name(), pack_weight_size_);
  init_code.CodeFunction("RowMajor2Col8MajorFp16", ori_weight_addr, packed_weight_str, out_channel,
                         in_channel * kernel_plane, false);

  if (input_tensors_.size() == kInputSize2) {
    auto bias_data_size = static_cast<size_t>(oc_block_num * DataTypeSize(data_type_));
    bias_data_ =
      allocator_->Malloc(data_type_, kOnlineSize, kOnlinePackWeight, bias_tensor_->tensor_name() + "_online_pack");
    MS_CHECK_PTR(bias_data_);
    init_code.CodeBufferOffsetExpression(bias_data_, context->weight_name(), context->weight_offset_name(),
                                         context->weight_size_name(), bias_data_size);
    w_buf_size += bias_data_size;
    auto bias_data_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(bias_data_);
    std::string bias_tensor_str = allocator_->GetRuntimeAddr(bias_tensor_);
    init_code.CodeFunction("memcpy", bias_data_str, bias_tensor_str, bias_tensor_->Size());
  }

  context->AppendInitWeightSizeCode(w_buf_size);
  context->AppendInitCode(init_code.str());
  return RET_OK;
}

void ConvolutionFP16Coder::CollectFilesForFunc(CoderContext *const context) {
  Collect(context, {}, {},
          {
            "MatmulFp16.S",
            "MatmulFp16Opt.S",
            "MatVecMulFp16.S",
            "Matmul12X16Fp16.S",
          });
  Collect(context,
          {
            "nnacl/fp16/matmul_fp16.h",
            "nnacl/conv_parameter.h",
            "nnacl/op_base.h",
            "nnacl/fp16/conv_fp16.h",
          },
          {
            "common_func.c",
            "matmul_fp16.c",
            "pack_fp16.c",
            "conv_fp16.c",
          });
}

int ConvolutionFP16Coder::DoCode(CoderContext *const context) {
  CollectFilesForFunc(context);
  NNaclFp32Serializer code;
  // call the op function
  auto packed_input_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float16 *>(packed_input_));
  auto col_major_input_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float16 *>(col_major_input_));
  auto packed_weight_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float16 *>(packed_weight_));
  auto input_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(input_tensor_);
  auto output_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(output_tensor_);
  code.CodeFunction("memset", packed_input_str, "0", packed_input_size_);
  code.CodeFunction("memset", col_major_input_str, "0", col_major_input_size_);
  code.CodeStruct("conv_parameter", *conv_param_);
  if (output_tensor_->format() == NC4HW4) {
    code.CodeFunction("ConvOutNc8hw8Fp16", input_str, packed_input_str, packed_weight_str, bias_data_,
                      col_major_input_str, output_str, kDefaultTaskId, "&conv_parameter");
  } else {
    code.CodeFunction("ConvFp16", input_str, packed_input_str, packed_weight_str, bias_data_, col_major_input_str,
                      output_str, kDefaultTaskId, "&conv_parameter");
  }
  context->AppendCode(code.str());
  return RET_OK;
}
}  // namespace mindspore::lite::micro::nnacl
