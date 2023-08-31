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

#include "coder/opcoders/nnacl/fp16/matmul_fp16_base_coder.h"
#include <string>
#include <vector>
#include "coder/log.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/nnacl/dequant/de_quant.h"
#include "base/float16.h"

using mindspore::schema::PrimitiveType_MatMulFusion;

namespace mindspore::lite::micro::nnacl {
int MatMulFP16BaseCoder::InitBufferForBias() {
  if (bias_ptr_) {
    return RET_OK;
  }
  bias_pack_ptr_size_ = static_cast<size_t>(params_.col_align_ * DataTypeSize(data_type_));
  if (input_tensors_.size() == C3NUM) {
    bias_ptr_ =
      allocator_->Malloc(data_type_, kOnlineSize, kOnlinePackWeight, bias_tensor_->tensor_name() + "_online_pack");
    MS_CHECK_PTR(bias_ptr_);
  } else {
    bias_ptr_ = allocator_->Malloc(data_type_, kOnlineSize, kOnlinePackWeight, node_->name_ + "_bias_online_pack");
    MS_CHECK_PTR(bias_ptr_);
  }
  return RET_OK;
}

int MatMulFP16BaseCoder::InitBufferA() {
  if (a_pack_ptr_ != nullptr || vec_matmul_) {
    return RET_OK;
  }
  a_pack_ptr_size_ =
    static_cast<size_t>(params_.a_batch_ * params_.row_align_ * params_.deep_ * DataTypeSize(data_type_));
  if (params_.a_const_) {
    a_pack_ptr_ = allocator_->GetSharedWeightAddr(input_tensor_);
    if (a_pack_ptr_ == nullptr) {
      a_pack_ptr_ =
        allocator_->Malloc(data_type_, kOnlineSize, kOnlinePackWeight, input_tensor_->tensor_name() + "_online_pack");
      allocator_->MarkSharedWeight(input_tensor_, a_pack_ptr_);
    } else {
      a_packed_ = true;
    }
  } else {
    a_pack_ptr_ = allocator_->Malloc(data_type_, a_pack_ptr_size_, kWorkspace);
  }
  MS_CHECK_PTR(a_pack_ptr_);
  return RET_OK;
}

int MatMulFP16BaseCoder::InitBufferB() {
  if (target_ != kARM64) {
    if (vec_matmul_ && params_.b_transpose_) {
      return RET_OK;
    }
  }
  return MatMulFP32BaseCoder::InitBufferB();
}

std::string MatMulFP16BaseCoder::InitBiasData(NNaclFp32Serializer *const init_code, CoderContext *const context,
                                              size_t *w_buf) {
  init_code->CodeBufferOffsetExpression(bias_ptr_, context->weight_name(), context->weight_offset_name(),
                                        context->weight_size_name(), bias_pack_ptr_size_);
  *w_buf = *w_buf + bias_pack_ptr_size_;
  std::string bias_str = allocator_->GetRuntimeAddr(bias_ptr_);
  if (input_tensors_.size() == DIMENSION_3D) {
    auto origin_bias_str = allocator_->GetRuntimeAddr(bias_tensor_);
    init_code->CodeFunction("memcpy", bias_str, origin_bias_str, bias_tensor_->Size());
  } else {
    init_code->CodeFunction("memset", bias_str, 0, bias_pack_ptr_size_);
  }
  return bias_str;
}

std::string MatMulFP16BaseCoder::InitMatrixA(NNaclFp32Serializer *const code, NNaclFp32Serializer *const init_code,
                                             CoderContext *const context, size_t *w_buf) {
  if (vec_matmul_) {
    return allocator_->GetRuntimeAddr(input_tensor_, input_tensor_->IsConst());
  }
  std::string input_a_str = allocator_->GetRuntimeAddr(input_tensor_);
  std::string input_a_pack_str = allocator_->GetRuntimeAddr(static_cast<float16 *>(a_pack_ptr_));
  if (params_.a_const_) {
    init_code->CodeBufferOffsetExpression(a_pack_ptr_, context->weight_name(), context->weight_offset_name(),
                                          context->weight_size_name(), a_pack_ptr_size_);
    *w_buf = *w_buf + a_pack_ptr_size_;
  }
  NNaclFp32Serializer &pack_code = params_.a_const_ ? *init_code : *code;
  if (params_.a_batch_ == 1) {
    if (params_.a_transpose_) {
      if (target_ == kARM64) {
        pack_code.CodeFunction("RowMajor2RowNMajorFp16", input_a_str, input_a_pack_str, params_.deep_, params_.row_,
                               "false");
      } else {
        pack_code.CodeFunction("RowMajor2Row12MajorFp16", input_a_str, input_a_pack_str, params_.deep_, params_.row_,
                               false);
      }
    } else {
      if (target_ == kARM64) {
        pack_code.CodeFunction("RowMajor2ColNMajorFp16", input_a_str, input_a_pack_str, params_.row_, params_.deep_,
                               "false");
      } else {
        pack_code.CodeFunction("RowMajor2Col12MajorFp16", input_a_str, input_a_pack_str, params_.row_, params_.deep_,
                               false);
      }
    }
  } else {
    pack_code << "  for (int i = 0; i < " << params_.a_batch_ << "; i++) {\n"
              << "    float16_t *src = " << input_a_str << " + i * " << params_.deep_ * params_.row_ << ";\n"
              << "    float16_t *dst = " << input_a_pack_str << " + i * " << params_.deep_ * params_.row_align_
              << ";\n";
    if (params_.a_transpose_) {
      if (target_ == kARM64) {
        pack_code << "    RowMajor2RowNMajorFp16(src, dst, " << params_.deep_ << ", " << params_.row_ << ", false);\n";
      } else {
        pack_code << "    RowMajor2Row12MajorFp16(src, dst, " << params_.deep_ << ", " << params_.row_ << ", false);\n";
      }
    } else {
      if (target_ == kARM64) {
        pack_code << "    RowMajor2ColNMajorFp16(src, dst, " << params_.row_ << ", " << params_.deep_ << ", false);\n";
      } else {
        pack_code << "    RowMajor2Col12MajorFp16(src, dst, " << params_.row_ << ", " << params_.deep_ << ", false);\n";
      }
    }
    pack_code << "  }\n";
  }
  return input_a_pack_str;
}

std::string MatMulFP16BaseCoder::InitMatrixB(NNaclFp32Serializer *const code, NNaclFp32Serializer *const init_code,
                                             CoderContext *const context, size_t *w_buf) {
  bool not_pack = target_ != kARM64 && vec_matmul_ && params_.b_transpose_;
  if (not_pack) {
    return allocator_->GetRuntimeAddr(filter_tensor_, filter_tensor_->IsConst());
  }
  std::string input_b_str = allocator_->GetRuntimeAddr(filter_tensor_);
  std::string input_b_pack_str = allocator_->GetRuntimeAddr(static_cast<float16 *>(b_pack_ptr_));
  if (params_.b_const_) {
    init_code->CodeBufferOffsetExpression(b_pack_ptr_, context->weight_name(), context->weight_offset_name(),
                                          context->weight_size_name(), b_pack_ptr_size_);
    *w_buf = *w_buf + b_pack_ptr_size_;
  }
  NNaclFp32Serializer &pack_code = params_.b_const_ ? *init_code : *code;
  if (target_ != kARM64) {
    if (vec_matmul_) {
      if (params_.b_batch_ == 1) {
        pack_code.CodeFunction("RowMajor2ColMajorFp16", input_b_str, input_b_pack_str, params_.deep_, params_.col_,
                               false);
      } else {
        pack_code << "  for (int i = 0; i < " << params_.b_batch_ << "; i++) {\n"
                  << "    float16_t *src = " << input_b_str << " + i * " << params_.deep_ * params_.col_ << ";\n"
                  << "    float16_t *dst = " << input_b_pack_str << " + i * " << params_.deep_ * params_.col_ << ";\n"
                  << "    RowMajor2ColMajorFp16(src, dst, " << params_.deep_ << ", " << params_.col_ << ", false);\n"
                  << "  }\n";
      }
      return input_b_pack_str;
    }
  }

  if (params_.b_batch_ == 1) {
    if (params_.b_transpose_) {
      pack_code.CodeFunction("RowMajor2Col8MajorFp16", input_b_str, input_b_pack_str, params_.col_, params_.deep_,
                             false);
    } else {
      pack_code.CodeFunction("RowMajor2Row8MajorFp16", input_b_str, input_b_pack_str, params_.deep_, params_.col_,
                             false);
    }
  } else {
    pack_code << "  for (int i = 0; i < " << params_.b_batch_ << "; i++) {\n"
              << "    float16_t *src = " << input_b_str << " + i * " << params_.deep_ * params_.col_ << ";\n"
              << "    float16_t *dst = " << input_b_pack_str << " + i * " << params_.deep_ * params_.col_align_
              << ";\n";
    if (params_.b_transpose_) {
      pack_code << "    RowMajor2Col8MajorFp16(src, dst, " << params_.col_ << ", " << params_.deep_ << ", false);\n";
    } else {
      pack_code << "    RowMajor2Row8MajorFp16(src, dst, " << params_.deep_ << ", " << params_.col_ << ", false);\n";
    }
    pack_code << "  }\n";
  }
  return input_b_pack_str;
}

int MatMulFP16BaseCoder::Prepare(CoderContext *const context) {
  if (input_tensor_->data_type() != kNumberTypeFloat16 || filter_tensor_->data_type() != kNumberTypeFloat16) {
    MS_LOG(INFO) << "Input tensor data type is invalid";
    return RET_INPUT_PARAM_INVALID;
  }
  row_tile_ = C12NUM;
  if (target_ == kARM64) {
    row_tile_ = C4NUM;
  }

  auto ret = InitAShape();
  MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "init A-metrics' info failed");
  ret = InitBShape();
  MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "init B-metrics' info failed");
  if (params_.row_ == 1) {
    vec_matmul_ = true;
  }
  if (vec_matmul_) {
    params_.row_align_ = 1;
    params_.col_align_ = (target_ == kARM64) ? UP_ROUND(params_.col_, C8NUM) : params_.col_;
  } else {
    params_.row_align_ = UP_ROUND(params_.row_, row_tile_);
    params_.col_align_ = UP_ROUND(params_.col_, C8NUM);
  }
  MS_CHECK_RET_CODE(InitBufferA(), "InitBufferA failed");
  MS_CHECK_RET_CODE(InitBufferB(), "InitBufferB failed");
  MS_CHECK_RET_CODE(InitBufferForBias(), "InitBufferForBias failed");
  return RET_OK;
}

int MatMulFP16BaseCoder::CollectFilesForTarget(CoderContext *const context) {
  Collect(context,
          {
            "nnacl/fp16/pack_fp16.h",
            "nnacl/fp16/matmul_fp16.h",
          },
          {
            "pack_fp16.c",
            "matmul_fp16.c",
          });
  if (target_ == kARM32) {
    Collect(context, {}, {},
            {
              "Matmul12x8Fp16.S",
              "MatVecMulFp16.S",
            });
  } else if (target_ == kARM64) {
    Collect(context, {}, {},
            {
              "MatmulFp16.S",
              "MatmulFp16Opt.S",
              "MatVecMulFp16.S",
              "Matmul12X16Fp16.S",
              "MatmulBaseFp16Neon.S",
              "MatmulWinogradFp16.S",
              "VecMatmulFp16.S",
            });
  }
  return RET_OK;
}

int MatMulFP16BaseCoder::DoCode(CoderContext *const context) {
  CollectFilesForTarget(context);
  NNaclFp32Serializer code, init_code;
  size_t w_buf_size = 0;

  auto bias_str = InitBiasData(&init_code, context, &w_buf_size);
  auto input_a_str = InitMatrixA(&code, &init_code, context, &w_buf_size);
  auto input_b_str = InitMatrixB(&code, &init_code, context, &w_buf_size);
  auto output_str = allocator_->GetRuntimeAddr(output_tensor_);
  code << "    for (int i = 0; i < " << params_.batch << "; ++i) {\n";
  if (vec_matmul_) {
    code << "      const float16_t *batch_a_ptr = " << input_a_str << " + i * " << params_.deep_ << ";\n";
    if (params_.b_batch_ != 1) {
      code << "      const float16_t *batch_b_ptr = " << input_b_str << " + i * "
           << params_.deep_ * (target_ == kARM64 ? params_.col_align_ : params_.col_) << ";\n";
    } else {
      code << "      const float16_t *batch_b_ptr = " << input_b_str << ";\n";
    }
    code << "      float16_t *batch_c_ptr = " << output_str << " + i * " << params_.row_ * params_.col_ << ";\n  ";
    code.CodeFunction(target_ == kARM64 ? "VecMatmulFp16" : "MatVecMulFp16", "batch_a_ptr", "batch_b_ptr",
                      "batch_c_ptr", bias_str, params_.act_type_, params_.deep_, params_.col_);
  } else {
    code << "      const float16_t *batch_a_ptr = " << input_a_str << " + i * " << params_.row_align_ * params_.deep_
         << ";\n";
    code << "      const float16_t *batch_b_ptr = " << input_b_str << " + i * " << params_.deep_ * params_.col_align_
         << ";\n";
    code << "      float16_t *batch_c_ptr = " << output_str << " + i * " << params_.row_ * params_.col_ << ";\n  ";
    code.CodeFunction(target_ == kARM64 ? "MatmulBaseFp16Neon" : "MatMulFp16", "batch_a_ptr", "batch_b_ptr",
                      "batch_c_ptr", bias_str, params_.act_type_, params_.deep_, params_.row_, params_.col_,
                      params_.col_, OutType_Nhwc);
  }
  code << "  }\n";
  context->AppendInitWeightSizeCode(w_buf_size);
  context->AppendCode(code.str());
  context->AppendInitCode(init_code.str());
  return RET_OK;
}
}  // namespace mindspore::lite::micro::nnacl
