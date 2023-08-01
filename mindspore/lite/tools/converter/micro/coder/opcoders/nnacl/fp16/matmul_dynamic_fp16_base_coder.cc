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

#include "tools/converter/micro/coder/opcoders/nnacl/fp16/matmul_dynamic_fp16_base_coder.h"
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include "tools/converter/micro/coder/log.h"
#include "tools/converter/micro/coder/opcoders/file_collector.h"
#include "tools/common/string_util.h"
#include "base/float16.h"

using mindspore::schema::PrimitiveType_MatMulFusion;

namespace mindspore::lite::micro::nnacl {
int MatMulDynamicFP16BaseCoder::Prepare(CoderContext *const context) {
  row_tile_ = C4NUM;
  col_tile_ = C8NUM;
  auto ret = InitAShape();
  MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "init A-metrics' info failed");
  ret = InitBShape();
  MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "init B-metrics' info failed");
  params_.col_align_ = UP_ROUND(params_.col_, C8NUM);
  return RET_OK;
}

int MatMulDynamicFP16BaseCoder::DoCode(CoderContext *const context) {
  CollectFilesForTarget(context);
  auto ret = InitMatrixB(context);
  MS_CHECK_TRUE_MSG(ret != RET_OK, ret, "InitMatrixB failed.");
  ret = InitBiasData(context);
  MS_CHECK_TRUE_MSG(ret != RET_OK, ret, "InitBiasData failed.");
  if (!params_.a_const_) {
    ret = ComputeMatrixAWorkspace();
    MS_CHECK_TRUE_MSG(ret != RET_OK, ret, "InitMatrixA failed.");
  }

  auto input_a_str = dynamic_mem_manager_->GetVarTensorAddr(input_tensor_);
  MS_CHECK_TRUE_MSG(!input_a_str.empty(), RET_ERROR, "Matmul cannot get matrixA");
  auto output_str = dynamic_mem_manager_->GetVarTensorAddr(output_tensor_);
  MS_CHECK_TRUE_MSG(!output_str.empty(), RET_ERROR, "Matmul cannot get output");
  NNaclFp32Serializer code;
  code << "  if (" << dynamic_params_.row_ << " * " << dynamic_params_.batch_ << " == 1) {\n";
  code.CodeFunction("VecMatmulFp16", "(float16_t *)(" + input_a_str + ")", input_b_pack_str_, output_str, bias_str_,
                    params_.act_type_, params_.deep_, params_.col_);
  code << "  } else {\n";
  if (params_.a_transpose_) {
    code << "    if (" << dynamic_params_.row_ << " == 1) {\n";
    code.CodeFunction("RowMajor2RowNMajorFp16", "(float16_t *)(" + input_a_str + ")",
                      "(float16_t *)(" + buffers_start_ + ")", params_.deep_, dynamic_params_.row_);
    code.CodeFunction("MatmulBaseFp16Neon", "(float16_t *)(" + input_a_str + ")", input_b_pack_str_,
                      "(float16_t *)(" + output_str + ")", bias_str_, params_.act_type_, params_.deep_,
                      dynamic_params_.batch_, params_.col_, params_.col_, OutType_Nhwc);
    code << "    } else {\n"
         << "      int in_stride = " << dynamic_params_.row_ << " * " << params_.deep_ << ";\n"
         << "      int out_stride = " << dynamic_params_.row_ << " * " << params_.col_ << ";\n"
         << "      for (int i = 0; i < " << dynamic_params_.batch_ << "; i++) {\n"
         << "        float16_t *src = " << input_a_str << " + i * in_stride;\n"
         << "        float16_t *dst = " << output_str << " + i * out_stride;\n";
    code.CodeFunction("RowMajor2RowNMajorFp16", "src", "(float16_t *)(" + buffers_start_ + ")", params_.deep_,
                      dynamic_params_.row_);
    code.CodeFunction("MatmulBaseFp16Neon", "src", input_b_pack_str_, "dst", bias_str_, params_.act_type_,
                      params_.deep_, dynamic_params_.row_, params_.col_, params_.col_, OutType_Nhwc);
    code << "      }\n"
         << "    }\n";
  } else {
    code << "    if (" << dynamic_params_.row_ << " == 1) {\n";
    code.CodeFunction("RowMajor2ColNMajorFp16", "(float16_t *)(" + input_a_str + ")",
                      "(float16_t *)(" + buffers_start_ + ")", dynamic_params_.row_, params_.deep_);
    code.CodeFunction("MatmulBaseFp16Neon", "(float16_t *)(" + input_a_str + ")", input_b_pack_str_,
                      "(float16_t *)(" + output_str + ")", bias_str_, params_.act_type_, params_.deep_,
                      dynamic_params_.batch_, params_.col_, params_.col_, OutType_Nhwc);
    code << "    } else {\n"
         << "      int in_stride = " << dynamic_params_.row_ << " * " << params_.deep_ << ";\n"
         << "      int out_stride = " << dynamic_params_.row_ << " * " << params_.col_ << ";\n"
         << "      for (int i = 0; i < " << dynamic_params_.batch_ << "; i++) {\n"
         << "        float16_t *src = " << input_a_str << " + i * in_stride;\n"
         << "        float16_t *dst = " << output_str << " + i * out_stride;\n";
    code.CodeFunction("RowMajor2ColNMajorFp16", "src", "(float16_t *)(" + buffers_start_ + ")", params_.deep_,
                      dynamic_params_.row_);
    code.CodeFunction("MatmulBaseFp16Neon", "src", input_b_pack_str_, "dst", bias_str_, params_.act_type_,
                      params_.deep_, dynamic_params_.row_, params_.col_, params_.col_, OutType_Nhwc);
    code << "      }\n"
         << "    }\n";
  }
  code << "  }\n";
  context->AppendCode(code.str());
  return RET_OK;
}

int MatMulDynamicFP16BaseCoder::InitBiasData(CoderContext *const context) {
  if (bias_ptr_ != nullptr) {
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
  NNaclFp32Serializer init_code;
  init_code.CodeBufferOffsetExpression(bias_ptr_, context->weight_name(), context->weight_offset_name(),
                                       context->weight_size_name(), bias_pack_ptr_size_);
  bias_str_ = allocator_->GetRuntimeAddr(bias_ptr_);
  if (input_tensors_.size() == DIMENSION_3D) {
    auto origin_bias_str = allocator_->GetRuntimeAddr(bias_tensor_);
    init_code.CodeFunction("memcpy", bias_str_, origin_bias_str, bias_tensor_->Size());
  } else {
    init_code.CodeFunction("memset", bias_str_, 0, bias_pack_ptr_size_);
  }
  context->AppendInitWeightSizeCode(bias_pack_ptr_size_);
  context->AppendInitCode(init_code.str());
  return RET_OK;
}

int MatMulDynamicFP16BaseCoder::ComputeMatrixAWorkspace() {
  auto a_shape = shape_info_container_->GetTemplateShape(input_tensor_);
  std::map<std::string, std::vector<int>> real_nums;
  size_t scene_num = 0;
  for (auto &dim_template : a_shape) {
    auto dim_nums = shape_info_container_->GetRealNums(dim_template);
    MS_CHECK_TRUE_MSG(!dim_nums.empty(), RET_ERROR, "Dynamic shape's num must be greater than 0.");
    real_nums[dim_template] = dim_nums;
    scene_num = std::max(scene_num, dim_nums.size());
  }
  for (size_t i = 0; i < scene_num; ++i) {
    std::vector<int> real_shape(a_shape.size());
    for (size_t j = 0; j < a_shape.size(); ++j) {
      if (IsNumber(a_shape[j])) {
        real_shape[j] = std::stoi(a_shape[j]);
      } else {
        real_shape[j] = a_shape[j][i % real_nums[a_shape[j]].size()];
      }
    }
    int a_batch = 1;
    for (size_t j = 0; j < a_shape.size() - C2NUM; ++j) {
      MS_CHECK_INT_MUL_NOT_OVERFLOW(a_batch, real_shape[j], RET_ERROR);
      a_batch *= real_shape[j];
    }
    int row = params_.a_transpose_ ? real_shape.back() : real_shape[real_shape.size() - C2NUM];
    int deep = params_.a_transpose_ ? real_shape[real_shape.size() - C2NUM] : real_shape.back();
    MS_CHECK_TRUE_MSG(deep == params_.deep_, RET_INPUT_PARAM_INVALID,
                      "For MatMul, the deep of matrixA must be equal to the deep of MatrixB. Now MatrixA's deep is "
                        << deep << ", but MatrixB's deep is " << params_.deep_);

    int workspace = 0;
    if (params_.a_transpose_ && a_batch * row != C1NUM) {
      workspace = UP_ROUND(row, row_tile_) * deep;
    }
    buffers_start_ = dynamic_mem_manager_->AllocWorkSpace(workspace, i);
    MS_CHECK_TRUE_MSG(!buffers_start_.empty(), RET_ERROR, "MatMul cannot alloc workspace.");
  }
  return RET_OK;
}

int MatMulDynamicFP16BaseCoder::InitMatrixB(CoderContext *const context) {
  if (b_pack_ptr_ != nullptr) {
    return RET_OK;
  }
  b_pack_ptr_size_ = static_cast<size_t>(params_.col_align_ * params_.deep_ * DataTypeSize(data_type_));
  b_pack_ptr_ = allocator_->GetSharedWeightAddr(filter_tensor_);
  if (b_pack_ptr_ == nullptr) {
    b_pack_ptr_ = allocator_->Malloc(data_type_, b_pack_ptr_size_, kOnlinePackWeight,
                                     filter_tensor_->tensor_name() + "_online_pack");
    allocator_->MarkSharedWeight(filter_tensor_, b_pack_ptr_);
  }
  MS_CHECK_PTR(b_pack_ptr_);
  std::string input_b_str = allocator_->GetRuntimeAddr(filter_tensor_);
  input_b_pack_str_ = allocator_->GetRuntimeAddr(static_cast<float16 *>(b_pack_ptr_));
  NNaclFp32Serializer init_code;
  init_code.CodeBufferOffsetExpression(b_pack_ptr_, context->weight_name(), context->weight_offset_name(),
                                       context->weight_size_name(), b_pack_ptr_size_);
  if (b_batch_ == C1NUM) {
    if (params_.b_transpose_) {
      init_code.CodeFunction("RowMajor2Col8MajorFp16", input_b_str, input_b_pack_str_, params_.col_, params_.deep_,
                             false);
    } else {
      init_code.CodeFunction("RowMajor2Row8MajorFp16", input_b_str, input_b_pack_str_, params_.deep_, params_.col_,
                             false);
    }
  } else {
    init_code << "  for (int i = 0; i < " << b_batch_ << "; i++) {\n"
              << "    float16_t *src = " << input_b_str << " + i * " << params_.deep_ * params_.col_ << ";\n"
              << "    float16_t *dst = " << input_b_pack_str_ << " + i * " << params_.deep_ * params_.col_align_
              << ";\n";
    if (params_.b_transpose_) {
      init_code << "    RowMajor2Col8MajorFp16(src, dst, " << params_.col_ << ", " << params_.deep_ << ", false);\n";
    } else {
      init_code << "    RowMajor2Row8MajorFp16(src, dst, " << params_.deep_ << ", " << params_.col_ << ", false);\n";
    }
    init_code << "  }\n";
  }
  context->AppendInitWeightSizeCode(b_pack_ptr_size_);
  context->AppendInitCode(init_code.str());
  return RET_OK;
}

int MatMulDynamicFP16BaseCoder::CollectFilesForTarget(CoderContext *const context) {
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
}  // namespace mindspore::lite::micro::nnacl
