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

#include "coder/opcoders/nnacl/int8/affine_int8_coder.h"
#include <string>
#include <algorithm>
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"
#include "coder/log.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/nnacl/int8/matmul_int8_coder.h"
#include "tools/converter/micro/coder/wrapper/base/affine_wrapper.h"
#include "tools/converter/micro/coder/opcoders/op_coder_builder.h"
#include "src/litert/kernel/cpu/fp32/affine_fp32.h"
#include "coder/opcoders/nnacl/fp32/affine_fp32_coder.h"

using mindspore::schema::PrimitiveType_Affine;

namespace mindspore::lite::micro::nnacl {
int AffineInt8Coder::ReSize(CoderContext *const context) { return RET_OK; }

void AffineInt8Coder::PrepareFirstRunCode(CoderContext *const context) {
  auto global_code = context->global_code_blocks();
  std::string to_add_first_run_code = "bool first_run = true;";
  // only add once at global scope
  if (std::find(global_code.begin(), global_code.end(), to_add_first_run_code) == global_code.end()) {
    global_code.emplace_back(to_add_first_run_code);
    context->set_global_code_blocks(global_code);
  }

  auto after_inference_code = context->after_inference_code_blocks();
  std::string to_add_after_inference_code = "first_run = false;";
  // only add once after inference code
  if (std::find(after_inference_code.begin(), after_inference_code.end(), to_add_after_inference_code) ==
      after_inference_code.end()) {
    after_inference_code.emplace_back(to_add_after_inference_code);
    context->set_after_inference_code_blocks(after_inference_code);
  }
}

int AffineInt8Coder::PrepareSpliceOp() {
  affine_param_ = reinterpret_cast<AffineParameter *>(parameter_);
  auto input_shape = input_tensors_.front()->shape();
  int out_dim = affine_param_->output_dim_;
  int context_min = affine_param_->context_[0];
  int context_max = affine_param_->context_[affine_param_->context_size_ - 1];
  full_input_ =
    allocator_->MallocTensor(kNumberTypeInt8, {1, input_shape.at(1) - (context_max - context_min), out_dim});
  increment_input_ = allocator_->MallocTensor(kNumberTypeInt8, {1, 1, out_dim});

  // init splice param
  splice_param_ = new SpliceWrapperParam();
  if (affine_param_->context_size_ > MAX_SHAPE_SIZE) {
    MS_LOG(ERROR) << "Context size should be less than MAX_SHAPE_SIZE.";
    return RET_ERROR;
  }
  for (int i = 0; i < affine_param_->context_size_; i++) {
    splice_param_->context[i] = affine_param_->context_[i];
  }
  splice_param_->context_size = affine_param_->context_size_;
  splice_param_->src_to_dst_row_offset =
    *std::min_element(affine_param_->context_, affine_param_->context_ + affine_param_->context_size_);

  std::vector<int> src_shape = input_tensors_.at(kInputIndex)->shape();
  std::vector<int> dst_shape = full_input_->shape();
  if (src_shape.size() != dst_shape.size() || src_shape.size() != kInputSize2 || dst_shape.size() != kInputSize2) {
    MS_LOG(ERROR) << "splice kernel src_shape size not equal to dst_shape size";
    return RET_ERROR;
  }
  // src and dst shape: {batch, row, col}
  splice_param_->src_row = src_shape.at(kernel::kInputRow);
  splice_param_->src_col = src_shape.at(kernel::kInputCol);
  splice_param_->dst_row = dst_shape.at(kernel::kInputRow);
  splice_param_->dst_col = dst_shape.at(kernel::kInputCol);
  if (splice_param_->src_col * splice_param_->context_size != splice_param_->dst_col) {
    MS_LOG(ERROR) << "splice kernel src_col not match dst_col";
    return RET_ERROR;
  }
  for (int r = 0; r < splice_param_->dst_row; ++r) {
    for (int off = 0; off < affine_param_->context_size_; ++off) {
      int r_off = r - splice_param_->src_to_dst_row_offset + affine_param_->context_[off];
      if (r_off < 0) {
        MS_LOG(ERROR) << "splice row index out of range";
        return RET_ERROR;
      }
      if (r_off >= splice_param_->src_row) {
        MS_LOG(ERROR) << "splice row index out of range";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

int AffineInt8Coder::PrepareFullMatmulOp(CoderContext *const context) {
  if (input_tensors_.size() < 2) {
    MS_LOG(ERROR) << "wrong affine input size";
    return RET_ERROR;
  }

  std::vector<lite::Tensor *> matmul_inputs;
  full_input_->set_quant_params(input_tensors_.at(0)->quant_params());
  if (input_tensors_.size() == kernel::kAffineMaxInputNum) {
    matmul_inputs = {full_input_, input_tensors_.at(kWeightIndex), input_tensors_.at(kBiasIndex)};
  } else {
    matmul_inputs = {full_input_, input_tensors_.at(kWeightIndex)};
  }
  affine_param_->matmul_parameter_->act_type_ = static_cast<ActType>(affine_param_->activation_type_);
  auto matmul_primitive = CreateMatmulPrimitive();
  allocated_.emplace_back(matmul_primitive);
  matmul_node_ = CreateMatmulNode(matmul_primitive, node_->name_);
  OpCoderBuilder builder;
  full_matmul_coder_ = builder.inputs(matmul_inputs)
                         .outputs(output_tensors_)
                         .node(node_)
                         .parameter(reinterpret_cast<OpParameter *>(affine_param_->matmul_parameter_))
                         .target(target_)
                         .support_parallel(support_parallel_)
                         .data_type(kNumberTypeInt8)
                         .build(schema_version_);
  delete (node_);
  MS_CHECK_RET_CODE(full_matmul_coder_->Prepare(context), "full matmul coder prepare failed.");
  previous_output_ = reinterpret_cast<int8_t *>(
    allocator_->MallocWeightTensor(kNumberTypeInt8, output_tensor_->ElementsNum() * sizeof(int8_t), kWorkspace));
  NNaclInt8Serializer init_code;
  init_code.CodeMallocExpression(previous_output_, output_tensor_->ElementsNum() * sizeof(int8_t));
  init_code.CodeFunction("memset", previous_output_, 0, output_tensor_->ElementsNum() * sizeof(int8_t));
  context->AppendInitCode(init_code.str());
  return RET_OK;
}

int AffineInt8Coder::GenFullAffineCode(CoderContext *context, std::string *code) {
  NNaclInt8Serializer splice_code;
  splice_code.CodeStruct("splice_param", *splice_param_);
  splice_code.CodeFunction("FullSpliceRunInt8", input_tensors_.at(0), full_input_, "&splice_param");
  MS_CHECK_RET_CODE(full_matmul_coder_->DoCode(context), "full matmul coder docode failed.");
  auto full_matmul_code_block = context->code_blocks().back();

  // memcpy to previous output
  NNaclInt8Serializer memcpy_code;
  memcpy_code.CodeFunction("memcpy", previous_output_, output_tensor_,
                           std::to_string(matmul_row_ * matmul_col_ * sizeof(int8_t)));

  std::string affine_code = splice_code.str() + full_matmul_code_block + memcpy_code.str();
  auto all_code = context->code_blocks();
  all_code.pop_back();
  context->set_code_blocks(all_code);

  std::string start_if_line = "if (first_run) {\n";
  std::string end_line = "}";
  *code = start_if_line + affine_code + end_line;
  return RET_OK;
}

int AffineInt8Coder::GenIncrementAffineCode(CoderContext *context, std::string *code) {
  NNaclInt8Serializer splice_code;
  splice_code.CodeStruct("splice_param", *splice_param_);
  splice_code.CodeFunction("IncrementSpliceRunInt8", input_tensors_.at(0), increment_input_, "&splice_param");
  MS_CHECK_RET_CODE(increment_matmul_coder_->DoCode(context), "increment matmul coder docode failed.");

  // memcpy to previous output
  NNaclInt8Serializer memcpy_code;
  memcpy_code.CodeFunction(
    "memcpy", output_tensor_,
    MemoryAllocator::GetInstance()->GetRuntimeAddr(previous_output_, true) + "+" + std::to_string(matmul_col_),
    std::to_string((matmul_row_ - 1) * matmul_col_ * sizeof(int8_t)));
  memcpy_code.CodeFunction("memcpy", previous_output_,
                           MemoryAllocator::GetInstance()->GetRuntimeAddr(output_tensor_, true) + "+" +
                             std::to_string((matmul_row_ - 1) * matmul_col_),
                           std::to_string(matmul_col_ * sizeof(int8_t)));
  memcpy_code.CodeFunction("memcpy", previous_output_, output_tensor_,
                           std::to_string(matmul_row_ * matmul_col_ * sizeof(int8_t)));

  auto increment_code_block = context->code_blocks().back();
  std::string affine_code = splice_code.str() + increment_code_block + memcpy_code.str();
  auto all_code = context->code_blocks();
  all_code.pop_back();
  context->set_code_blocks(all_code);

  std::string start_if_line = " else {\n";
  std::string end_line = "}";
  *code = start_if_line + affine_code + end_line;
  return RET_OK;
}

int AffineInt8Coder::PrepareIncreMatmulOp(CoderContext *const context) {
  if (input_tensors_.size() < 2) {
    MS_LOG(ERROR) << "wrong affine input size";
    return RET_ERROR;
  }

  std::vector<lite::Tensor *> matmul_inputs;
  increment_input_->set_quant_params(input_tensors_.at(0)->quant_params());
  if (input_tensors_.size() == kernel::kAffineMaxInputNum) {
    matmul_inputs = {increment_input_, input_tensors_.at(kWeightIndex), input_tensors_.at(kBiasIndex)};
  } else {
    matmul_inputs = {increment_input_, input_tensors_.at(kWeightIndex)};
  }
  matmul_col_ = output_tensors_.front()->shape().at(2);
  matmul_row_ = output_tensors_.front()->shape().at(1);
  increment_output_ = allocator_->MallocTensor(kNumberTypeInt8, {1, 1, matmul_col_});
  increment_output_->set_quant_params(output_tensor_->quant_params());
  auto matmul_param = reinterpret_cast<MatMulParameter *>(malloc(sizeof(*affine_param_->matmul_parameter_)));
  if (matmul_param == nullptr) {
    MS_LOG(ERROR) << "malloc matmul_param failed.";
    return RET_ERROR;
  }
  allocated_.emplace_back(matmul_param);

  if (EOK != memcpy_s(matmul_param, sizeof(*affine_param_->matmul_parameter_), affine_param_->matmul_parameter_,
                      sizeof(*affine_param_->matmul_parameter_))) {
    MS_LOG(ERROR) << "memcpy_s failed.";
    return RET_MEMORY_FAILED;
  }
  matmul_param->act_type_ = static_cast<ActType>(affine_param_->activation_type_);
  OpCoderBuilder builder;
  increment_matmul_coder_ = builder.inputs(matmul_inputs)
                              .outputs({increment_output_})
                              .node(matmul_node_)
                              .parameter(reinterpret_cast<OpParameter *>(matmul_param))
                              .target(target_)
                              .support_parallel(support_parallel_)
                              .data_type(kNumberTypeInt8)
                              .build(schema_version_);
  delete (node_);
  MS_CHECK_RET_CODE(increment_matmul_coder_->Prepare(context), "increment matmul coder prepare failed.");
  return RET_OK;
}

int AffineInt8Coder::Prepare(CoderContext *const context) {
  PrepareFirstRunCode(context);
  MS_CHECK_RET_CODE(PrepareSpliceOp(), "prepare splice op failed.");
  MS_CHECK_RET_CODE(PrepareFullMatmulOp(context), "prepare full matmul op failed.");
  MS_CHECK_RET_CODE(PrepareIncreMatmulOp(context), "prepare increment matmul op failed.");
  return ReSize(context);
}

int AffineInt8Coder::DoCode(CoderContext *context) {
  Collect(context,
          {
            "wrapper/base/affine_wrapper.h",
          },
          {"affine_wrapper.c"});
  std::string full_affine_code, increment_affine_code;
  MS_CHECK_RET_CODE(GenFullAffineCode(context, &full_affine_code), "GenFullAffineCode failed.");
  MS_CHECK_RET_CODE(GenIncrementAffineCode(context, &increment_affine_code), "GenIncrementAffineCode failed.");
  std::string affine_code = full_affine_code + increment_affine_code;
  context->AppendCode(affine_code);
  return RET_OK;
}  // namespace mindspore::lite::micro::nnacl

REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_Affine, CPUOpCoderCreator<AffineInt8Coder>)
}  // namespace mindspore::lite::micro::nnacl
