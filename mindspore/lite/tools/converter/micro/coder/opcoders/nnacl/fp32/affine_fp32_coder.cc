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

#include "coder/opcoders/nnacl/fp32/affine_fp32_coder.h"
#include <string>
#include <algorithm>
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/log.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/nnacl/fp32/matmul_fp32_coder.h"
#include "tools/converter/micro/coder/wrapper/base/affine_wrapper.h"
#include "src/litert/kernel/cpu/fp32/affine_fp32.h"
#include "tools/converter/micro/coder/opcoders/op_coder_builder.h"
#include "coder/utils/common.h"

using mindspore::schema::PrimitiveType_Affine;

namespace mindspore::lite::micro::nnacl {
int AffineFP32Coder::ReSize(CoderContext *const context) { return RET_OK; }

int AffineFP32Coder::PrepareSpliceOp() {
  affine_param_ = reinterpret_cast<AffineParameter *>(parameter_);
  auto input_shape = input_tensors_.front()->shape();
  int out_dim = affine_param_->output_dim_;
  int context_min = affine_param_->context_[0];
  int context_max = affine_param_->context_[affine_param_->context_size_ - 1];
  std::vector<int> splice_output_shape = {1, input_shape.at(1) - (context_max - context_min), out_dim};
  full_input_ = allocator_->MallocTensor(kNumberTypeFloat32, splice_output_shape);

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

  // check if splice_param context out of range
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

void *CreateMatmulPrimitive() {
  flatbuffers::FlatBufferBuilder fbb(k1024);
  auto val_offset = schema::CreateMatMulFusion(fbb);
  auto prim_offset = schema::CreatePrimitive(fbb, schema::PrimitiveType_MatMulFusion, val_offset.o);
  fbb.Finish(prim_offset);
  auto tmp_buf = fbb.GetBufferPointer();
  void *prim_buf = malloc(fbb.GetSize());
  if (prim_buf == nullptr) {
    return nullptr;
  }
  memcpy(prim_buf, tmp_buf, fbb.GetSize());
  fbb.Clear();
  return prim_buf;
}

LiteGraph::Node *CreateMatmulNode(void *prim_buf, const std::string &name) {
  auto primitive = flatbuffers::GetRoot<schema::Primitive>(prim_buf);
  auto node = new LiteGraph::Node();
  node->primitive_ = primitive;
  node->name_ = name;
  return node;
}

int AffineFP32Coder::Prepare(CoderContext *const context) {
  int ret = PrepareSpliceOp();
  if (RET_OK != ret) {
    MS_LOG(ERROR) << "prepare splice op failed.";
    return ret;
  }

  // FullMatmulKernelCreate
  if (input_tensors_.size() < DIMENSION_2D) {
    MS_LOG(ERROR) << "wrong affine input size";
    return RET_ERROR;
  }

  std::vector<lite::Tensor *> matmul_inputs;
  // For affine op, the possible inputs are:
  // { input, weight, bias}
  // { input, weight}
  if (input_tensors_.size() == kernel::kAffineMaxInputNum) {
    matmul_inputs = {full_input_, input_tensors_.at(kWeightIndex), input_tensors_.at(kBiasIndex)};
  } else {
    matmul_inputs = {full_input_, input_tensors_.at(kWeightIndex)};
  }
  auto *matmul_param = new MatMulParameter(*affine_param_->matmul_parameter_);
  matmul_param->act_type_ = static_cast<ActType>(affine_param_->activation_type_);
  affine_param_->matmul_parameter_->act_type_ = static_cast<ActType>(affine_param_->activation_type_);
  OpCoderBuilder builder;

  matmul_primitive_ = CreateMatmulPrimitive();
  matmul_node_ = CreateMatmulNode(matmul_primitive_, node_->name_);

  matmul_coder_ = builder.inputs(matmul_inputs)
                    .outputs(output_tensors_)
                    .node(matmul_node_)
                    .parameter(reinterpret_cast<OpParameter *>(matmul_param))
                    .target(target_)
                    .support_parallel(support_parallel_)
                    .data_type(kNumberTypeFloat32)
                    .build(schema_version_);
  MS_CHECK_RET_CODE(matmul_coder_->Prepare(context), "matmul coder prepare failed.");
  return ReSize(context);
}

std::string AffineFP32Coder::GenSpliceCode() {
  NNaclFp32Serializer splice_code;
  splice_code.CodeStruct("splice_param", *splice_param_);
  splice_code.CodeFunction("FullSpliceRunFp32", input_tensors_.at(0), full_input_, "&splice_param");
  return splice_code.str();
}

int AffineFP32Coder::DoCode(CoderContext *context) {
  Collect(context,
          {
            "wrapper/base/affine_wrapper.h",
          },
          {"affine_wrapper.c"});
  std::string splice_code = GenSpliceCode();
  MS_CHECK_RET_CODE(matmul_coder_->DoCode(context), "matmul coder docode failed.");
  auto all_code_blocks = context->code_blocks();
  auto matmul_code_block = all_code_blocks.back();
  all_code_blocks[all_code_blocks.size() - 1] = splice_code + matmul_code_block;
  context->set_code_blocks(all_code_blocks);
  return RET_OK;
}  // namespace mindspore::lite::micro::nnacl

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Affine, CPUOpCoderCreator<AffineFP32Coder>)
}  // namespace mindspore::lite::micro::nnacl
