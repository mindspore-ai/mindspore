/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/fp32/matmul_fp32_base.h"
#include <algorithm>
#include "nnacl/fp32/matmul_fp32.h"
#include "nnacl/fp32/pack_fp32.h"
#include "nnacl/fp32/pack_fp32_opt.h"
#if defined(PARALLEL_INFERENCE) && defined(ENABLE_MINDRT)
#include "thread/parallel_thread_pool_manager.h"
#endif

using mindspore::lite::kCHWDimNumber;
using mindspore::lite::kHWDimNumber;
using mindspore::lite::kNCHWDimNumber;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_MatMulFusion;
namespace mindspore::kernel {
int MatmulRun(void *cdata, int task_id, float, float) {
  CHECK_NULL_RETURN(cdata);
  auto op = reinterpret_cast<const MatmulFp32BaseCPUKernel *>(cdata);
  auto error_code = (op->*(op->parallel_fun_))(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "MatmulRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

MatmulFp32BaseCPUKernel::~MatmulFp32BaseCPUKernel() {
  // packed const-matrix will be delete by framework.
  if (out_need_aligned_ && output_data_ != nullptr) {
    free(output_data_);
    output_data_ = nullptr;
  }
  if (matrix_c_.pack_ptr != nullptr) {
    free(matrix_c_.pack_ptr);
    matrix_c_.pack_ptr = nullptr;
  }
  if (params_->a_const_) {
    if (is_sharing_pack_) {
      lite::PackWeightManager::GetInstance()->Free(matrix_a_.pack_ptr);
    } else {
      free(matrix_a_.pack_ptr);
    }
  }
  if (params_->b_const_) {
    if (!matrix_b_.need_pack && weight_is_packed_) {
      return;
    }
    if (is_sharing_pack_) {
      lite::PackWeightManager::GetInstance()->Free(matrix_b_.pack_ptr);
    } else {
      free(matrix_b_.pack_ptr);
    }
  }
}

void MatmulFp32BaseCPUKernel::InitGlobalVariable() {
  matrix_a_.need_pack = true;
  matrix_b_.need_pack = !weight_is_packed_;
  matrix_a_pack_fun_ = params_->a_transpose_ ? RowMajor2Row12MajorParallel : RowMajor2Col12MajorParallel;
  matrix_b_pack_fun_ = params_->b_transpose_ ? RowMajor2Col8MajorParallel : RowMajor2Row8MajorParallel;
  row_tile_ = C12NUM;
  col_tile_ = C8NUM;
  col_min_unit_ = C8NUM;
}

int MatmulFp32BaseCPUKernel::PackMatrixAImplOpt() {
  MS_LOG(ERROR) << "Matmul: don't support optimized-packing, only support single-thread currently.";
  return RET_ERROR;
}

int MatmulFp32BaseCPUKernel::ParallelRunByBatch(int task_id) const {
  int start_batch = task_id * batch_stride_;
  int end_batch = MSMIN(params_->batch, start_batch + batch_stride_);
  int func_flag{0};
  if (params_->row_ == 1) {
    func_flag += (!params_->b_const_ && params_->col_ <= C128NUM) ? C2NUM : C1NUM;
  }

  for (int index = start_batch; index < end_batch; ++index) {
    const float *a = matrix_a_.pack_ptr + a_offset_[index] * params_->row_align_ * params_->deep_;
    const float *b = matrix_b_.pack_ptr + b_offset_[index] * params_->deep_ * params_->col_align_;
    float *c = output_data_ + index * params_->row_ * col_step_;

    auto bias = (matrix_c_.pack_ptr == nullptr) ? nullptr : matrix_c_.pack_ptr;
    if (func_flag == 0) {
      MatMulOpt(a, b, c, bias, params_->act_type_, params_->deep_, params_->row_, col_step_, params_->col_,
                OutType_Nhwc);
    } else if (func_flag == C1NUM) {
      MatVecMulFp32Block8(a, b, c, bias, params_->act_type_, params_->deep_, col_step_);
    } else {
      MatVecMulNoPackFp32(a, b, c, bias, params_->act_type_, params_->deep_, col_step_, col_step_);
    }
  }
  return RET_OK;
}

int MatmulFp32BaseCPUKernel::ParallelRunByRow(int task_id) const {
  (void)task_id;
  return RET_ERROR;
}

int MatmulFp32BaseCPUKernel::ParallelRunByOC(int task_id) const {
  if (task_id < 0 || task_id >= thread_num_) {
    MS_LOG(ERROR) << "task_id " << task_id << " is out of range, node is " << name_;
    return RET_ERROR;
  }
  int start_oc = split_points_[task_id];
  int end_oc = col_step_;
  if (task_id < (thread_num_ - 1)) {
    end_oc = split_points_[task_id + 1];
  }
  int compute_oc = end_oc - start_oc;
  if (compute_oc <= 0) {
    return RET_OK;
  }
  int func_flag{0};
  if (params_->row_ == 1) {
    func_flag += (!params_->b_const_ && params_->col_ <= C128NUM) ? C2NUM : C1NUM;
  }
  int b_stride = func_flag == C2NUM ? 1 : params_->deep_;
  for (int i = 0; i < params_->batch; ++i) {
    auto a = matrix_a_.pack_ptr + a_offset_[i] * params_->row_align_ * params_->deep_;
    auto b = matrix_b_.pack_ptr + b_offset_[i] * params_->deep_ * params_->col_align_ + start_oc * b_stride;
    auto c = output_data_ + i * params_->row_ * col_step_ + start_oc;
    auto bias = (matrix_c_.pack_ptr == nullptr) ? nullptr : matrix_c_.pack_ptr + start_oc;
    if (func_flag == 0) {
      MatMulOpt(a, b, c, bias, params_->act_type_, params_->deep_, params_->row_, compute_oc, params_->col_,
                OutType_Nhwc);
    } else if (func_flag == C1NUM) {
      MatVecMulFp32Block8(a, b, c, bias, params_->act_type_, params_->deep_, compute_oc);
    } else {
      MatVecMulNoPackFp32(a, b, c, bias, params_->act_type_, params_->deep_, compute_oc, col_step_);
    }
  }
  return RET_OK;
}

bool MatmulFp32BaseCPUKernel::CheckThreadCuttingByRow() { return false; }

int MatmulFp32BaseCPUKernel::BackupConstMatrix(MatrixInfo *matrix_info, int index) {
  MS_CHECK_TRUE_MSG(index < static_cast<int>(in_tensors_.size()), RET_ERROR, "matrix is not existing.");
  auto element_num = in_tensors_[index]->ElementsNum();
  MS_CHECK_TRUE_MSG(element_num > 0, RET_ERROR, "matrix is invalid.");
  matrix_info->origin_ptr = reinterpret_cast<float *>(ms_context_->allocator->Malloc(element_num * sizeof(float)));
  MS_CHECK_TRUE_MSG(matrix_info->origin_ptr != nullptr, RET_ERROR, "matrix is invalid.");
  auto src_ptr = in_tensors_[index]->data();
  MS_CHECK_TRUE_MSG(src_ptr != nullptr, RET_ERROR, "matrix is invalid.");
  (void)memcpy(matrix_info->origin_ptr, src_ptr, element_num * sizeof(float));
  matrix_info->has_origin = true;
  return RET_OK;
}

int MatmulFp32BaseCPUKernel::PackMatrixA() {
  if (!params_->a_const_) {
    if (!matrix_a_.need_pack) {
      matrix_a_.pack_ptr = reinterpret_cast<float *>(in_tensors_[FIRST_INPUT]->data());
      return RET_OK;
    }
    if (op_parameter_->is_train_session_) {
      matrix_a_.pack_ptr = reinterpret_cast<float *>(workspace());
    } else {
      matrix_a_.pack_ptr =
        reinterpret_cast<float *>(ms_context_->allocator->Malloc(matrix_a_.pack_size * sizeof(float)));
    }
  } else {
    bool is_packed = false;
    void *data = nullptr;
    if (is_sharing_pack_) {
      data = lite::PackWeightManager::GetInstance()->GetPackData(
        in_tensors()[FIRST_INPUT]->data(), static_cast<size_t>(matrix_a_.pack_size) * sizeof(float), &is_packed);
    } else {
      data = malloc(static_cast<size_t>(matrix_a_.pack_size) * sizeof(float));
    }
    matrix_a_.pack_ptr = reinterpret_cast<float *>(data);
    if (matrix_a_.pack_ptr == nullptr) {
      MS_LOG(ERROR) << "matrix a pack ptr is nullptr.";
      return RET_ERROR;
    }
    if (is_packed) {
      return RET_OK;
    }
  }
  if (pack_opt_) {
    return PackMatrixAImplOpt();  // currently, only arm64 support.
  }
  return PackMatrixAImpl();
}

int MatmulFp32BaseCPUKernel::PackMatrixAImpl() {
  auto src_ptr =
    matrix_a_.has_origin ? matrix_a_.origin_ptr : reinterpret_cast<float *>(in_tensors_[FIRST_INPUT]->data());
  MS_CHECK_TRUE_MSG(src_ptr != nullptr, RET_ERROR, "matrix-a source ptr is a nullptr.");
  MS_CHECK_TRUE_MSG(matrix_a_.pack_ptr != nullptr, RET_ERROR, "matrix-a pack ptr is a nullptr.");
  MS_CHECK_TRUE_MSG(matrix_a_pack_fun_ != nullptr, RET_ERROR, "matrix-a func is a nullptr.");
  for (int i = 0; i < a_batch_; i++) {
    const float *src = src_ptr + i * params_->deep_ * params_->row_;
    float *dst = matrix_a_.pack_ptr + i * params_->deep_ * params_->row_align_;
    if (params_->a_transpose_) {
      matrix_a_pack_fun_(src, dst, params_->deep_, params_->row_, 0, params_->deep_);
    } else {
      matrix_a_pack_fun_(src, dst, params_->row_, params_->deep_, 0, params_->row_);
    }
  }
  return RET_OK;
}

void MatmulFp32BaseCPUKernel::FreePackedMatrixA() {
  if (matrix_a_.need_pack && !op_parameter_->is_train_session_ && matrix_a_.pack_ptr != nullptr) {
    ms_context_->allocator->Free(matrix_a_.pack_ptr);
  }
  matrix_a_.pack_ptr = nullptr;
}

int MatmulFp32BaseCPUKernel::PackMatrixB() {
  if (!params_->b_const_) {
    if (!matrix_b_.need_pack) {
      matrix_b_.pack_ptr = reinterpret_cast<float *>(in_tensors_[SECOND_INPUT]->data());
      return RET_OK;
    }
    if (op_parameter_->is_train_session_) {
      matrix_b_.pack_ptr = reinterpret_cast<float *>(workspace()) + matrix_a_.pack_size;
    } else {
      matrix_b_.pack_ptr =
        reinterpret_cast<float *>(ms_context_->allocator->Malloc(matrix_b_.pack_size * sizeof(float)));
    }
  } else {
    if (!matrix_b_.need_pack && weight_is_packed_) {
      matrix_b_.pack_ptr = reinterpret_cast<float *>(in_tensors_[SECOND_INPUT]->data());
      return RET_OK;
    }
    bool is_packed = false;
    void *data = nullptr;
    if (is_sharing_pack_) {
      data = lite::PackWeightManager::GetInstance()->GetPackData(
        in_tensors()[SECOND_INPUT]->data(), static_cast<size_t>(matrix_b_.pack_size) * sizeof(float), &is_packed);
    } else {
      data = malloc(static_cast<size_t>(matrix_b_.pack_size) * sizeof(float));
    }
    matrix_b_.pack_ptr = reinterpret_cast<float *>(data);
    if (matrix_b_.pack_ptr == nullptr) {
      MS_LOG(ERROR) << "matrix b pack ptr is nullptr.";
      return RET_ERROR;
    }
    if (is_packed) {
      return RET_OK;
    }
  }
  return PackMatrixBImpl();
}

int PackMatrixBRun(void *cdata, int task_id, float, float) {
  CHECK_NULL_RETURN(cdata);
  auto op = reinterpret_cast<const MatmulFp32BaseCPUKernel *>(cdata);
  auto error_code = op->PackMatrixBParallelRunByBatch(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "PackMatrixBRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int MatmulFp32BaseCPUKernel::PackMatrixBParallelRunByBatch(int task_id) const {
  int start = task_id * pack_b_stride_;
  if (params_->b_transpose_) {
    int end = MSMIN(params_->col_, start + pack_b_stride_);
    matrix_b_pack_fun_(pack_b_src_, pack_b_dst_, params_->col_, params_->deep_, start, end);
  } else {
    int end = MSMIN(params_->deep_, start + pack_b_stride_);
    matrix_b_pack_fun_(pack_b_src_, pack_b_dst_, params_->deep_, params_->col_, start, end);
  }
  return RET_OK;
}

int MatmulFp32BaseCPUKernel::PackMatrixBImpl() {
  auto src_ptr = matrix_b_.has_origin
                   ? matrix_b_.origin_ptr
                   : (conv1x1_origin_weight_ != nullptr ? conv1x1_origin_weight_
                                                        : reinterpret_cast<float *>(in_tensors_[SECOND_INPUT]->data()));
  MS_CHECK_TRUE_MSG(src_ptr != nullptr, RET_ERROR, "matrix-b source ptr is a nullptr.");
  MS_CHECK_TRUE_MSG(matrix_b_.pack_ptr != nullptr, RET_ERROR, "matrix-b pack ptr is a nullptr.");
  MS_CHECK_TRUE_MSG(matrix_b_pack_fun_ != nullptr, RET_ERROR, "matrix-b func is a nullptr.");
  for (int i = 0; i < b_batch_; i++) {
    if (params_->b_transpose_) {
      pack_b_stride_ = UP_DIV(params_->col_, op_parameter_->thread_num_);
    } else {
      pack_b_stride_ = UP_DIV(params_->deep_, op_parameter_->thread_num_);
    }
    pack_b_src_ = src_ptr + i * params_->deep_ * params_->col_;
    pack_b_dst_ = matrix_b_.pack_ptr + i * params_->deep_ * params_->col_align_;
    auto ret = ParallelLaunch(this->ms_context_, PackMatrixBRun, this, op_parameter_->thread_num_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "MatmulRun failed in split by batch";
      return ret;
    }
  }
  return RET_OK;
}

void MatmulFp32BaseCPUKernel::FreePackedMatrixB() {
  if (matrix_b_.need_pack && !op_parameter_->is_train_session_ && matrix_b_.pack_ptr != nullptr) {
    ms_context_->allocator->Free(matrix_b_.pack_ptr);
  }
  matrix_b_.pack_ptr = nullptr;
}

int MatmulFp32BaseCPUKernel::PackBiasMatrix() {
  if (in_tensors_.size() != FOURTH_INPUT) {
    return RET_OK;
  }
  if (matrix_c_.has_packed) {
    if (matrix_c_.pack_size < params_->col_align_) {
      MS_LOG(ERROR) << "matmul don't support that column is dynamic.";
      return RET_ERROR;
    }
    return RET_OK;
  }
  auto bias_tensor = in_tensors_[THIRD_INPUT];
  if (bias_tensor == nullptr) {
    MS_LOG(ERROR) << "bias_tensor invalid";
    return RET_ERROR;
  }
  auto bias_src =
    matrix_c_.has_origin
      ? matrix_c_.origin_ptr
      : (conv1x1_origin_bias_ != nullptr ? conv1x1_origin_bias_ : reinterpret_cast<float *>(bias_tensor->data()));
  MS_CHECK_TRUE_MSG(bias_src != nullptr, RET_ERROR, "matrix-c is a nullptr.");
  auto bias_num = bias_tensor->ElementsNum();
  MS_CHECK_TRUE_MSG(bias_num > 0 && params_->col_align_ >= bias_num, RET_ERROR, "matrix-c is invalid.");
  matrix_c_.pack_size = params_->col_align_;
  matrix_c_.pack_ptr = reinterpret_cast<float *>(malloc(static_cast<size_t>(matrix_c_.pack_size) * sizeof(float)));
  MS_CHECK_TRUE_MSG(matrix_c_.pack_ptr != nullptr, RET_ERROR, "matrix-c malloc failed.");
  if (bias_num == 1) {
    for (int i = 0; i < matrix_c_.pack_size; ++i) {
      matrix_c_.pack_ptr[i] = bias_src[0];
    }
  } else {
    (void)memcpy(matrix_c_.pack_ptr, bias_src, bias_num * static_cast<int>(sizeof(float)));
    (void)memset(matrix_c_.pack_ptr + bias_num, 0, (matrix_c_.pack_size - bias_num) * sizeof(float));
  }
  if (matrix_c_.has_origin) {
    ms_context_->allocator->Free(matrix_c_.origin_ptr);
    matrix_c_.origin_ptr = nullptr;
    matrix_c_.has_origin = false;
  }
  return RET_OK;
}

int MatmulFp32BaseCPUKernel::ParallelRunIsNotPackByBatch(int task_id) const {
  int start_batch = task_id * batch_stride_;
  int end_batch = MSMIN(params_->batch, start_batch + batch_stride_);
  float bias = 0;
  if (matrix_c_.pack_ptr != nullptr) {
    bias = matrix_c_.pack_ptr[0];
  }
  for (int index = start_batch; index < end_batch; ++index) {
    const float *a = matrix_a_.pack_ptr + a_offset_[index] * params_->row_ * params_->deep_;
    const float *b = matrix_b_.pack_ptr + b_offset_[index] * params_->deep_ * params_->col_;
    float *c = output_data_ + index * params_->row_ * params_->col_;
    gemmIsNotPackFun(a, b, c, &bias, params_->row_, params_->deep_, params_->act_type_);
  }
  return RET_OK;
}

int MatmulFp32BaseCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  MS_CHECK_TRUE_MSG(in_tensors_[FIRST_INPUT]->data_type() == kNumberTypeFloat32, RET_ERROR,
                    "matrix-a's data type is invalid.");
  MS_CHECK_TRUE_MSG(in_tensors_[SECOND_INPUT]->data_type() == kNumberTypeFloat32, RET_ERROR,
                    "matrix-b's data type is invalid.");
  if (in_tensors_.size() == FOURTH_INPUT) {
    MS_CHECK_TRUE_MSG(in_tensors_[THIRD_INPUT]->IsConst() || (conv1x1_origin_bias_ != nullptr), RET_ERROR,
                      "matrix-c must be const when existing.");
    MS_CHECK_TRUE_MSG(in_tensors_[THIRD_INPUT]->data_type() == kNumberTypeFloat32, RET_ERROR,
                      "matrix-c's data type is invalid.");
  }
  auto act_type = params_->act_type_;
  if (act_type != ActType_No && act_type != ActType_Relu && act_type != ActType_Relu6) {
    MS_LOG(ERROR) << "matmul don't support the act-type: " << act_type;
    return RET_ERROR;
  }
  auto ret = InitParameter();
  MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "Init parameters failed.");
  if (params_->a_const_) {
    ret = PackMatrixA();
    MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "pack const-matrix a failed.");
    matrix_a_.has_packed = true;
  }
  if (params_->b_const_) {
    ret = PackMatrixB();
    MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "pack const-matrix b failed.");
    matrix_b_.has_packed = true;
  }
  if (!InferShapeDone()) {
    if (in_tensors_.size() == FOURTH_INPUT && !op_parameter_->is_train_session_) {
      ret = BackupConstMatrix(&matrix_c_, THIRD_INPUT);
      MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "backup matrix-c failed.");
    }
    return RET_OK;
  }
  return RET_OK;
}

int MatmulFp32BaseCPUKernel::FullConnectionPrepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  params_->a_const_ = in_tensors_[kInputIndex]->IsConst() && !op_parameter_->is_train_session_;
  params_->b_const_ = in_tensors_[kWeightIndex]->IsConst() && !op_parameter_->is_train_session_;

  if (params_->a_const_ || InferShapeDone()) {
    auto a_shape = in_tensors_.at(0)->shape();
    CHECK_LESS_RETURN(a_shape.size(), C2NUM);
    params_->row_ = a_shape[0];
    params_->deep_ = a_shape[1];
  }

  if (params_->b_const_ || InferShapeDone()) {
    auto b_shape = in_tensors_.at(1)->shape();
    CHECK_LESS_RETURN(b_shape.size(), C2NUM);
    params_->col_ = b_shape[0];
    params_->deep_ = b_shape[1];
  }

  params_->batch = 1;
  a_offset_.resize(params_->batch, 0);
  b_offset_.resize(params_->batch, 0);
  a_batch_ = 1;
  b_batch_ = 1;
  params_->a_transpose_ = false;
  params_->b_transpose_ = true;

  auto ret = MatmulFp32BaseCPUKernel::Prepare();
  if (ret != RET_OK) {
    return ret;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return FullConnectionReSize();
}

void MatmulFp32BaseCPUKernel::InitShapeA() {
  auto a_shape = in_tensors_[kInputIndex]->shape();
  int batch = 1;
  MS_CHECK_TRUE_RET_VOID(a_shape.size() >= C2NUM);
  for (size_t i = 0; i < a_shape.size() - C2NUM; ++i) {
    batch *= a_shape[i];
  }
  a_batch_ = batch;
  params_->row_ = params_->a_transpose_ ? a_shape[a_shape.size() - 1] : a_shape[a_shape.size() - C2NUM];
  params_->deep_ = params_->a_transpose_ ? a_shape[a_shape.size() - C2NUM] : a_shape[a_shape.size() - 1];
}

void MatmulFp32BaseCPUKernel::InitShapeB() {
  auto b_shape = in_tensors_[kWeightIndex]->shape();
  int batch = 1;
  MS_CHECK_TRUE_RET_VOID(b_shape.size() >= C2NUM);
  for (size_t i = 0; i < b_shape.size() - C2NUM; ++i) {
    batch *= b_shape[i];
  }
  b_batch_ = batch;
  params_->col_ = params_->b_transpose_ ? b_shape[b_shape.size() - C2NUM] : b_shape[b_shape.size() - 1];
  params_->deep_ = params_->b_transpose_ ? b_shape[b_shape.size() - 1] : b_shape[b_shape.size() - C2NUM];
}

int MatmulFp32BaseCPUKernel::MatmulPrepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  params_->a_const_ = in_tensors_[kInputIndex]->IsConst() && !op_parameter_->is_train_session_;
  params_->b_const_ = in_tensors_[kWeightIndex]->IsConst() && !op_parameter_->is_train_session_;

  if (params_->a_const_ || InferShapeDone()) {
    InitShapeA();
  }

  if (params_->b_const_ || InferShapeDone()) {
    InitShapeB();
  }

  auto ret = MatmulFp32BaseCPUKernel::Prepare();
  if (ret != RET_OK) {
    return ret;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return MatmulReSize();
}

int MatmulFp32BaseCPUKernel::Conv1x1Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), C1NUM);

  if (InferShapeDone() || params_->a_const_) {
    auto input = in_tensors_.at(0);
    params_->row_ = in_tensors_.at(0)->Batch() * input->Height() * input->Width();
    params_->deep_ = input->Channel();
  }

  if (InferShapeDone() || params_->b_const_) {
    auto weight = in_tensors_.at(1);
    params_->col_ = weight->Batch();
    params_->deep_ = weight->Channel();
  }

  a_batch_ = 1;
  b_batch_ = 1;
  params_->batch = 1;
  a_offset_.resize(params_->batch, 0);
  b_offset_.resize(params_->batch, 0);

  params_->a_transpose_ = false;
  params_->b_transpose_ = true;

  auto ret = MatmulFp32BaseCPUKernel::Prepare();
  if (ret != RET_OK) {
    return ret;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return Conv1x1ReSize();
}

int MatmulFp32BaseCPUKernel::ReSize() {
  auto ret = InitParameter();
  MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "Init parameters failed.");
  if (op_parameter_->is_train_session_) {
    set_workspace_size((matrix_a_.pack_size + matrix_b_.pack_size) * static_cast<int>(sizeof(float)));
  }
  thread_num_ = op_parameter_->thread_num_;
  ret = GetThreadCuttingPolicy();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ThreadCuttingPolicy error!";
    return ret;
  }
  if (!matrix_c_.has_packed) {
    ret = PackBiasMatrix();
    MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "pack const-matrix c failed.");
    matrix_c_.has_packed = true;
  }
  ret = InitTmpOutBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InitTmpOutBuffer error!";
    return ret;
  }
  return RET_OK;
}

int MatmulFp32BaseCPUKernel::InitBroadcastParams(const std::vector<int> &a_shape_const,
                                                 const std::vector<int> &b_shape_const, MatMulParameter *params,
                                                 std::vector<int> *a_offsets, std::vector<int> *b_offsets) {
  size_t max_dim_size = std::max(a_shape_const.size(), b_shape_const.size());
  max_dim_size = std::max(max_dim_size, static_cast<size_t>(kNCHWDimNumber));
  std::vector<int> a_shape = a_shape_const;
  if (a_shape.size() < max_dim_size) {
    size_t add_nums = max_dim_size - a_shape.size();
    for (size_t i = 0; i < add_nums; ++i) {
      (void)a_shape.insert(a_shape.begin(), 1);
    }
  }
  std::vector<int> b_shape = b_shape_const;
  if (b_shape.size() < max_dim_size) {
    size_t add_nums = max_dim_size - b_shape.size();
    for (size_t i = 0; i < add_nums; ++i) {
      (void)b_shape.insert(b_shape.begin(), 1);
    }
  }

  int batch_sizes[MAX_SHAPE_SIZE] = {0};
  int a_batch_sizes[MAX_SHAPE_SIZE] = {0};
  int b_batch_sizes[MAX_SHAPE_SIZE] = {0};
  for (int i = a_shape.size() - kCHWDimNumber; i >= 0; --i) {
    if (static_cast<int>(a_shape.size() - kCHWDimNumber) == i) {
      batch_sizes[i] = std::max(a_shape[i], b_shape[i]);
      a_batch_sizes[i] = a_shape[i];
      b_batch_sizes[i] = b_shape[i];
    } else {
      batch_sizes[i] = batch_sizes[i + 1] * std::max(a_shape[i], b_shape[i]);
      a_batch_sizes[i] = a_batch_sizes[i + 1] * a_shape[i];
      b_batch_sizes[i] = b_batch_sizes[i + 1] * b_shape[i];
    }
  }

  int out_batch = 1;
  for (size_t i = 0; i < a_shape.size() - kHWDimNumber; ++i) {
    int max_v = MSMAX(a_shape[i], b_shape[i]);
    int min_v = MSMIN(a_shape[i], b_shape[i]) > 0 ? MSMIN(a_shape[i], b_shape[i]) : 1;
    out_batch *= max_v;
    if (max_v != min_v && max_v % min_v != 0) {
      MS_LOG(ERROR) << "matmul don't support broadcast for dimension " << a_shape << " and " << b_shape;
      return RET_ERROR;
    }
  }
  params->batch = out_batch;

  a_offsets->resize(params->batch, 0);
  b_offsets->resize(params->batch, 0);
  for (int i = 0; i < params->batch; ++i) {
    int delta = i;
    int a_offset = 0;
    int b_offset = 0;
    for (size_t j = 0; j < a_shape.size() - kHWDimNumber; ++j) {
      if (j > 0) {
        delta = delta % batch_sizes[j];
      }
      if (j < (a_shape.size() - kCHWDimNumber)) {
        a_offset += (delta / batch_sizes[j + 1] * a_shape[j] / std::max(a_shape[j], b_shape[j])) * a_batch_sizes[j + 1];
        b_offset += (delta / batch_sizes[j + 1] * b_shape[j] / std::max(a_shape[j], b_shape[j])) * b_batch_sizes[j + 1];
      } else {
        a_offset += (delta * a_shape[j] / std::max(a_shape[j], b_shape[j]));
        b_offset += (delta * b_shape[j] / std::max(a_shape[j], b_shape[j]));
      }
    }
    (*a_offsets)[i] = a_offset;
    (*b_offsets)[i] = b_offset;
  }

  return RET_OK;
}

int MatmulFp32BaseCPUKernel::MatmulReSize() {
  InitShapeA();
  InitShapeB();
  auto ret = MatmulFp32BaseCPUKernel::InitBroadcastParams(
    in_tensors_[kInputIndex]->shape(), in_tensors_[kWeightIndex]->shape(), params_, &a_offset_, &b_offset_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InitBroadcastParams failed.";
    return RET_ERROR;
  }
  return MatmulFp32BaseCPUKernel::ReSize();
}

int MatmulFp32BaseCPUKernel::FullConnectionReSize() {
  MS_CHECK_TRUE_MSG(out_tensors_.at(0)->shape().size() > 0, RET_ERROR, "Invalid output tensor shape");
  int row = 1;
  for (size_t i = 0; i < out_tensors_.at(0)->shape().size() - 1; ++i) {
    row *= (out_tensors_.at(0)->shape())[i];
  }
  params_->row_ = row;
  params_->col_ = out_tensors_.at(0)->shape().back();
  params_->deep_ = (in_tensors_.at(1)->shape()).at(1);

  return MatmulFp32BaseCPUKernel::ReSize();
}

int MatmulFp32BaseCPUKernel::Conv1x1ReSize() {
  auto input = in_tensors_.at(0);
  params_->row_ = in_tensors_.at(0)->Batch() * input->Height() * input->Width();
  params_->deep_ = input->Channel();

  auto weight = in_tensors_.at(1);
  params_->col_ = weight->Batch();

  return MatmulFp32BaseCPUKernel::ReSize();
}

bool MatmulFp32BaseCPUKernel::CheckRow1OptimalConditions() {
  return params_->row_ == 1 && !(SupportMulBatchCuttingByRow() && (a_batch_ > 1 && b_batch_ == 1));
}

int MatmulFp32BaseCPUKernel::InitParameter() {
  InitGlobalVariable();
  if (CheckRow1OptimalConditions()) {
    row_tile_ = 1;
    matrix_a_pack_fun_ = params_->a_transpose_ ? RowMajor2ColMajorParallel : RowMajor2RowMajorParallel;
    matrix_a_.need_pack = false;
    pack_opt_ = false;
    if (!params_->b_const_ && params_->col_ <= C128NUM) {
      col_tile_ = 1;
      out_need_aligned_ = false;
      matrix_b_pack_fun_ = params_->b_transpose_ ? RowMajor2ColMajorParallel : RowMajor2RowMajorParallel;
      matrix_b_.need_pack = params_->b_transpose_;
    }
  }
  if (params_->col_ == 1 && !params_->a_const_) {
    out_need_aligned_ = false;
    row_tile_ = 1;
    col_tile_ = 1;
    matrix_a_pack_fun_ = params_->a_transpose_ ? RowMajor2ColMajorParallel : RowMajor2RowMajorParallel;
    matrix_b_pack_fun_ = params_->b_transpose_ ? RowMajor2ColMajorParallel : RowMajor2RowMajorParallel;
    matrix_a_.need_pack = params_->a_transpose_ && params_->row_ != 1;
    matrix_b_.need_pack = false;
    pack_opt_ = false;
  }
  params_->row_align_ = UP_ROUND(params_->row_, row_tile_);
  params_->col_align_ = UP_ROUND(params_->col_, col_tile_);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(a_batch_, params_->row_align_, RET_ERROR);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(a_batch_ * params_->row_align_, params_->deep_, RET_ERROR);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(a_batch_, params_->col_align_, RET_ERROR);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(a_batch_ * params_->col_align_, params_->deep_, RET_ERROR);
  auto a_pack_size = a_batch_ * params_->row_align_ * params_->deep_;
  auto b_pack_size = b_batch_ * params_->col_align_ * params_->deep_;
  if ((matrix_a_.has_packed && matrix_a_.pack_size != a_pack_size) ||
      (matrix_b_.has_packed && matrix_b_.pack_size != b_pack_size)) {
    MS_LOG(ERROR) << "matmul don't support dynamic packing if matrix is a constant.";
    return RET_ERROR;
  }
  matrix_a_.pack_size = a_pack_size;
  matrix_b_.pack_size = b_pack_size;
  params_->row_align_ = UP_ROUND(params_->row_, row_tile_);
  out_need_aligned_ = (out_need_aligned_ && ((params_->col_ % col_tile_) != 0));
  col_step_ = out_need_aligned_ ? params_->col_align_ : params_->col_;
  MS_CHECK_FALSE(INT_MUL_OVERFLOW(a_batch_, params_->row_), RET_ERROR);
  row_num_ = a_batch_ * params_->row_;
  return RET_OK;
}

int MatmulFp32BaseCPUKernel::InitTmpOutBuffer() {
  if (out_need_aligned_) {
    if (output_data_ != nullptr) {
      free(output_data_);
    }
    // avx need to malloc dst aligned to C8NUM
    // avx512 need to malloc dst aligned to C16NUM
    int out_channel = params_->col_;
    int oc_block_num = UP_DIV(out_channel, col_tile_);
    output_data_ = reinterpret_cast<float *>(
      malloc(params_->batch * params_->row_ * oc_block_num * col_tile_ * static_cast<int>(sizeof(float))));
    if (output_data_ == nullptr) {
      MS_LOG(ERROR) << "malloc tmp output data failed.";
      return RET_NULL_PTR;
    }
  }
  return RET_OK;
}

int MatmulFp32BaseCPUKernel::GetThreadCuttingPolicy() {
#if defined(PARALLEL_INFERENCE) && defined(ENABLE_MINDRT)
  constexpr int kNumDeepThreshold = 512;
  if (params_->deep_ < kNumDeepThreshold) {
    auto num = ParallelThreadPoolManager::GetInstance()->GetThreadPoolSize(
      static_cast<const lite::InnerContext *>(ms_context_)->thread_pool_);
    params_->op_parameter_.thread_num_ = num != -1 ? num : params_->op_parameter_.thread_num_;
  }
#endif
  if ((a_batch_ >= thread_num_ && (b_batch_ == a_batch_ || !SupportMulBatchCuttingByRow())) || params_->col_ == 1) {
    batch_stride_ = UP_DIV(params_->batch, thread_num_);
    parallel_fun_ = &MatmulFp32BaseCPUKernel::ParallelRunByBatch;
    if (params_->col_ != 1 || params_->a_const_) {
      return RET_OK;
    }
    parallel_fun_ = &MatmulFp32BaseCPUKernel::ParallelRunIsNotPackByBatch;
    if (params_->deep_ == 1) {
      gemmIsNotPackFun = GemmIsNotPack;
    } else {
      gemmIsNotPackFun = GemmIsNotPackOptimize;
      if (CheckThreadCuttingByRow()) {
        parallel_fun_ = &MatmulFp32BaseCPUKernel::ParallelRunByRow;
        GetThreadCuttingInfoByRow();
      }
    }
    return RET_OK;
  } else if ((a_batch_ >= thread_num_ && b_batch_ == 1) || CheckThreadCuttingByRow()) {
    parallel_fun_ = &MatmulFp32BaseCPUKernel::ParallelRunByRow;
    GetThreadCuttingInfoByRow();
  } else {
    int total_col_unit = UP_DIV(params_->col_align_, col_min_unit_);
    thread_num_ = MSMIN(thread_num_, total_col_unit);
    int block_col_unit = UP_DIV(total_col_unit, thread_num_);
    split_points_.clear();
    int split_point = 0;
    while (split_point < total_col_unit) {
      split_points_.push_back(split_point * col_min_unit_);
      split_point += block_col_unit;
    }
    thread_num_ = split_points_.size();
    parallel_fun_ = &MatmulFp32BaseCPUKernel::ParallelRunByOC;
  }
  return RET_OK;
}

void MatmulFp32BaseCPUKernel::GetThreadCuttingInfoByRow() {
  int row_step = MSMAX(row_num_ / thread_num_, row_min_unit_);
  int row_remaining = row_num_ - row_step * thread_num_;
  split_points_.clear();
  int split_point = 0;
  while (split_point < row_num_) {
    split_points_.push_back(split_point);
    split_point += row_step;
    if (row_remaining > 0) {
      ++split_point;
      --row_remaining;
    }
  }
  thread_num_ = split_points_.size();
}

int MatmulFp32BaseCPUKernel::Run() {
  auto out_data = reinterpret_cast<float *>(out_tensors_.front()->data());
  CHECK_NULL_RETURN(out_data);
  if (!out_need_aligned_) {
    output_data_ = out_data;
  }
  if (!params_->a_const_) {
    auto ret = PackMatrixA();
    MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "pack const-matrix a failed.");
  }
  if (!params_->b_const_) {
    auto ret = PackMatrixB();
    MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "pack const-matrix b failed.");
  }
  MS_CHECK_TRUE_MSG(matrix_a_.pack_ptr != nullptr, RET_ERROR, "matrix-a pack ptr is a nullptr.");
  MS_CHECK_TRUE_MSG(matrix_b_.pack_ptr != nullptr, RET_ERROR, "matrix-b pack ptr is a nullptr.");

  auto ret = ParallelLaunch(this->ms_context_, MatmulRun, this, thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "MatmulRun failed in split by batch";
    return ret;
  }

  if (out_need_aligned_) {
    PackNHWCXToNHWCFp32(output_data_, out_data, params_->batch, params_->row_, params_->col_, col_tile_);
  } else {
    output_data_ = nullptr;
  }
  if (!params_->a_const_) {
    FreePackedMatrixA();
  }

  if (!params_->b_const_) {
    FreePackedMatrixB();
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
