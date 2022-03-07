/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "src/runtime/kernel/arm/fp32/arithmetic_fp32.h"
#include "src/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Eltwise;

namespace mindspore::kernel {
namespace {
#ifdef SERVER_INFERENCE
const std::map<std::pair<int, int>, float> dt_arithmetic_cost_map_ = {
  // {{PrimitiveType_MulFusion, schema::ActivationType_RELU}, 1.0f},
  // {{PrimitiveType_MulFusion, schema::ActivationType_RELU6}, 1.0f},
  // {{PrimitiveType_MulFusion, schema::ActivationType_NO_ACTIVATION}, 1.0f},

  {{PrimitiveType_AddFusion, schema::ActivationType_RELU}, 1.806f},
  {{PrimitiveType_AddFusion, schema::ActivationType_RELU6}, 1.806f},
  {{PrimitiveType_AddFusion, schema::ActivationType_NO_ACTIVATION}, 1.275f},

  {{PrimitiveType_SubFusion, schema::ActivationType_RELU}, 1.806f},
  {{PrimitiveType_SubFusion, schema::ActivationType_RELU6}, 1.806f},
  {{PrimitiveType_SubFusion, schema::ActivationType_NO_ACTIVATION}, 1.275f},

  // {{PrimitiveType_DivFusion, schema::ActivationType_RELU}, 1.0f},
  // {{PrimitiveType_DivFusion, schema::ActivationType_RELU6}, 1.0f},
  // {{PrimitiveType_DivFusion, schema::ActivationType_NO_ACTIVATION}, 1.0f},

  // {{PrimitiveType_RealDiv, schema::ActivationType_RELU}, 1.0f},
  // {{PrimitiveType_RealDiv, schema::ActivationType_RELU6}, 1.0f},
  // {{PrimitiveType_RealDiv, schema::ActivationType_NO_ACTIVATION}, 1.0f},

  // {{PrimitiveType_LogicalAnd, schema::ActivationType_NO_ACTIVATION}, 1.0f},
  // {{PrimitiveType_LogicalOr, schema::ActivationType_NO_ACTIVATION}, 1.0f},
  // {{PrimitiveType_Maximum, schema::ActivationType_NO_ACTIVATION}, 1.0f},
  // {{PrimitiveType_Minimum, schema::ActivationType_NO_ACTIVATION}, 1.0f},
  // {{PrimitiveType_FloorMod, schema::ActivationType_NO_ACTIVATION}, 1.0f},
  // {{PrimitiveType_FloorDiv, schema::ActivationType_NO_ACTIVATION}, 1.0f},
  // {{PrimitiveType_Mod, schema::ActivationType_NO_ACTIVATION}, 1.0f},
  // {{PrimitiveType_SquaredDifference, schema::ActivationType_NO_ACTIVATION}, 1.0f},
};
#endif
}  // namespace

#ifdef SERVER_INFERENCE
int ArithmeticCPUKernel::SetThreadCostContext() {
  std::pair<int, int> fusion_type = std::make_pair(param_->op_parameter_.type_, param_->activation_type_);
  if (dt_arithmetic_cost_map_.count(fusion_type) > 0) {
    thread_cost_context = std::make_unique<ThreadCostContext>();
    thread_cost_context->per_unit_load_num_ = 1;
    thread_cost_context->per_unit_store_num_ = 1;
    thread_cost_context->per_unit_compute_cost_ = dt_arithmetic_cost_map_.at(fusion_type);
  }
  return RET_OK;
}
#endif

int ArithmeticCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);

#ifdef SERVER_INFERENCE
  if (SetThreadCostContext() != RET_OK) {
    return RET_ERROR;
  }
#endif

  auto primitive_type = param_->op_parameter_.type_;
  if (primitive_type == schema::PrimitiveType_Eltwise) {
    switch (param_->eltwise_mode_) {
      case schema::EltwiseMode_PROD:
        primitive_type = schema::PrimitiveType_MulFusion;
        break;
      case schema::EltwiseMode_SUM:
        primitive_type = schema::PrimitiveType_AddFusion;
        break;
      case schema::EltwiseMode_MAXIMUM:
        primitive_type = schema::PrimitiveType_Maximum;
        break;
      default:
        MS_LOG(ERROR) << "Eltwise mode not support, mode:" << param_->eltwise_mode_;
        return RET_ERROR;
    }
  }
  InitRunFunction(primitive_type);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}
bool ArithmeticCPUKernel::IsScalarClac() {
  if (param_->in_elements_num0_ == 1 || param_->in_elements_num1_ == 1) {
    return true;
  }
  return false;
}
int ArithmeticCPUKernel::ReSize() {
#ifdef SERVER_INFERENCE
  if (thread_cost_context != nullptr) {
    thread_cost_context->total_unit_num_ = in_tensors_.at(0)->ElementsNum();
    thread_num_ = UpdateThreadNum(this->ms_context_, thread_cost_context.get(), op_parameter_->thread_num_);
  }
#endif
  CalcMultiplesAndStrides(param_);
  scalar_ = IsScalarClac();
  int ret = RET_OK;
  if (!scalar_) {
    ret = ConstTensorBroadCast();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "failed to init const tensor";
      return ret;
    }
  }
  if (!scalar_ && param_->broadcasting_) {
    ret = InitIndexOffsetInfo();
  }
  data_type_len_ = lite::DataTypeSize(in_tensors_.at(0)->data_type());

  return ret;
}

bool ArithmeticCPUKernel::IsBatchScalarCalc() {  // 1 32 240 240,  2 32 1 1
  int last_batch_axis0 = ARITHMETIC_SUPPORT_DIMS_NUM + 1;
  int last_batch_axis1 = ARITHMETIC_SUPPORT_DIMS_NUM + 1;
  if (param_->in_shape0_[param_->ndim_ - 1] == 1) {
    for (int i = static_cast<int>(param_->ndim_) - 1; i >= 0 && i < ARITHMETIC_SUPPORT_DIMS_NUM; --i) {
      if (param_->in_shape0_[i] != 1) {
        last_batch_axis0 = i;
        break;
      }
    }
  }
  if (param_->in_shape1_[param_->ndim_ - 1] == 1) {
    for (int i = static_cast<int>(param_->ndim_) - 1; i >= 0 && i < ARITHMETIC_SUPPORT_DIMS_NUM; --i) {
      if (param_->in_shape1_[i] != 1) {
        last_batch_axis1 = i;
        break;
      }
    }
  }
  int min_axis = MSMIN(last_batch_axis0, last_batch_axis1);
  if (min_axis < static_cast<int>(param_->ndim_) - 1) {
    last_batch_axis_ = min_axis;
    if (last_batch_axis0 < last_batch_axis1) {
      param_->in_elements_num0_ = 1;
    } else {
      param_->in_elements_num1_ = 1;
    }
    return true;
  }
  return false;
}

int ArithmeticCPUKernel::InitIndexOffsetInfo() {
  split_by_batch_ = true;
  for (int i = static_cast<int>(param_->ndim_) - 1; i >= 0 && i < ARITHMETIC_SUPPORT_DIMS_NUM; --i) {
    if (param_->in_shape0_[i] != param_->in_shape1_[i]) {
      break_pos_ = i;
      break;
    }
  }

  std::vector<int> a_shape;
  std::vector<int> b_shape;
  std::vector<int> c_shape = out_tensors_[0]->shape();
  size_t dim = c_shape.size();
  for (size_t i = 0; i < dim; ++i) {
    a_shape.push_back(param_->in_shape0_[i]);
    b_shape.push_back(param_->in_shape1_[i]);
  }
  batch_scalar_ = IsBatchScalarCalc();

  a_stride_size_ = 1;
  b_stride_size_ = 1;
  c_stride_size_ = 1;
  int last_batch_axis = batch_scalar_ ? last_batch_axis_ : break_pos_;
  for (int i = static_cast<int>(param_->ndim_) - 1; i > last_batch_axis && i < ARITHMETIC_SUPPORT_DIMS_NUM; --i) {
    a_stride_size_ *= a_shape[i];
    b_stride_size_ *= b_shape[i];
    c_stride_size_ *= c_shape[i];
  }

  out_batch_ = 1;
  int batch_size[ARITHMETIC_SUPPORT_DIMS_NUM] = {};
  int a_batch_size[ARITHMETIC_SUPPORT_DIMS_NUM] = {};
  int b_batch_size[ARITHMETIC_SUPPORT_DIMS_NUM] = {};
  for (int i = last_batch_axis; i >= 0; --i) {
    out_batch_ *= c_shape[i];
    if (i == last_batch_axis) {
      batch_size[i] = c_shape[i];
      a_batch_size[i] = a_shape[i];
      b_batch_size[i] = b_shape[i];
    } else {
      batch_size[i] = batch_size[i + 1] * c_shape[i];
      a_batch_size[i] = a_batch_size[i + 1] * a_shape[i];
      b_batch_size[i] = b_batch_size[i + 1] * b_shape[i];
    }
  }

  a_offset_.resize(out_batch_, 0);
  b_offset_.resize(out_batch_, 0);
  for (int i = 0; i < out_batch_; ++i) {
    int delta = i;
    int a_offset = 0;
    int b_offset = 0;
    for (int j = 0; j <= last_batch_axis; ++j) {
      if (j > 0) {
        delta = delta % batch_size[j];
      }
      if (j < last_batch_axis) {
        a_offset += (delta / batch_size[j + 1] * a_shape[j] / c_shape[j]) * a_batch_size[j + 1];
        b_offset += (delta / batch_size[j + 1] * b_shape[j] / c_shape[j]) * b_batch_size[j + 1];
      } else {
        a_offset += (delta * a_shape[j] / c_shape[j]);
        b_offset += (delta * b_shape[j] / c_shape[j]);
      }
    }
    a_offset_[i] = a_offset;
    b_offset_[i] = b_offset;
  }
  return RET_OK;
}

int ArithmeticCPUKernel::CheckDataType() {
  auto in0_dataType = in_tensors_.at(0)->data_type();
  auto in1_dataType = in_tensors_.at(1)->data_type();
  if (in0_dataType != in1_dataType) {
    MS_LOG(ERROR) << "The dataTypes of input tensor0 and input tensor1 should be the same. input 0 dataType: "
                  << in0_dataType << " input 1 dataType: " << in1_dataType;
    return RET_ERROR;
  }
  if (op_parameter_->is_train_session_) {
    data_type_len_ = lite::DataTypeSize(in_tensors_.at(0)->data_type());
  }
  return RET_OK;
}

int ArithmeticCPUKernel::ConstTensorBroadCast() {
  /* if const node need broadcast and all need-broadcast-node are const, broadcast in resize */
  if (!param_->broadcasting_) {
    return RET_OK;
  }

  /* [1, 1, 2] + [1, 2, 1] -> [1, 2, 2], need broadcast both input */
  FreeConstTileBuff();
  if (in_tensors_[0]->IsConst() && param_->in_elements_num0_ != param_->out_elements_num_) {
    input0_ptr_ = malloc(param_->out_elements_num_ * data_type_len_);
    if (input0_ptr_ == nullptr) {
      return RET_ERROR;
    }
    CHECK_NULL_RETURN(in_tensors_[0]->data());
    TileConstTensor(in_tensors_[0]->data(), input0_ptr_, param_->ndim_, param_->in_shape0_, param_->in_strides0_,
                    param_->out_strides_, param_->multiples0_);
    input0_broadcast_ = true;
    param_->in_elements_num0_ = param_->out_elements_num_;
    // shape must be equal to out
    for (size_t i = 0; i < param_->ndim_; ++i) {
      param_->in_shape0_[i] = param_->out_shape_[i];
      param_->in_strides0_[i] = param_->out_strides_[i];
    }
  }
  if (in_tensors_[1]->IsConst() && param_->in_elements_num1_ != param_->out_elements_num_) {
    input1_ptr_ = malloc(param_->out_elements_num_ * data_type_len_);
    if (input1_ptr_ == nullptr) {
      FreeConstTileBuff();
      return RET_ERROR;
    }
    CHECK_NULL_RETURN(in_tensors_[1]->data());
    TileConstTensor(in_tensors_[1]->data(), input1_ptr_, param_->ndim_, param_->in_shape1_, param_->in_strides1_,
                    param_->out_strides_, param_->multiples1_);
    input1_broadcast_ = true;
    param_->in_elements_num1_ = param_->out_elements_num_;
    // shape must be equal to out
    for (size_t i = 0; i < param_->ndim_; ++i) {
      param_->in_shape1_[i] = param_->out_shape_[i];
      param_->in_strides1_[i] = param_->out_strides_[i];
    }
  }
  // broadcast input and get new break_pos_
  for (int i = static_cast<int>(param_->ndim_) - 1; i >= 0; --i) {
    if (param_->in_shape0_[i] != param_->in_shape1_[i]) {
      break_pos_ = i;
      break;
    }
  }
  if (param_->in_elements_num0_ == param_->out_elements_num_ &&
      param_->in_elements_num1_ == param_->out_elements_num_) {
    param_->broadcasting_ = false;
  }
  return RET_OK;
}

void ArithmeticCPUKernel::TileConstTensor(const void *in_data, void *out_data, size_t ndim, const int *in_shape,
                                          const int *in_strides, const int *out_strides, const int *multiple) {
  TileOneDimensionFp32(reinterpret_cast<const float *>(in_data), reinterpret_cast<float *>(out_data), 0, ndim, in_shape,
                       in_strides, out_strides, multiple);
}

void ArithmeticCPUKernel::FreeConstTileBuff() {
  if (input0_broadcast_ == true && input0_ptr_ != nullptr) {
    free(input0_ptr_);
    input0_ptr_ = nullptr;
    input0_broadcast_ = false;
  }
  if (input1_broadcast_ == true && input1_ptr_ != nullptr) {
    free(input1_ptr_);
    input1_ptr_ = nullptr;
    input1_broadcast_ = false;
  }
}

void ArithmeticCPUKernel::InitRunFunction(int primitive_type) {
  ARITHMETIC_FUNC_INFO_FP32 fun_table[] = {
    {PrimitiveType_MulFusion, schema::ActivationType_RELU, ElementMulRelu, ElementMulReluInt, nullptr,
     ElementOptMulRelu, ElementOptMulReluInt, nullptr},
    {PrimitiveType_MulFusion, schema::ActivationType_RELU6, ElementMulRelu6, ElementMulRelu6Int, nullptr,
     ElementOptMulRelu6, ElementOptMulRelu6Int, nullptr},
    {PrimitiveType_MulFusion, schema::ActivationType_NO_ACTIVATION, ElementMul, ElementMulInt, nullptr, ElementOptMul,
     ElementOptMulInt, nullptr},
    {PrimitiveType_AddFusion, schema::ActivationType_RELU, ElementAddRelu, nullptr, nullptr, ElementOptAddRelu, nullptr,
     nullptr},
    {PrimitiveType_AddFusion, schema::ActivationType_RELU6, ElementAddRelu6, nullptr, nullptr, ElementOptAddRelu6,
     nullptr, nullptr},
    {PrimitiveType_AddFusion, schema::ActivationType_NO_ACTIVATION, ElementAdd, ElementAddInt, nullptr, ElementOptAdd,
     ElementOptAddInt, nullptr},
    {PrimitiveType_SubFusion, schema::ActivationType_RELU, ElementSubRelu, nullptr, nullptr, ElementOptSubRelu, nullptr,
     nullptr},
    {PrimitiveType_SubFusion, schema::ActivationType_RELU6, ElementSubRelu6, nullptr, nullptr, ElementOptSubRelu6,
     nullptr, nullptr},
    {PrimitiveType_SubFusion, schema::ActivationType_NO_ACTIVATION, ElementSub, ElementSubInt, nullptr, ElementOptSub,
     ElementOptSubInt, nullptr},
    {PrimitiveType_DivFusion, schema::ActivationType_RELU, ElementDivRelu, nullptr, nullptr, ElementOptDivRelu, nullptr,
     nullptr},
    {PrimitiveType_DivFusion, schema::ActivationType_RELU6, ElementDivRelu6, nullptr, nullptr, ElementOptDivRelu6,
     nullptr, nullptr},
    {PrimitiveType_DivFusion, schema::ActivationType_NO_ACTIVATION, ElementDiv, nullptr, nullptr, ElementOptDiv,
     ElementOptDivInt, nullptr},
    {PrimitiveType_RealDiv, schema::ActivationType_RELU, ElementDivRelu, nullptr, nullptr, ElementOptDivRelu, nullptr,
     nullptr},
    {PrimitiveType_RealDiv, schema::ActivationType_RELU6, ElementDivRelu6, nullptr, nullptr, ElementOptDivRelu6,
     nullptr, nullptr},
    {PrimitiveType_RealDiv, schema::ActivationType_NO_ACTIVATION, ElementDiv, nullptr, nullptr, ElementOptDiv,
     ElementOptDivInt, nullptr},
    {PrimitiveType_LogicalAnd, schema::ActivationType_NO_ACTIVATION, ElementLogicalAnd, ElementLogicalAndInt,
     ElementLogicalAndBool, ElementOptLogicalAnd, ElementOptLogicalAndInt, ElementOptLogicalAndBool},
    {PrimitiveType_LogicalOr, schema::ActivationType_NO_ACTIVATION, ElementLogicalOr, nullptr, ElementLogicalOrBool,
     nullptr, nullptr, ElementOptLogicalOrBool},
    {PrimitiveType_Maximum, schema::ActivationType_NO_ACTIVATION, ElementMaximum, ElementMaximumInt, nullptr,
     ElementOptMaximum, ElementOptMaximumInt, nullptr},
    {PrimitiveType_Minimum, schema::ActivationType_NO_ACTIVATION, ElementMinimum, ElementMinimumInt, nullptr,
     ElementOptMinimum, ElementOptMinimumInt, nullptr},
    {PrimitiveType_FloorMod, schema::ActivationType_NO_ACTIVATION, ElementFloorMod, ElementFloorModInt, nullptr,
     ElementOptFloorMod, ElementOptFloorModInt, nullptr},
    {PrimitiveType_FloorDiv, schema::ActivationType_NO_ACTIVATION, ElementFloorDiv, ElementFloorDivInt, nullptr,
     ElementOptFloorDiv, ElementOptFloorDivInt, nullptr},
    {PrimitiveType_Mod, schema::ActivationType_NO_ACTIVATION, ElementMod, ElementModInt, nullptr, ElementOptMod,
     ElementOptModInt, nullptr},
    {PrimitiveType_SquaredDifference, schema::ActivationType_NO_ACTIVATION, ElementSquaredDifference, nullptr, nullptr,
     ElementOptSquaredDifference, nullptr, nullptr}};

  size_t length = sizeof(fun_table) / sizeof(ARITHMETIC_FUNC_INFO_FP32);
  for (size_t i = 0; i < length; i++) {
    if (fun_table[i].primitive_type_ == primitive_type && fun_table[i].activation_type_ == param_->activation_type_) {
      arithmetic_run_ = fun_table[i].func_;
      arithmetic_run_int_ = fun_table[i].int_func_;
      arithmetic_run_bool_ = fun_table[i].bool_func_;
      arithmetic_opt_run_ = fun_table[i].opt_func_;
      arithmetic_opt_run_int_ = fun_table[i].opt_int_func_;
      arithmetic_opt_run_bool_ = fun_table[i].opt_bool_func_;
      return;
    }
  }
}

int ArithmeticCPUKernel::DoExecute(const void *input0, const void *input1, void *output, int size, bool is_opt) {
  int ret = RET_OK;
  if (in_tensors_[0]->data_type() == kNumberTypeFloat32) {
    if (is_opt) {
      CHECK_NULL_RETURN(arithmetic_opt_run_);
      ret = arithmetic_opt_run_(reinterpret_cast<const float *>(input0), reinterpret_cast<const float *>(input1),
                                reinterpret_cast<float *>(output), size, param_);
    } else {
      CHECK_NULL_RETURN(arithmetic_run_);
      ret = arithmetic_run_(reinterpret_cast<const float *>(input0), reinterpret_cast<const float *>(input1),
                            reinterpret_cast<float *>(output), size);
    }
  } else if (in_tensors_[0]->data_type() == kNumberTypeBool) {
    if (is_opt) {
      CHECK_NULL_RETURN(arithmetic_opt_run_bool_);
      ret = arithmetic_opt_run_bool_(reinterpret_cast<const bool *>(input0), reinterpret_cast<const bool *>(input1),
                                     reinterpret_cast<bool *>(output), size, param_);
    } else {
      CHECK_NULL_RETURN(arithmetic_run_bool_);
      ret = arithmetic_run_bool_(reinterpret_cast<const bool *>(input0), reinterpret_cast<const bool *>(input1),
                                 reinterpret_cast<bool *>(output), size);
    }
  } else {
    if (is_opt) {
      CHECK_NULL_RETURN(arithmetic_opt_run_int_);
      ret = arithmetic_opt_run_int_(reinterpret_cast<const int *>(input0), reinterpret_cast<const int *>(input1),
                                    reinterpret_cast<int *>(output), size, param_);
    } else {
      CHECK_NULL_RETURN(arithmetic_run_int_);
      ret = arithmetic_run_int_(reinterpret_cast<const int *>(input0), reinterpret_cast<const int *>(input1),
                                reinterpret_cast<int *>(output), size);
    }
  }
  return ret;
}

int ArithmeticCPUKernel::CalcArithmeticByBatch(int task_id) {
  int batch_per_thread = UP_DIV(out_batch_, thread_num_);
  int start_batch = batch_per_thread * task_id;
  int end_batch = MSMIN(start_batch + batch_per_thread, out_batch_);
  for (int i = start_batch; i < end_batch; i++) {
    int ret = RET_ERROR;
    auto batch_a_ptr = static_cast<uint8_t *>(input0_ptr_) + a_offset_[i] * a_stride_size_ * data_type_len_;
    auto batch_b_ptr = static_cast<uint8_t *>(input1_ptr_) + b_offset_[i] * b_stride_size_ * data_type_len_;
    auto batch_c_ptr = static_cast<uint8_t *>(output_ptr_) + i * c_stride_size_ * data_type_len_;
    if (batch_scalar_) {
      ret = DoExecute(batch_a_ptr, batch_b_ptr, batch_c_ptr, c_stride_size_, true);
    } else {
      ret = DoExecute(batch_a_ptr, batch_b_ptr, batch_c_ptr, c_stride_size_, false);
    }
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "failed to calculate.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int ArithmeticCPUKernel::DoArithmetic(int task_id) {
  if (split_by_batch_) {
    return CalcArithmeticByBatch(task_id);
  }

  int64_t element_num = out_tensors_[0]->ElementsNum();
  auto ret = RET_ERROR;
  int stride = UP_DIV(element_num, thread_num_);
  int count = MSMIN(stride, element_num - stride * task_id);
  if (count <= 0) {
    return RET_OK;
  }
  CHECK_LESS_RETURN(ARITHMETIC_SUPPORT_DIMS_NUM, param_->ndim_);
  int offset = stride * task_id * data_type_len_;
  if (scalar_) {
    if (param_->in_elements_num0_ == 1) {
      ret = DoExecute(batch_a_ptr_, batch_b_ptr_ + offset, batch_c_ptr_ + offset, count, true);
    } else {
      ret = DoExecute(batch_a_ptr_ + offset, batch_b_ptr_, batch_c_ptr_ + offset, count, true);
    }
  } else {
    ret = DoExecute(batch_a_ptr_ + offset, batch_b_ptr_ + offset, batch_c_ptr_ + offset, count, false);
  }
  return ret;
}

int ArithmeticsRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto kernel = reinterpret_cast<ArithmeticCPUKernel *>(cdata);
  auto ret = kernel->DoArithmetic(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ArithmeticsRun error task_id[" << task_id << "] error_code[" << ret << "]";
  }
  return ret;
}

int ArithmeticCPUKernel::Run() {
  if (CheckDataType() != RET_OK) {
    MS_LOG(ERROR) << "ArithmeticCPUKernel check dataType failed, kernel name: " << this->name();
    return RET_ERROR;
  }

  if (!input0_broadcast_) {
    input0_ptr_ = in_tensors_[0]->data();
    CHECK_NULL_RETURN(input0_ptr_);
  }
  if (!input1_broadcast_) {
    input1_ptr_ = in_tensors_[1]->data();
    CHECK_NULL_RETURN(input1_ptr_);
  }
  output_ptr_ = out_tensors_[0]->data();
  CHECK_NULL_RETURN(output_ptr_);

  batch_a_ptr_ = static_cast<uint8_t *>(input0_ptr_);
  batch_b_ptr_ = static_cast<uint8_t *>(input1_ptr_);
  batch_c_ptr_ = static_cast<uint8_t *>(output_ptr_);

  auto ret = ParallelLaunch(this->ms_context_, ArithmeticsRun, this, thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "arithmetic failed";
    return RET_ERROR;
  }

  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_MulFusion, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_MulFusion, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_AddFusion, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_AddFusion, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_AddFusion, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SubFusion, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_SubFusion, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_DivFusion, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_RealDiv, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Mod, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Mod, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_LogicalAnd, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_LogicalAnd, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_LogicalAnd, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_LogicalOr, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_LogicalOr, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Maximum, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Minimum, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Maximum, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Minimum, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_FloorDiv, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_FloorMod, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_FloorDiv, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_FloorMod, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SquaredDifference, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Eltwise, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_DivFusion, LiteKernelCreator<ArithmeticCPUKernel>)
}  // namespace mindspore::kernel
