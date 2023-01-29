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
#include "sparse_apply_adagrad_da.h"

#include <securec.h>

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "kernel_log.h"
#include "status.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 9;
const char *kSparseApplyAdagradDA = "SparseApplyAdagradDA";
#define DO_COMPUTE_CASE(DTYPE, TYPE, ITYPE, CTX)                       \
  case (DTYPE): {                                                      \
    uint32_t ret = KERNEL_STATUS_OK;                                   \
    if ((ITYPE) == DT_INT32) {                                         \
      ret = DoCompute<TYPE, int32_t>(CTX);                             \
    } else {                                                           \
      ret = DoCompute<TYPE, int64_t>(CTX);                             \
    }                                                                  \
    if (ret != KERNEL_STATUS_OK) {                                     \
      KERNEL_LOG_ERROR("SparseApplyAdagradDA kernel compute failed."); \
      return ret;                                                      \
    }                                                                  \
    break;                                                             \
  }
}  // namespace

namespace aicpu {
uint32_t SparseApplyAdagradDACpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "SparseApplyAdagradDA check input or output is failed.");
  KERNEL_HANDLE_ERROR(ValidParam(ctx), "[%s] check params failed.", kSparseApplyAdagradDA);
  auto data_type = ctx.Input(0)->GetDataType();
  auto data_type_indices = ctx.Input(4)->GetDataType();
  KERNEL_CHECK_FALSE((data_type_indices == DT_INT32 || data_type_indices == DT_INT64), KERNEL_STATUS_PARAM_INVALID,
                     "indices data type[%s] is unsupported", DTypeStr(data_type_indices).c_str());

  switch (data_type) {
    DO_COMPUTE_CASE(DT_DOUBLE, double, data_type_indices, ctx);
    DO_COMPUTE_CASE(DT_FLOAT, float, data_type_indices, ctx);
    DO_COMPUTE_CASE(DT_FLOAT16, Eigen::half, data_type_indices, ctx);
    DO_COMPUTE_CASE(DT_INT8, int8_t, data_type_indices, ctx);
    DO_COMPUTE_CASE(DT_INT16, int16_t, data_type_indices, ctx);
    DO_COMPUTE_CASE(DT_INT32, int32_t, data_type_indices, ctx);
    DO_COMPUTE_CASE(DT_INT64, int64_t, data_type_indices, ctx);
    DO_COMPUTE_CASE(DT_UINT8, uint8_t, data_type_indices, ctx);
    DO_COMPUTE_CASE(DT_UINT16, uint16_t, data_type_indices, ctx);
    DO_COMPUTE_CASE(DT_UINT32, uint32_t, data_type_indices, ctx);
    DO_COMPUTE_CASE(DT_UINT64, uint64_t, data_type_indices, ctx);
    default:
      KERNEL_LOG_ERROR("SparseApplyAdagradDA kernel data type[%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t SparseApplyAdagradDACpuKernel::ValidParam(CpuKernelContext &ctx) {
  Tensor *var_tensor = ctx.Input(0);
  Tensor *accm_tensor = ctx.Input(1);
  Tensor *square_accum_tensor = ctx.Input(2);
  Tensor *grad_tensor = ctx.Input(3);
  Tensor *indices_tensor = ctx.Input(4);
  Tensor *lr_tensor = ctx.Input(5);
  Tensor *l1_tensor = ctx.Input(6);
  Tensor *l2_tensor = ctx.Input(7);
  Tensor *global_step_tensor = ctx.Input(8);
  Tensor *output_tensor = ctx.Output(0);

  auto var_shape = var_tensor->GetTensorShape();
  auto accm_shape = accm_tensor->GetTensorShape();
  auto square_accum_shape = square_accum_tensor->GetTensorShape();
  auto grad_shape = grad_tensor->GetTensorShape();
  auto indices_shape = indices_tensor->GetTensorShape();
  auto lr_shape = lr_tensor->GetTensorShape();
  auto l1_shape = l1_tensor->GetTensorShape();
  auto l2_shape = l2_tensor->GetTensorShape();
  auto global_step_shape = global_step_tensor->GetTensorShape();
  auto output_shape = output_tensor->GetTensorShape();

  std::map<std::string, Tensor *> tensor_types;
  tensor_types.insert({"accm", accm_tensor});
  tensor_types.insert({"square_accum", square_accum_tensor});
  tensor_types.insert({"grad", grad_tensor});
  tensor_types.insert({"lr", lr_tensor});
  tensor_types.insert({"l1", l1_tensor});
  tensor_types.insert({"l2", l2_tensor});
  tensor_types.insert({"output var", output_tensor});

  for (auto iter = tensor_types.begin(); iter != tensor_types.end(); iter++) {
    KERNEL_CHECK_FALSE(var_tensor->GetDataType() == iter->second->GetDataType(), KERNEL_STATUS_PARAM_INVALID,
                       "The data type of %s [%s] need be same with input var [%s].", iter->first.c_str(),
                       DTypeStr(iter->second->GetDataType()).c_str(), DTypeStr(var_tensor->GetDataType()).c_str());
  }

  std::map<std::string, std::shared_ptr<TensorShape>> tensor_shapes;
  tensor_shapes.insert({"accm", accm_shape});
  tensor_shapes.insert({"square_accum", square_accum_shape});
  tensor_shapes.insert({"output var", output_shape});
  for (auto iter = tensor_shapes.begin(); iter != tensor_shapes.end(); iter++) {
    KERNEL_CHECK_FALSE(var_shape->GetDimSizes() == iter->second->GetDimSizes(), KERNEL_STATUS_PARAM_INVALID,
                       "The %s shape size should be same as the input var shape size.", iter->first.c_str());
  }

  std::vector<int64_t> var_size = var_shape->GetDimSizes();
  for (size_t i = 1; i < var_size.size(); ++i) {
    KERNEL_CHECK_FALSE(var_shape->GetDimSize(i) == grad_shape->GetDimSize(i), KERNEL_STATUS_PARAM_INVALID,
                       "input grad should be equal in dimension %s with input var", i);
  }

  std::map<std::string, std::shared_ptr<TensorShape>> scalar_shapes;
  scalar_shapes.insert({"lr", lr_shape});
  scalar_shapes.insert({"l1", l1_shape});
  scalar_shapes.insert({"l2", l2_shape});
  scalar_shapes.insert({"global_step", global_step_shape});
  for (auto iter = scalar_shapes.begin(); iter != scalar_shapes.end(); iter++) {
    KERNEL_CHECK_FALSE(iter->second->GetDims() == 0, KERNEL_STATUS_PARAM_INVALID,
                       "The input %s should be as scalar, got dim size [%d].", iter->first.c_str(),
                       iter->second->GetDims());
  }

  KERNEL_CHECK_FALSE(var_shape->GetDims() >= 1, KERNEL_STATUS_PARAM_INVALID,
                     "The input var must be at least 1 dimensional, got dims [%d].", var_shape->GetDims());

  KERNEL_CHECK_FALSE(indices_shape->GetDims() == 1, KERNEL_STATUS_PARAM_INVALID,
                     "The input indices must be one-dimensional, but got dims [%d].", indices_shape->GetDims());

  KERNEL_CHECK_FALSE(grad_shape->GetDimSize(0) == indices_shape->GetDimSize(0), KERNEL_STATUS_PARAM_INVALID,
                     "The input grad must be the same size as indices in the "
                     "first dimension.");

  return KERNEL_STATUS_OK;
}

template <typename T, typename TI>
uint32_t SparseApplyAdagradDACpuKernel::DoCompute(CpuKernelContext &ctx) {
  Tensor *var = ctx.Input(0);
  TI first_dim_size = var->GetTensorShape()->GetDimSize(0);
  auto var_shape = var->GetTensorShape();
  Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> var_flat((T *)var->GetData(), var_shape->GetDimSize(0),
                                                                  var_shape->NumElements() / var_shape->GetDimSize(0));

  Tensor *grad_accum = ctx.Input(1);
  auto grad_accum_shape = grad_accum->GetTensorShape();
  Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> grad_accum_flat(
    (T *)grad_accum->GetData(), grad_accum_shape->GetDimSize(0),
    grad_accum_shape->NumElements() / grad_accum_shape->GetDimSize(0));

  Tensor *grad_square_accum = ctx.Input(2);
  auto grad_square_accum_shape = grad_square_accum->GetTensorShape();
  Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> grad_square_accum_flat(
    (T *)grad_square_accum->GetData(), grad_square_accum_shape->GetDimSize(0),
    grad_square_accum_shape->NumElements() / grad_square_accum_shape->GetDimSize(0));

  Tensor *grad = ctx.Input(3);
  auto grad_shape = grad->GetTensorShape();
  Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> grad_flat(
    (T *)grad->GetData(), grad_shape->GetDimSize(0), grad_shape->NumElements() / grad_shape->GetDimSize(0));

  Tensor *indices_tensor = ctx.Input(4);
  EigenTensor indices(indices_tensor, indices_tensor->GetData());
  auto indices_vec = indices.flat<TI>();

  Tensor *lr = ctx.Input(5);
  T lr_scalar = *(reinterpret_cast<const T *>(lr->GetData()));

  Tensor *l1 = ctx.Input(6);
  T l1_scalar = *(reinterpret_cast<const T *>(l1->GetData()));

  Tensor *l2 = ctx.Input(7);
  T l2_scalar = *(reinterpret_cast<const T *>(l2->GetData()));

  Tensor *global_step = ctx.Input(8);
  int64_t global_step_scalar_int64 = *(reinterpret_cast<const int64_t *>(global_step->GetData()));
  T global_step_scalar = static_cast<T>(global_step_scalar_int64);
  int64_t inner_dim = 1;
  for (int d = 1; d < var_shape->GetDims(); d++) {
    KERNEL_CHECK_FALSE(var_shape->GetDimSize(d) == grad_shape->GetDimSize(d), KERNEL_STATUS_PARAM_INVALID,
                       "var and grad must match in dimension [%d]", d);
    inner_dim *= grad_shape->GetDimSize(d);
  }
  TI indeces_dim1 = indices_tensor->GetTensorShape()->GetDimSize(0);

  const T gs_lr = global_step_scalar * lr_scalar;
  for (TI i = 0; i < indeces_dim1; i++) {
    const TI index = SubtleMustCopy(indices_vec(i));
    KERNEL_CHECK_FALSE(index < first_dim_size, KERNEL_STATUS_PARAM_INVALID,
                       "The value of indices[%d]:[%d] is out of range:[%d]", i, index, first_dim_size);
    auto ga = grad_accum_flat.template chip<0>(index);
    auto da = grad_square_accum_flat.template chip<0>(index);
    auto g = grad_flat.template chip<0>(i);
    auto v = var_flat.template chip<0>(index);
    ga += g;
    da += g.square();
    if (l1_scalar > static_cast<T>(0.0)) {
      v = ga.constant(static_cast<T>(-1.0)) * ga.sign() *
          ((ga.abs() / ga.constant(global_step_scalar)) - ga.constant(l1_scalar)).cwiseMax(static_cast<T>(0.0)) /
          (v.constant(l2_scalar) + da.sqrt() / v.constant(gs_lr));
    } else {
      v = ga.constant(static_cast<T>(-1.0)) * (ga / ga.constant(global_step_scalar)) /
          (v.constant(l2_scalar) + da.sqrt() / v.constant(gs_lr));
    }
  }
  auto var_data = var->GetData();
  auto output_data = ctx.Output(0)->GetData();
  auto copy_size = var->GetDataSize();
  auto mem_ret = memcpy_s(output_data, copy_size, var_data, copy_size);
  KERNEL_CHECK_FALSE(mem_ret == EOK, KERNEL_STATUS_INNER_ERROR, "Memcpy size[%zu] from input var to output var failed.",
                     copy_size);
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kSparseApplyAdagradDA, SparseApplyAdagradDACpuKernel);
}  // namespace aicpu
