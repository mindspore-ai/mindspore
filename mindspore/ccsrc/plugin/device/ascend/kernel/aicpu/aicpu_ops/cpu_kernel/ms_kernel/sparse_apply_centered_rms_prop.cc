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
#include "sparse_apply_centered_rms_prop.h"

#include <securec.h>

#include <iostream>

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "kernel_log.h"
#include "status.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const int32_t kInputNum = 10;
const int32_t kOutputNum = 1;
const char *kSparseApplyCenteredRMSProp = "SparseApplyCenteredRMSProp";
#define DO_COMPUTE_CASE(DTYPE, TYPE, ITYPE, CTX)                             \
  case (DTYPE): {                                                            \
    uint32_t ret = KERNEL_STATUS_OK;                                         \
    if ((ITYPE) == DT_INT32) {                                               \
      ret = DoCompute<TYPE, int32_t>(CTX);                                   \
    } else {                                                                 \
      ret = DoCompute<TYPE, int64_t>(CTX);                                   \
    }                                                                        \
    if (ret != KERNEL_STATUS_OK) {                                           \
      KERNEL_LOG_ERROR("SparseApplyCenteredRMSProp kernel compute failed."); \
      return ret;                                                            \
    }                                                                        \
    break;                                                                   \
  }
}  // namespace

namespace aicpu {
uint32_t SparseApplyCenteredRMSPropCpuKernel::ValidParam(CpuKernelContext &ctx) {
  Tensor *var_tensor = ctx.Input(0);
  Tensor *mg_tensor = ctx.Input(1);
  Tensor *ms_tensor = ctx.Input(2);
  Tensor *mom_tensor = ctx.Input(3);
  Tensor *lr_tensor = ctx.Input(4);
  Tensor *rho_tensor = ctx.Input(5);
  Tensor *momentum_tensor = ctx.Input(6);
  Tensor *epsilon_tensor = ctx.Input(7);
  Tensor *grad_tensor = ctx.Input(8);
  Tensor *indices_tensor = ctx.Input(9);
  Tensor *output_tensor = ctx.Output(0);

  auto var_shape = var_tensor->GetTensorShape();
  auto mg_shape = mg_tensor->GetTensorShape();
  auto ms_shape = ms_tensor->GetTensorShape();
  auto mom_shape = mom_tensor->GetTensorShape();
  auto lr_shape = lr_tensor->GetTensorShape();
  auto rho_shape = rho_tensor->GetTensorShape();
  auto momentum_shape = momentum_tensor->GetTensorShape();
  auto epsilon_shape = epsilon_tensor->GetTensorShape();
  auto grad_shape = grad_tensor->GetTensorShape();
  auto indices_shape = indices_tensor->GetTensorShape();
  auto output_shape = output_tensor->GetTensorShape();

  std::map<std::string, Tensor *> tensor_types;
  tensor_types.insert({"mg", mg_tensor});
  tensor_types.insert({"ms", ms_tensor});
  tensor_types.insert({"mom", mom_tensor});
  tensor_types.insert({"lr", lr_tensor});
  tensor_types.insert({"rho", rho_tensor});
  tensor_types.insert({"momentum", momentum_tensor});
  tensor_types.insert({"epsilon", epsilon_tensor});
  tensor_types.insert({"grad", grad_tensor});
  tensor_types.insert({"output var", output_tensor});
  for (auto iter = tensor_types.begin(); iter != tensor_types.end(); iter++) {
    KERNEL_CHECK_FALSE(var_tensor->GetDataType() == iter->second->GetDataType(), KERNEL_STATUS_PARAM_INVALID,
                       "The data type of %s [%s] need be same with input var [%s].", iter->first.c_str(),
                       DTypeStr(iter->second->GetDataType()).c_str(), DTypeStr(var_tensor->GetDataType()).c_str());
  }

  std::map<std::string, std::shared_ptr<TensorShape>> tensor_shapes;
  tensor_shapes.insert({"mg", mg_shape});
  tensor_shapes.insert({"ms", ms_shape});
  tensor_shapes.insert({"mom", mom_shape});
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
  scalar_shapes.insert({"rho", rho_shape});
  scalar_shapes.insert({"momentum", momentum_shape});
  scalar_shapes.insert({"epsilon", epsilon_shape});
  for (auto iter = scalar_shapes.begin(); iter != scalar_shapes.end(); iter++) {
    KERNEL_CHECK_FALSE(
      iter->second->GetDims() == 0 || (iter->second->GetDims() == 1 && iter->second->NumElements() == 1),
      KERNEL_STATUS_PARAM_INVALID, "The input %s should be a scalar, got dim size [%d].", iter->first.c_str(),
      iter->second->GetDims());
  }

  KERNEL_CHECK_FALSE(grad_shape->GetDims() >= 1, KERNEL_STATUS_PARAM_INVALID,
                     "The input grad must be at least 1 dimensional, got dims [%d].", grad_shape->GetDims());

  KERNEL_CHECK_FALSE(indices_shape->GetDims() == 1, KERNEL_STATUS_PARAM_INVALID,
                     "The input indices must be one-dimensional, but got dims [%d].", indices_shape->GetDims());

  KERNEL_CHECK_FALSE(grad_shape->GetDimSize(0) == indices_shape->GetDimSize(0), KERNEL_STATUS_PARAM_INVALID,
                     "The input grad must be the same size as indices in the "
                     "first dimension.");

  return KERNEL_STATUS_OK;
}

uint32_t SparseApplyCenteredRMSPropCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "SparseApplyCenteredRMSProp check input or output is failed.");
  KERNEL_HANDLE_ERROR(ValidParam(ctx), "[%s] check params failed.", kSparseApplyCenteredRMSProp);
  auto data_type = ctx.Input(0)->GetDataType();
  auto data_type_indices = ctx.Input(9)->GetDataType();
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
      KERNEL_LOG_ERROR("SparseApplyCenteredRMSProp kernel data type[%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T, typename TI>
uint32_t SparseApplyCenteredRMSPropCpuKernel::DoCompute(CpuKernelContext &ctx) {
  Tensor *var = ctx.Input(0);
  auto var_shape = var->GetTensorShape();
  Tensor *mg = ctx.Input(1);
  auto mg_shape = mg->GetTensorShape();
  Tensor *ms = ctx.Input(2);
  auto ms_shape = ms->GetTensorShape();
  Tensor *mom = ctx.Input(3);
  auto mom_shape = mom->GetTensorShape();

  Tensor *lr = ctx.Input(4);
  Tensor *rho = ctx.Input(5);
  Tensor *momentum = ctx.Input(6);
  Tensor *epsilon = ctx.Input(7);

  Tensor *grad = ctx.Input(8);
  auto grad_shape = grad->GetTensorShape();

  Tensor *indices_tensor = ctx.Input(9);
  EigenTensor indices(indices_tensor, indices_tensor->GetData());
  TI indeces_dim1 = indices_tensor->GetTensorShape()->GetDimSize(0);
  if (indeces_dim1 > 0) {
    TI first_dim_size = var->GetTensorShape()->GetDimSize(0);
    auto indices_vec = indices.flat<TI>();

    for (TI i = 0; i < indeces_dim1; i++) {
      const TI index = SubtleMustCopy(indices_vec(i));
      KERNEL_CHECK_FALSE(index >= 0 && index < first_dim_size, KERNEL_STATUS_PARAM_INVALID,
                         "Index [%d] at offset [%d] in indices is out of range[%d].", index, i, first_dim_size);
    }

    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> var_flat(
      (T *)var->GetData(), var_shape->GetDimSize(0), var_shape->NumElements() / var_shape->GetDimSize(0));
    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> ms_flat((T *)ms->GetData(), ms_shape->GetDimSize(0),
                                                                   ms_shape->NumElements() / ms_shape->GetDimSize(0));
    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> mg_flat((T *)mg->GetData(), mg_shape->GetDimSize(0),
                                                                   mg_shape->NumElements() / mg_shape->GetDimSize(0));
    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> mom_flat(
      (T *)mom->GetData(), mom_shape->GetDimSize(0), mom_shape->NumElements() / mom_shape->GetDimSize(0));
    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> grad_flat(
      (T *)grad->GetData(), grad_shape->GetDimSize(0), grad_shape->NumElements() / grad_shape->GetDimSize(0));

    T lr_scalar = *(reinterpret_cast<const T *>(lr->GetData()));
    T rho_scalar = *(reinterpret_cast<const T *>(rho->GetData()));
    T epsilon_scalar = *(reinterpret_cast<const T *>(epsilon->GetData()));
    T momentum_scalar = *(reinterpret_cast<const T *>(momentum->GetData()));

    for (TI i = 0; i < indeces_dim1; i++) {
      const TI index = SubtleMustCopy(indices_vec(i));

      auto ms_ = ms_flat.template chip<0>(index);
      auto mom_ = mom_flat.template chip<0>(index);
      auto grad_ = grad_flat.template chip<0>(i);

      ms_ = ms_ * ms_.constant(rho_scalar) + grad_.square() * grad_.constant(static_cast<T>(1) - rho_scalar);

      auto mg_ = mg_flat.template chip<0>(index);
      mg_ = mg_ * mg_.constant(rho_scalar) + grad_ * grad_.constant(static_cast<T>(1) - rho_scalar);
      auto denom_ = ms_ + ms_.constant(epsilon_scalar) - mg_.square();
      mom_ = mom_ * mom_.constant(momentum_scalar) + denom_.rsqrt() * ms_.constant(lr_scalar) * grad_;
      auto v = var_flat.template chip<0>(index);
      v -= mom_;
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
REGISTER_CPU_KERNEL(kSparseApplyCenteredRMSProp, SparseApplyCenteredRMSPropCpuKernel);
}  // namespace aicpu