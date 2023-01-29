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

#include "sparse_apply_proximal_gradient_descent.h"

#include <securec.h>
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "kernel_log.h"
#include "status.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace {
const int32_t kInputNum = 6;
const int32_t kOutputNum = 1;
const char *kSparseApplyProximalGradientDescent = "SparseApplyProximalGradientDescent";
#define DO_COMPUTE_CASE(DTYPE, TYPE, ITYPE, CTX)                                     \
  case (DTYPE): {                                                                    \
    uint32_t ret = KERNEL_STATUS_OK;                                                 \
    if ((ITYPE) == DT_INT32) {                                                       \
      ret = DoCompute<TYPE, int32_t>(CTX);                                           \
    } else {                                                                         \
      ret = DoCompute<TYPE, int64_t>(CTX);                                           \
    }                                                                                \
    if (ret != KERNEL_STATUS_OK) {                                                   \
      KERNEL_LOG_ERROR("SparseApplyProximalGradientDescent kernel compute failed."); \
      return ret;                                                                    \
    }                                                                                \
    break;                                                                           \
  }
}  // namespace

namespace aicpu {
uint32_t SparseApplyProximalGradientDescentCpuKernel::ValidParam(CpuKernelContext &ctx) {
  Tensor *var_tensor = ctx.Input(0);
  Tensor *alpha_tensor = ctx.Input(1);
  Tensor *l1_tensor = ctx.Input(2);
  Tensor *l2_tensor = ctx.Input(3);
  Tensor *grad_tensor = ctx.Input(4);
  Tensor *indices_tensor = ctx.Input(5);
  Tensor *output_tensor = ctx.Output(0);

  auto var_shape = var_tensor->GetTensorShape();
  auto alpha_shape = alpha_tensor->GetTensorShape();
  auto l1_shape = l1_tensor->GetTensorShape();
  auto l2_shape = l2_tensor->GetTensorShape();
  auto grad_shape = grad_tensor->GetTensorShape();
  auto indices_shape = indices_tensor->GetTensorShape();
  auto output_shape = output_tensor->GetTensorShape();

  std::map<std::string, Tensor *> tensor_types;
  tensor_types.insert({"alpha", alpha_tensor});
  tensor_types.insert({"l1", l1_tensor});
  tensor_types.insert({"l2", l2_tensor});
  tensor_types.insert({"grad", grad_tensor});
  tensor_types.insert({"output var", output_tensor});
  for (auto iter = tensor_types.begin(); iter != tensor_types.end(); iter++) {
    KERNEL_CHECK_FALSE(var_tensor->GetDataType() == iter->second->GetDataType(), KERNEL_STATUS_PARAM_INVALID,
                       "The data type of %s [%s] need be same with input var [%s].", iter->first.c_str(),
                       DTypeStr(iter->second->GetDataType()).c_str(), DTypeStr(var_tensor->GetDataType()).c_str());
  }

  std::map<std::string, std::shared_ptr<TensorShape>> tensor_shapes;
  tensor_shapes.insert({"grad", grad_shape});
  tensor_shapes.insert({"output var", output_shape});
  for (auto iter = tensor_shapes.begin(); iter != tensor_shapes.end(); iter++) {
    KERNEL_CHECK_FALSE(var_shape->GetDimSizes() == iter->second->GetDimSizes(), KERNEL_STATUS_PARAM_INVALID,
                       "The %s shape size should be same as the input var shape size.", iter->first.c_str());
  }

  std::map<std::string, std::shared_ptr<TensorShape>> scalar_shapes;
  scalar_shapes.insert({"alpha", alpha_shape});
  scalar_shapes.insert({"l1", l1_shape});
  scalar_shapes.insert({"l2", l2_shape});
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
                     "The input grad must be the same size as indices in the first dimension.");

  return KERNEL_STATUS_OK;
}

uint32_t SparseApplyProximalGradientDescentCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "SparseApplyProximalGradientDescent check input or output is failed.");
  KERNEL_HANDLE_ERROR(ValidParam(ctx), "[%s] check params failed.", kSparseApplyProximalGradientDescent);
  auto data_type = ctx.Input(0)->GetDataType();
  auto data_type_indices = ctx.Input(5)->GetDataType();
  KERNEL_CHECK_FALSE((data_type_indices == DT_INT32 || data_type_indices == DT_INT64), KERNEL_STATUS_PARAM_INVALID,
                     "indices data type[%s] is unsupported", DTypeStr(data_type_indices).c_str());
  switch (data_type) {
    DO_COMPUTE_CASE(DT_FLOAT16, Eigen::half, data_type_indices, ctx);
    DO_COMPUTE_CASE(DT_FLOAT, float, data_type_indices, ctx);
    DO_COMPUTE_CASE(DT_DOUBLE, double, data_type_indices, ctx);
    DO_COMPUTE_CASE(DT_INT8, int8_t, data_type_indices, ctx);
    DO_COMPUTE_CASE(DT_INT16, int16_t, data_type_indices, ctx);
    DO_COMPUTE_CASE(DT_INT32, int32_t, data_type_indices, ctx);
    DO_COMPUTE_CASE(DT_INT64, int64_t, data_type_indices, ctx);
    DO_COMPUTE_CASE(DT_UINT8, uint8_t, data_type_indices, ctx);
    DO_COMPUTE_CASE(DT_UINT16, uint16_t, data_type_indices, ctx);
    DO_COMPUTE_CASE(DT_UINT32, uint32_t, data_type_indices, ctx);
    DO_COMPUTE_CASE(DT_UINT64, uint64_t, data_type_indices, ctx);
    default:
      KERNEL_LOG_ERROR("SparseApplyProximalGradientDescent kernel data type[%s] not support.",
                       DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T, typename TI>
uint32_t SparseApplyProximalGradientDescentCpuKernel::DoCompute(CpuKernelContext &ctx) {
  Tensor *var = ctx.Input(0);
  auto var_shape = var->GetTensorShape();

  Tensor *alpha = ctx.Input(1);
  Tensor *l1 = ctx.Input(2);
  Tensor *l2 = ctx.Input(3);

  Tensor *grad = ctx.Input(4);
  auto grad_shape = grad->GetTensorShape();

  Tensor *indices_tensor = ctx.Input(5);
  EigenTensor indices(indices_tensor, indices_tensor->GetData());
  const TI N = indices_tensor->GetTensorShape()->GetDimSize(0);

  int64_t inner_dim = 1;
  for (int d = 1; d < var_shape->GetDims(); d++) {
    inner_dim *= grad_shape->GetDimSize(d);
  }

  if (N > 0) {
    auto indices_vec = indices.flat<TI>();

    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> var_flat(
      (T *)var->GetData(), var_shape->GetDimSize(0), var_shape->NumElements() / var_shape->GetDimSize(0));
    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> grad_flat(
      (T *)grad->GetData(), grad_shape->GetDimSize(0), grad_shape->NumElements() / grad_shape->GetDimSize(0));

    T alpha_scalar = *(reinterpret_cast<const T *>(alpha->GetData()));
    T l1_scalar = *(reinterpret_cast<const T *>(l1->GetData()));
    T l2_scalar = *(reinterpret_cast<const T *>(l2->GetData()));

    for (TI i = 0; i < N; i++) {
      TI index = SubtleMustCopy(indices_vec(i));
      auto g = grad_flat.template chip<0>(i);
      auto v = var_flat.template chip<0>(index);
      auto learning_rate = v.constant(alpha_scalar);
      auto prox_v = v;
      prox_v -= g * learning_rate;
      if (l1_scalar > static_cast<T>(0.0)) {
        v = prox_v.sign() * (prox_v.abs() - learning_rate * prox_v.constant(l1_scalar)).cwiseMax(static_cast<T>(0.0)) /
            (v.constant(static_cast<T>(1.0)) + v.constant(l2_scalar) * learning_rate);
      } else {
        v = prox_v / (v.constant(static_cast<T>(1.0)) + v.constant(l2_scalar) * learning_rate);
      }
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
REGISTER_CPU_KERNEL(kSparseApplyProximalGradientDescent, SparseApplyProximalGradientDescentCpuKernel);
}  // namespace aicpu
