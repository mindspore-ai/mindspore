/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "cpu_kernel/ms_kernel/self_adjoint_eig.h"
#include <vector>
#include <complex>
#include <Eigen/Dense>
#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "utils/kernel_util.h"
#include "utils/kernel_util.h"
#include "Eigen/Core"

namespace {
const char *kSelfAdjointEig = "SelfAdjointEig";
const uint32_t kInputNum = 1;
const uint32_t kOutputNum = 2;
}  // namespace
namespace aicpu {
uint32_t SelfAdjointEigCpuKernel::Compute(CpuKernelContext &ctx) {
  if (NormalCheck(ctx, kInputNum, kOutputNum) != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  Tensor *input0 = ctx.Input(0);
  if ((input0->GetDataSize() == 0)) {
    KERNEL_LOG_INFO("[%s] Input is empty tensor.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_OK;
  }
  uint32_t ret = KERNEL_STATUS_OK;
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    case DT_FLOAT:
      ret = SelfAdjointEigCompute<float>(ctx);
      break;
    case DT_DOUBLE:
      ret = SelfAdjointEigCompute<double>(ctx);
      break;
    case DT_COMPLEX64:
      ret = SelfAdjointEigCompute<std::complex<float>>(ctx);
      break;
    case DT_COMPLEX128:
      ret = SelfAdjointEigCompute<std::complex<double>>(ctx);
      break;
    default:
      KERNEL_LOG_ERROR("[%s] Data type of input is not support, input data type is [%s].", ctx.GetOpType().c_str(),
                       DTypeStr(data_type).c_str());
      ret = KERNEL_STATUS_PARAM_INVALID;
  }
  return ret;
}

template <typename T>
uint32_t SelfAdjointEigCpuKernel::SelfAdjointEigCompute(const CpuKernelContext &ctx) {
  auto input_tensor = ctx.Input(0);
  auto output_tensor0 = ctx.Output(0);
  auto output_tensor1 = ctx.Output(1);
  auto input_tensor_shape = input_tensor->GetTensorShape();
  auto inputData = reinterpret_cast<T *>(input_tensor->GetData());
  int64_t rank = input_tensor_shape->GetDims();
  std::vector<int64_t> input_dims = input_tensor_shape->GetDimSizes();
  const int32_t m = input_dims[rank - 1];
  int64_t num_array = input_tensor_shape->NumElements() / (m * m);
  using MatrixMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

  if (rank <= 2) {
    MatrixMap input0(inputData, m, m);
    MatrixMap output0(reinterpret_cast<T *>(output_tensor0->GetData()), m, 1);
    MatrixMap output1(reinterpret_cast<T *>(output_tensor1->GetData()), m, m);
    AttrValue *attr = ctx.GetAttr("compute_v");
    bool attr_ = (attr == nullptr) ? true : attr->GetBool();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> es(
      input0, attr_ ? Eigen::ComputeEigenvectors : Eigen::EigenvaluesOnly);
    output0 = es.eigenvalues().template cast<T>();
    if (attr_) {
      output1 = es.eigenvectors();
    }
  } else {
    auto outputData0 = reinterpret_cast<T *>(output_tensor0->GetData());
    auto outputData1 = reinterpret_cast<T *>(output_tensor1->GetData());
    for (int64_t batch = 0; batch < num_array; ++batch) {
      AttrValue *attr = ctx.GetAttr("compute_v");
      bool attr_ = (attr == nullptr) ? true : attr->GetBool();
      T *inputDataMap = reinterpret_cast<T *>(new T[m * m]);
      T *outputDataMap0 = reinterpret_cast<T *>(new T[m]);
      T *outputDataMap1 = reinterpret_cast<T *>(new T[m * m]);
      for (int64_t i = 0; i < m * m; ++i) {
        inputDataMap[i] = inputData[batch * m * m + i];
        outputDataMap1[i] = outputData1[batch * m * m + i];
      }
      for (int64_t i = 0; i < m; ++i) {
        outputDataMap0[i] = outputData0[batch * m + i];
      }
      MatrixMap input0(inputDataMap, m, m);
      MatrixMap output0(outputDataMap0, m, 1);
      MatrixMap output1(outputDataMap1, m, m);
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> es(
        input0, attr_ ? Eigen::ComputeEigenvectors : Eigen::EigenvaluesOnly);
      output0 = es.eigenvalues().template cast<T>();
      for (int64_t i = 0; i < m; i++) {
        *(outputData0 + batch * m + i) = output0(i, 0);
      }
      if (attr_) {
        output1 = es.eigenvectors();
        for (int64_t i = 0; i < m; i++) {
          for (int64_t j = 0; j < m; j++) {
            *(outputData1 + batch * m * m + i * m + j) = output1(i, j);
          }
        }
      }
    }
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kSelfAdjointEig, SelfAdjointEigCpuKernel);
}  // namespace aicpu
