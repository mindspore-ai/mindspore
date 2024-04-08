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
#include "meshgrid.h"
#include <vector>
#include <string>
#include <map>
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"
#include "inc/kernel_log.h"
#include "context/inc/cpu_kernel_utils.h"

namespace aicpu {
namespace {
const char *kMeshgrid = "Meshgrid";
}

template <typename T>
uint32_t MeshgridTask(CpuKernelContext &ctx, const std::string &indexing, size_t ndim, const std::vector<int> &bcast) {
  auto shards = [&](const int64_t begin, const int64_t end) {
    for (int i = begin; i < end; ++i) {  // 0~ndim
      auto new_i = i;
      auto s = bcast;
      if (indexing == "xy" && i < 2) {
        new_i = 1 - i;
        auto tmp = s[0];
        s[0] = s[1];
        s[1] = tmp;
      }
      size_t row_ = 1;
      size_t col_ = 1;
      for (int j = 0; j <= new_i; j++) {
        row_ *= s[j];
      }
      for (int j = new_i + 1; j < static_cast<int>(s.size()); j++) {
        col_ *= s[j];
      }

      Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> input_map(reinterpret_cast<T *>(ctx.Input(i)), bcast[i],
                                                                       1);
      const auto &input = Eigen::Tensor<T, 2, Eigen::RowMajor>(input_map);
      Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> output(reinterpret_cast<T *>(ctx.Output(i)), row_, col_);

      Eigen::Tensor<T, 2, Eigen::RowMajor> origin(bcast[i], row_ * col_ / bcast[i]);
      for (int c = 0; c < bcast[i]; ++c) {
        for (int r = 0; r < static_cast<int>(row_ * col_ / bcast[i]); ++r) {
          origin(c, r) = input(c, 0);
        }
      }

      for (size_t j = 0; j < row_ * col_ / bcast[i] / col_; ++j) {
        Eigen::array<int64_t, 2> offsets_in = {0, static_cast<int64_t>(col_ * j)};
        Eigen::array<int64_t, 2> offsets_out = {static_cast<int64_t>(bcast[i] * j), 0};
        Eigen::array<int64_t, 2> extents = {static_cast<int64_t>(bcast[i]), static_cast<int64_t>(col_)};
        output.slice(offsets_out, extents) = origin.slice(offsets_in, extents);
      }
    }
  };

  return CpuKernelUtils::ParallelFor(ctx, ndim, 1, shards);
}

uint32_t MeshgridKernel::Compute(CpuKernelContext &ctx) {
  RETURN_IF_FAILURE(ParseKernelParam(ctx));
  std::map<int, std::function<uint32_t(CpuKernelContext &, std::string &, size_t &, std::vector<int> &)>> calls;
  calls[DT_INT8] = MeshgridTask<int8_t>;
  calls[DT_INT16] = MeshgridTask<int16_t>;
  calls[DT_INT32] = MeshgridTask<int32_t>;
  calls[DT_INT64] = MeshgridTask<int64_t>;
  calls[DT_FLOAT16] = MeshgridTask<Eigen::half>;
  calls[DT_FLOAT] = MeshgridTask<float>;
  calls[DT_DOUBLE] = MeshgridTask<double>;
  calls[DT_UINT8] = MeshgridTask<uint8_t>;
  calls[DT_UINT16] = MeshgridTask<uint16_t>;
  calls[DT_UINT32] = MeshgridTask<uint32_t>;
  calls[DT_UINT64] = MeshgridTask<uint64_t>;
  calls[DT_BOOL] = MeshgridTask<bool>;
  return calls[input_type_](ctx, indexing_, ndim_, bcast_);
}

uint32_t MeshgridKernel::ParseKernelParam(CpuKernelContext &ctx) {
  auto indexing_attr = ctx.GetAttr("indexing");
  CUST_KERNEL_CHECK_NULLPTR(ctx, indexing_attr, KERNEL_STATUS_INNER_ERROR, "Failed to get attr [indexing].");
  indexing_ = indexing_attr->GetString();

  input_type_ = ctx.Input(0)->GetDataType();

  ndim_ = ctx.GetInputsSize();
  bcast_.resize(ndim_);
  for (size_t n = 0; n < ndim_; ++n) {
    auto input_tensor = ctx.Input(n);
    auto input_shape = input_tensor->GetTensorShape()->GetDimSizes();
    if (input_shape.size() != 1) {
      CUST_AICPU_LOGE(ctx, "input tensor should be 1-D.");
    }
    bcast_[n] = input_shape.front();
  }

  return KERNEL_STATUS_OK;
}
REGISTER_MS_CPU_KERNEL(kMeshgrid, MeshgridKernel);
}  // namespace aicpu