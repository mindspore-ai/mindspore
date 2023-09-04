/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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
#include "cpu_kernel/ms_kernel/masked_select.h"

#include <array>
#include <atomic>
#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

#include "Eigen/Core"
#include "securec/include/securec.h"
#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "cpu_kernel/inc/cpu_types.h"
#include "mindspore/ccsrc/plugin/device/ascend/kernel/aicpu/aicpu_ops/common/kernel_log.h"
#include "cpu_kernel/common/status.h"
#include "utils/broadcast_iterator.h"
#include "utils/kernel_util.h"

namespace {
constexpr uint32_t kMaskedSelectInputNum = 2;
constexpr uint32_t kMaskedSelectOutputNum = 1;
constexpr int64_t kParallelDataNums = 32 * 1000;
const char *const kMaskedSelect = "MaskedSelect";
struct OutputInfo {
  int64_t startIdx;
  int64_t len;
  OutputInfo() {
    startIdx = 0;
    len = 0;
  }
};

bool CompareFunc(const OutputInfo &a, const OutputInfo &b) { return a.startIdx <= b.startIdx; }

// calculate the index stride of dataShape.
// dataShape:[m, 1, k] and broadcastShape:[j, m, n, k] --> index_stride:[0, k, 0, 1]
std::vector<int64_t> CalIndexStride(const std::vector<int64_t> &dataShape, const std::vector<int64_t> &broadcastShape) {
  int broadcastDimNum = broadcastShape.size();
  int dataDimNum = dataShape.size();
  int diffDimNum = broadcastDimNum - dataDimNum;
  std::vector<int64_t> indexStride(broadcastDimNum, 0);
  indexStride[broadcastDimNum - 1] = 1;
  for (int i = broadcastDimNum - 1; i > diffDimNum; i--) {
    indexStride[i - 1] = indexStride[i] * dataShape[i];
  }
  for (int i = 0; i < dataDimNum; i++) {
    if (dataShape[i] == 1) {
      indexStride[i + diffDimNum] = 0;
    }
  }
  return indexStride;
}

// calculate the index stride of shape.
// shape:[m, n, k] --> index_stride:[n*k, k, 1]
std::vector<int64_t> CalIndexStride(const std::vector<int64_t> &shape) {
  int dimNum = shape.size();
  std::vector<int64_t> indexStride(dimNum, 1);
  for (int i = dimNum - 1; i > 0; i--) {
    indexStride[i - 1] = indexStride[i] * shape[i];
  }
  return indexStride;
}

// calculate the original index of data.
// shape:[7,8,9] indexStride:[72,9,1] and flatten_index:11--> ori_index:[0,1,2]
bool CalIndexInfo(const std::vector<int64_t> &indexStride, int64_t flattenIndex, std::vector<int64_t> *oriIndex,
                  int dimNum) {
  for (int i = 0; i < dimNum - 1; i++) {
    if (indexStride[i] == 0) {
      return false;
    }
    (*oriIndex)[i] = flattenIndex / indexStride[i];
    flattenIndex = flattenIndex % indexStride[i];
  }
  (*oriIndex)[dimNum - 1] = flattenIndex;
  return true;
}

inline int64_t CalFlattenIndex(const std::vector<int64_t> &indexStride, const std::vector<int64_t> &oriIndex,
                               int dimNum) {
  int64_t flattenIndex = 0;
  for (int i = 0; i < dimNum; i++) {
    flattenIndex += indexStride[i] * oriIndex[i];
  }
  return flattenIndex;
}

void UpdateIndexByCarry(std::vector<int64_t> *preIndex, const std::vector<int64_t> &shape, int dimNum) {
  // shape:[7,3,10,17] and last index:[0,0,9,16] -> next index:[0,1,0,0]
  constexpr int64_t carryBit = 1;
  for (int i = dimNum - 1; i >= 0; i--) {
    (*preIndex)[i] = (*preIndex)[i] + carryBit;
    if ((*preIndex)[i] < shape[i]) {
      break;
    } else {
      (*preIndex)[i] = (*preIndex)[i] - shape[i];
    }
  }
  return;
}
}  // namespace

namespace aicpu {
uint32_t MaskedSelectCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kMaskedSelectInputNum, kMaskedSelectOutputNum), "[%s] check params failed.",
                      kMaskedSelect);

  // choose compute function depend on dataType
  auto data_type0 = static_cast<DataType>(ctx.Input(kFirstInputIndex)->GetDataType());
  auto data_type1 = static_cast<DataType>(ctx.Input(kSecondInputIndex)->GetDataType());
  auto data_type2 = static_cast<DataType>(ctx.Output(kFirstOutputIndex)->GetDataType());
  if (data_type1 != DT_BOOL) {
    KERNEL_LOG_ERROR("[%s] Data type of mask requires bool, but got data type [%s].", ctx.GetOpType().c_str(),
                     DTypeStr(data_type1).c_str());
    return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
  }
  if (data_type0 != data_type2) {
    KERNEL_LOG_ERROR("[%s] Data type of x and y requires same, but got data type [%s] and [%s].",
                     ctx.GetOpType().c_str(), DTypeStr(data_type0).c_str(), DTypeStr(data_type2).c_str());
    return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
  }
  switch (data_type0) {
    case DT_FLOAT16:
      return MaskedSelectCompute<Eigen::half>(ctx);
    case DT_FLOAT:
      return MaskedSelectCompute<float>(ctx);
    case DT_DOUBLE:
      return MaskedSelectCompute<double>(ctx);
    case DT_INT8:
      return MaskedSelectCompute<int8_t>(ctx);
    case DT_INT16:
      return MaskedSelectCompute<int16_t>(ctx);
    case DT_INT32:
      return MaskedSelectCompute<int32_t>(ctx);
    case DT_INT64:
      return MaskedSelectCompute<int64_t>(ctx);
    case DT_UINT8:
      return MaskedSelectCompute<uint8_t>(ctx);
    case DT_UINT16:
      return MaskedSelectCompute<uint16_t>(ctx);
    case DT_UINT32:
      return MaskedSelectCompute<uint32_t>(ctx);
    case DT_UINT64:
      return MaskedSelectCompute<uint64_t>(ctx);
    case DT_BOOL:
      return MaskedSelectCompute<bool>(ctx);
    default:
      KERNEL_LOG_ERROR("[%s] Data type of input is not support, input data type is [%s].", ctx.GetOpType().c_str(),
                       DTypeStr(data_type0).c_str());
      return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
  }
}

template <typename T>
uint32_t MaskedSelectCpuKernel::ParallelCompute(const CpuKernelContext &ctx, const std::vector<int64_t> &inputShapeX,
                                                const std::vector<int64_t> &inputShapeMask,
                                                const std::vector<int64_t> &outputShape, int64_t dataNum) {
  T *x = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  bool *mask = reinterpret_cast<bool *>(ctx.Input(1)->GetData());
  T *y = reinterpret_cast<T *>(ctx.Output(0)->GetData());

  std::atomic<int> threadNum{0};
  std::atomic<bool> taskFlag(true);
  constexpr int queueLen = 100;
  std::array<OutputInfo, queueLen> outputIndexList;

  std::vector<int64_t> indexStrideX = CalIndexStride(inputShapeX, outputShape);
  std::vector<int64_t> indexStrideMask = CalIndexStride(inputShapeMask, outputShape);
  std::vector<int64_t> indexStrideOutput = CalIndexStride(outputShape);
  KERNEL_LOG_DEBUG("index stride of x[%s].", VectorToString(indexStrideX).c_str());
  KERNEL_LOG_DEBUG("index stride of mask[%s].", VectorToString(indexStrideMask).c_str());

  auto work = [=, &threadNum, &taskFlag, &outputIndexList](int64_t start, int64_t end) {
    int64_t cnt = 0;
    int dimNum = outputShape.size();
    std::vector<int64_t> indexValue(dimNum, 0);
    if (!CalIndexInfo(indexStrideOutput, start, &indexValue, dimNum)) {
      taskFlag.store(false);
      KERNEL_LOG_ERROR("Invalid index stride, please check.");
      return;
    }

    for (int64_t i = start; i < end; ++i) {
      int64_t maskFlatIndex = CalFlattenIndex(indexStrideMask, indexValue, dimNum);
      int64_t xFlatIndex = CalFlattenIndex(indexStrideX, indexValue, dimNum);
      if (mask[maskFlatIndex]) {
        y[start + cnt] = x[xFlatIndex];
        cnt++;
      }
      UpdateIndexByCarry(&indexValue, outputShape, dimNum);
    }
    int idx = threadNum.fetch_add(1, std::memory_order_relaxed);
    if (idx >= queueLen) {
      taskFlag.store(false);
      return;
    }
    outputIndexList[idx].startIdx = start;
    outputIndexList[idx].len = cnt;
    KERNEL_LOG_DEBUG("outputIndexList[%d] startIdx is [%lld], len is  [%lld].", idx, outputIndexList[idx].startIdx,
                     outputIndexList[idx].len);
  };
  constexpr int perUnitSize = 1000;
  KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, dataNum, perUnitSize, work), "MaskedSelect calculate failed.");

  if (!taskFlag.load()) {
    KERNEL_LOG_ERROR("Invalid array.");
    return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
  }

  int validNum = threadNum.load();
  std::sort(outputIndexList.begin(), outputIndexList.begin() + validNum, CompareFunc);

  int validOffset = outputIndexList[0].len;
  int ret = 0;
  for (int i = 1; i < validNum; i++) {
    int64_t copyLen = outputIndexList[i].len;
    if (copyLen <= 0) {
      continue;
    }
    int64_t byteLen = copyLen * static_cast<int64_t>(sizeof(T));
    ret = memmove_s(y + validOffset, byteLen, y + outputIndexList[i].startIdx, byteLen);
    KERNEL_CHECK_FALSE((ret == EOK), KERNEL_STATUS_PARAM_INVALID, "Memmove failed, result = [%d].", ret);
    validOffset += copyLen;
  }
  ctx.Output(0)->GetTensorShape()->SetDimSizes({validOffset});
  return static_cast<uint32_t>(KERNEL_STATUS_OK);
}

template <typename T>
uint32_t MaskedSelectCpuKernel::MaskedSelectCompute(const CpuKernelContext &ctx) {
  T *x = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  KERNEL_CHECK_NULLPTR(x, static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID), "[%s] get input_data[0] failed.",
                       kMaskedSelect);
  bool *mask = reinterpret_cast<bool *>(ctx.Input(1)->GetData());
  KERNEL_CHECK_NULLPTR(mask, static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID), "[%s] get input_data[1] failed.",
                       kMaskedSelect);
  T *y = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  KERNEL_CHECK_NULLPTR(y, static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID), "[%s] get output_data[0] failed.",
                       kMaskedSelect);

  auto input_shape_a = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  auto input_shape_b = ctx.Input(1)->GetTensorShape()->GetDimSizes();
  if (IsScalar(input_shape_a) && IsScalar(input_shape_b)) {
    if (mask[0]) {
      y[0] = x[0];
      ctx.Output(0)->GetTensorShape()->SetDimSizes({1});
    } else {
      ctx.Output(0)->GetTensorShape()->SetDimSizes({0});
    }
    return static_cast<uint32_t>(KERNEL_STATUS_OK);
  }
  std::vector<int64_t> output_shape;
  auto ret = GetBroadcastShape(input_shape_a, input_shape_b, output_shape);
  KERNEL_CHECK_FALSE(ret == KERNEL_STATUS_OK, static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID),
                     "Shape of x and mask can't be broadcast.");

  int64_t tensor_size = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int64_t>());
  if (tensor_size >= kParallelDataNums) {
    ret = ParallelCompute<T>(ctx, input_shape_a, input_shape_b, output_shape, tensor_size);
    return ret;
  }

  int64_t j = 0;
  BroadcastIterator iter(input_shape_a, input_shape_b, output_shape);
  iter.SetPos(0);
  for (int64_t i = 0; i < tensor_size; ++i) {
    if (mask[iter.GetInputPosB()]) {
      y[j++] = x[iter.GetInputPosA()];
    }
    iter.GenNextPos();
  }
  ctx.Output(0)->GetTensorShape()->SetDimSizes({j});
  return static_cast<uint32_t>(KERNEL_STATUS_OK);
}
REGISTER_CPU_KERNEL(kMaskedSelect, MaskedSelectCpuKernel);
}  // namespace aicpu
