/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AICPU_AICPU_UTIL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AICPU_AICPU_UTIL_H_

#include <cstdint>
#include <utility>
#include <memory>
#include <vector>
#include <map>
#include <set>
#include <string>
#include "kernel/kernel.h"
namespace mindspore {
namespace kernel {
constexpr auto kLibAicpuKernelSoName = "libaicpu_kernels.so";
constexpr auto kLibCpuKernelSoName = "libcpu_kernels.so";
constexpr auto kFormat = "format";
constexpr auto kDataFormat = "data_format";
constexpr auto kInitDataSetQueue = "InitDataSetQueue";
constexpr auto kInitData = "InitData";
constexpr auto kGetNext = "GetNext";
constexpr auto kPrint = "Print";
constexpr auto kPack = "Pack";
constexpr auto kCumSum = "CumSum";
constexpr auto kCumProd = "CumProd";
constexpr auto kMeshgrid = "Meshgrid";
constexpr auto kOutputTypes = "output_types";
constexpr auto kOutputShapes = "output_shapes";
constexpr auto kChannelName = "channel_name";
constexpr auto kSharedName = "shared_name";
constexpr auto kShapes = "shapes";
constexpr auto kTypes = "types";
constexpr auto kQueueName = "queue_name";
constexpr auto kNameRangeV2 = "RangeV2";
constexpr auto kSeed = "seed";
constexpr auto kSeed0 = "Seed0";
constexpr auto kSeed1 = "Seed1";
constexpr auto kSeed2 = "seed2";
constexpr auto kTopK = "TopK";
constexpr auto kTopKV2 = "TopKV2";
constexpr auto kStack = "Stack";
constexpr auto kUnstack = "Unstack";
constexpr auto kStackInit = "StackInit";
constexpr auto kStackPush = "StackPush";
constexpr auto kStackPop = "StackPop";
constexpr auto kStackDestroy = "StackDestroy";
constexpr auto kEditDistance = "EditDistance";
constexpr auto kGatherD = "GatherD";
constexpr auto kGather = "Gather";
constexpr auto kHistogram = "Histogram";
constexpr auto kIdentity = "Identity";
constexpr auto kIdentityN = "IdentityN";
constexpr auto kRandomChoiceWithMask = "RandomChoiceWithMask";
constexpr auto kGatherDGradV2 = "GatherDGradV2";
constexpr auto kResizeNearestNeighborV2 = "ResizeNearestNeighborV2";
constexpr auto kResizeNearestNeighborV2Grad = "ResizeNearestNeighborV2Grad";
constexpr auto kUpdateCache = "UpdateCache";
constexpr auto kCacheSwapTable = "CacheSwapTable";
constexpr auto kSubAndFilter = "SubAndFilter";
constexpr auto kPadAndShift = "PadAndShift";
constexpr auto kCpuRunApi = "RunCpuKernel";
constexpr auto kDropout2D = "Dropout2D";
constexpr auto kDropout3D = "Dropout3D";
constexpr auto kNonMaxSuppressionV3 = "NonMaxSuppressionV3";
constexpr auto kMaskedSelect = "MaskedSelect";
constexpr auto kMaskedSelectGrad = "MaskedSelectGrad";
constexpr auto kDynamicStitch = "DynamicStitch";
constexpr auto kSearchSorted = "SearchSorted";
constexpr auto kResizeBilinear = "ResizeBilinear";
constexpr auto kResizeBilinearGrad = "ResizeBilinearGrad";
constexpr auto kTensorScatterElements = "TensorScatterElements";
constexpr auto kExtractGlimpse = "ExtractGlimpse";
constexpr auto kUpsampleNearest3D = "UpsampleNearest3D";
constexpr auto kUpsampleNearest3DGrad = "UpsampleNearest3DGrad";
constexpr auto kEnvironCreate = "EnvironCreate";
constexpr auto kEnvironSet = "EnvironSet";
constexpr auto kEnvironGet = "EnvironGet";
constexpr auto kEnvironDestroyAll = "EnvironDestroyAll";
constexpr auto kSampleDistortedBoundingBoxV2 = "SampleDistortedBoundingBoxV2";
constexpr auto kSparseToDenseV2 = "SparseToDenseV2";
constexpr auto kSparseSoftmaxCrossEntropyWithLogitsV2 = "SparseSoftmaxCrossEntropyWithLogitsV2";
constexpr auto kPriorityReplayBufferCreate = "PriorityReplayBufferCreate";
constexpr auto kPriorityReplayBufferPush = "PriorityReplayBufferPush";
constexpr auto kPriorityReplayBufferSample = "PriorityReplayBufferSample";
constexpr auto kPriorityReplayBufferUpdate = "PriorityReplayBufferUpdate";
constexpr auto kPriorityReplayBufferDestroy = "PriorityReplayBufferDestroy";
constexpr auto kReLUV3 = "ReLUV3";
constexpr auto kNonZero = "NonZero";
constexpr auto kMaxPoolV1 = "MaxPoolV1";
constexpr auto kMaxPoolGradV1 = "MaxPoolGradV1";
constexpr auto kAvgPoolV1 = "AvgPoolV1";
constexpr auto kAvgPoolGradV1 = "AvgPoolGradV1";
constexpr auto kUniqueConsecutive = "UniqueConsecutive";
constexpr auto kRandomShuffle = "RandomShuffle";

const std::set<std::string> kCpuKernelOps{kIdentity,           kMaskedSelect,          kMaskedSelectGrad,
                                          kDynamicStitch,      kSearchSorted,          kResizeBilinear,
                                          kResizeBilinearGrad, kTensorScatterElements, kUniqueConsecutive};
const std::set<std::string> kCacheKernelOps{kUpdateCache, kCacheSwapTable,      kSubAndFilter, kPadAndShift, kDropout3D,
                                            kDropout2D,   kNonMaxSuppressionV3, kGetNext,      kInitData,    kPrint};
const std::set<std::string> kCpuKernelBaseOps{kRandomChoiceWithMask,
                                              kEnvironCreate,
                                              kEnvironSet,
                                              kEnvironGet,
                                              kEnvironDestroyAll,
                                              kPriorityReplayBufferCreate,
                                              kPriorityReplayBufferPush,
                                              kPriorityReplayBufferSample,
                                              kPriorityReplayBufferUpdate,
                                              kPriorityReplayBufferDestroy,
                                              kGatherDGradV2,
                                              kRandomShuffle};
const std::set<std::string> kDynamicInputOps{
  kPrint,           kPack,           kMeshgrid,      kStackInitOpName,          kStackDestroyOpName,
  kStackPushOpName, kStackPopOpName, kDynamicStitch, kPriorityReplayBufferPush, kPriorityReplayBufferSample,
  kIdentityN};
const std::map<std::string, std::string> kOpNameToAicpuOpNameMap{
  {kHistogram, "HistogramD"},
  {kMaxPoolV1, "MaxPool"},
  {kMaxPoolGradV1, "MaxPoolGrad"},
  {kUpsampleNearest3D, "UpsampleNearest3d"},
  {kUpsampleNearest3DGrad, "UpsampleNearest3dGrad"},
  {kNameRangeV2, "Range"},
  {kReLUV3, "Relu"},
  {kStack, "Pack"},
  {kUnstack, "Unpack"},
  {kGather, "GatherV2"},
  {kCumSum, "Cumsum"},
  {kCumProd, "Cumprod"},
  {kSampleDistortedBoundingBoxV2, "SampleDistortedBoundingBoxExt2"},
  {kSparseSoftmaxCrossEntropyWithLogitsV2, "SparseSoftmaxCrossEntropyWithLogits"},
  {kSparseToDenseV2, "SparseToDense"},
  {kAvgPoolV1, "AvgPool"},
  {kNonZero, "Where"},
  {kAvgPoolGradV1, "AvgPoolGrad"},
  {kTensorScatterElements, "ScatterElements"}};
struct AicpuParamHead {
  uint32_t length;         // Total length: include cunstom message
  uint32_t ioAddrNum;      // Input and output address number
  uint32_t extInfoLength;  // extInfo struct Length
  uint64_t extInfoAddr;    // extInfo address
} __attribute__((packed));

// Extent info ShapeAndType
const uint32_t kMaxShapeDims = 8;
struct ShapeAndType {
  int32_t type;
  int64_t dims[kMaxShapeDims];
} __attribute__((packed));

// Extend info structure for extInfoAddr
const uint32_t kExtInfoHeadSize = 8;
struct ExtInfo {
  int32_t infoType;  // extend type
  uint32_t infoLen;  // length for infoMsg
  char infoMsg[0];   // extend value
} __attribute__((packed));

// Extend Info type for task
enum FWKTaskExtInfoType {
  FWK_ADPT_EXT_SHAPE_TYPE = 0,
  FWK_ADPT_EXT_INPUT_SHAPE,
  FWK_ADPT_EXT_OUTPUT_SHAPE,
  FWK_ADPT_EXT_INVALID
};

// for unknown shape op type
enum UnknowShapeOpType {
  DEPEND_IN_SHAPE = 1,     // op out shape get by input shape
  DEPEND_CONST_VALUE = 2,  // op out shape get by const op value
  DEPEND_SHAPE_RANGE = 3,  // op out shape get by range
  DEPEND_COMPUTE = 4       // op out shape get by totally computing
};

class AicpuOpUtil {
 public:
  static int MsTypeToProtoType(TypeId ms_type);
  static int ProtoTypeToMsType(int proto_type);

 private:
  // kernel id
  static uint64_t KernelId_;
};

class OpKernelBin {
 public:
  OpKernelBin(std::string name, std::vector<char> &&data) : name_(std::move(name)), data_(std::move(data)) {}
  ~OpKernelBin() = default;

  const std::string &GetName() const { return name_; }
  const uint8_t *GetBinData() const { return reinterpret_cast<const uint8_t *>(data_.data()); }
  size_t GetBinDataSize() const { return data_.size(); }
  OpKernelBin(const OpKernelBin &) = delete;
  const OpKernelBin &operator=(const OpKernelBin &) = delete;

  bool loaded() const { return loaded_; }
  void SetLoaded(bool flag) { loaded_ = flag; }

 private:
  std::string name_;
  std::vector<char> data_;
  bool loaded_{false};
};

using OpKernelBinPtr = std::shared_ptr<OpKernelBin>;
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AICPU_AICPU_UTIL_H_
