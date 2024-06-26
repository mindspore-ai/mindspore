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

#include "include/common/utils/utils.h"

#if defined(_WIN32) || defined(_WIN64) || defined(__APPLE__)
#else
#include <sys/statfs.h>
#endif
#include <set>
#include <string>
#include "include/common/utils/parallel_context.h"
#include "ops/array_op_name.h"
#include "ops/ascend_op_name.h"
#include "ops/conv_pool_op_name.h"
#include "ops/framework_op_name.h"
#include "ops/image_op_name.h"
#include "ops/math_op_name.h"
#include "ops/nn_op_name.h"
#include "ops/nn_optimizer_op_name.h"
#include "ops/other_op_name.h"
#include "ops/random_op_name.h"
#include "ops/sequence_op_name.h"
#include "ops/sparse_op_name.h"
#include "ops/structure_op_name.h"
#include "utils/convert_utils_base.h"
#include "utils/ms_context.h"

#if !defined(BUILD_LITE)
#include "pybind11/pybind11.h"
namespace py = pybind11;
#endif

namespace mindspore {
namespace {
constexpr size_t kKBToByte = 1024;
constexpr size_t kLineMaxSize = 1024;
}  // namespace
bool IsOneOfPosteriorOperator(const std::string &name) {
  static const std::set<std::string> kPosteriorOperatorSet = {kPullOpName};

  auto iter = kPosteriorOperatorSet.find(name);
  return iter != kPosteriorOperatorSet.end();
}

bool IsOneOfCacheBlackList(const std::string &name) {
  static const std::set<std::string> kOpCacheBlackList = {kUniformCandidateSamplerOpName, kInitDatasetQueueOpName,
                                                          kGetNextOpName};

  auto iter = kOpCacheBlackList.find(name);
  return iter != kOpCacheBlackList.end();
}

std::vector<std::tuple<std::string, int, std::string>> GetPythonStack_() {
  std::vector<std::tuple<std::string, int, std::string>> all_stacks;
#if !defined(BUILD_LITE)
  try {
    const size_t func_name_index = 2;
    const size_t min_frame_info_size = 3;
    py::gil_scoped_acquire gil_acquire;
    py::module traceback_module = py::module::import("traceback");
    py::list extracted_stack = traceback_module.attr("extract_stack")();
    for (size_t i = 0; i < extracted_stack.size(); ++i) {
      py::tuple frame_info = extracted_stack[i].cast<py::tuple>();
      if (frame_info.size() < min_frame_info_size) {
        MS_LOG(ERROR) << "frame_info size is invalid, frame_info size:" << frame_info.size();
        continue;
      }
      // frame_info: (filename, line number, function name, code_context)
      std::string file_name = frame_info[0].cast<std::string>();
      int line_number = frame_info[1].cast<int>();
      std::string func_name = frame_info[func_name_index].cast<std::string>();
      (void)all_stacks.emplace_back(std::tuple(file_name, line_number, func_name));
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Error while accessing Python stack: " << e.what();
  }
#endif

  return all_stacks;
}

std::string GetPythonStackStr_() {
  const auto &stacks = GetPythonStack_();
  const size_t func_name_index = 2;

  std::stringstream ss;
  for (const auto &stack_info : stacks) {
    ss << "File:" << std::get<0>(stack_info) << ";Line:" << std::get<1>(stack_info)
       << ";Function:" << std::get<func_name_index>(stack_info) << '|';
  }
  return ss.str();
}

bool IsOneOf3DFormat(const std::string &format) {
  static const std::set<std::string> k3DFormatSet = {kOpFormat_NCDHW, kOpFormat_NDC1HWC0, kOpFormat_FRACTAL_Z_3D,
                                                     kOpFormat_NDHWC, kOpFormat_DHWCN,    kOpFormat_DHWNC};

  auto iter = k3DFormatSet.find(format);
  return iter != k3DFormatSet.end();
}

bool IsOneOfNoPaddingFormat(const std::string &format) {
  static const std::set<std::string> kNoPaddingFormatSet = {
    kOpFormat_ChannelLast, kOpFormat_FRAC_NZ, kOpFormat_FRACTAL_ZN_RNN, kOpFormat_ND_RNN_BIAS, kOpFormat_DEFAULT};

  auto iter = kNoPaddingFormatSet.find(format);
  return iter != kNoPaddingFormatSet.end();
}

bool IsOneOfDynamicShapeConstInputToAttrGPU(const std::string &name) {
  static const std::set<std::string> DynamicShapeConstInputToAttrGPU = {
    kCastOpName,      kExpandDimsOpName, kReshapeOpName,    kEmbeddingLookupOpName, kTransposeOpName,
    kReduceSumOpName, kReduceMinOpName,  kReduceMeanOpName, kReduceMaxOpName,       kReduceAllOpName,
    kReduceAnyOpName, kConcatOpName,     kScatterNdOpName,  kGatherOpName,          kAvgPool3DGradOpName};

  auto iter = DynamicShapeConstInputToAttrGPU.find(name);
  return iter != DynamicShapeConstInputToAttrGPU.end();
}

bool IsOneOfCustomAkgType(const std::string &name) {
  const std::set<std::string> kCustomTypeAkg = {"ir_builder", "tvm_compute", "hybrid"};

  auto iter = kCustomTypeAkg.find(name);
  return iter != kCustomTypeAkg.end();
}

bool IsOneOfOperator(const std::string &name) {
  static const std::set<std::string> kOptOperatorSet = {kMomentumOpName,
                                                        kApplyMomentumOpName,
                                                        kApplyMomentumDOpName,
                                                        kApplyAdadeltaOpName,
                                                        kApplyAdadeltaDOpName,
                                                        kApplyAdagradOpName,
                                                        kApplyAdagradDOpName,
                                                        kApplyAdagradDAOpName,
                                                        kApplyAdagradDADOpName,
                                                        kAdamOpName,
                                                        kApplyAdamDOpName,
                                                        kApplyAdamOpName,
                                                        kApplyAdaMaxOpName,
                                                        kApplyAdaMaxDOpName,
                                                        kApplyAddSignOpName,
                                                        kApplyAddSignDOpName,
                                                        kApplyCenteredRMSPOpName,
                                                        kApplyFtrlOpName,
                                                        kApplyFtrlDOpName,
                                                        kApplyFtrlV2OpName,
                                                        kApplyFtrlV2DOpName,
                                                        kApplyGradientDescentOpName,
                                                        kApplyPowerSignOpName,
                                                        kApplyPowerSignDOpName,
                                                        kApplyProximalAdagradOpName,
                                                        kApplyProximalAdagradDOpName,
                                                        kApplyProximalGradientDescentOpName,
                                                        kApplyRMSPropOpName,
                                                        kApplyRMSPropDOpName,
                                                        kAdamApplyOneWithDecayOpName,
                                                        kAdamApplyOneWithDecayAssignOpName,
                                                        kFusedAdamWeightDecayOpName,
                                                        kAdamWeightDecayOpName,
                                                        kFusedCastAdamWeightDecayOpName,
                                                        kFusedAdamOpName,
                                                        kFusedAdaFactorOpName,
                                                        kFusedAdaFactorWithGlobalNormOpName,
                                                        kFusedSparseAdamOpName,
                                                        kFusedMulApplyMomentumOpName,
                                                        kFusedWeightScaleApplyMomentumOpName,
                                                        kFusedScaleApplyMomentumOpName,
                                                        kApplyCenteredRMSPropOpName,
                                                        kApplyCenteredRMSPropDOpName,
                                                        kFusedSparseFtrlOpName,
                                                        kFusedSparseProximalAdagradOpName,
                                                        kFusedSparseLazyAdamOpName,
                                                        kSparseApplyFtrlOpName,
                                                        kSparseApplyFtrlDOpName,
                                                        kSparseApplyFtrlV2OpName,
                                                        kSparseApplyFtrlV2DOpName,
                                                        kSGDOpName,
                                                        kLARSUpdateOpName,
                                                        kLarsV2UpdateOpName,
                                                        kCombineWeightDecayScaleMomentumOpName,
                                                        kCombineScaleMomentumOpName,
                                                        kCombineMomentumOpName,
                                                        kScatterAddOpName,
                                                        kScatterUpdateOpName,
                                                        kSparseApplyProximalAdagradOpName,
                                                        kSparseApplyProximalAdagradDOpName,
                                                        kAdaptiveMaxPool2dOpName,
                                                        kApplyKerasMomentumDOpName};

  auto iter = kOptOperatorSet.find(name);
  return iter != kOptOperatorSet.end();
}

bool IsOneOfNotSupportedTransFormat(const std::string &format) {
  static const std::set<std::string> kNotSupportedFormat = {kOpFormat_DHWCN, kOpFormat_NDHWC, kOpFormat_CHWN};
  return (kNotSupportedFormat.find(format) != kNotSupportedFormat.end());
}

bool IsOneOfComputeDepend(const std::string &name) {
  static const std::set<std::string> kComputeDepend = {kUniqueOpName,
                                                       kUniqueConsecutiveOpName,
                                                       kComputeAccidentalHitsOpName,
                                                       kSubAndFilterOpName,
                                                       kPadAndShiftOpName,
                                                       kCTCGreedyDecoderOpName,
                                                       kMaskedSelectOpName,
                                                       kDynamicStitchOpName,
                                                       kGetNextOpName,
                                                       kListDiffOpName,
                                                       kNonMaxSuppressionV3OpName,
                                                       kNonMaxSuppressionWithOverlapsOpName,
                                                       kCoalesceOpName,
                                                       kTruncatedNormalOpName,
                                                       kNonDeterministicIntsOpName,
                                                       kFractionalAvgPoolGradOpName,
                                                       kDenseToDenseSetOperationOpName,
                                                       kDenseToSparseSetOperationOpName,
                                                       kSegmentMaxOpName,
                                                       kCSRSparseMatrixToSparseTensorOpName,
                                                       kSegmentMinOpName,
                                                       kLuUnpackOpName,
                                                       kSegmentSumOpName,
                                                       kResizeBicubicOpName,
                                                       kResizeAreaOpName,
                                                       kSegmentMeanOpName,
                                                       kSegmentProdOpName,
                                                       kSparseSliceOpName,
                                                       kNonZeroOpName,
                                                       kSparseSparseMinimumOpName,
                                                       kSparseSparseMaximumOpName,
                                                       kRpcRecvOpName,
                                                       kSparseFillEmptyRowsOpName,
                                                       kSparseCrossOpName,
                                                       kAdaptiveMaxPool3DOpName,
                                                       kDynamicBroadcastGradientArgsOpName};

  auto iter = kComputeDepend.find(name);
  return iter != kComputeDepend.end();
}

bool IsOneOfUnsignedType(const TypeId &type_id) {
  static const std::set<TypeId> unsigned_types{kNumberTypeUInt8, kNumberTypeUInt16, kNumberTypeUInt32,
                                               kNumberTypeUInt64};
  return unsigned_types.count(type_id) > 0;
}

bool IsOneOfHWSpecialFormat(const std::string &format) {
  static const std::set<std::string> kHWSpecialFormatSet = {
    kOpFormat_FRACTAL_Z_3D,   kOpFormat_NC1KHKWHWC0, kOpFormat_NC1HWC0,       kOpFormat_FRAC_NZ,
    kOpFormat_C1HWNCoC0,      kOpFormat_NC1HWC0_C04, kOpFormat_FRACTAL_Z_C04, kOpFormat_FRACTAL_ZN_LSTM,
    kOpFormat_FRACTAL_ZN_RNN, kOpFormat_NDC1HWC0,    kOpFormat_FRAC_Z};

  auto iter = kHWSpecialFormatSet.find(format);
  return iter != kHWSpecialFormatSet.end();
}

bool IsOneOfDefaultFormat(const std::string &format) {
  static const std::set<std::string> kOpDefaultFormatList = {kOpFormat_DEFAULT, kOpFormat_ND, kOpFormat_NCDHW,
                                                             kOpFormat_NCHW};
  return kOpDefaultFormatList.find(format) != kOpDefaultFormatList.end();
}

bool IsOneOfFormat(const std::string &format) {
  static const std::set<std::string> kOpFormatList = {
    kOpFormat_DEFAULT,        kOpFormat_NC1KHKWHWC0,  kOpFormat_ND,
    kOpFormat_NCHW,           kOpFormat_NHWC,         kOpFormat_HWCN,
    kOpFormat_CHWN,           kOpFormat_NC1HWC0,      kOpFormat_FRAC_Z,
    kOpFormat_C1HWNCoC0,      kOpFormat_FRAC_NZ,      kOpFormat_NC1HWC0_C04,
    kOpFormat_FRACTAL_Z_C04,  kOpFormat_NDHWC,        kOpFormat_FRACTAL_ZN_LSTM,
    kOpFormat_FRACTAL_ZN_RNN, kOpFormat_ND_RNN_BIAS,  kOpFormat_NDC1HWC0,
    kOpFormat_NCDHW,          kOpFormat_FRACTAL_Z_3D, kOpFormat_DHWNC,
    kOpFormat_DHWCN};

  auto iter = kOpFormatList.find(format);
  return iter != kOpFormatList.end();
}

bool IsOneOfServerFormatC04(const std::string &format) {
  static const std::set<std::string> kServerFormatC04List = {kOpFormat_NC1HWC0_C04, kOpFormat_FRACTAL_Z_C04};
  return kServerFormatC04List.find(format) != kServerFormatC04List.end();
}

bool IsOneOfDynRankNeedPadShape(const std::string &format) {
  const std::set<std::string> kOpFormats = {kOpFormat_NC1HWC0,      kOpFormat_NDC1HWC0,      kOpFormat_FRAC_Z,
                                            kOpFormat_NDC1HWC0,     kOpFormat_C1HWNCoC0,     kOpFormat_NC1HWC0_C04,
                                            kOpFormat_FRACTAL_Z_3D, kOpFormat_FRACTAL_Z_C04, kOpFormat_NCDHW};
  return kOpFormats.find(format) != kOpFormats.end();
}

bool IsEnableRefMode() {
  static bool ret = !(common::GetEnv("MS_DISABLE_REF_MODE") == "1");
  return ret;
}

bool IsMemoryPoolRecycle() {
  static bool optimize_mem = !common::IsDisableAlllocConfig(common::kAllocMemoryRecycle);
  static bool enable_ref_mode = IsEnableRefMode();
  auto context_ptr = MsContext::GetInstance();
  auto mode = context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE);
  auto task_sink = context_ptr->get_param<bool>(MS_CTX_ENABLE_TASK_SINK);
  return optimize_mem && enable_ref_mode && mode == kGraphMode && task_sink;
}

size_t GetSystemMemorySize(const std::string &key) {
#if defined(_WIN32) || defined(_WIN64) || defined(__APPLE__)
  return SIZE_MAX;
#else
  FILE *file = fopen("/proc/meminfo", "r");
  if (file == nullptr) {
    MS_LOG(ERROR) << "Get system meminfo failed.";
    return 0;
  }

  size_t mem_size = 0;
  char buf[kLineMaxSize] = {0};
  while (fgets(buf, kLineMaxSize, file)) {
    // Get mem title.
    std::string line(buf);
    auto title_end_pos = line.find(":");
    auto title = line.substr(0, title_end_pos);
    // Get mem size.
    if (title == key) {
      auto mem_size_end_pos = line.find_last_of(" ");
      auto mem_size_begin_pos = line.find_last_of(" ", mem_size_end_pos - 1);
      if ((mem_size_end_pos != std::string::npos) && (mem_size_begin_pos != std::string::npos)) {
        auto mem_size_string = line.substr(mem_size_begin_pos, mem_size_end_pos - mem_size_begin_pos);
        mem_size = LongToSize(std::stol(mem_size_string));
      }
      break;
    }
    if (memset_s(buf, kLineMaxSize, 0, kLineMaxSize) != EOK) {
      MS_LOG(ERROR) << "Set system meminfo failed.";
      (void)fclose(file);
      return 0;
    }
  }
  (void)fclose(file);

  MS_LOG(INFO) << "Get system memory(" << key << "): " << mem_size << " kB";
  return mem_size * kKBToByte;
#endif
}

size_t GetSystemFreeDiskSize(const std::string &path) {
#if defined(_WIN32) || defined(_WIN64) || defined(__APPLE__)
  // Do not implement
  return 0;
#else
  struct statfs disk_info;
  int ret = statfs(path.c_str(), &disk_info);
  if (ret != 0) {
    MS_LOG(INFO) << "Failed to get disk directory " << path << " size, check whether the directory is created.";
    return 0;
  }
  size_t block_size = static_cast<size_t>(disk_info.f_bsize);
  size_t fb_size = static_cast<size_t>(disk_info.f_bfree);
  return block_size * fb_size;
#endif
}

bool SkipOrResetCopyAction(bool need_reset) {
  static bool copy_action = false;
  if (need_reset) {
    MS_LOG(INFO) << "Step end, reset copy action flag";
    copy_action = false;
    return true;
  }
  if (!copy_action) {
    copy_action = true;
    return true;
  }
  return false;
}

bool SkipOrResetSyncAction(bool need_reset) {
  static bool sync_action = false;
  if (need_reset) {
    MS_LOG(INFO) << "Step end, reset sync action flag";
    sync_action = false;
    return true;
  }
  if (!sync_action) {
    sync_action = true;
    return true;
  }
  return false;
}

std::vector<std::tuple<std::string, int, std::string>> GetPythonStack() {
  std::vector<std::tuple<std::string, int, std::string>> all_stacks;
#if !defined(BUILD_LITE)
  try {
    const size_t func_name_index = 2;
    const size_t min_frame_info_size = 3;
    py::gil_scoped_acquire gil_acquire;
    py::module traceback_module = py::module::import("traceback");
    py::list extracted_stack = traceback_module.attr("extract_stack")();
    for (size_t i = 0; i < extracted_stack.size(); ++i) {
      py::tuple frame_info = extracted_stack[i].cast<py::tuple>();
      if (frame_info.size() < min_frame_info_size) {
        MS_LOG(ERROR) << "frame_info size is invalid, frame_info size:" << frame_info.size();
        continue;
      }

      // frame_info: (filename, line number, function name, code_context)
      std::string file_name = frame_info[0].cast<std::string>();
      int line_number = frame_info[1].cast<int>();
      std::string func_name = frame_info[func_name_index].cast<std::string>();
      (void)all_stacks.emplace_back(std::tuple(file_name, line_number, func_name));
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Error while accessing Python stack: " << e.what();
  }
#endif

  return all_stacks;
}

std::string GetPythonStackStr() {
  const auto &stacks = GetPythonStack();
  const size_t func_name_index = 2;

  std::stringstream ss;
  for (const auto &stack_info : stacks) {
    ss << "File:" << std::get<0>(stack_info) << ";Line:" << std::get<1>(stack_info)
       << ";Function:" << std::get<func_name_index>(stack_info) << '|';
  }
  return ss.str();
}
}  // namespace mindspore
