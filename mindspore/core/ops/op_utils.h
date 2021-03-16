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

#ifndef MINDSPORE_CORE_OPS_OP_UTILS_H
#define MINDSPORE_CORE_OPS_OP_UTILS_H
#include <string>
#include <set>
#include <vector>
#include <algorithm>
#include <memory>
#include "abstract/primitive_infer_map.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kAlpha = "alpha";
constexpr auto kActivation = "activation";
constexpr auto kActivationType = "activation_type";
constexpr auto kAddress = "address";
constexpr auto kAlignCorners = "align_corners";
constexpr auto kAspectRatios = "aspect_ratios";
constexpr auto kAxes = "axes";
constexpr auto kAxis = "axis";
constexpr auto kAxisType = "axis_type";
constexpr auto kBaseSize = "base_size";
constexpr auto kBatchDim = "batch_dim";
constexpr auto kBeginMask = "begin_mask";
constexpr auto kBeginNormAxis = "begin_norm_axis";
constexpr auto kBeginParamsAxis = "begin_params_axis";
constexpr auto kBeta = "beta";
constexpr auto kBias = "bias";
constexpr auto kBidirectional = "bidirectional";
constexpr auto kBlockSize = "block_size";
constexpr auto kBlockShape = "block_shape";
constexpr auto kCellClip = "cell_clip";
constexpr auto kCellDepth = "cell_depth";
constexpr auto kCenterPointBox = "center_point_box";
constexpr auto kClip = "clip";
constexpr auto kCondition = "condition";
constexpr auto kCrops = "crops";
constexpr auto kCustom = "custom";
constexpr auto kDampening = "dampening";
constexpr auto kDataType = "data_type";
constexpr auto kDctCoeffNum = "dct_coeff_num";
constexpr auto kDelta = "delta";
constexpr auto kDependMode = "depend_mode";
constexpr auto kDepthRadius = "depth_radius";
constexpr auto kDetectionsPerClass = "detections_per_class";
constexpr auto kDilation = "dilation";
constexpr auto kDropout = "dropout";
constexpr auto kDstT = "dst_t";
constexpr auto kDType = "d_type";
constexpr auto kEllipsisMask = "ellipsis_mask";
constexpr auto kEndMask = "end_mask";
constexpr auto kEps = "eps";
constexpr auto kEpsilon = "epsilon";
constexpr auto kElement_dtype = "element_dtype";
constexpr auto kFeatStride = "feat_stride";
constexpr auto kFftLength = "fft_length";
constexpr auto kFilterBankChannelNum = "filter_bank_channel_num";
constexpr auto kFlip = "flip";
constexpr auto kFormat = "format";
constexpr auto kFreqLowerLimit = "freq_lower_limit";
constexpr auto kFreqUpperLimit = "freq_upper_limit";
constexpr auto kFreezeBn = "freeze_bn";
constexpr auto kGateOrder = "gate_order";
constexpr auto kGlobal = "global";
constexpr auto kGrad = "grad";
constexpr auto kIsGrad = "is_grad";
constexpr auto kGradientScale = "gradient_scale";
constexpr auto kGradX = "grad_x";
constexpr auto kGradY = "grad_y";
constexpr auto kGroup = "group";
constexpr auto kHasBias = "has_bias";
constexpr auto kHiddenSize = "hidden_size";
constexpr auto kId = "id";
constexpr auto kImageSizeH = "image_size_h";
constexpr auto kImageSizeW = "image_size_w";
constexpr auto kIncludeALLGrams = "include_all_grams";
constexpr auto kInputSize = "input_size";
constexpr auto kInChannel = "in_channel";
constexpr auto kInputShape = "input_shape";
constexpr auto kIoFormat = "io_format";
constexpr auto kIsScale = "is_scale";
constexpr auto kIsTraining = "is_training";
constexpr auto kKeepDims = "keep_dims";
constexpr auto kKeepProb = "keep_prob";
constexpr auto kKernelSize = "kernel_size";
constexpr auto kLimit = "limit";
constexpr auto kMagSquare = "mag_square";
constexpr auto kMax = "max";
constexpr auto kMaxSizes = "max_sizes";
constexpr auto kMaxSkipSize = "max_skip_size";
constexpr auto kMaxClassesPerDetection = "max_classes_per_detection";
constexpr auto kMaxDetections = "max_detections";
constexpr auto kMaxNorm = "max_norm";
constexpr auto kMin = "min";
constexpr auto kMinSize = "min_size";
constexpr auto kMinSizes = "min_sizes";
constexpr auto kMode = "mode";
constexpr auto kMomentum = "momentum";
constexpr auto kN = "n";
constexpr auto kNarrowRange = "narrow_range";
constexpr auto kNesterov = "nesterov";
constexpr auto kNewAxisMask = "new_axis_mask";
constexpr auto kNgramSize = "ngram_size";
constexpr auto kNmsThresh = "nms_thresh";
constexpr auto kNormRegion = "norm_region";
constexpr auto kNumLayers = "num_layers";
constexpr auto kNumElements = "num_elements";
constexpr auto kNumBits = "num_bits";
constexpr auto kNumDirections = "num_directions";
constexpr auto kNumProj = "num_proj";
constexpr auto kOffset = "offset";
constexpr auto kNmsIouThreshold = "nms_iou_threshold";
constexpr auto kNmsScoreThreshold = "nms_score_threshold";
constexpr auto kNumClasses = "num_classes";
constexpr auto kOffsets = "offsets";
constexpr auto kOffsetA = "offset_a";
constexpr auto kOrder = "order";
constexpr auto kOutChannel = "out_channel";
constexpr auto kOutMaxValue = "out_max_value";
constexpr auto kOutputChannel = "output_channel";
constexpr auto kOutputNum = "output_num";
constexpr auto koutputPaddings = "output_paddings";
constexpr auto kOutputType = "output_type";
constexpr auto kOutQuantized = "out_quantized";
constexpr auto kP = "p";
constexpr auto kPad = "pad";
constexpr auto kPadding = "padding";
constexpr auto kPaddingsElementSize = "paddings_element_size";
constexpr auto kPaddingsSize = "paddings_size";
constexpr auto kPadItem = "pad_item";
constexpr auto kPadList = "pad_list";
constexpr auto kPadMode = "pad_mode";
constexpr auto kPads = "pads";
constexpr auto kPadSize = "pad_size";
constexpr auto kPooledH = "pooled_h";
constexpr auto kPooledW = "pooled_w";
constexpr auto kPoolMode = "pool_mode";
constexpr auto kPostNmsTopn = "post_nms_topn";
constexpr auto kPower = "power";
constexpr auto kPreNmsTopn = "pre_nms_topn";
constexpr auto kRatio = "ratio";
constexpr auto kReduction = "reduction";
constexpr auto kRootRank = "root_rank";
constexpr auto kRoundMode = "round_mode";
constexpr auto kSame = "same";
constexpr auto kScale = "scale";
constexpr auto kSeed = "seed";
constexpr auto kSeed2 = "seed2";
constexpr auto kSeqDim = "seq_dim";
constexpr auto kSetattrFlag = "setattr_flag";
constexpr auto kShape = "shape";
constexpr auto kShapeSize = "shape_size";
constexpr auto kShift = "shift";
constexpr auto kShrinkAxisMask = "shrink_axis_mask";
constexpr auto kSize = "size";
constexpr auto kSorted = "sorted";
constexpr auto kSrcT = "src_t";
constexpr auto kStart = "start";
constexpr auto kStepH = "step_h";
constexpr auto kStepW = "step_w";
constexpr auto kStride = "stride";
constexpr auto kStrides = "strides";
constexpr auto kShapeType = "shape_type";
constexpr auto kSubGraphIndex = "sub_graph_index";
constexpr auto kSummarize = "summarize";
constexpr auto kTimeMajor = "time_major";
constexpr auto kTopK = "top_k";
constexpr auto kTransposeA = "transpose_a";
constexpr auto kTransposeB = "transpose_b";
constexpr auto kNegativeSlope = "negative_slope";
constexpr auto kType = "type";
constexpr auto kUseAxis = "use_axis";
constexpr auto kUseLocking = "use_locking";
constexpr auto kUseNesterov = "use_nesterov";
constexpr auto kUseNesteroy = "use_nesteroy";
constexpr auto kUseRegularNms = "use_regular_nms";
constexpr auto kValid = "valid";
constexpr auto kValue = "value";
constexpr auto kVariances = "variances";
constexpr auto kWeightDecay = "weight_decay";
constexpr auto kWeightThreshold = "weight_threshold";
constexpr auto kWindow = "window";
constexpr auto kWindowSize = "window_size";
constexpr auto kPaddings = "paddings";
constexpr auto kInput_size = "input_size";
constexpr auto kHidden_size = "hidden_size";
constexpr auto kChannelShared = "channel_shared";
constexpr auto kSlope = "slope";
constexpr auto kBase = "base";
constexpr auto kConstantValue = "constant_value";
constexpr auto kSizeSplits = "size_splits";
constexpr auto kDims = "dims";
constexpr auto kPaddingMode = "padding_mode";
constexpr auto kLargest = "largest";
constexpr auto kElementwiseAffine = "elementwise_affine";
constexpr auto kMinVal = "min_val";
constexpr auto kMaxVal = "max_val";
constexpr auto kMethod = "method";
constexpr auto kNewHeight = "new_height";
constexpr auto kNewWidth = "new_width";
constexpr auto kPreserveAspectRatio = "preserve_aspect_ratio";
constexpr auto kCoordinateTransformMode = "coordinate_transform_mode";
constexpr auto kCubicCoeff = "cubic_coeff";
constexpr auto kExcludeOutside = "exclude_outside";
constexpr auto kExtrapolationValue = "extrapolation_value";
constexpr auto kNearestMode = "nearest_mode";
constexpr auto kReduceToEnd = "reduce_to_end";
constexpr auto kResetAfter = "reset_after";
constexpr auto kCoeff = "coeff";
constexpr auto kIsDepthWise = "is_depth_wise";
constexpr auto kZoneoutCell = "zoneout_cell";
constexpr auto kZoneoutHidden = "zoneout_hidden";
constexpr auto kSpliceContext = "context";
constexpr auto kSpliceForwardIndexes = "forward_indexes";
constexpr auto kSpliceOutputDims = "output_dim";

const std::set<TypeId> common_valid_types = {
  kNumberTypeInt8,   kNumberTypeInt16,  kNumberTypeInt32,   kNumberTypeInt64,   kNumberTypeUInt8,  kNumberTypeUInt16,
  kNumberTypeUInt32, kNumberTypeUInt64, kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64};

const std::set<TypeId> all_types = {
  kNumberTypeBool,    kNumberTypeInt,     kNumberTypeInt8,    kNumberTypeInt16,     kNumberTypeInt32,  kNumberTypeInt64,
  kNumberTypeUInt,    kNumberTypeUInt8,   kNumberTypeUInt16,  kNumberTypeUInt32,    kNumberTypeUInt64, kNumberTypeFloat,
  kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64, kNumberTypeComplex64,
};

abstract::ShapePtr BroadCastInferShape(const std::string &op_name, const std::vector<AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_OP_UTILS_H
