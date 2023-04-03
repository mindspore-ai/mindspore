/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_COMMON_OP_ATTR_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_COMMON_OP_ATTR_H_

namespace mindspore {
namespace dpico {
constexpr auto kAcrossSpatial = "across_spatial";
constexpr auto kAcrossChannels = "across_channels";
constexpr auto kActivateAlpha = "activate_alpha";
constexpr auto kActivateBeta = "activate_beta";
constexpr auto kActivateType = "activate_type";
constexpr auto kAfClip = "af_clip";
constexpr auto kBiasTerm = "bias_term";
constexpr auto kBilinear = "bilinear";
constexpr auto kBlockHeight = "block_height";
constexpr auto kBlockWidth = "block_width";
constexpr auto kChannelShared = "channel_shared";
constexpr auto kClip = "clip";
constexpr auto kCoeffs = "coeffs";
constexpr auto kCustomName = "custom_";
constexpr auto kCustomParamSize = "custom_param_size";
constexpr auto kDetectionBackgroundLabelId = "detection_background_label_id";
constexpr auto kDetectionBiasVec = "detection_bias_vec";
constexpr auto kDetectionCalcMode = "detection_calc_mode";
constexpr auto kDetectionClipBbox = "detection_clip_bbox";
constexpr auto kDetectionCodeType = "detection_code_type";
constexpr auto kDetectionMultiClassSorting = "detection_multi_class_sorting";
constexpr auto kDetectionOutputParam = "detection_output_param";
constexpr auto kDetectionOutputParamSize = "detection_output_param_size";
constexpr auto kDetectionProposalParamType = "detection_proposal_param_type";
constexpr auto kDetectionReportFlag = "detection_report_flag";
constexpr auto kDetectionShareLocation = "detection_share_location";
constexpr auto kDetectionShareVariance = "detection_share_variance";
constexpr auto kDetectionTop = "detection_top";
constexpr auto kDetectionTopK = "detection_top_k";
constexpr auto kDetectionVarianceVec = "detection_variance_vec";
constexpr auto kDecBBoxParam = "decbbox_param";
constexpr auto kDim1 = "dim_1";
constexpr auto kDim2 = "dim_2";
constexpr auto kDim3 = "dim_3";
constexpr auto kDirection = "direction";
constexpr auto kEndAxis = "end_axis";
constexpr auto kExposeHidden = "expose_hidden";
constexpr auto kExtendedOpType = "extended_op_type";
constexpr auto kFmod = "fmod";
constexpr auto kGraphName = "graph_name";
constexpr auto kGroupSize = "group_size";
constexpr auto kGruWeightOrderZrhFlag = "gru_weight_order_zrh_flag";
constexpr auto kHasOutputGateFlag = "has_output_gate_flag";
constexpr auto kHasSplitBiasFlag = "has_split_bias_flag";
constexpr auto kHasSplitHWeightFlag = "has_split_h_weight_flag";
constexpr auto kHiddenSize = "hidden_size";
constexpr auto kInferDone = "infer_done";
constexpr auto kInitialCOnlineFlag = "initial_c_online_flag";
constexpr auto kInitialHOnlineFlag = "initial_h_online_flag";
constexpr auto kInputsShape = "inputs_shape";
constexpr auto kInterpolationMode = "interpolation_mode";
constexpr auto kIsMainGraph = "is_main_graph";
constexpr auto kIsMapperSupported = "is_mapper_supported";
constexpr auto kKeepDirectionDimFlag = "keep_direction_dim_flag";
constexpr auto kKernelShape = "kernel_shape";
constexpr auto kLrnK = "lrn_k";
constexpr auto kLstmWeightOrderIofcFlag = "lstm_weight_order_iofc_flag";
constexpr auto kMultiples = "multiples";
constexpr auto kNearest = "nearest";
constexpr auto kNegativeSlope = "negative_slope";
constexpr auto kNetType = "net_type";
constexpr auto kNormalizeVariance = "normalize_variance";
constexpr auto kNumAnchors = "num_anchors";
constexpr auto kNumAxes = "num_axes";
constexpr auto kNumBboxesPerGrid = "num_bboxes_per_grid";
constexpr auto kNumClasses = "num_classes";
constexpr auto kNumCoords = "num_coords";
constexpr auto kNumGridsWidth = "num_grid_width";
constexpr auto kNumGridsHeight = "num_grid_height";
constexpr auto kNumOutput = "num_output";
constexpr auto kOnnxModeOutFlag = "onnx_model_out_flag";
constexpr auto kOperatorType = "operator_type";
constexpr auto kOutputChannel = "output_channel";
constexpr auto kOutputDim = "output_dim";
constexpr auto kOutputHeight = "output_height";
constexpr auto kOutputWidth = "output_width";
constexpr auto kOutputsNames = "outputs_names";
constexpr auto kOutputsShape = "outputs_shape";
constexpr auto kOutputsFormat = "outputs_format";
constexpr auto kLastDimStride = "internal_stride";
constexpr auto kOutputLastFrameFlag = "output_last_frame_flag";
constexpr auto kPadBeg = "pad_beg";
constexpr auto kPadEnd = "pad_end";
constexpr auto kPads = "pads";
constexpr auto kPeepHoleFlag = "peep_hole_flag";
constexpr auto kPerm = "perm";
constexpr auto kPoolMethod = "pool_method";
constexpr auto kPyramidHeight = "pyramid_height";
constexpr auto kRecurrentDirection = "recurrent_direction";
constexpr auto kSamplingRatio = "sampling_ratio";
constexpr auto kSelectLastIndex = "select_last_index";
constexpr auto kSequenceLensOnlineFlag = "sequence_lens_online_flag";
constexpr auto kShrinkFactor = "shrink_factor";
constexpr auto kSlicePointBegin = "slice_point_begin";
constexpr auto kSlicePointEnd = "slice_point_end";
constexpr auto kSpatialScale = "spatial_scale";
constexpr auto kSqrtA = "sqrt_a";
constexpr auto kStartAxis = "start_axis";
constexpr auto kThreshold = "threshold";
constexpr auto kUpsampleH = "upsample_h";
constexpr auto kUpsampleW = "upsample_w";
constexpr auto kUseDefaultInitialCFlag = "use_default_initial_c_flag";
constexpr auto kUseDefaultInitialHFlag = "use_default_initial_h_flag";
constexpr auto kUseGlobalStats = "use_global_stats";
constexpr auto kZoomFactor = "zoom_factor";
}  // namespace dpico
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_COMMON_OP_ATTR_H_
