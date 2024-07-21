/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_UTILS_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_UTILS_H_

#include <fcntl.h>
#include <sys/stat.h>
#ifndef _MSC_VER
#include <sys/time.h>
#endif
#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <sstream>
#include <vector>
#include <tuple>

#include "include/common/visible.h"
#include "include/common/utils/stream_util.h"
#include "ir/dtype/type.h"
#include "utils/log_adapter.h"

#ifndef MS_UNLIKELY
#ifdef _MSC_VER
#define MS_UNLIKELY(x) (x)
#else
#define MS_UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif
#endif

#ifndef MS_LIKELY
#ifdef _MSC_VER
#define MS_LIKELY(x) (x)
#else
#define MS_LIKELY(x) __builtin_expect(!!(x), 1)
#endif
#endif

namespace mindspore {
// attr key name
constexpr auto kAttrInternalSepcialFormat = "internal_special_format";
constexpr auto kAttrSegment = "segment";
constexpr auto kAttrAlignCorners = "align_corners";
constexpr auto kAttrHalfPixelCenters = "half_pixel_centers";
constexpr auto kAttrInputNames = "input_names";
constexpr auto kAttrAttrNames = "attr_names";
constexpr auto kAttrAnyTypeCast = "any_type";
constexpr auto kAttrBins = "bins";
constexpr auto kAttrMin = "min";
constexpr auto kAttrMax = "max";
constexpr auto kAttrCopyData = "need_copy";
constexpr auto kAttrInputDefaultFormat = "input_default_format";
constexpr auto kAttrOutputDefaultFormat = "output_default_format";
constexpr auto kAttrConvertAttrNode = "convert_attr_node";
constexpr auto kAttrNeedCast = "need_cast";
constexpr auto kAttrIsAiCpuKernel = "is_AICPU_kernel";
constexpr auto kIsBackendCast = "is_backend_cast";
constexpr auto kAttrOutputNames = "output_names";
constexpr auto kAttrAsync = "async";
constexpr auto kAttrOffload = "offload";
constexpr auto kAttrOutIdx = "out_idx";
constexpr auto kAttrVisited = "visited";
constexpr auto kAttrReshapePaddingAxis = "reshape_padding_axis";
constexpr auto kAttrBeginNormAxis = "begin_norm_axis";
constexpr auto kAttrBeginParamsAxis = "begin_params_axis";
constexpr auto kAttrShape = "shape";
constexpr auto kAttrMomentum = "momentum";
constexpr auto kAttrEps = "eps";
constexpr auto kAttrEpsilon = "epsilon";
constexpr auto kAttrFactor = "factor";
constexpr auto kAttrIsRef = "isRef";
constexpr auto kAttrDataShape = "data_shape";
constexpr auto kAttrFormat = "format";
constexpr auto kAttrOriginFormat = "origin_format";
constexpr auto kAttrReshapeType = "reshape_type";
constexpr auto kAttrAxis = "axis";
constexpr auto kAttrAxes = "axes";
constexpr auto kAttrAlpha = "alpha";
constexpr auto kAttrAclSpecialFormat = "acl_special_format";
constexpr auto kAttrAclSpecialInputFormat = "acl_special_input_format";
constexpr auto kAttrAclInconsistentInputDtype = "acl_inconsistent_input_dtype";
constexpr auto kAttrBatchDims = "batch_dims";
constexpr auto kAttrKeepDims = "keep_dims";
constexpr auto kTransposeA = "transpose_a";
constexpr auto kTransposeB = "transpose_b";
constexpr auto kAttrSkipMode = "skip_mode";
constexpr auto kAttrShapeGamma = "shape_gamma";
constexpr auto kAttrPerm = "perm";
constexpr auto kAttrTransposeFirst = "transpose_first";
constexpr auto kAttrTbeFusionType = "tbe_fusion_type";
constexpr auto kAttrAtomicAddMemSize = "automic_add_mem_size";
constexpr auto kAttrAtomicOutputIndexs = "atomic_output_clean_indexs";
constexpr auto kAttrNeedAtomic = "need_atomic";
constexpr auto kAttrAtomicWorkspaceIndexs = "atomic_workspace_clean_indexs";
constexpr auto kAttrSwitchCondition = "switch_condition";
constexpr auto kAttrDataType = "data_type";
constexpr auto kAttrDType = "dtype";
constexpr auto kAttrActiveTarget = "active_target";
constexpr auto kAttrActiveStreamId = "active_stream_id";
constexpr auto kAttrActiveStreamList = "active_stream_list";
constexpr auto kAttrTrueBranchStream = "true_branch_stream";
constexpr auto kAttrStreamSwitchKind = "stream_switch_kind";
constexpr auto kAttrEventId = "event_id";
constexpr auto kAttrLabelId = "label_id";
constexpr auto kAttrLogicId = "logic_id";
constexpr auto kAttrNodeInfo = "node_info";
constexpr auto kAttrNodeName = "node_name";
constexpr auto kAttrDynInput = "dynamic";
constexpr auto kAttrDynInputSizes = "dyn_input_sizes";
constexpr auto kAttrChannelName = "channel_name";
constexpr auto kAttrTupleInputStructural = "tuple_input_structural";
constexpr auto kAttrListStartIndex = "list_start_index";
constexpr auto kAttrPyExecuteNeedUpdateShape = "pyexecute_need_update_shape";
constexpr auto kAttrPyExecuteOutput = "pyexecute_output";
constexpr auto kAttrSrcFormat = "src_format";
constexpr auto kAttrDstFormat = "dst_format";
constexpr auto kAttrFixPrecision = "fix_precision";
constexpr auto kAttrOutputPrecision = "output_precision";
constexpr auto kAttrOutputUsedNum = "output_used_num";
constexpr auto kAttrHasBias = "has_bias";
constexpr auto kAttrN = "n";
constexpr auto kAttrLabelForInsertStreamActive = "label_for_insert_stream_active";
constexpr auto kAttrFpBpEnd = "fpbp_end";
constexpr auto kAttrFusion = "fusion";
constexpr auto kAttrCommInputDepend = "comm_input_depend";
constexpr auto kAttrRecomputeCommDepend = "recompute_comm_depend";
constexpr auto kAttrNotDelayFusion = "not_delay_fusion";
constexpr auto kAttrGroup = "group";
constexpr auto kAttrRankList = "rank_list";
constexpr auto kAttrGroups = "groups";
constexpr auto kAttrGroupBack = "group_back";
constexpr auto kAttrFracZGroup = "fracz_group";
constexpr auto kAttrFracZGroupIdx = "fracz_group_idx";
constexpr auto kAttrOp = "op";
constexpr auto kAttrDestRank = "dest_rank";
constexpr auto kAttrSrcRank = "src_rank";
constexpr auto kAttrSrTag = "sr_tag";
constexpr auto kAttrRootRank = "root_rank";
constexpr auto kAttrComm = "comm";
constexpr auto kAttrIsTraining = "is_training";
constexpr auto kAttrFusionId = "fusion_id";
constexpr auto kAttrDuplicated = "duplicated";
constexpr auto kAttrGradOutputIndex = "grad_output_index";
constexpr auto kAttrLabelIndex = "label_index";
constexpr auto kAttrLabelSwitchList = "label_switch_list";
constexpr auto kAttrBeginMask = "begin_mask";
constexpr auto kAttrEndMask = "end_mask";
constexpr auto kAttrEllipsisMask = "ellipsis_mask";
constexpr auto kAttrNewAxisMask = "new_axis_mask";
constexpr auto kAttrShrinkAxisMask = "shrink_axis_mask";
constexpr auto kAttrDatadumpOriginalNames = "_datadump_original_names";
constexpr auto kAttrDatadumpIsMultiop = "_datadump_is_multiop";
constexpr auto kAttrNeedRecordEvent = "need_record_event";
constexpr auto kAttrStreamId = "stream_id";
constexpr auto kAttrRecomputeId = "recompute_id";
constexpr auto kAttrRecordEvent = "record_event";
constexpr auto kAttrAccumulatedAttention = "AccumulatedAttention";
constexpr auto kAttrWaitEvent = "wait_event";
constexpr auto kAttrRecordEventStream = "record_event_stream";
constexpr auto kAttrWaitEventStream = "wait_event_stream";
constexpr auto kAttrRecordWaitEventStreamPairId = "record_wait_event_stream_pair_id";
constexpr auto kAttrInputMultiStreamSafe = "input_multi_thread_safe";
constexpr auto kAttrStream = "stream";
constexpr auto kAttrIndex = "index";
constexpr auto kAttrSplit = "split";
constexpr auto kAttrSplitDim = "split_dim";
constexpr auto kAttrNumSplit = "num_split";
constexpr auto kAttrNumGroups = "num_groups";
constexpr auto kAttrActivateSilu = "activate_silu";
constexpr auto kAttrReduction = "reduction";
constexpr auto kAttrOutputNum = "output_num";
constexpr auto kAttrOutputSize = "output_size";
constexpr auto kAttrScales = "scales";
constexpr auto kAttrScale = "scale";
constexpr auto kAttrZeroPoint = "zero_point";
constexpr auto kAttrScaleVec = "scale_vec";
constexpr auto kAttrZeroPointVec = "zero_point_vec";
constexpr auto kAttrSizeSplits = "size_splits";
constexpr auto kAttrOutputDefault = "output_default";
constexpr auto kAttrPrimitiveTarget = "primitive_target";
constexpr auto kAttrUseLocking = "use_locking";
constexpr auto kAttrReduceScatterFlag = "reduce_scatter_flag";
constexpr auto kAttrOffset = "offset";
constexpr auto kAttrCacheEnable = "cache_enable";
constexpr auto kAttrPsKey = "ps_key";
constexpr auto kAttrOptimizerType = "optim_type";
constexpr auto kAttrChildGraph = "child_graph";
constexpr auto kAttrInputNums = "inputNums";
constexpr auto kAttrT = "T";
constexpr auto kAttrNum = "num";
constexpr auto kAttrRecvType = "recv_type";
constexpr auto kAttrConcatDim = "concat_dim";
constexpr auto kAttrSplitCount = "split_count";
constexpr auto kAttrSendRankIds = "send_rank_ids";
constexpr auto kAttrRecvRankIds = "recv_rank_ids";
constexpr auto kAttrSendLens = "send_lens";
constexpr auto kAttrRecvLens = "recv_lens";
constexpr auto kAttrRankSize = "rank_size";
constexpr auto kAttrPadDimSize = "pad_dim_size";
constexpr auto kAttrPaddings = "paddings";
constexpr auto kAttrNumSegments = "num_segments";
constexpr auto kAttrStackOpName = "stack_op_name";
constexpr auto kAttrBegin = "begin";
constexpr auto kAttrEnd = "end";
constexpr auto kAttrSize = "size";
constexpr auto kAttrSizes = "sizes";
constexpr auto kAttrKsizes = "ksizes";
constexpr auto kAttrIsKernelDynamicImpl = "is_kernel_dynamic_impl";
constexpr auto kAttrIsKernelDynamicShape = "is_kernel_dynamic_shape";
constexpr auto kAttrIsPyboostTupleInput = "is_pyboost_tuple_input";
constexpr auto kAttrIsDynamicRank = "is_dynamic_rank";
constexpr auto kAttrInputIsDynamicRank = "input_is_dynamic_rank";
constexpr auto kAttrOutputIsDynamicRank = "output_is_dynamic_rank";
constexpr auto kAttrInputIsDynamicShape = "input_is_dynamic_shape";
constexpr auto kAttrOutputIsDynamicShape = "output_is_dynamic_shape";
constexpr auto kAttrPynativeNextOpName = "next_op";
constexpr auto kAttrPynativeNextIndex = "next_index";
constexpr auto kAttrMutableOpName = "mutable";
constexpr auto kAttrMutableKernel = "mutable_kernel";
constexpr auto kAttrAclHighPrecision = "acl_high_precision";
constexpr auto kAttrCompileInfo = "compile_info";
constexpr auto kAttrFusionType = "fusion_type";
constexpr auto kAttrStride = "stride";
constexpr auto kAttrStrides = "strides";
constexpr auto kAttrShapex = "shapex";
constexpr auto kAttrKernelSize = "kernel_size";
constexpr auto kAttrDilation = "dilation";
constexpr auto kAttrDatFormat = "data_format";
constexpr auto kAttrPadMode = "pad_mode";
constexpr auto kAttPaddingMode = "padding_mode";
constexpr auto kAttrPad = "pad";
constexpr auto kAttrPadding = "padding";
constexpr auto kAttrMode = "mode";
constexpr auto kAttrWindow = "window";
constexpr auto kAttrCeilMode = "ceil_mode";
constexpr auto kAttrGlobalPooling = "global_pooling";
constexpr auto kAttrNonTask = "non_task";
constexpr auto kAttrIsGrad = "is_grad";
constexpr auto kAttrRecompute = "recompute";
constexpr auto kAttrCheckpoint = "checkpoint";
constexpr auto kAttrSliceActivation = "slice_activation";
constexpr auto kAttrNeedCseAfterRecompute = "need_cse_after_recompute";
constexpr auto kAttrParallelDimInfo = "parallel_dim_info";
constexpr auto kAttrParallelFusionType = "parallel_fusion_type";
constexpr auto kAttrParallelTypeInfo = "parallel_type_info";
constexpr auto kAttrCompositeType = "composite_type";
constexpr auto kAttrStitch = "stitch";
constexpr auto kAttrTopoSortRhsFirst = "topo_sort_rhs_first";
constexpr auto kAttrIgnoreSideEffect = "ignore_side_effect";
constexpr auto kAttrSwitchLayer = "switch_layer";
constexpr auto kAttrReturn = "return";
constexpr auto kAttrRecursiveStart = "recursive_start";
constexpr auto kAttrRecursiveEnd = "recursive_end";
constexpr auto kAttrRecursive = "recursive";
constexpr auto kAttrMultiCallEnd = "multicall_end";
constexpr auto kAttrProfilingIterEnd = "PROFILING_ITER_END";
constexpr auto kAttrHiddenSize = "hidden_size";
constexpr auto kAttrInputSize = "input_size";
constexpr auto kAttrDstType = "dst_type";
constexpr auto kAttrDump = "dump";
constexpr auto kAttrUselessInput = "useless_input";
constexpr auto kAttrSkipNopOpAddr = "skip_nop_op_addr";
constexpr auto kAttrSkipNopOpExecution = "skip_nop_op_execution";
constexpr auto kAttrFixedInputFormat = "fixed_input_format";
constexpr auto kAttrFixedOutputFormat = "fixed_output_format";
constexpr auto kAttrFixedInputDeviceShape = "fixed_input_device_shape";
constexpr auto kAttrFixedOutputDeviceShape = "fixed_output_device_shape";
constexpr auto kAttrFuncType = "func_type";
constexpr auto kAttrFuncName = "func_name";
constexpr auto kAttrFunctor = "functor";
constexpr auto kAttrCustAicpu = "cust_aicpu";
constexpr auto kAttrIsInternalOutputNopNode = "is_internal_output_nop_node";
constexpr auto kAttrIsUBFusionOp = "is_ub_fusion_op";
constexpr auto kAttrNopOp = "nop_op";
constexpr auto kAttrPlaceHolderIndex = "placeholder_index";
constexpr auto kAttrMicro = "micro";
constexpr auto kAttrJsonFileName = "json_file_name";
constexpr auto kAttrNeedDropInput = "need_drop_input";
constexpr auto kAttrNeedConvertToValueNode = "need_convert_to_value_node";
constexpr auto kAttrSendSrcNodeName = "send_src_node_name";
constexpr auto kAttrSendDstNodeName = "send_dst_node_name";
constexpr auto kAttrSendDstRanks = "send_dst_ranks";
constexpr auto kAttrSendDstRoles = "send_dst_roles";
constexpr auto kAttrRecvSrcNodeName = "recv_src_node_name";
constexpr auto kAttrRecvDstNodeName = "recv_dst_node_name";
constexpr auto kAttrRecvSrcRanks = "recv_src_ranks";
constexpr auto kAttrRecvSrcRoles = "recv_src_roles";
constexpr auto kAttrInterProcessEdgeNames = "inter_process_edge_names";
constexpr auto kAttrInterProcessEdgeLabel = "inter_process_edge_label";
constexpr auto kAttrIsMuxRpcKernel = "is_mux_rpc_kernel";
constexpr auto kAttrGroupRankIds = "group_rank_ids";
constexpr auto kAttrReuseCommunication = "reuse_communication_node";
constexpr auto kAttrPrecisionFlag = "precision_flag";
constexpr auto kAttrDfmGroup = "deformable_groups";
constexpr auto kAttrModulated = "modulated";
constexpr auto kAttrDilations = "dilations";
constexpr auto kAttrDataFormat = "data_format";
constexpr auto kAttrPads = "pads";
constexpr auto kAttrKsize = "ksize";
constexpr auto kAttrOnlyUseFirstOutput = "only_use_first_output";
constexpr auto kAttrOnlyUseSecondOutput = "only_use_second_output";
constexpr auto kAttrOpAdaptationProcessed = "op_adaptation_processed";
constexpr auto kAttrAbstractAdaptationProcessed = "abstract_adaptation_processed";
constexpr auto kAttrMeOpName = "me_op_name";
constexpr auto kAttrIRChange = "ir_change";
constexpr auto kParamterIsSequence = "param_is_sequence";
constexpr auto kAttrZeroInfinity = "zero_infinity";
constexpr auto kAttrBlank = "blank";
constexpr auto kAttrUpdateSlots = "update_slots";
constexpr auto kAttrLr = "lr";
constexpr auto kAttrWithBiasAdd = "with_bias_add";
constexpr auto kAttrWithRelu = "with_relu";
constexpr auto kAttrNeedGradFlagOfInputs = "need_grad_flag_of_inputs";
constexpr auto kAttrIsCNodeNeedGrad = "is_cnode_need_grad";
constexpr auto kAttrJitLevel = "jit_level";
constexpr auto kAttrJitLevelO0 = "O0";
constexpr auto kAttrJitLevelO1 = "O1";
constexpr auto kAttrJitLevelO2 = "O2";
constexpr auto kAttrCellJitConfigDict = "_jit_config_dict";
constexpr auto kAttrBinaryOutput = "binary_output";
constexpr auto kAttrMinLength = "minlength";
constexpr auto kAttrMaxLength = "maxlength";
constexpr auto kAttrIouThreshold = "iou_threshold";
constexpr auto kAttrEnableEmbeddingStorage = "enable_embedding_storage";
constexpr auto kAttrParameterKey = "parameter_key";
constexpr auto kAttrJitCallNode = "jit_call_node";
constexpr auto kAttrInsertDefaultValue = "insert_default_value";
constexpr auto kAttrIsSparse = "IsSparse";
constexpr auto kAttrKernelBackoffWithFailureInfo = "kernel_backoff_with_failure_info";
constexpr auto kAttrKernelBackoffWithFailureType = "kernel_backoff_with_failure_type";
constexpr auto kAttrKernelGraph = "kernel_graph";
constexpr auto kAttrPreKernelGraph = "pre_kernel_graph";
constexpr auto kAttrKernelGraphBoundary = "kernel_graph_boundary";
constexpr auto kAttrKernelPacketNode = "kernel_packet_node";
constexpr auto kAttrNeedInline = "need_inline";
constexpr auto kAttrOriFusionName = "ori_fusion_name";
constexpr auto kAttrDynamicLenName = "is_dynamic_len";
constexpr auto kAttrAnyOutputName = "is_any_output";
constexpr auto kAttrForFormatChange = "for_format_change";
constexpr auto kAttrReplaceRealKernelInBackend = "replace_real_kernel_in_backend";
constexpr auto kAttrRefNodeMonadOutputIdx = "ref_node_monad_output_idx";
constexpr auto kAttrRandomOpSnapShot = "random_op_snapshot";
constexpr auto kAttrTbeOpAtomicDtypes = "tbe_op_atomic_dtypes";
constexpr auto kAttrTbeOpAtomicInt64Values = "tbe_op_atomic_int64_values";
constexpr auto kAttrTbeOpAtomicFloatValues = "tbe_op_atomic_float_values";
constexpr auto kAttrDtypes = "dtypes";
constexpr auto kAttrValuesInt = "values_int";
constexpr auto kAttrValuesFloat = "values_float";
constexpr auto kAttrRecomputeSubGraph = "recompute_sub_graph";
constexpr auto kAttrExpandDimsMask = "expand_dims_mask";
constexpr auto kAttrTupleIndexTypes = "tuple_index_types";
constexpr auto kAttrTupleIndexAxis = "tuple_index_axis";
constexpr auto kAttrInitByNone = "init_by_none";
constexpr auto kAttrExpandDimsCnt = "expand_dims_cnt";
constexpr auto kAttrEmptyIndicesOut = "empty_indices_out";
constexpr auto kAttrHasTrue = "has_true";
constexpr auto kAttrHasSequence = "has_sequence";
constexpr auto kAttrOriginIndexType = "origin_index_type";
constexpr auto kIntIndex = "int_index";
constexpr auto kTensorIndexSequenceIndex = "tensor_index_sequence_index";
constexpr auto kNoneIndex = "none_index";
constexpr auto kBoolSequenceIndex = "bool_sequence_index";
constexpr auto kSliceIndex = "slice_index";
constexpr auto kEllipsisIndex = "ellipsis_index";
constexpr auto kSetitemByTupleWithTensor = "setitem_by_tuple_with_tensor";
constexpr auto kSetitemByTuple = "setitem_by_tuple";
constexpr auto kPreSetitemByTuple = "pre_setitem_by_tuple";
constexpr auto kAttrTupleIndexInfoType = "tuple_index_info_type";
constexpr auto kAttrSimpleSliceInfo = "simple_slice_info";
constexpr auto kAttrNotCut = "not_cut";
constexpr auto kAttrNotSupportOpForDevice = "not_support_op_for_device";
constexpr auto kAttrGraphSplitGroup = "graph_split_group";
constexpr const char kAttrNeedAllGather[] = "parallel_optimizer_allgather";
constexpr const char kAttrNodeCloseFollowing[] = "node_close_following";
constexpr const char kAttrNodeWithoutOutput[] = "node_without_output";
constexpr char kAttrInputLayout[] = "input_layout";
constexpr char kAttrKeepProb[] = "keep_prob";
constexpr char kAttrHeadNum[] = "head_num";
constexpr auto kAttrFuncGraph = "func_graph";
constexpr char kAttrScaleValue[] = "scale_value";
constexpr char kAttrPreTokens[] = "pre_tokens";
constexpr char kAttrNextTokens[] = "next_tokens";
constexpr char kAttrSparseMode[] = "sparse_mode";
constexpr char kAttrEnableLoadBalance[] = "enable_load_balance";
constexpr char kAttrIsTransA[] = "is_trans_a";
constexpr char kAttrIsTransB[] = "is_trans_b";
constexpr char kAttrReduceOp[] = "reduce_op";
constexpr char kAttrTransposeX1[] = "transpose_x1";
constexpr char kAttrTransposeX2[] = "transpose_x2";
constexpr char kAttrCommTurn[] = "comm_turn";
constexpr char kAttrGatherIndex[] = "gather_index";
constexpr char kAttrBranchOutputNum[] = "branch_output_num";
constexpr char kAttrBranchGraphName[] = "branch_graph_name";
constexpr char kInlineSubGraphName[] = "inline_sub_graph_name";
constexpr char kAttrBpropAutoMonadLevel[] = "bprop_auto_monad_level";
constexpr char kAttrSideEffectBpropAppPropagate[] = "side_effect_bprop_app_propagate";
constexpr char kAttrSideEffectBpropApp[] = "side_effect_bprop_app";
constexpr auto kAttrOriginOutputShape = "origin_output_shape";
constexpr auto kAttrOriginInputShapes = "origin_input_shapes";
constexpr char kAttrNotRemove[] = "not_remove";
constexpr const char kAttrValueDepend[] = "value_depend";
constexpr const char kAttrOnlyDependShape[] = "only_depend_shape";
constexpr char kAttrFineGrainedInterleavedBlockIndex[] = "fine_grained_interleaved_index";

// FuncGraph Flags
constexpr auto kFlagIsPynativeBpropGraph = "is_pynative_bprop_graph";
constexpr auto kFlagPyNativeRunInGraph = "pynative_run_in_graph";
constexpr auto kFlagNeedRenormalize = "need_renormalize";
constexpr auto kFlagEnableZeroCopyInGraph = "enable_zero_copy_in_graph";
constexpr auto kFlagPyNativeBpropGraphWithBpropCut = "pynative_bprop_graph_with_bprop_cut";
constexpr auto kFlagPyNativeBpropGraphIsDynamic = "pynative_bprop_graph_is_dynamic";
constexpr auto kFlagEnableRunGraphBySingleOp = "enable_run_graph_by_single_op";
constexpr auto kFlagIsPyNativeBpropKernelGraph = "is_pynative_bprop_kernel_graph";
constexpr auto kFlagPyNativeWithJitCallGraph = "pynative_with_jit_call_graph";
constexpr auto kFlagJitCallGraph = "jit_call_graph";
constexpr auto kFlagJitGraph = "jit_graph";
constexpr auto kFlagSwitchInline = "switch_inline_graph";
constexpr auto kFlagIsControlFlow = "is_control_flow";

// custom operator func type
constexpr auto kCustomTypeAOT = "aot";
constexpr auto kCustomTypeJULIA = "julia";
constexpr auto kCustomTypePyfunc = "pyfunc";
constexpr auto kCustomTypeTbe = "tbe";
constexpr auto kCustomTypeAICPU = "aicpu";
constexpr auto kCustomTypeHybrid = "hybrid";

// primal attr key name
constexpr auto kPrimalAttrForwardNodeName = "forward_node_name";
constexpr auto kPrimalAttrMicro = "micro";
constexpr auto kPrimalAttrChunk = "chunk";
constexpr auto kPrimalAttrPipelineParam = "pipeline_param";
constexpr auto kPrimalAttrBackwardMicroEnd = "backward_micro_end";
constexpr auto kPrimalAttrForwardEnd = "forward_end";
constexpr auto kPrimalAttrSegmentMax = "segment_max";
constexpr auto kPrimalAttrUniqueId = "unique_id";
constexpr auto kPrimalAttrForwardUniqueId = "forward_unique_id";
constexpr auto kPrimalAttrForwardCommNodeUniqueId = "forward_comm_node_unique_id";
constexpr auto kPrimalAttrMirrorUserId = "mirror_user_id";

// attr value
constexpr auto kValueTargetSwitch = "target_switch";
constexpr auto kValueTargetOther = "target_other";
constexpr auto kValueTrue = "true";
constexpr auto kTensorValueIsType = "tensor_value_is_type";
constexpr auto kTensorValueIsEmpty = "tensor_value_is_empty";
constexpr auto kTensorUserDataIsSensTensor = "is_sens_tensor";
constexpr auto kFakeTensorPos = "fake_tensor_pos";
constexpr auto kFakeTensorListPos = "fake_tensor_list_pos";
constexpr auto kChannelNameNpuLog = "_npu_log";

// env key
constexpr auto kCompilerCacheEnable = "MS_COMPILER_CACHE_ENABLE";
constexpr auto kCompilerCachePath = "MS_COMPILER_CACHE_PATH";
constexpr auto kSimulationLevel = "MS_SIMULATION_LEVEL";
constexpr auto kSimulationLevelCompileGraph = "0";
constexpr auto kSimulationLevelCompileKernel = "1";

// comm
constexpr auto kHCCLWorldGroup = "hccl_world_group";
constexpr auto kNCCLWorldGroup = "nccl_world_group";
constexpr auto kEnvRankSize = "RANK_SIZE";
constexpr auto kEnvRankId = "RANK_ID";
constexpr auto kEnvLocalRankSize = "LOCAL_RANK_SIZE";
constexpr auto kEnvLocalRankId = "LOCAL_RANK_ID";

// some size
const size_t kShape4dDims = 4;
const size_t kShape3dDims = 3;
const size_t kShape2dDims = 2;
const size_t kShape5dDims = 5;
const size_t kShape6dDims = 6;
const size_t kShape1dDims = 1;
const size_t kCubeSize = 16;
const size_t kCubeSize_C04 = 4;
const size_t kNiSize = 16;
const size_t kMemAlignSize = 512;
const size_t kBNChannelMultipleFactor = 4;
constexpr auto kNCHWShapeSize = 4;
const size_t kMaxTensorIndexDimNums = 8;

// define special index in special node
constexpr auto kAnfPrimitiveIndex = 0;
constexpr auto kFirstDataInputIndex = 1;
constexpr auto kRealInputNodeIndexInTupleGetItem = 1;
constexpr auto kInputNodeOutputIndexInTupleGetItem = 2;
constexpr auto kSparseGetAttrInputSize = 2;
constexpr auto kTupleGetItemInputSize = 3;

// index define of kTupleSetItem
constexpr auto kTupleSetItemTupleIndex = 1;
constexpr auto kTupleSetItemIndexIndex = 2;
constexpr auto kTupleSetItemValueIndex = 3;
constexpr auto kTupleSetItemInputSize = 4;
// index define of partial
constexpr auto kPartialMinInputSize = 2;
constexpr auto kPartialGraphIndex = 1;

// index define of switch
constexpr auto kSwitchInputSize = 4;
constexpr auto kSwitchCondIndex = 1;
constexpr auto kSwitchTrueBranchIndex = 2;
constexpr auto kSwitchFalseBranchIndex = 3;
constexpr auto kSwitchBranchesNum = 2;

// index define of GridSampler & GridSamplerGrad
constexpr int kGridSamplerInputNum = 5;
constexpr int kGridSamplerOutputNum = 1;
constexpr int kGridSamplerGradInputNum = 6;
constexpr int kGridSamplerGradOutputNum = 2;

// index define of switch_layer
constexpr auto kSwitchLayerInputSize = 3;
constexpr auto kSwitchLayerSelectIndex = 1;
constexpr auto kSwitchLayerBranchesIndex = 2;

// index define of depend
constexpr auto kRealInputIndexInDepend = 1;
constexpr auto kDependAttachNodeIndex = 2;
constexpr auto kDependInputSize = 3;
// index define of UpdateState
constexpr auto kUpdateStateStateInput = 1;
constexpr auto kUpdateStateRealInput = 2;
// index define of Load
constexpr auto kLoadRealInput = 1;
constexpr auto kLoadStateInput = 2;
constexpr auto kGenerateEodMaskOpName = "GenerateEodMask";
// time transfer unit
constexpr int kBasicTimeTransferUnit = 1000;
constexpr int kMaxVectorSize = 10000;
// index of input or output
constexpr size_t kIndex0 = 0;
constexpr size_t kIndex1 = 1;
constexpr size_t kIndex2 = 2;
constexpr size_t kIndex3 = 3;
constexpr size_t kIndex4 = 4;
constexpr size_t kIndex5 = 5;
constexpr size_t kIndex6 = 6;
constexpr size_t kIndex7 = 7;
constexpr size_t kIndex8 = 8;
constexpr size_t kIndex9 = 9;
constexpr size_t kIndex10 = 10;
constexpr size_t kIndex11 = 11;
constexpr size_t kIndex12 = 12;
constexpr size_t kIndex13 = 13;
constexpr size_t kIndex14 = 14;
constexpr size_t kIndex15 = 15;
constexpr size_t kIndex16 = 16;
constexpr size_t kIndex17 = 17;
constexpr size_t kIndex18 = 18;
constexpr size_t kIndex19 = 19;
constexpr size_t kIndex20 = 20;
constexpr size_t kIndex21 = 21;
constexpr size_t kIndex22 = 22;
constexpr size_t kIndex23 = 23;
constexpr size_t kIndex24 = 24;
constexpr size_t kIndex25 = 25;
constexpr size_t kIndex26 = 26;
constexpr size_t kIndex27 = 27;
constexpr size_t kIndex28 = 28;
// dim of shape
constexpr size_t kDim0 = 0;
constexpr size_t kDim1 = 1;
constexpr size_t kDim2 = 2;
constexpr size_t kDim3 = 3;
constexpr size_t kDim4 = 4;
constexpr size_t kDim5 = 5;
constexpr size_t kDim6 = 6;
// format
constexpr auto kOpFormat_DEFAULT = "DefaultFormat";
constexpr auto kOpFormat_ChannelFirst = "ChannelFirst";
constexpr auto kOpFormat_ChannelLast = "ChannelLast";
constexpr auto kOpFormat_NC1KHKWHWC0 = "NC1KHKWHWC0";
constexpr auto kOpFormat_ND = "ND";
constexpr auto kOpFormat_NCHW = "NCHW";
constexpr auto kOpFormat_NHWC = "NHWC";
constexpr auto kOpFormat_HWCN = "HWCN";
constexpr auto kOpFormat_CHWN = "CHWN";
constexpr auto kOpFormat_NC1HWC0 = "NC1HWC0";
constexpr auto kOpFormat_FRAC_Z = "FRACTAL_Z";
constexpr auto kOpFormat_FRACTAL_Z = "FRACTAL_Z";
constexpr auto kOpFormat_FRAC_NZ = "FRACTAL_NZ";
constexpr auto kOpFormat_C1HWNCoC0 = "C1HWNCoC0";
constexpr auto kOpFormat_NC1HWC0_C04 = "NC1HWC0_C04";
constexpr auto kOpFormat_FRACTAL_Z_C04 = "FRACTAL_Z_C04";
constexpr auto kOpFormat_NDHWC = "NDHWC";
constexpr auto kOpFormat_NCDHW = "NCDHW";
constexpr auto kOpFormat_DHWNC = "DHWNC";
constexpr auto kOpFormat_DHWCN = "DHWCN";
constexpr auto kOpFormat_NDC1HWC0 = "NDC1HWC0";
constexpr auto kOpFormat_FRACTAL_Z_3D = "FRACTAL_Z_3D";
constexpr auto kOpFormat_FRACTAL_ZN_LSTM = "FRACTAL_ZN_LSTM";
constexpr auto kOpFormat_FRACTAL_ZN_RNN = "FRACTAL_ZN_RNN";
constexpr auto kOpFormat_ND_RNN_BIAS = "ND_RNN_BIAS";
constexpr auto kOpFormat_NCL = "NCL";
constexpr auto kSliceStart = "start";
constexpr auto kSliceStop = "stop";
constexpr auto kSliceStep = "step";

// graph parse
constexpr auto kClassTensorObject = "class_tensor_object";

// graph type
constexpr auto kFuncGraphTypeName = "FuncGraph";
constexpr auto kKernelGraphTypeName = "KernelGraph";

// graph group
constexpr auto kDefaultGroup = "DefaultGroup";
constexpr auto kKernelGroup = "KernelGroup";
constexpr auto kGraphGroup = "GraphGroup";

// compile cache
constexpr auto kUniqueCacheName = "UniqueCacheName";
constexpr auto kDistributedSplit = "distribtued_split";
constexpr auto kValidate = "validate";
constexpr auto kGraphId = "graph_id";
constexpr auto kBackendFrontAnf = "backend_front_anf";
constexpr auto kInternalParameterToFrontNode = "internal_parameter_to_front_node";
constexpr auto kOutInRef = "out_in_ref";
constexpr auto kIsFeatureMap = "is_feature_map";
constexpr auto kGraphValueNodes = "graph_value_nodes";
constexpr auto kExecutionOrder = "execution_order";
constexpr auto kChildGraphOrder = "child_graph_order";
constexpr auto kRunMode = "run_mode";
constexpr auto kIsLoopCountSink = "is_loop_count_sink";
constexpr auto kIsDynamicShape = "is_dynamic_shape";
constexpr auto kInputs = "inputs";
constexpr auto kParameters = "parameters";
constexpr auto kForwardOutput = "forward_output";
constexpr auto kChildGraphResult = "child_graph_result";
constexpr auto kDeviceTarget = "device_target";
constexpr auto kRootGraphId = "root_graph_id";
constexpr auto kExecutable = "executable";
constexpr auto kValidInputs = "valid_inputs";
constexpr auto kNeedInline = "need_inline";
constexpr auto kStartLabel = "start_label";
constexpr auto kEndGoto = "end_goto";
constexpr auto kPreGraphs = "pre_graphs";
constexpr auto kPostGraphs = "post_graphs";
constexpr auto kHasRecursiveCall = "has_recursive_call";
constexpr auto kHasSubgraphMultiCall = "has_subgraph_multicall";
constexpr auto kIsNeedGil = "is_need_gil";
constexpr auto kIsFromSingleOp = "is_from_single_op";
constexpr auto kCommSubGraphIds = "comm_sub_graph_ids";
constexpr auto kNodesKernelInfo = "nodes_kernel_info";
constexpr auto kAllInputFormat = "all_input_format";
constexpr auto kAllOutputFormat = "all_output_format";
constexpr auto kAllInputDeviceType = "all_input_device_type";
constexpr auto kAllOutputDeviceType = "all_output_device_type";
constexpr auto kAllInputReshapeType = "all_input_reshape_type";
constexpr auto kAllOutputReshapeType = "all_output_reshape_type";
constexpr auto kOutputDataDesc = "output_data_desc";
constexpr auto kCoreType = "core_type";
constexpr auto kRuntimeCacheValid = "runtime_cache_valid";
constexpr auto kRuntimeCacheDeviceTarget = "runtime_cache_device_target";
constexpr auto kRuntimeCacheOutputTensorNum = "runtime_cache_output_tensor_num";
constexpr auto kRuntimeCacheIsRealKernel = "runtime_cache_is_real_kernel";
constexpr auto kRuntimeCachePrevOutputs = "runtime_cache_prev_outputs";
constexpr auto kCorrespondFrontendGraph = "correspond_frontend_graph";
constexpr auto kReturnNode = "_return_node";
constexpr auto kReturnPrimNode = "_return_prim_node";
constexpr auto kOriginDataFormat = "origin_data_format";
constexpr auto kKernelType = "kernel_type";
constexpr auto kOpType = "op_type";
constexpr auto kFusionType = "fusion_type";
constexpr auto kOpPattern = "op_pattern";
constexpr auto kProcessor = "processor";
constexpr auto kKernelBuildInfoValid = "kernel_build_info_valid";
constexpr auto kInputKernelObjectTypes = "input_kernel_object_types";
constexpr auto kOutputKernelObjectTypes = "output_kernel_object_types";
constexpr auto kOutputElementsKernelObjectTypes = "output_elements_kernel_object_types";
constexpr auto kInputSizeList = "input_size_list";
constexpr auto kOutputSizeList = "output_size_list";
constexpr auto kJsonName = "json_name";
constexpr auto kHasSelectKernelBuildInfo = "has_select_kernel_build_info";
constexpr auto kBackendParamToFrontendParamIndex = "backend_param_to_frontend_param_index_";
constexpr auto kLabelNum = "label_num";
constexpr auto kParameterUniqueNameToName = "param_unique_name_to_name";
constexpr auto kRefInOutMap = "ref_in_out_map";
constexpr auto kRetryIntervalMilliSeconds = 500;
constexpr auto kSummaryNodes = "summary_nodes";
constexpr auto kSummaryNodeExist = "summary_node_exist";
constexpr auto kGeCache = "ge_cache";
constexpr auto kGeGraphKey = "ge.graph_key";
constexpr auto kGeGraphCompilerCacheDir = "ge.graph_compiler_cache_dir";
constexpr auto kIsRefGraph = "is_ref_graph";
constexpr auto kFromRefGraph = "from_ref_graph";

// recompute and parallel
constexpr auto kRecomputeInsert = "recompute_insert";
constexpr auto kAddedRecomputeDependAttr = "added_recompute_depend";
constexpr auto kCondidateOverlapBlockId = "condidate_overlap_block_id";
constexpr auto kNcclWorldGroup = "nccl_world_group";
constexpr auto kHcclWorldGroup = "hccl_world_group";
constexpr auto kSyncBnGroup = "sync_bn_group";
constexpr auto kRankID = "RANK_ID";

// User data key.

// pyexecute.
constexpr auto kSyncUserDataHandler = "sync_user_data_handler";

constexpr auto kRealElementsSize = "real_elements_size";

// For expander and pynative grad graph
enum class InputType {
  // Scala or Constant tensor, no need to grad
  kConstant = 0,
  // Weight parameter tensor
  kParameter,
  // Net input tensor
  kInput,
  // Other op output tensor
  kOpOutput,
  // Default
  kUnkown,
};

// Return vec<filename, line number, function name>
COMMON_EXPORT std::vector<std::tuple<std::string, int, std::string>> GetPythonStack_();
COMMON_EXPORT std::string GetPythonStackStr_();

COMMON_EXPORT bool IsOneOfCustomAkgType(const std::string &name);
COMMON_EXPORT bool IsOneOfOperator(const std::string &name);
COMMON_EXPORT bool IsOneOfNotSupportedTransFormat(const std::string &format);
COMMON_EXPORT bool IsOneOfPosteriorOperator(const std::string &name);
COMMON_EXPORT bool IsOneOfCacheBlackList(const std::string &name);
COMMON_EXPORT bool IsOneOf3DFormat(const std::string &format);
COMMON_EXPORT bool IsOneOfNoPaddingFormat(const std::string &format);
COMMON_EXPORT bool IsOneOfDynamicShapeConstInputToAttrGPU(const std::string &name);
COMMON_EXPORT bool IsOneOfComputeDepend(const std::string &name);
COMMON_EXPORT bool IsOneOfHWSpecialFormat(const std::string &format);
COMMON_EXPORT bool IsOneOfFormat(const std::string &format);
COMMON_EXPORT bool IsOneOfDefaultFormat(const std::string &format);
COMMON_EXPORT bool IsOneOfServerFormatC04(const std::string &format);
COMMON_EXPORT bool IsOneOfDynRankNeedPadShape(const std::string &format);
COMMON_EXPORT bool IsOneOfUnsignedType(const TypeId &type_id);

COMMON_EXPORT size_t GetSystemMemorySize(const std::string &key);
COMMON_EXPORT size_t GetSystemFreeDiskSize(const std::string &path);

COMMON_EXPORT bool IsEnableRefMode();
COMMON_EXPORT bool IsMemoryPoolRecycle();

// copy once flag, and reset copy flag when step end
COMMON_EXPORT bool SkipOrResetCopyAction(bool need_reset = false);
// only sync once flag
COMMON_EXPORT bool SkipOrResetSyncAction(bool need_reset = false);
// Return vec<filename, line number, function name>
COMMON_EXPORT std::vector<std::tuple<std::string, int, std::string>> GetPythonStack();
COMMON_EXPORT std::string GetPythonStackStr();

// The map between kernel's output and input ref relationship.
// Key is the output index while the value is input index which will be used as the reference of output.
using OutputInputRefMap = std::map<size_t, size_t>;

using HashTableExportData = std::vector<std::shared_ptr<std::vector<char>>>;

static inline double GetCurrentUSec() {
  auto time_now = std::chrono::system_clock::now();
  auto tv_usec = std::chrono::duration_cast<std::chrono::microseconds>(time_now.time_since_epoch()).count();
  return static_cast<double>(tv_usec);
}

#define PROF_START(stage) double start_usec_##stage = mindspore::GetCurrentUSec()
#define PROF_END(stage)                                                                                        \
  do {                                                                                                         \
    double end_usec_##stage = mindspore::GetCurrentUSec();                                                     \
    std::ostringstream oss;                                                                                    \
    oss << "[PROF]" << #stage << " costs " << (end_usec_##stage - start_usec_##stage) / kBasicTimeTransferUnit \
        << " msec.";                                                                                           \
    if (common::IsEnableRuntimeConfig(common::kRuntimeCompileStat)) {                                          \
      std::cout << oss.str() << std::endl;                                                                     \
    }                                                                                                          \
    MS_LOG(INFO) << oss.str();                                                                                 \
  } while (0)

#define PROF_MULTI_DEFINE(stage)       \
  do {                                 \
    static uint64_t total_##stage = 0; \
    static uint64_t count_##stage = 0; \
  } while (0)

#define PROF_LOCAL_DEFINE(stage) \
  do {                           \
    uint64_t total_##stage = 0;  \
    uint64_t count_##stage = 0;  \
  } while (0)

#define PROF_MULTI_START(stage) uint64_t start_usec_##stage = mindspore::GetCurrentUSec()

#define PROF_MULTI_END(stage)                                 \
  do {                                                        \
    ++count_##stage;                                          \
    uint64_t end_usec_##stage = mindspore::GetCurrentUSec();  \
    total_##stage += (end_usec_##stage - start_usec_##stage); \
  } while (0)

#define PROF_MULTI_PRINT(stage)                                                                             \
  do {                                                                                                      \
    MS_LOG(INFO) << #stage << " called " << count_##stage << " times, costs " << total_##stage << " usec."; \
  } while (0)

#define SET_FLAG(value, flag) ((value) = ((value) | (flag)))
#define TEST_FLAG(value, flag) (((value) & (flag)) == (flag))
#define CLEAR_FLAG(value, flag) ((value) = ((value) & (~(flag))))

#define _STRING_COMPILE_OPT(x) #x
#define STRING_COMPILE_OPT(x) _STRING_COMPILE_OPT(x)
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_UTILS_H_
