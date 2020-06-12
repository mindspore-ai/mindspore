/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PARALLEL_OPS_INFO_OPS_UTILS_H_
#define MINDSPORE_CCSRC_PARALLEL_OPS_INFO_OPS_UTILS_H_

namespace mindspore {
namespace parallel {
constexpr size_t PRELU_INPUTS_SIZE = 2;
constexpr size_t PRELU_OUTPUTS_SIZE = 1;
constexpr size_t PRELU_SECOND_INPUT_SIZE = 1;
constexpr int32_t PRELU_CHANNEL_INDEX = 1;
constexpr int32_t PRELU_CHANNEL_STRATEGY = 1;
constexpr int32_t NO_SPLIT_MAP = -1;
constexpr int32_t NO_SPLIT_STRATEGY = 1;
constexpr int32_t SPLIT_FLAG = 1;
constexpr int32_t NO_SPLIT_FLAG = 0;
constexpr size_t MATMUL_ATTRS_SIZE = 2;
constexpr size_t MATMUL_INPUTS_SIZE = 2;
constexpr size_t MATMUL_OUTPUTS_SIZE = 1;
constexpr size_t ACTIVATION_ATTR_SIZE = 1;
constexpr size_t SOFTMAX_ATTR_SIZE = 1;
constexpr size_t ACTIVATION_INPUTS_SIZE = 1;
constexpr size_t ACTIVATION_OUTPUTS_SIZE = 1;
constexpr size_t EXPANDDIMS_INPUT_SIZE = 2;
constexpr size_t DROPOUT_DO_MASK_CNODE_INPUT_SIZE = 4;
constexpr size_t DROPOUT_GEN_MASK_CNODE_INPUT_SIZE = 3;
constexpr size_t DROPOUT_GEN_MASK_INDEX = 2;
constexpr size_t DROPOUT_DO_MASK_KEEP_PROB_INDEX = 3;
constexpr size_t SoftmaxCrossEntropyWithLogitsAttrSize = 1;
constexpr size_t SoftmaxCrossEntropyWithLogitsInputsSize = 2;
constexpr size_t SoftmaxCrossEntropyWithLogitsOutputsSize = 2;
constexpr double EPS = 1e-6;
constexpr double INF = 1e20;

constexpr char AUTO_PARALLEL_RUN_ONCE_ONLY[] = "auto_parallel_run_once_only";
constexpr char SEMI_AUTO_PARALLEL_RUN_ONCE_ONLY[] = "semi_auto_parallel_run_once_only";
constexpr char CHECK_SET_STRATEGY_VALID_ONCE_ONLY[] = "check_set_strategy_valid_once_only";
constexpr char STRATEGY[] = "strategy";
constexpr char GEN_STRATEGY[] = "gen_strategy";
constexpr char REDUCE_OP_SUM[] = "sum";
constexpr char REDUCE_OP_MAX[] = "max";
constexpr char REDUCE_OP_MIN[] = "min";
constexpr char OP_PATH[] = "mindspore.ops.operations";
constexpr char GET_OP_FUNCTION_PATH[] = "mindspore.parallel._utils";
constexpr char GET_OP_FUNCTION[] = "_get_python_op";
constexpr char KEEP_DIMS[] = "keep_dims";
constexpr char CROSS_BATCH[] = "cross_batch";
constexpr char STEP_PARALLEL_BEGIN[] = "step_parallel_begin";
constexpr char STEP_PARALLEL_END[] = "step_parallel_end";
constexpr char STEP_AUTO_PARALLEL_BEGIN[] = "step_auto_parallel_begin.dot";
constexpr char REQUIRES_GRAD[] = "requires_grad";
constexpr char PARAM_NAME[] = "name";

constexpr char RELU_TYPE[] = "relu";
constexpr char RELU6_TYPE[] = "relu6";
constexpr char SIGMOID_TYPE[] = "sigmoid";
constexpr char OP[] = "op";
constexpr char IDENTITY_INFO[] = "identity_info";
constexpr char DIVISOR[] = "divisor";
constexpr char NONE[] = "None";
constexpr char DEPEND[] = "Depend";
constexpr char BATCH_PARALLEL[] = "BatchParallel";

constexpr char ACTIVATION_TYPE[] = "activation_type";
constexpr char TARGET[] = "primitive_target";
constexpr char CPU[] = "CPU";
constexpr char TRANSPOSE_A[] = "transpose_a";
constexpr char TRANSPOSE_B[] = "transpose_b";
constexpr char SHAPE[] = "shape";
constexpr char BEGIN_MASK[] = "begin_mask";
constexpr char END_MASK[] = "end_mask";
constexpr char ELLIPSIS_MASK[] = "ellipsis_mask";
constexpr char NEW_AXIS_MASK[] = "new_axis_mask";
constexpr char SHRINK_AXIS_MASK[] = "shrink_axis_mask";
constexpr char BEGIN[] = "begin";
constexpr char END[] = "end";
constexpr char STRIDES[] = "strides";
constexpr char GROUP[] = "group";
constexpr char AXIS[] = "axis";
constexpr char OUTPUT_NUM[] = "output_num";
constexpr char SPLIT_COUNT[] = "split_count";
constexpr char SPLIT_DIM[] = "split_dim";
constexpr char CONCAT_DIM[] = "concat_dim";
constexpr char FORWARD[] = "forward";
constexpr char BACKWARD[] = "backward";
constexpr char REDISTRIBUTION[] = "redistribution";
constexpr char REPLACE[] = "replace";
constexpr char CONNSYMBOL[] = "/";
constexpr char INSTANCE_NAME[] = "instance_name";
constexpr char SPLIT_SENS[] = "split_sens";
constexpr char SPLIT_TENSOR[] = "split_tensor";
constexpr char DEV_MAT[] = "dev_mat";
constexpr char TENSOR_MAP[] = "tensor_map";
constexpr char SEED0[] = "Seed0";
constexpr char SEED1[] = "Seed1";
constexpr char KEEP_PROB[] = "keep_prob";
constexpr char SRC[] = "src";
constexpr char CLONE_INFO[] = "clone_info";
constexpr char CLONED[] = "cloned";
constexpr char BE_CLONED[] = "be_cloned";
constexpr char CLONED_INDEX[] = "cloned_index";
constexpr char BE_CLONED_INDEX[] = "be_cloned_index";
constexpr char GROUP_RANKS[] = "group_ranks";
constexpr char IS_IN_FORWARD[] = "is_in_forward";
constexpr char DEFAULT_INPUT[] = "default_input";
constexpr char DTYPE[] = "DType";
constexpr char DEV_NUM[] = "dev_num";
constexpr char MEAN_FLAG[] = "mean_flag";
constexpr char TYPES[] = "types";
constexpr char SHAPES[] = "shapes";
constexpr char GETNEXT_NUM[] = "output_num";
constexpr char SHARED_NAME[] = "shared_name";
constexpr char MIRROR_OP[] = "mirror_op";
constexpr char FORWARD_OP[] = "forward_op";
constexpr char REDISTRIBUTION_OP[] = "redistribution_op";
constexpr char DARA_PARALLEL[] = "data_parallel";
constexpr char FORWARD_REDUCE_SCATTER[] = "forward_reduce_scatter";

// Operator
constexpr char VIRTUAL_DIV[] = "_VirtualDiv";
constexpr char GET_TENSOR_SLICE[] = "_GetTensorSlice";
constexpr char SPLIT[] = "Split";
constexpr char ALL_TO_ALL[] = "_AlltoAll";
constexpr char PERMUTE_BY_AXIS[] = "PermuteByAxis";
constexpr char CONCAT_BY_AXIS[] = "ConcatByAxis";
constexpr char SPLIT_BY_AXIS[] = "SplitByAxis";
constexpr char ALL_REDUCE[] = "AllReduce";
constexpr char MIRROR_OPERATOR[] = "_MirrorOperator";
constexpr char STRIDED_SLICE[] = "StridedSlice";
constexpr char ALL_GATHER[] = "AllGather";
constexpr char REDUCE_SCATTER[] = "ReduceScatter";
constexpr char HOST_REDUCE_SCATTER[] = "HostReduceScatter";
constexpr char EMBEDDING_LOOKUP[] = "EmbeddingLookup";
constexpr char CONCAT[] = "Concat";
constexpr char SOFTMAX_CROSS_ENTROPY_WITH_LOGITS[] = "SoftmaxCrossEntropyWithLogits";
constexpr char SIGMOID_CROSS_ENTROPY_WITH_LOGITS[] = "SigmoidCrossEntropyWithLogits";
constexpr char MATMUL[] = "MatMul";
constexpr char GELU[] = "Gelu";
constexpr char TANH[] = "Tanh";
constexpr char SOFTMAX[] = "Softmax";
constexpr char LOG_SOFTMAX[] = "LogSoftmax";
constexpr char ACTIVATION[] = "Activation";
constexpr char PRELU[] = "PReLU";
constexpr char FLOORDIV[] = "FloorDiv";
constexpr char MAXPOOL[] = "MaxPool";
constexpr char MAXPOOLV2[] = "MaxPoolV2";
constexpr char L2_NORMALIZE[] = "L2Normalize";
constexpr char TRANSPOSE[] = "Transpose";
constexpr char RESHAPE[] = "Reshape";
constexpr char TENSOR_ADD[] = "TensorAdd";
constexpr char BIAS_ADD[] = "BiasAdd";
constexpr char SUB[] = "Sub";
constexpr char MUL[] = "Mul";
constexpr char DIV[] = "Div";
constexpr char REAL_DIV[] = "RealDiv";
constexpr char ASSIGN_SUB[] = "AssignSub";
constexpr char GREATER[] = "Greater";
constexpr char VIRTUAL_DATA_SET[] = "_VirtualDataset";
constexpr char VIRTUAL_DATA_SET_INFO[] = "VirtualDatasetInfo";
constexpr char SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS[] = "SparseSoftmaxCrossEntropyWithLogits";
constexpr char RELU[] = "ReLU";
constexpr char ONEHOT[] = "OneHot";
constexpr char DROPOUT_DO_MASK[] = "DropoutDoMask";
constexpr char DROPOUT_GEN_MASK[] = "DropoutGenMask";
constexpr char REDUCE_MAX[] = "ReduceMax";
constexpr char REDUCE_MIN[] = "ReduceMin";
constexpr char REDUCE_SUM[] = "ReduceSum";
constexpr char REDUCE_MEAN[] = "ReduceMean";
constexpr char ARGMAXWITHVALUE[] = "ArgMaxWithValue";
constexpr char ARGMINWITHVALUE[] = "ArgMinWithValue";
constexpr char CONV2D[] = "Conv2D";
constexpr char FUSE_BATCH_NORM[] = "FusedBatchNorm";
constexpr char BATCH_NORM[] = "BatchNorm";
constexpr char LAYER_NORM[] = "LayerNorm";
constexpr char POOLING[] = "Pooling";
constexpr char CAST[] = "Cast";
constexpr char MAX_POOL_WITH_ARGMAX[] = "MaxPoolWithArgmax";
constexpr char SIMPLE_MEAN[] = "SimpleMean";
constexpr char FLATTEN[] = "Flatten";
constexpr char J[] = "J";
constexpr char TMPIDENTITY_INFO_NAME[] = "identity_info";
constexpr char COS[] = "Cos";
constexpr char ACOS[] = "ACos";
constexpr char EXP[] = "Exp";
constexpr char LOG[] = "Log";
constexpr char SIGMOID[] = "Sigmoid";
constexpr char POW[] = "Pow";
constexpr char MAXIMUM[] = "Maximum";
constexpr char MINIMUM[] = "Minimum";
constexpr char EQUAL[] = "Equal";
constexpr char NOT_EQUAL[] = "NotEqual";
constexpr char LOGICALNOT[] = "LogicalNot";
constexpr char GATHERV2[] = "GatherV2";
constexpr char SPARSE_GATHERV2[] = "SparseGatherV2";
constexpr char STRIDEDSLICE[] = "StridedSlice";
constexpr char BROADCAST[] = "Broadcast";
constexpr char SQRT[] = "Sqrt";
constexpr char ASSIGN[] = "Assign";
constexpr char GET_NEXT[] = "GetNext";
constexpr char SQUEEZE[] = "Squeeze";
constexpr char NEG[] = "Neg";
constexpr char BATCH_MATMUL[] = "BatchMatMul";
constexpr char EXPAND_DIMS[] = "ExpandDims";
constexpr char SQUARE[] = "Square";
constexpr char BATCHMATMUL[] = "BatchMatMul";
constexpr char TOPK[] = "TopK";
constexpr char IN_TOPK[] = "InTopK";
constexpr char PACK[] = "Pack";
constexpr char GATHER_ND[] = "GatherNd";
constexpr char UNSORTEF_SEGMENT_MIND[] = "UnsortedSegmentMinD";
constexpr char UNSORTEF_SEGMENT_PRODD[] = "UnsortedSegmentProdD";

// Parallel don't care
constexpr char TUPLE_GETITEM[] = "tuple_getitem";
constexpr char STRING_EQUAL[] = "string_equal";
constexpr char MAKE_TUPLE[] = "make_tuple";
constexpr char MAKE_LIST[] = "make_list";
constexpr char MAKE_DICT[] = "make_dict";
constexpr char MAKE_SLICE[] = "make_slice";
constexpr char MAKE_RECORD[] = "make_record";
constexpr char LIST_GETITEM[] = "list_getitem";
constexpr char ARRAY_GETITEM[] = "array_getitem";
constexpr char TUPLE_SETITEM[] = "tuple_setitem";
constexpr char LIST_SETITEM[] = "list_setitem";
constexpr char ARRAY_SETITEM[] = "array_setitem";
constexpr char DICT_GETITEM[] = "dict_getitem";
constexpr char LIST_APPEND[] = "list_append";
constexpr char LIST_MAP[] = "list_map";
constexpr char LIST_REDUCE[] = "list_reduce";
constexpr char TUPLE_REVERSED[] = "tuple_reversed";
constexpr char TILE_SHAPE[] = "tile_shape";
constexpr char REDUCED_SHAPE[] = "reduced_shape";
constexpr char TUPLE_DIV[] = "tuple_div";
constexpr char TUPLE_TO_ARRAY[] = "tuple_to_array";
constexpr char VIRTUALLOSS[] = "VirtualLoss";
constexpr char RETURN[] = "return";
constexpr char ENV_GETITEM[] = "env_getitem";
constexpr char IDENTITY[] = "identity";
constexpr char PARTIAL[] = "partial";
constexpr char ENVSETITEM[] = "env_setitem";
constexpr char ENVGETITEM[] = "env_getitem";
constexpr char ENVADD[] = "env_add";
constexpr char MAKEREFKEY[] = "MakeRefKey";
constexpr char MAKEREF[] = "make_ref";
constexpr char GETREFKEY[] = "get_ref_key";
constexpr char GETREFVALUE[] = "get_ref_value";
constexpr char GETREFORIGIN[] = "get_ref_origin";
constexpr char STATESETITEM[] = "state_setitem";
constexpr char SCALARSUMMARY[] = "ScalarSummary";
constexpr char IMAGESUMMARY[] = "ImageSummary";
constexpr char TENSORSUMMARY[] = "TensorSummary";
constexpr char HISTOGRAMSUMMARY[] = "HistogramSummary";
constexpr char BROADCASTGRADIENTARGS[] = "BroadcastGradientArgs";
constexpr char INVERTPERMUTATION[] = "InvertPermutation";
constexpr char CONTROLDEPEND[] = "ControlDepend";
constexpr char DOT[] = "dot";
constexpr char IM2COL[] = "im2col";
constexpr char COL2IM[] = "col2im";
constexpr char IM2COLV1[] = "im2col_v1";
constexpr char COL2IMV1[] = "col2im_v1";
constexpr char RESOLVE[] = "resolve";
constexpr char EMBED[] = "embed";
constexpr char CREATINSTANCE[] = "create_instance";
constexpr char ZEROSLIKE[] = "ZerosLike";
constexpr char REF_TO_EMBED[] = "RefToEmbed";
constexpr char STOP_GRADIENT[] = "stop_gradient";

constexpr size_t LAST_INDEX(size_t s) { return s - 1; }
constexpr size_t SECOND_FROM_END(size_t s) { return s - 2; }
constexpr size_t THIRD_FROM_END(size_t s) { return s - 3; }
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PARALLEL_OPS_INFO_OPS_UTILS_H_
