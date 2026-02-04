/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_GRAPH_KERNEL_GRAPH_KERNEL_JSON_FLAGS_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_GRAPH_KERNEL_GRAPH_KERNEL_JSON_FLAGS_H_
namespace mindspore::graphkernel {
constexpr auto kJsonKeyOpDesc = "op_desc";
constexpr auto kJsonKeyAttr = "attr";
constexpr auto kJsonKeyInputDesc = "input_desc";
constexpr auto kJsonKeyFormat = "format";
constexpr auto kJsonKeyInferDataType = "infer_data_type";
constexpr auto kJsonKeyInferShape = "infer_shape";
constexpr auto kJsonKeyShape = "shape";
constexpr auto kJsonKeySymbolicShape = "symbolic_shape";
constexpr auto kJsonKeySymbolCalcExpr = "symbol_calc_expr";
constexpr auto kJsonKeyDataType = "data_type";
constexpr auto kJsonKeyDataformat = "data_format";
constexpr auto kJsonKeyOutputDesc = "output_desc";
constexpr auto kJsonKeyName = "name";
constexpr auto kJsonKeyTensorName = "tensor_name";
constexpr auto kJsonKeyValue = "value";
constexpr auto kJsonKeyImplPath = "impl_path";
constexpr auto kJsonKeyProcess = "process";
constexpr auto kJsonKeyComposite = "composite";
constexpr auto kJsonKeyId = "id";
constexpr auto kJsonKeyOp = "op";
constexpr auto kJsonKeyPtrAddress = "ptr_address";
constexpr auto kJsonKeyCompositeGraph = "composite_graph";
constexpr auto kJsonKeyPlatform = "platform";
constexpr auto kJsonKeyOpFullName = "op_full_name";
constexpr auto kJsonKeyParallelFusion = "parallel_fusion";
constexpr auto kJsonKeyFusionType = "fusion_type";
constexpr auto kJsonKeySubGraph = "sub_graph";
constexpr auto kJsonKeyCoreNum = "core_num";
constexpr auto kJsonKeyTypeInfo = "type_info";
constexpr auto kJsonKeyRecomputeOps = "recompute_ops";
constexpr auto kJsonKeyBufferStitch = "buffer_stitch";
constexpr auto kJsonKeyStitchOp = "stitch_op";
constexpr auto kJsonKeyStitchAtomicOp = "stitch_atomic_op";
constexpr auto kJsonKeyVersion = "version";
constexpr auto kJsonKeyTargetInfo = "target_info";
constexpr auto kJsonKeyComputeCapability = "compute_capability";
constexpr auto kJsonKeySmCount = "sm_count";
constexpr auto kJsonKeySystem = "system";
constexpr auto kJsonKeyArch = "arch";
constexpr auto kJsonKeyCpuFeature = "feature";
constexpr auto kJsonKeyCpuType = "cpu";
constexpr auto kJsonKeyNodeName = "node_name";
constexpr auto kJsonKeyDynamicInputIndex = "dynamic_input_index";
}  // namespace mindspore::graphkernel
#endif
