/**
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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

#include "transform/graph_ir/convert.h"

#include <algorithm>
#include <unordered_set>
#include <vector>

#include "op_proto/inc/array_ops.h"
#include "op_proto/inc/elewise_calculation_ops.h"
#include "op_proto/inc/save_ops.h"
#include "op_proto/inc/state_ops.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/config_manager.h"
#include "include/common/utils/utils.h"
#include "include/transform/graph_ir/utils.h"
#include "ir/graph_utils.h"
#include "ops/array_ops.h"
#include "ops/conv_pool_ops.h"
#include "ops/framework_ops.h"
#include "ops/image_ops.h"
#include "ops/math_op_name.h"
#include "ops/nn_ops.h"
#include "ops/nn_optimizer_ops.h"
#include "ops/other_ops.h"
#include "ops/sequence_ops.h"
#include "ops/structure_ops.h"
#include "ops/lite_ops.h"
#include "ops/op_def.h"
#include "ops/auto_generate/gen_ops_primitive.h"
#include "plugin/device/ascend/hal/hardware/ascend_collective_comm_lib.h"
#include "plugin/device/ascend/hal/hardware/dummy_ascend_collective_comm_lib.h"
#include "plugin/device/ascend/hal/hardware/ge_utils.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"
#include "transform/graph_ir/op_adapter.h"
#include "transform/graph_ir/op_adapter_desc.h"
#include "transform/graph_ir/op_adapter_map.h"
#include "transform/graph_ir/storage_format_convertor.h"
#include "utils/anf_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "utils/symbolic.h"
#include "utils/singleton.h"

namespace mindspore::transform {
using ::ge::Operator;
using mindspore::kValueAny;
using std::make_shared;
using std::shared_ptr;
using std::string;
using std::vector;
using Variable = ::ge::op::Variable;
using Constant = ::ge::op::Constant;
using Assign = ::ge::op::Assign;
using Data = ::ge::op::Data;
using RefData = ::ge::op::RefData;
using std::endl;
using std::static_pointer_cast;

constexpr int64_t kInputOffset = 2;
constexpr size_t kSwitchInputSize = 4;
constexpr size_t kSwitchBodyIndex = 2;
constexpr size_t kSwitchAfterIndex = 3;
constexpr size_t kAfterIndexInCache = 2;
constexpr size_t kCnodeInputSizeOne = 1;
constexpr size_t kDataInputIndex = 1;
constexpr size_t kInputSize2 = 2;
constexpr size_t kMergeInputSize = 2;
constexpr size_t kNoOpOptThreshold = 3;
constexpr auto kHcclFusionByFusionID = 2;
constexpr auto kHcclFusionDefault = 1;
constexpr auto kTypeNoOp = "NoOp";
constexpr auto kTypeIdentity = "Identity";
constexpr auto kTypeIdentityN = "IdentityN";
constexpr auto kTypeMerge = "Merge";
constexpr auto kTypeIf = "If";
constexpr auto kTypeVariable = "Variable";
constexpr auto kParallelGroup = "_parallel_group";
constexpr auto kParallelGroupId = "_parallel_group_id";
constexpr auto kTypeRefData = "RefData";
constexpr auto kBroadcast = "broadcast";
constexpr auto kInit = "init";
constexpr auto kTypeData = "Data";
constexpr auto kTypeIndex = "index";
constexpr auto kTypeY = "y";
constexpr auto kTypeX = "x";
constexpr auto kProcessNodeEngineID = "_process_node_engine_id";
constexpr auto kIsFreeVariable = "_is_free_variable";

namespace {
const std::map<TypeId, TypeId> kReduceRaiseMap = {{kNumberTypeInt64, kNumberTypeInt32}};
mindspore::HashMap<std::string, size_t> branches_repeat_times = {};
mindspore::HashMap<std::string, size_t> call_subgraphs_repeat_times = {};
// {node name | {{input_index, dst_type}...}}
const std::map<std::string, std::vector<std::pair<size_t, TypeId>>> kTransInputDTypeMap = {
  {kResizeNearestNeighborGradOpName, {{2, kNumberTypeInt32}}},
  {kResizeNearestNeighborOpName, {{2, kNumberTypeInt32}}},
  {kResizeNearestNeighborV2OpName, {{2, kNumberTypeInt32}}},
  {kResizeNearestNeighborV2GradOpName, {{2, kNumberTypeInt32}}},
  {kResizeBicubicOpName, {{2, kNumberTypeInt32}}},
  {kConv2DBackpropFilterOpName, {{3, kNumberTypeInt32}}},
  {kConv2DBackpropInputOpName, {{3, kNumberTypeInt32}}},
  {kOneHotOpName, {{2, kNumberTypeInt32}}},
  {kLinSpaceOpName, {{3, kNumberTypeInt32}}},
  {kResizeNearestNeighborV2GradOpName, {{2, kNumberTypeInt32}}},
  {kResizeBilinearV2OpName, {{2, kNumberTypeInt32}}},
  {kCol2ImOpName, {{2, kNumberTypeInt32}}}};

// {node name | {{attr_name, dst_type}...}}
const std::map<std::string, std::vector<std::pair<std::string, TypeId>>> kTransAttrDTypeMap = {
  {kResizeBilinearOpName, {{"size", kNumberTypeInt32}}},
  {kSpaceToBatchNDOpName, {{"block_shape", kNumberTypeInt32}}},
  {kBatchToSpaceNDOpName, {{"block_shape", kNumberTypeInt32}}},
  {kSplitVOpName, {{"split_dim", kNumberTypeInt32}}},
  {kSplitVDOpName, {{"split_dim", kNumberTypeInt32}}}};

bool IsValidConversion(TypeId src_type, TypeId dst_type) {
  if (src_type == dst_type) {
    MS_LOG(DEBUG) << "No need convert, src type and dst type is same, type:" << TypeIdToString(src_type);
    return false;
  }
  auto iter = kReduceRaiseMap.find(src_type);
  if (iter != kReduceRaiseMap.end() && iter->second == dst_type) {
    MS_LOG(INFO) << "Convert data type from " << TypeIdToString(src_type) << " to " << TypeIdToString(dst_type);
    return true;
  }
  MS_LOG(DEBUG) << "Unsupported conversion. src_type:" << TypeIdToString(src_type)
                << ", dst_type:" << TypeIdToString(dst_type);
  return false;
}

template <typename T>
ValuePtr CreateNewValue(const ValuePtr &value, const std::vector<T> &values, const TypeId &dst_type) {
  MS_EXCEPTION_IF_NULL(value);
  if (dst_type == kNumberTypeInt32) {
    if (value->isa<ValueSequence>()) {
      std::vector<int32_t> result;
      std::for_each(values.begin(), values.end(),
                    [&result](const auto &elem) { result.emplace_back(static_cast<int32_t>(elem)); });
      return MakeValue(result);
    }
    return MakeValue(static_cast<int32_t>(values[0]));
  } else {
    MS_LOG(EXCEPTION) << "Invalid dst type:" << TypeIdToString(dst_type);
  }
  return value;
}

template <typename T>
std::vector<T> GetAllValues(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  std::vector<T> result;
  if (value->isa<ValueSequence>()) {
    auto value_seq = value->cast<ValueSequencePtr>();
    MS_EXCEPTION_IF_NULL(value_seq);
    for (const auto &elem : value_seq->value()) {
      auto value_list = GetAllValues<T>(elem);
      std::copy(value_list.begin(), value_list.end(), std::back_inserter(result));
    }
  } else {
    result.emplace_back(GetValue<T>(value));
  }
  return result;
}

TypeId GetElemType(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::Tensor>()) {
    auto tensor_ptr = value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor_ptr);
    return tensor_ptr->data_type();
  }
  if (!value->isa<ValueList>() && !value->isa<ValueTuple>()) {
    return value->type()->type_id();
  }

  auto elems = value->isa<ValueTuple>() ? value->cast<ValueTuplePtr>()->value() : value->cast<ValueListPtr>()->value();
  if (elems.empty()) {
    MS_LOG(EXCEPTION) << "Value:" << value->ToString() << " is empty, check pls.";
  }
  return GetElemType(elems.at(0));
}

ValuePtr CastDstValue(const ValuePtr &value, const TypeId &dst_type) {
  MS_EXCEPTION_IF_NULL(value);
  auto src_type = GetElemType(value);
  if (!IsValidConversion(src_type, dst_type)) {
    return nullptr;
  }
  if (src_type == kNumberTypeInt64) {
    if (value->isa<tensor::Tensor>()) {
      auto tensor_ptr = value->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor_ptr);
      auto tensor_size = tensor_ptr->Size() / sizeof(int64_t);
      int64_t *data = static_cast<int64_t *>(tensor_ptr->data_c());
      std::vector<int32_t> v;
      for (size_t i = 0; i < tensor_size; i++) {
        (void)v.emplace_back(LongToInt(data[i]));
      }
      return MakeValue(v);
    }
    auto values = GetAllValues<int64_t>(value);
    return CreateNewValue<int64_t>(value, values, dst_type);
  } else {
    MS_LOG(EXCEPTION) << "Invalid src type:" << value->type()->ToString();
  }
  return value;
}

// If mark_fv is true, set the kIsFreeVariable flag for all free variables and their inputs.
AnfNodeWeakPtrList SuccIncludeFv(const FuncGraphPtr &fg, const AnfNodePtr &node, bool mark_fv = false) {
  AnfNodeWeakPtrList vecs;
  if (node == nullptr) {
    return vecs;
  }

  if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    bool is_fv = mark_fv && node->has_user_data(kIsFreeVariable);
    auto &weak_inputs = cnode->weak_inputs();

    // Check if free variables used.
    for (const auto &weak_input : weak_inputs) {
      auto input = weak_input.lock();
      MS_EXCEPTION_IF_NULL(input);
      if (is_fv) {
        input->set_user_data(kIsFreeVariable, std::make_shared<bool>(true));
      }
      auto input_fg = GetValueNode<FuncGraphPtr>(input);
      if (input_fg) {
        for (auto &fv : input_fg->free_variables_nodes()) {
          if (fv->func_graph() == fg && fg->nodes().contains(fv)) {
            if (mark_fv) {
              fv->set_user_data(kIsFreeVariable, std::make_shared<bool>(true));
            }
            (void)vecs.emplace_back(fv);
          }
        }
      }
    }

    (void)vecs.insert(vecs.end(), weak_inputs.begin(), weak_inputs.end());
  }

  return vecs;
}

std::vector<AnfNodePtr> GetOrderedCNodes(const FuncGraphPtr fg, const AnfNodePtr node = nullptr) {
  MS_EXCEPTION_IF_NULL(fg);
  auto succ_include_fv = [&fg](const AnfNodePtr &node) -> AnfNodeWeakPtrList { return SuccIncludeFv(fg, node); };

  return (node == nullptr) ? TopoSort(fg->get_return(), succ_include_fv) : TopoSort(node, succ_include_fv);
}

std::set<std::string> GetFvNames(const FuncGraphPtr fg) {
  MS_EXCEPTION_IF_NULL(fg);
  auto succ_include_fv = [&fg](const AnfNodePtr &node) -> AnfNodeWeakPtrList { return SuccIncludeFv(fg, node, true); };

  std::set<std::string> fvs;
  auto nodes = TopoSort(fg->get_return(), succ_include_fv);
  for (const auto &node : nodes) {
    if (node->has_user_data(kIsFreeVariable)) {
      node->set_user_data(kIsFreeVariable, std::shared_ptr<bool>(nullptr));
      fvs.emplace(node->fullname_with_scope());
    }
  }

  return fvs;
}

int64_t GetDynInputNum(const OpAdapterPtr &adpt, bool is_call, std::vector<int64_t> dyn_input_sizes,
                       size_t real_input_idx, size_t input_size, const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(adpt);
  MS_EXCEPTION_IF_NULL(node);
  int64_t dyn_input_num = -1;
  if (!dyn_input_sizes.empty()) {
    dyn_input_num = dyn_input_sizes.at(real_input_idx - 1);
  } else if (adpt->IsDynInputOp(real_input_idx)) {
    if (is_call) {
      auto &input = node->inputs().back();
      // the first input of Call node is Primitive, the second input is kernel_graph,
      // which should not be members of input args, so the dyn_input_num need to minus 2 in default.
      if (IsPrimitiveCNode(input, prim::kPrimUpdateState)) {
        // For PartitionedCall, Monod should not be a member of input args, so here dyn_input_num need to minus 3.
        dyn_input_num = SizeToLong(input_size) - 3;
      } else {
        dyn_input_num = SizeToLong(input_size) - 2;
      }
      return dyn_input_num;
    }
    dyn_input_num = 1;
  }
  return dyn_input_num;
}

bool IsBranchNode(const AnfNodePtr &node) { return IsIfNode(node) || IsCaseNode(node); }

std::vector<AnfNodePtr> GetAnfCallInputs(bool is_kernel_graph, const CNodePtr &c_node) {
  std::vector<AnfNodePtr> inputs;
  if (is_kernel_graph) {
    (void)std::copy(c_node->inputs().begin() + kInputOffset, c_node->inputs().end(), std::back_inserter(inputs));
  } else {
    if (c_node->input(0)->isa<CNode>()) {
      auto in0 = c_node->input(0)->cast<CNodePtr>();
      (void)std::copy(in0->inputs().begin() + kInputOffset, in0->inputs().end(), std::back_inserter(inputs));
    }
    (void)std::copy(c_node->inputs().begin() + 1, c_node->inputs().end(), std::back_inserter(inputs));
  }
  return inputs;
}

bool HasSubgraph(const std::shared_ptr<AnfGraph> &func_graph) {
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto sub_graph = GetCNodeFuncGraph(node);
    if (sub_graph != nullptr) {
      return true;
    }
  }
  return false;
}

bool IsMakeTupleWithNullValue(const AnfNodePtr &node, const AnfNodePtr &input) {
  MS_EXCEPTION_IF_NULL(input);
  if (IsPrimitiveCNode(node, prim::kPrimMakeTuple) && input->isa<ValueNode>()) {
    auto type = input->Type();
    MS_EXCEPTION_IF_NULL(type);
    if (type->isa<Tuple>()) {
      auto tuple_type = type->cast<std::shared_ptr<Tuple>>();
      MS_EXCEPTION_IF_NULL(tuple_type);
      if (tuple_type->elements().empty()) {
        return true;
      }
    }
  }
  return false;
}

bool IsMonad(const AnfNodePtr &node) {
  return IsValueNode<UMonad>(node) || IsValueNode<IOMonad>(node) || HasAbstractMonad(node);
}

bool IsOverFlowNode(const AnfNodePtr &node, const AnfNodePtr &input) {
  return IsPrimitiveCNode(input, prim::kPrimNPUClearFloatStatusV2) ||
         IsPrimitiveCNode(node, prim::kPrimNPUClearFloatStatusV2) ||
         IsPrimitiveCNode(node, prim::kPrimNPUGetFloatStatusV2);
}

std::string SelectParamOriFormat(const FuncGraphManagerPtr &manager, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(manager);
  MS_EXCEPTION_IF_NULL(node);
  std::deque<AnfNodePtr> todo{node};
  while (!todo.empty()) {
    auto &curr_node = todo.front();
    todo.pop_front();
    const auto &nodes = manager->node_users()[curr_node];
    for (const auto &node_pair : nodes) {
      if (IsPrimitiveCNode(node_pair.first, prim::kPrimLoad)) {
        todo.emplace_back(node_pair.first);
      } else if (node_pair.first->isa<CNode>()) {
        auto visited_format = GetOpIOFormat(node_pair.first);
        if (visited_format != kOpFormat_DEFAULT) {
          return visited_format;
        }
      }
    }
  }
  return kOpFormat_DEFAULT;
}

std::vector<int> GetGeTensorOrders(const mindspore::HashMap<int, int> &ge_input_to_ms_input,
                                   const std::vector<int64_t> &dyn_input_sizes, const int &ge_input_size,
                                   std::vector<int64_t> *new_dyn_input_sizes) {
  std::vector<int> ge_tensor_orders(ge_input_size, -1);
  for (int ge_idx = 0; ge_idx < ge_input_size; ++ge_idx) {
    int ms_idx = ge_input_to_ms_input.at(ge_idx);
    new_dyn_input_sizes->at(ge_idx) = dyn_input_sizes[ms_idx];
    int begin_idx = 0;
    for (int i = 0; i < ms_idx; ++i) {
      begin_idx += dyn_input_sizes[i] == -1 ? 1 : dyn_input_sizes[i];
    }
    ge_tensor_orders[ge_idx] = begin_idx;
  }
  return ge_tensor_orders;
}

bool IsNeedToUpdateTensorDesc(const std::string &op_type, const AnfNodePtr &node) {
  // When IdentityN's input is Function or IdentityN, it can not
  // find GEType mapping to MSType. There are ERROR logs that do not affect the result. So it no need to set OutputDesc
  // of IdentityN, it can be inferred by GE. eg: MakeTuple-->MakeTuple. Output node should set OpDesc.
  if (op_type == kTypeIdentityN && !IsPrimitiveCNode(node, prim::kPrimReturn)) {
    MS_LOG(DEBUG) << "No need to set the OpDesc of Identity except return, node: " << node->fullname_with_scope();
    return false;
  }
  // NoOp has not output, so it no need to set OutputDesc.
  if (op_type == kTypeNoOp) {
    MS_LOG(DEBUG) << "No need to set the OpDesc of NoOp, node: " << node->fullname_with_scope();
    return false;
  }
  return true;
}

template <typename T>
void SetXDataIndex(const OperatorPtr &op, T idx) {
  MS_EXCEPTION_IF_NULL(op);
  op->SetAttr(kTypeIndex, static_cast<int64_t>(idx));
}

bool ParamCompare(const std::string &l, const std::string &r, const mindspore::HashMap<std::string, AnfNodePtr> &params,
                  const NodeUsersMap &node_users) {
  auto lpram_iter = params.find(l);
  auto rpram_iter = params.find(r);
  if (lpram_iter == params.end() && rpram_iter == params.end()) {
    return l.compare(r) < 0;
  } else if (lpram_iter == params.end()) {
    return true;
  } else if (rpram_iter == params.end()) {
    return false;
  }

  bool lused_as_accum = (GetMomentumVarByAccum(lpram_iter->second, node_users) != nullptr);
  bool rused_as_accum = (GetMomentumVarByAccum(rpram_iter->second, node_users) != nullptr);
  if (lused_as_accum ^ rused_as_accum) {
    return rused_as_accum;
  }

  return l.compare(r) < 0;
}

bool IsESNodeWithNoOutput(const AnfNodePtr &node) {
  const mindspore::HashSet<PrimitivePtr, PrimitiveHasher, PrimitiveEqual> no_output_prims = {
    prim::kPrimInitPartitionMap,          prim::kPrimInitEmbeddingHashmap,      prim::kPrimEmbeddingTableImport,
    prim::kPrimEmbeddingComputeVarExport, prim::kPrimEmbeddingComputeVarImport, prim::kPrimEmbeddingTableExport};
  if (IsOneOfPrimitiveCNode(node, no_output_prims)) {
    return true;
  }
  return false;
}

std::vector<AnfNodePtr> GetEmbeddingApplyAdamOutput(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  std::vector<AnfNodePtr> ret_nodes;
  auto depend = node->input(1);
  MS_EXCEPTION_IF_NULL(depend);
  if (!IsPrimitiveCNode(depend, prim::kPrimDepend)) {
    MS_LOG(EXCEPTION) << "Need Depend ops, but get " << depend->fullname_with_scope();
  }
  auto depend_cnode = depend->cast<CNodePtr>();
  auto tuple = depend_cnode->input(2);
  MS_EXCEPTION_IF_NULL(tuple);
  if (!IsPrimitiveCNode(tuple, prim::kPrimMakeTuple)) {
    MS_LOG(EXCEPTION) << "Need MakeTuple ops, but get " << tuple->fullname_with_scope();
  }
  auto tuple_cnode = tuple->cast<CNodePtr>();
  auto output_nodes = tuple_cnode->inputs();
  ret_nodes.emplace_back(depend_cnode->input(1));
  ret_nodes.insert(ret_nodes.end(), output_nodes.begin() + 1, output_nodes.end());
  return ret_nodes;
}
}  // namespace

DfGraphPtr GenExampleGraph(const std::string &name) {
  MS_LOG(INFO) << "Gen example graph name is " << name;
  auto graph = std::make_shared<DfGraph>(name);
  MS_EXCEPTION_IF_NULL(graph);
  auto shape_data = std::vector<int64_t>({1, 1, 1, 1});
  GeTensorDesc desc_data(ge::Shape(shape_data), ge::FORMAT_ND, ge::DT_FLOAT16);
  auto data = ge::op::Data("data");
  data.set_attr_index(0);
  data.update_input_desc_x(desc_data);
  data.update_output_desc_y(desc_data);
  auto abs = ge::op::Abs("abs").set_input_x(data);
  std::vector<Operator> inputs{data};
  std::vector<Operator> outputs{abs};
  graph->SetInputs(inputs);
  graph->SetOutputs(outputs);
  return graph;
}

// ---------------implement of DfGraphConvertor-------------

bool IsDynamicShapeNode(const AnfNodePtr node) {
  auto shape = node->Shape();
  if (shape == nullptr) {
    return false;
  }
  if (!shape->isa<abstract::Shape>()) {  // do not accept tuple shape as call node input
    return false;
  }
  if (shape->IsDynamic()) {
    return true;
  }
  return false;
}

bool DfGraphConvertor::InitLoopVar(std::vector<::ge::Operator> *init_input) {
  MS_EXCEPTION_IF_NULL(init_input);
  if (!this->training_) {
    return false;
  }
  bool is_sink_size_repeat = false;
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  int64_t value = 0;
  if (ConfigManager::GetInstance().dataset_mode() == DS_SINK_MODE) {
    static int64_t sink_size = 0;
    if (!ms_context->get_param<bool>(MS_CTX_ENABLE_LOOP_SINK)) {
      return false;
    }
    value = ConfigManager::GetInstance().iter_num();
    if (sink_size == value) {
      is_sink_size_repeat = true;
    }
    sink_size = value;
  } else {
    MS_LOG(INFO) << "Run with normal(non-sink) mode, the iterator number will always be 1";
    ConfigManager::GetInstance().ResetIterNum();
    return false;
  }
  GeTensorDesc desc(GeShape(), ::ge::FORMAT_NCHW, ::ge::DT_INT64);
  auto var_iter_num = std::make_shared<Variable>("npu_runconfig/iterations_per_loop");
  auto var_loop_cond = std::make_shared<Variable>("npu_runconfig/loop_cond");
  auto var_one = std::make_shared<Variable>("npu_runconfig/one");
  auto var_zero = std::make_shared<Variable>("npu_runconfig/zero");
  (void)var_iter_num->update_output_desc_y(desc);
  (void)var_loop_cond->update_output_desc_y(desc);
  (void)var_one->update_output_desc_y(desc);
  (void)var_zero->update_output_desc_y(desc);
  vars_["npu_runconfig/iterations_per_loop"] = var_iter_num;
  vars_["npu_runconfig/loop_cond"] = var_loop_cond;
  vars_["npu_runconfig/one"] = var_one;
  vars_["npu_runconfig/zero"] = var_zero;
  auto const_iter_num = std::make_shared<Constant>("const/npu_runconfig/iterations_per_loop");
  value -= 1;  // iteration start from 0, the max iteration number for n loop should be n-1
  (void)const_iter_num->set_attr_value(GeTensor(desc, reinterpret_cast<uint8_t *>(&value), sizeof(int64_t)));

  auto const_loop_cond = std::make_shared<Constant>("const/npu_runconfig/loop_cond");
  value = 0;
  (void)const_loop_cond->set_attr_value(GeTensor(desc, reinterpret_cast<uint8_t *>(&value), sizeof(int64_t)));

  auto const_one = std::make_shared<Constant>("const/npu_runconfig/one");
  value = 1;
  (void)const_one->set_attr_value(GeTensor(desc, reinterpret_cast<uint8_t *>(&value), sizeof(int64_t)));

  auto const_zero = std::make_shared<Constant>("const/npu_runconfig/zero");
  value = 0;
  (void)const_zero->set_attr_value(GeTensor(desc, reinterpret_cast<uint8_t *>(&value), sizeof(int64_t)));

  (void)const_iter_num->update_output_desc_y(desc);
  (void)const_loop_cond->update_output_desc_y(desc);
  (void)const_one->update_output_desc_y(desc);
  (void)const_zero->update_output_desc_y(desc);

  auto assign_iter_num = std::make_shared<Assign>("assign/npu_runconfig/iterations_per_loop");
  (void)assign_iter_num->set_input_ref(*var_iter_num).set_input_value(*const_iter_num);
  auto assign_loop_cond = std::make_shared<Assign>("assign/npu_runconfig/loop_cond");
  (void)assign_loop_cond->set_input_ref(*var_loop_cond).set_input_value(*const_loop_cond);
  auto assign_one = std::make_shared<Assign>("assign/npu_runconfig/one");
  (void)assign_one->set_input_ref(*var_one).set_input_value(*const_one);
  auto assign_zero = std::make_shared<Assign>("assign/npu_runconfig/zero");
  (void)assign_zero->set_input_ref(*var_zero).set_input_value(*const_zero);

  init_input->emplace_back(*var_iter_num);
  init_input->emplace_back(*var_loop_cond);
  init_input->emplace_back(*var_one);
  init_input->emplace_back(*var_zero);
  init_ops_.emplace_back(var_iter_num);
  init_ops_.emplace_back(var_loop_cond);
  init_ops_.emplace_back(var_one);
  init_ops_.emplace_back(var_zero);
  init_ops_.emplace_back(const_iter_num);
  init_ops_.emplace_back(const_loop_cond);
  init_ops_.emplace_back(const_one);
  init_ops_.emplace_back(const_zero);
  init_ops_.emplace_back(assign_iter_num);
  init_ops_.emplace_back(assign_loop_cond);
  init_ops_.emplace_back(assign_one);
  init_ops_.emplace_back(assign_zero);
  return is_sink_size_repeat;
}

void DfGraphConvertor::DrawParamInitSubGraph(const std::string &name, const AnfNodePtr &it) {
  // draw init subgraph
  init_sout_ << "op_assign" << it.get() << "[label=<";
  init_sout_ << "<table border='1' cellborder='1'>" << endl;
  init_sout_ << "<tr>";
  init_sout_ << "<td port='1'>resource</td>";
  init_sout_ << "<td port='2'>value</td>";
  init_sout_ << "</tr>" << endl;
  init_sout_ << "<tr><td colspan=\"2\">"
             << "\"assign_" << name << "\"</td></tr>" << endl;
  init_sout_ << "</table>> shape=plaintext]" << endl;
  init_sout_ << "param" << it.get() << "[shape=octagon, label=\"" << name << "\"]" << endl;
  init_sout_ << "const" << it.get() << "[label= \"" << name << "_const"
             << "\" shape=ellipse]" << endl;
  init_sout_ << "param" << it.get() << "->"
             << "op_assign" << it.get() << ":1" << endl;
  init_sout_ << "const" << it.get() << "->"
             << "op_assign" << it.get() << ":2" << endl;
}

void DfGraphConvertor::SetupParamInitSubGraph(const TensorOrderMap &tensors,
                                              const std::vector<::ge::Operator> *const init_input,
                                              bool is_sink_size_repeat) {
  DfGraphPtr init_graph = std::make_shared<DfGraph>(kInit);
  MS_EXCEPTION_IF_NULL(init_graph);
  std::vector<AnfNodePtr> nodes = GetOrderedCNodes(anf_graph_);

  for (auto &it : nodes) {
    MS_EXCEPTION_IF_NULL(it);
    if (it->isa<ValueNode>()) {
      if (IsValueNode<SymbolicKeyInstance>(it)) {
        auto symbolic = GetValueNode<SymbolicKeyInstancePtr>(it);
        auto name = std::static_pointer_cast<Parameter>(symbolic->node())->name();
        auto iter = vars_.find(name);  // get corresponding variable op
        if (iter != vars_.end()) {
          op_cache_[it.get()] = iter->second;
          // #ifdef DRAW_GE_GRAPH
          compute_sout_ << op_draw_name_[params_[name].get()] << " -> " << op_draw_name_[it.get()]
                        << "[style=\"dotted\"]" << endl;
          // #endif
        }
      } else if (IsValueNode<RefKey>(it)) {
        auto refkey = GetValueNode<StringImmPtr>(it);
        MS_EXCEPTION_IF_NULL(refkey);
        auto name = refkey->value();
        auto iter = vars_.find(name);  // get corresponding variable op
        if (iter != vars_.end()) {
          op_cache_[it.get()] = iter->second;
          compute_sout_ << op_draw_name_[params_[name].get()] << " -> " << op_draw_name_[it.get()]
                        << "[style=\"dotted\"]" << endl;
        }
      }
    }
  }

  for (auto &it : tensors) {
    if (vars_.find(it.first) == vars_.end()) {
      MS_LOG(WARNING) << "Init parameter " << it.first << " didn't appear in graph.";
      vars_[it.first] = nullptr;
    }
  }

  // set up init sub graph
  MS_EXCEPTION_IF_NULL(init_input);
  if (!init_input->empty() && !is_sink_size_repeat) {
    // init sub graph needs no input
    MS_LOG(INFO) << "Build data init subgraph.";
    (void)init_graph->SetInputs(*init_input);
    this->init_graph_ = init_graph;
  } else {
    this->init_graph_ = nullptr;
  }
}

void DfGraphConvertor::SetupParamInitSubGraph() {
  DfGraphPtr init_graph = std::make_shared<DfGraph>("init");
  std::vector<AnfNodePtr> nodes = GetOrderedCNodes(anf_graph_);
  MS_EXCEPTION_IF_NULL(init_graph);

  for (auto &it : nodes) {
    MS_EXCEPTION_IF_NULL(it);
    if (it->isa<ValueNode>()) {
      if (IsValueNode<SymbolicKeyInstance>(it)) {
        auto symbolic = GetValueNode<SymbolicKeyInstancePtr>(it);
        MS_EXCEPTION_IF_NULL(symbolic);
        MS_EXCEPTION_IF_NULL(std::static_pointer_cast<Parameter>(symbolic->node()));
        auto name = std::static_pointer_cast<Parameter>(symbolic->node())->name();
        auto iter = vars_.find(name);  // get corresponding variable op
        if (iter != vars_.end()) {
          op_cache_[it.get()] = iter->second;
        }
      } else if (IsValueNode<RefKey>(it)) {
        auto refkey = GetValueNode<StringImmPtr>(it);
        MS_EXCEPTION_IF_NULL(refkey);
        auto name = refkey->value();
        auto iter = vars_.find(name);  // get corresponding variable op
        if (iter != vars_.end()) {
          op_cache_[it.get()] = iter->second;
        }
      }
    }
  }

  // set up init sub graph
  std::vector<::ge::Operator> init_input;
  bool is_sink_size_repeat = InitLoopVar(&init_input);
  if (!init_input.empty() && !is_sink_size_repeat) {
    // init sub graph needs no input
    MS_LOG(INFO) << "Build data init subgraph.";
    (void)init_graph->SetInputs(init_input);
    this->init_graph_ = init_graph;
  } else {
    this->init_graph_ = nullptr;
  }
}

void DfGraphConvertor::SetupBroadcast(const OperatorPtr &broadcast, const std::vector<GeTensorDesc> &broadcast_desc,
                                      const DfGraphPtr &broadcast_graph, std::vector<::ge::Operator> broadcast_input) {
  MS_LOG(INFO) << "build broadcast subgraph";
  if (broadcast_desc.size() != broadcast_input.size()) {
    MS_LOG(EXCEPTION) << "Desc number of BroadCast is not equal to number of Input";
  }
  MS_EXCEPTION_IF_NULL(broadcast);
  (void)broadcast->DynamicInputRegister(kTypeX, (static_cast<unsigned int>(broadcast_input.size())));
  (void)broadcast->DynamicOutputRegister(kTypeY, static_cast<unsigned int>(broadcast_desc.size()));
  for (unsigned int i = 0; i < broadcast_input.size(); i++) {
    (void)broadcast->SetInput(kTypeX, i, broadcast_input[i]);
    (void)broadcast->UpdateDynamicOutputDesc(kTypeY, i, broadcast_desc[i]);
  }
  MS_EXCEPTION_IF_NULL(broadcast_graph);
  (void)broadcast_graph->SetInputs(broadcast_input);
  this->broadcast_graph_ = broadcast_graph;
}

bool DfGraphConvertor::NodeInputKeepUpdate(const FuncGraphManagerPtr &manager, const AnfNodePtr &node) {
  if (manager == nullptr || node == nullptr) {
    MS_LOG(ERROR) << "Input argument manager or node is nullptr";
    return false;
  }
  if (offline_convert_) {
    return false;
  }
  if (std::find(extra_variables_names_.begin(), extra_variables_names_.end(), node->fullname_with_scope()) !=
      extra_variables_names_.end()) {
    return true;
  }
  const auto &node_users = manager->node_users();
  std::vector<PrimitivePtr> vec{
    prim::kPrimAssign,        prim::kPrimKVCacheMgr,     prim::kPrimScatterUpdate,       prim::kPrimScatterNdUpdate,
    prim::kPrimPromptKVCache, prim::kPrimDecoderKVCache, prim::kPrimKVCacheScatterUpdate};
  auto user_it = node_users.find(node);
  if (user_it != node_users.end()) {
    auto &users = user_it->second;
    for (auto &user_node : users) {
      auto &node_use = user_node.first;
      if (node_use && std::any_of(vec.begin(), vec.end(),
                                  [&node_use](const PrimitivePtr &prim) { return IsPrimitiveCNode(node_use, prim); })) {
        return true;
      }
      // check if node is ReshapeAndKVCache which is fused by akg.
      if (IsPrimitiveCNode(node_use, prim::kPrimCustom)) {
        auto prim_custom = GetCNodePrimitive(node_use);
        const std::string kAttrNameInfoPath = "info_path";

        if (!prim_custom->HasAttr(kAttrNameInfoPath)) {
          continue;
        }
        auto info_path_attr_node = prim_custom->GetAttr(kAttrNameInfoPath);
        if (info_path_attr_node == nullptr) {
          MS_LOG(EXCEPTION) << "attr node '" << kAttrNameInfoPath << "' is null";
        }
        std::string info_path = GetValue<std::string>(info_path_attr_node);
        const std::string kOpReshapeAndCache = "ReshapeAndCache";
        if (info_path.find(kOpReshapeAndCache) == std::string::npos) {
          continue;
        }

        MS_LOG(INFO) << "found ReshapeAndCache, make use inpu keep update";
        return true;
      }
    }
  }
  return false;
}

void DfGraphConvertor::JudgeParamTransType(const bool &node_will_update, bool *as_ref_data, bool *as_constant) const {
  if (ref_mode_) {
    if ((ref_mode_type_ == RefModeFlag::kRefModeAll || node_will_update) && !export_air_) {
      *as_ref_data = true;
    } else {  // When only variable will be treated as RefData, constant Parameter will be treated as Constant
      *as_constant = true;
    }
  } else if (!training_ && !node_will_update) {
    // parameter will be updated, lite inference mode will treat as variables
    *as_constant = true;
  }
}

void DfGraphConvertor::InitParamWithData(const TensorOrderMap &tensors) {
  int index = 0;
  std::vector<Operator> init_input;
  MS_EXCEPTION_IF_NULL(graph_manager_);
  // The format of Momentum's accum is updated according to format of Momentum's var, so here sort tensors to put
  // Momentum's accum parameter at last
  auto cmp = std::bind(ParamCompare, std::placeholders::_1, std::placeholders::_2, std::cref(params_),
                       graph_manager_->node_users());
  std::map<std::string, std::pair<int, tensor::TensorPtr>, decltype(cmp)> ordered_tensors(cmp);
  // NOTE: the sequence of parameters of init DfGraph is calculated by TensorOrderMap, see method `GetInputTensors`
  // defined in `mindspore/ccsrc/plugin/device/ascend/hal/hardware/ge_graph_executor.cc`
  for (auto &it : tensors) {
    ordered_tensors.insert({it.first, {index++, it.second}});
  }
  for (const auto &itor : ordered_tensors) {
    std::string name = itor.first;
    auto &it = itor.second;
    auto node_itor = params_.find(name);
    // if name not in params_, create a node in graph
    if (node_itor == params_.end()) {
      // In lite, param maybe not exist.
      MS_LOG(WARNING) << name << " is not in params, and create a new node.";
      ParameterPtr param = std::make_shared<Parameter>(nullptr);
      MS_EXCEPTION_IF_NULL(param);
      if (!ref_mode_) {
        name += "_temp";
      }
      param->set_name(name);
      (void)ConvertParameter(param);
      node_itor = params_.find(name);
    }
    auto node = node_itor->second;
    MS_EXCEPTION_IF_NULL(node);
    auto op_itor = op_cache_.find(node.get());
    if (op_itor == op_cache_.end()) {
      MS_LOG(EXCEPTION) << "Can not find op for node " << node->ToString() << ".";
    }

    MS_EXCEPTION_IF_NULL(it.second);
    bool as_ref_data = false;
    bool as_constant = false;
    auto node_will_update = NodeInputKeepUpdate(graph_manager_, node);
    JudgeParamTransType(node_will_update, &as_ref_data, &as_constant);

    auto shape = it.second->shape_c();
    if (as_ref_data && dyn_ref_data_func_ != nullptr) {
      shape = dyn_ref_data_func_(node, shape);
    }
    auto desc =
      TransformUtil::GetGeTensorDesc(shape, it.second->data_type(), SelectParamOriFormat(graph_manager_, node));
    if (desc == nullptr) {
      MS_LOG(WARNING) << "Create const " << name << " output descriptor failed!";
      continue;
    }
    if (as_ref_data) {
      StorageFormatConvertor::SetupStorageFormat(anf_graph_, node, desc);
      auto variable = std::make_shared<RefData>(name);
      MS_EXCEPTION_IF_NULL(variable);
      (void)variable->update_output_desc_y(*desc);
      (void)variable->update_input_desc_x(*desc);
      (void)variable->set_attr_index(SizeToInt(ref_datas_.size()));
      (void)ref_datas_.emplace_back(variable);
      ref_data_names_.emplace_back(name);
      // do not use read variable while variable sink
      MS_LOG(DEBUG) << "InitParam, op_name = " << name << ", var = " << variable->GetName() << ".";
      op_itor->second = variable;  // replace parameter with variable
      vars_[name] = variable;      // prevent the variable operator from being freed
    } else if (as_constant) {
      auto adpt_const = FindAdapter(kNameConst, training_);
      if (adpt_const == nullptr) {
        continue;
      }
      auto const_op = adpt_const->generate(name + "_const");
      (void)adpt_const->setAttr(const_op, "value", it.second);
      const_op->UpdateOutputDesc(kTypeY, *desc);
      const_op_to_value_[const_op] = it.second;
      vars_[name] = const_op;
      op_itor->second = const_op;
    } else {
      auto &infer_need_update_parameter_names =
        Singleton<mindspore::device::ascend::InferNeedUpdateParaNames>::Instance().GetInferParameterNames();
      // we need three variable ops for each graph with same name
      // build init subgraph
      auto adpt = FindAdapter(kNameParam, training_);
      if (adpt == nullptr) {
        continue;
      }
      auto param_op = adpt->generate(name + "_data");
      if (it.second->is_init() == 0) {
        SetXDataIndex(param_op, it.first);
        ProcessInputData(&init_input, &infer_need_update_parameter_names, param_op, name, desc);
      }

      auto variable = std::make_shared<Variable>(name);
      MS_EXCEPTION_IF_NULL(variable);
      (void)variable->update_output_desc_y(*desc);
      // do not use read variable while variable sink
      MS_LOG(DEBUG) << "InitParam, op_name = " << name << ", var = " << variable->GetName() << ".";
      op_itor->second = variable;  // replace parameter with variable
      vars_[name] = variable;      // prevent the variable operator from being freed
      DrawParamInitSubGraph(name, node);
    }
  }
  if (ref_mode_) {
    SetupParamInitSubGraph();
  } else {
    bool is_sink_size_repeat = InitLoopVar(&init_input);
    SetupParamInitSubGraph(tensors, &init_input, is_sink_size_repeat);
  }
}

void DfGraphConvertor::ProcessInputData(vector<Operator> *init_input,
                                        std::unordered_set<std::string> *infer_need_update_parameter_names,
                                        const OperatorPtr &param_op, const string &name,
                                        const std::shared_ptr<GeTensorDesc> &desc) {
  MS_EXCEPTION_IF_NULL(init_input);
  MS_EXCEPTION_IF_NULL(infer_need_update_parameter_names);
  auto init_var = std::make_shared<Variable>(name);
  auto assign_op = std::make_shared<Assign>("assign_" + name);
  MS_EXCEPTION_IF_NULL(init_var);
  MS_EXCEPTION_IF_NULL(assign_op);
  (void)init_var->update_output_desc_y(*desc);
  (void)assign_op->set_input_ref(*init_var).set_input_value(*param_op);
  init_input->emplace_back(*init_var);
  this->init_ops_.emplace_back(param_op);
  this->init_ops_.emplace_back(assign_op);
  this->init_ops_.emplace_back(init_var);
  this->init_data_names_.emplace_back(name);
  infer_need_update_parameter_names->insert(name);
}

// convert all parameter need initialize to variable
DfGraphConvertor &DfGraphConvertor::InitParam(const TensorOrderMap &tensors) {
  if (error_ != SUCCESS) {
    return *this;
  }
  if (anf_graph_ == nullptr || anf_graph_->output() == nullptr) {
    error_ = INVALID_ARGUMENT;
    MS_LOG(ERROR) << "Invalid AnfGraph in InitParam.";
    return *this;
  }

  InitParamWithData(tensors);
  init_sout_ << "}" << endl;
  return *this;
}

#if (defined ENABLE_D)
void DfGraphConvertor::BuildSaveCheckpointGraph() {
  std::vector<Operator> graph_inputs;
  ::ge::op::Save save_op("save_parms");
  int save_op_is_active = 0;
  size_t index = 0;
  string name;

  auto count_size = std::count_if(vars_.begin(), vars_.end(), [](const auto &it) {
    return LongToUlong(it.second == nullptr || it.first.find("/") != std::string::npos);
  });

  (void)save_op.create_dynamic_input_tensors(static_cast<uint32_t>(vars_.size() - static_cast<size_t>(count_size)));

  // for each "parameter" in anf graph excluding "input"
  for (const auto &it : vars_) {
    name = it.first;
    if (it.second == nullptr || name.find("/") != std::string::npos) {
      continue;
    }
    Variable variable(name);
    (void)variable.update_output_desc_y(it.second->GetOutputDesc(0));
    (void)save_op.set_dynamic_input_tensors(static_cast<uint32_t>(index++), variable);

    graph_inputs.emplace_back(variable);

    if (save_op_is_active == 0) {
      checkpoint_sout_ << "op_save" << &save_op << "[label=<";
      checkpoint_sout_ << "<table border='1' cellborder='1'>" << endl;
      checkpoint_sout_ << "<tr><td port='1'>tensor</td></tr>" << endl;
      checkpoint_sout_ << "<tr><td colspan=\"1\">"
                       << "\"saveop"
                       << "\"</td></tr>" << endl;
      checkpoint_sout_ << "</table>> shape=plaintext]" << endl;
    }

    checkpoint_sout_ << "param" << it.second << "[shape=octagon, label=\"" << name << "\"]" << endl;

    checkpoint_sout_ << "param" << it.second << "->"
                     << "op_save" << &save_op << ":1" << endl;
    save_op_is_active = 1;
  }
  if (save_op_is_active != 0) {
    std::vector<Operator> graph_output;
    (void)graph_output.emplace_back(save_op);
    DfGraphPtr checkpoint_graph = std::make_shared<DfGraph>("checkpoint");
    (void)checkpoint_graph->SetInputs(graph_inputs);
    (void)checkpoint_graph->SetOutputs(graph_output);
    this->save_ckp_graph_ = checkpoint_graph;
  } else {
    this->save_ckp_graph_ = nullptr;
  }

  checkpoint_sout_ << "}" << endl;
  return;
}
#endif

DfGraphConvertor &DfGraphConvertor::GenerateBroadcastGraph(const TensorOrderMap &tensors) {
  if (error_ != SUCCESS) {
    return *this;
  }
  if (anf_graph_ == nullptr || anf_graph_->output() == nullptr) {
    error_ = INVALID_ARGUMENT;
    MS_LOG(ERROR) << "Invalid AnfGraph in generate broadcast graph";
    return *this;
  }

  DfGraphPtr broadcast_graph = std::make_shared<DfGraph>(kBroadcast);
  // collect the operators create for broadcast sub graph, in order to avoid auto release
  std::vector<Operator> broadcast_input;
  std::vector<GeTensorDesc> broadcast_desc;
  auto adpt = FindAdapter(kNameBroadcast);
  if (!adpt) {
    MS_LOG(EXCEPTION) << "Get adpt failed, node type: HcomBroadcast";
  }
  auto broadcast = adpt->generate("broadcast_parameter");
  const int64_t root_rank_v = 0;
  (void)broadcast->SetAttr("root_rank", root_rank_v);
  (void)broadcast->SetAttr("group", "hccl_world_group");
  broadcast_ops_.emplace_back(broadcast);

  // find every parameter, build broadcast subgraph (or initialize the parameter with constant)
  for (auto &it : anf_graph_->parameters()) {
    auto op_itor = op_cache_.find(it.get());  // converted node
    if (it->isa<Parameter>() && op_itor != op_cache_.end()) {
      string name = std::static_pointer_cast<Parameter>(it)->name();
      auto tensor_itor = tensors.find(name);  // in init tensor map
      if (tensor_itor != tensors.end()) {
        auto tensor = tensor_itor->second;
        auto shape_ge = tensor->shape_c();

        // create tensor descriptor for output descriptor
        auto desc = TransformUtil::GetGeTensorDesc(shape_ge, tensor->data_type(), kOpFormat_DEFAULT);
        if (desc == nullptr) {
          MS_LOG(ERROR) << "Create variable " << name << " output descriptor failed!";
          continue;
        }

        // build broadcast subgraph
        if (distribute_) {
          auto broadcast_var = std::make_shared<Variable>(name);
          (void)broadcast_var->update_output_desc_y(*desc);
          broadcast_input.emplace_back(*broadcast_var);
          broadcast_desc.emplace_back(*desc);
          broadcast_ops_.emplace_back(broadcast_var);
        }
      }
    }
  }

  // set up broadcast sub graph
  if (!broadcast_input.empty()) {
    DfGraphConvertor::SetupBroadcast(broadcast, broadcast_desc, broadcast_graph, broadcast_input);
  } else {
    this->broadcast_graph_ = nullptr;
  }
  return *this;
}

DfGraphConvertor &DfGraphConvertor::GenerateCheckpointGraph() {
  if (error_ != SUCCESS) {
    MS_LOG(ERROR) << "Generate checkpoint graph failed, found error code " << error_ << ".";
    if (!unsupported_ops_names_.empty()) {
      MS_LOG(ERROR) << "===========================================";
      MS_LOG(ERROR) << unsupported_ops_names_.size() << " Operator(s) cannot be converted:";
      std::string unsupported_ops_list;
      for (const auto &unsupported_ops : unsupported_ops_names_) {
        if (!unsupported_ops_list.empty()) {
          unsupported_ops_list += ", ";
        }
        unsupported_ops_list += unsupported_ops;
      }
      MS_LOG(ERROR) << "Unsupported op type list: " << unsupported_ops_list;
      MS_LOG(ERROR) << "===========================================";
    }
    return *this;
  }
  if (anf_graph_ == nullptr || anf_graph_->output() == nullptr) {
    error_ = INVALID_ARGUMENT;
    MS_LOG(ERROR) << "Invalid AnfGraph in GenerateCheckpointGraph";
    return *this;
  }
#ifdef ENABLE_D
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->backend_policy() == "ge") {
    BuildSaveCheckpointGraph();
    // Restoring from checkpoint file is done by pyfront, not in graph now.
  }
#endif
  return *this;
}

DfGraphConvertor &DfGraphConvertor::ConvertAllNode() {
  if (error_ != SUCCESS) {
    return *this;
  }
  if (anf_graph_ == nullptr || anf_graph_->output() == nullptr) {
    MS_LOG(ERROR) << "Invalid AnfGraph";
    error_ = FAILED;
    return *this;
  }

  compute_sout_.clear();
  compute_sout_ << "digraph {" << endl;
  init_sout_.clear();
  init_sout_ << "digraph {" << endl;
#ifdef ENABLE_D
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->backend_policy() == "ge") {
    checkpoint_sout_.clear();
    checkpoint_sout_ << "digraph {" << endl;
  }
#endif
  restore_checkpoint_sout_.clear();
  restore_checkpoint_sout_ << "digraph {" << endl;
  // Trans data_type for some specific nodes' inputs and attr.
  TransDataType(anf_graph_);
  // Convert all anf node to Operator
  MS_LOG(INFO) << "Convert all node, graph: " << anf_graph_->ToString();
  std::vector<AnfNodePtr> nodes = GetOrderedCNodes(anf_graph_, while_cond_node_);
  if (ref_mode_) {
    // Ref mode need build all node(cnode && parameter).
    for (auto &p : anf_graph_->parameters()) {
      if (std::find(nodes.begin(), nodes.end(), p) == nodes.end()) {
        MS_LOG(INFO) << "Parameter " << p->DebugString() << " can not found in topo sort lists.";
        nodes.emplace_back(p);
      }
    }
  }
  for (auto &it : nodes) {
    if (IsSubGraph() && it->isa<Parameter>()) {
      continue;
    }
    if (IsSubGraph() && (IsPartialSuccNode(it) || IsPartialCNode(it))) {
      continue;
    }
    (void)Convert(it);
    if (this->error_ != SUCCESS) {
      MS_LOG(ERROR) << "Failed to convert node: " << it->DebugString() << ".";
    }
  }

  // return the data flow graph
  return *this;
}

void DfGraphConvertor::CacheWhileGraph(const CNodePtr &cnode) {
  if (while_graph_cache_.find(cnode) != while_graph_cache_.end()) {
    return;
  }
  ValueNodePtr graph_node = nullptr;
  if (is_kernel_graph_) {
    graph_node = cnode->input(1)->cast<ValueNodePtr>();
  } else {
    if (cnode->input(0)->isa<ValueNode>()) {
      graph_node = cnode->input(0)->cast<ValueNodePtr>();
    } else {
      auto partial_node = cnode->input(0);
      graph_node = partial_node->cast<CNodePtr>()->input(1)->cast<ValueNodePtr>();
    }
  }

  MS_EXCEPTION_IF_NULL(graph_node);
  FuncGraphPtr cond_graph = graph_node->value()->cast<FuncGraphPtr>();
  MS_EXCEPTION_IF_NULL(cond_graph);
  const auto &cond_set = cond_graph->nodes();
  for (auto beg = cond_set.begin(); beg != cond_set.end(); ++beg) {
    if (!((*beg)->isa<CNode>())) {
      continue;
    }
    auto c_beg = (*beg)->cast<CNodePtr>();
    if (GetCNodeFuncName(c_beg) == prim::kPrimSwitch->name()) {
      while_graph_cache_[cnode] = {c_beg->input(1), c_beg->input(kSwitchBodyIndex), c_beg->input(kSwitchAfterIndex)};
    }
  }
}

std::vector<Operator> DfGraphConvertor::GetWhileBodyOutputs() {
  std::vector<Operator> outputs;

  const auto &node = anf_graph_->get_return()->input(1);
  AnfNodePtr real_ret = node;
  MS_EXCEPTION_IF_NULL(real_ret);
  while (real_ret->isa<CNode>() && GetCNodeTargetFuncName(real_ret->cast<CNodePtr>()) == prim::kPrimDepend->name()) {
    real_ret = real_ret->cast<CNodePtr>()->input(1);
  }

  // skip input of UMonad, IOMonad
  if (HasAbstractUMonad(real_ret) || HasAbstractIOMonad(real_ret)) {
    return outputs;
  }

  // skip input of the None, UpdateState
  if (IsValueNode<None>(real_ret) || IsPrimitiveCNode(real_ret, prim::kPrimUpdateState)) {
    return outputs;
  }

  if (IsPrimitiveCNode(real_ret, prim::kPrimLoad)) {
    real_ret = ParseLoadInput(real_ret->cast<CNodePtr>());
  }

  if (!real_ret->isa<CNode>()) {
    return outputs;
  }

  auto c_node = real_ret->cast<CNodePtr>();
  std::vector<AnfNodePtr> inputs = GetAnfCallInputs(is_kernel_graph_, c_node);
  for (size_t i = 0; i < inputs.size(); i++) {
    auto j = inputs[i];
    MS_EXCEPTION_IF_NULL(j);
    if (!IsDataInput(c_node, j, 0)) {
      continue;
    }
    if (j->isa<Parameter>()) {
      int64_t idx = find(inputs_.begin(), inputs_.end(), j) - inputs_.begin();
      auto idx_cond = body_cond_map_[idx];
      if (while_used_input_index_.find(idx_cond) == while_used_input_index_.end() ||
          while_const_input_index_.find(idx_cond) != while_const_input_index_.end()) {
        continue;
      }
      outputs.emplace_back(*(subgraph_input_cache_[idx]));
    } else {
      outputs.emplace_back(*Convert(j));
    }
  }
  MS_LOG(DEBUG) << "get while body outputs size: " << outputs.size();
  return outputs;
}

std::shared_ptr<std::vector<Operator>> DfGraphConvertor::GetWhileSubGraphInput() {
  std::shared_ptr<std::vector<Operator>> graph_in = std::make_shared<std::vector<Operator>>();
  subgraph_input_cache_.clear();
  size_t i = 0;
  OperatorPtr op = nullptr;
  ParamIndexMap cond_body;
  std::string name_app = "_in_cond";
  if (graph_type_ == GraphType::kBody) {
    name_app = "_in_body";
    for (auto &p : body_cond_map_) {
      cond_body[p.second] = p.first;
    }
  }
  for (auto &idx : while_used_input_index_) {
    if (while_const_input_index_.find(idx) == while_const_input_index_.end()) {
      op = std::make_shared<Data>();
      MS_EXCEPTION_IF_NULL(op);
      SetXDataIndex(op, i);
      i++;
    } else {
      // No need to process ge tensor desc
      auto temp = while_const_input_index_[idx].op;
      auto name = temp->GetName();
      auto value = const_op_to_value_[temp];
      MS_EXCEPTION_IF_NULL(value);
      auto adpt_const = FindAdapter(kNameConst, training_);
      if (adpt_const == nullptr) {
        continue;
      }
      name += name_app;
      auto const_op = adpt_const->generate(name);
      (void)adpt_const->setAttr(const_op, "value", value);
      auto const_op_desc = TransformUtil::GetGeTensorDesc(value->shape_c(), value->data_type(), kOpFormat_DEFAULT);
      if (const_op_desc == nullptr) {
        MS_LOG(WARNING) << "Create variable " << name << " output descriptor failed!";
        continue;
      }
      const_op->UpdateOutputDesc(kTypeY, *const_op_desc);
      op = const_op;
    }
    graph_in->emplace_back(*op);
    if (graph_type_ == GraphType::kCond) {
      subgraph_input_cache_[idx] = op;
    } else if (graph_type_ == GraphType::kBody) {
      subgraph_input_cache_[cond_body[idx]] = op;
    }
  }
  MS_LOG(DEBUG) << "created " << subgraph_input_cache_.size() << " data node "
                << " in graph: " << anf_graph_->ToString();
  return graph_in;
}

void DfGraphConvertor::BuildWhileSubGraph() {
  // set up dependencies

  std::vector<Operator> graph_in = *GetWhileSubGraphInput();
  auto nodes = GetOrderedCNodes(anf_graph_, while_cond_node_);

  AnfNodePtr real_ret = anf_graph_->get_return()->input(1);
  while (real_ret->isa<CNode>() && GetCNodeTargetFuncName(real_ret->cast<CNodePtr>()) == prim::kPrimDepend->name()) {
    real_ret = real_ret->cast<CNodePtr>()->input(1);
  }
  for (auto &it : nodes) {
    if (IsBranchNode(it)) {
      auto node = it->cast<CNodePtr>();
      GetBranchNodeInput(node);
    }
  }

  for (auto &it : nodes) {
    if (it == real_ret || HasAbstractMonad(it)) {
      continue;
    }
    SetNodeInput(it);
    SetSubgraph(it);
    UpdateOpDesc(it);
  }
  std::vector<Operator> graph_out;
  auto graph_name = TransformUtil::NormOpName(cur_while_node_->fullname_with_scope());
  if (graph_type_ == GraphType::kCond) {
    if (op_cache_.find(while_cond_node_.get()) == op_cache_.end()) {
      return;
    }
    graph_name += "_cond_graph";
    graph_out.emplace_back(*(op_cache_[while_cond_node_.get()]));
  } else {
    graph_name += "_body_graph";
    graph_out = GetWhileBodyOutputs();
  }
  if (error_ == SUCCESS) {
    if (df_graph_->GetName() != graph_name) {
      MS_LOG(DEBUG) << "convert anf graph name : " << df_graph_->GetName() << " to df graph name: " << graph_name;
    }
    df_graph_ = make_shared<DfGraph>(graph_name);
  } else {
    return;
  }
  MS_LOG(DEBUG) << "Set while sub graph input num: " << graph_in.size();
  MS_LOG(DEBUG) << "Set while sub graph output num: " << graph_out.size();

  compute_sout_ << "}" << endl;
  (void)df_graph_->SetInputs(graph_in).SetOutputs(graph_out);
  IdentityOptimization();
}

void DfGraphConvertor::BuildWhileAfterSubGraph() {
  size_t i = 0;
  prev_cond_to_while_out_index_.clear();
  for (auto n : prev_while_used_input_index_) {
    if (prev_while_const_input_index_.find(n) == prev_while_const_input_index_.end()) {
      prev_cond_to_while_out_index_[n] = i;
      i++;
    }
  }
  GetCallNodeInputs(cur_while_node_);
  auto nodes = GetOrderedCNodes(anf_graph_);
  for (auto &it : nodes) {
    SetNodeInput(it);
    SetSubgraph(it);
    UpdateOpDesc(it);
  }
  if (graph_outputs_.empty()) {
    SetGraphOutputs();
  }
  compute_sout_ << "}" << endl;
  return;
}

void DfGraphConvertor::ConvertWhileBody(const AnfNodePtr &node) {
  if (!node->isa<CNode>() || GetCNodeFuncName(node->cast<CNodePtr>()) != prim::kPrimPartial->name()) {
    return;
  }
  auto graph_node = node->cast<CNodePtr>()->input(1)->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(graph_node);
  FuncGraphPtr anf_graph = graph_node->value()->cast<FuncGraphPtr>();
  MS_EXCEPTION_IF_NULL(anf_graph);
  DfGraphConvertor converter(anf_graph, phase_prefix_);
  converter.use_inputs_ = true;

  const auto &params = anf_graph->parameters();
  converter.inputs_ = params;

  converter.graph_type_ = GraphType::kBody;
  converter.cur_while_node_ = cur_while_node_;
  converter.body_cond_map_ = body_cond_map_;
  converter.while_const_input_index_ = while_const_input_index_;
  converter.while_used_input_index_ = while_used_input_index_;
  converter.const_op_to_value_ = const_op_to_value_;
  converter.ConvertAllNode().BuildWhileSubGraph();
  while_dfgraph_cache_[cur_while_node_]->emplace_back(*(converter.df_graph_));
  std::string name = graph_node->ToString() + "_ge_graph.dot";
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kFully)) {
    converter.DrawComputeGraph(name);
  }
  return;
}

void DfGraphConvertor::GetWhileUsedInputIndex(const std::vector<AnfNodePtr> &graphs) {
  if (!while_used_input_index_.empty()) {
    return;
  }

  auto cond_graph_node = graphs.at(0);
  auto graph = cond_graph_node->func_graph();
  MS_EXCEPTION_IF_NULL(graph);
  const auto &cond_params = graph->parameters();
  auto nodes = GetOrderedCNodes(graph, cond_graph_node);

  std::set<size_t> used_params_index;
  for (auto &n : nodes) {
    if (!n->isa<CNode>()) {
      continue;
    }
    auto c = n->cast<CNodePtr>();
    auto inputs = c->inputs();
    for (size_t idx = 1; idx < inputs.size(); idx++) {
      auto &i = inputs[idx];
      if (!i->isa<Parameter>() || HasAbstractMonad(i) || IsDynamicShapeNode(i)) {
        continue;
      }
      auto idx_cond = std::find(cond_params.begin(), cond_params.end(), i) - cond_params.begin();
      (void)used_params_index.insert(idx_cond);
    }
  }

  auto body_graph_node_in_cond = graphs.at(1)->cast<CNodePtr>();
  auto body_graph_node = body_graph_node_in_cond->input(1)->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(body_graph_node);
  graph = body_graph_node->value()->cast<FuncGraphPtr>();
  const auto &body_params = graph->parameters();

  auto real_ret = graph->get_return()->input(1);
  while (real_ret->isa<CNode>() && GetCNodeTargetFuncName(real_ret->cast<CNodePtr>()) == prim::kPrimDepend->name()) {
    real_ret = real_ret->cast<CNodePtr>()->input(1);
  }

  nodes = GetOrderedCNodes(graph);
  for (auto &n : nodes) {
    if (!n->isa<CNode>()) {
      continue;
    }
    auto c = n->cast<CNodePtr>();
    if (c == real_ret || c == real_ret->cast<CNodePtr>()->input(0)) {
      continue;
    }
    auto inputs = c->inputs();
    for (size_t idx = 1; idx < inputs.size(); idx++) {
      auto &i = inputs[idx];
      if (!i->isa<Parameter>() || HasAbstractMonad(i) || IsDynamicShapeNode(i)) {
        continue;
      }
      auto idx_body = std::find(body_params.begin(), body_params.end(), i) - body_params.begin();
      auto p = body_graph_node_in_cond->input(static_cast<size_t>(idx_body + kInputOffset));
      auto idx_cond = std::find(cond_params.begin(), cond_params.end(), p) - cond_params.begin();
      (void)used_params_index.insert(idx_cond);
    }
  }
  while_used_input_index_ = used_params_index;
}

void DfGraphConvertor::SetParamIndexMap(const std::vector<AnfNodePtr> &graphs) {
  auto cond_graph_node = graphs.at(0);
  MS_EXCEPTION_IF_NULL(cond_graph_node);
  auto cond_graph = cond_graph_node->func_graph();
  MS_EXCEPTION_IF_NULL(cond_graph);
  const auto &cond_params = cond_graph->parameters();

  auto body_graph_node = graphs.at(1);
  MS_EXCEPTION_IF_NULL(body_graph_node);
  if (!body_graph_node->isa<CNode>()) {
    return;
  }
  MS_EXCEPTION_IF_NULL(body_graph_node->cast<CNodePtr>());
  auto body_graph_node_inputs = body_graph_node->cast<CNodePtr>()->inputs();
  std::vector<AnfNodePtr> body_params;
  for (auto it = body_graph_node_inputs.begin() + kInputOffset; it != body_graph_node_inputs.end(); ++it) {
    body_params.emplace_back(*it);
  }

  for (size_t i = 0; i < body_params.size(); i++) {
    auto p = body_params[i];
    int64_t idx = find(cond_params.begin(), cond_params.end(), p) - cond_params.begin();
    body_cond_map_[i] = static_cast<size_t>(idx);
    MS_LOG(DEBUG) << "body_cond_map_'s key: " << i << " value: " << idx;
  }

  auto after_graph_node = graphs.at(kSwitchBodyIndex);
  MS_EXCEPTION_IF_NULL(after_graph_node);
  if (!after_graph_node->isa<CNode>()) {
    return;
  }
  MS_EXCEPTION_IF_NULL(after_graph_node->cast<CNodePtr>());
  auto after_graph_node_inputs = after_graph_node->cast<CNodePtr>()->inputs();
  std::vector<AnfNodePtr> after_params;
  for (auto it = after_graph_node_inputs.begin() + 2; it != after_graph_node_inputs.end(); ++it) {
    after_params.emplace_back(*it);
  }

  for (size_t i = 0; i < after_params.size(); i++) {
    auto p = after_params[i];
    int64_t idx = find(cond_params.begin(), cond_params.end(), p) - cond_params.begin();
    after_cond_map_[i] = static_cast<size_t>(idx);
    MS_LOG(DEBUG) << "after_cond_map_'s key: " << i << " value: " << idx;
  }
  return;
}

void DfGraphConvertor::ConvertWhileCond(const AnfNodePtr &node) {
  MS_LOG(DEBUG) << "begin to convert while node cond graph";
  auto func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);

  DfGraphConvertor converter(func_graph, phase_prefix_);
  converter.use_inputs_ = true;

  converter.inputs_ = func_graph->parameters();

  converter.graph_type_ = GraphType::kCond;
  converter.cur_while_node_ = cur_while_node_;
  converter.while_cond_node_ = node;
  converter.while_const_input_index_ = while_const_input_index_;
  converter.while_used_input_index_ = while_used_input_index_;
  converter.const_op_to_value_ = const_op_to_value_;
  converter.ConvertAllNode().BuildWhileSubGraph();
  MS_EXCEPTION_IF_NULL(while_dfgraph_cache_[cur_while_node_]);
  while_dfgraph_cache_[cur_while_node_]->emplace_back(*(converter.df_graph_));
  std::string name = func_graph->ToString() + "_ge_graph.dot";
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kFully)) {
    converter.DrawComputeGraph(name);
  }

  MS_LOG(DEBUG) << "convert while node cond graph end";
}

void DfGraphConvertor::SetWhileOutputHandle(const OperatorPtr &prev_while_op) {
  if (while_output_handle_cache_.find(prev_while_node_) != while_output_handle_cache_.end()) {
    return;
  }
  auto out_handler = std::make_shared<std::vector<OutHandler>>();
  MS_EXCEPTION_IF_NULL(out_handler);
  string str = "output";
  for (size_t i = 0; i < prev_while_node_out_size_; i++) {
    (void)out_handler->emplace_back(prev_while_op, str + std::to_string(i), prev_while_node_);
  }
  while_output_handle_cache_[prev_while_node_] = out_handler;
  return;
}

void DfGraphConvertor::ConvertWhileAfter(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>() || GetCNodeFuncName(node->cast<CNodePtr>()) != prim::kPrimPartial->name()) {
    return;
  }
  MS_LOG(DEBUG) << "begin to convert while node after graph";
  MS_EXCEPTION_IF_NULL(node->cast<CNodePtr>()->input(1));
  auto graph_node = node->cast<CNodePtr>()->input(1)->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(graph_node);
  MS_EXCEPTION_IF_NULL(graph_node->value());
  FuncGraphPtr anf_graph = graph_node->value()->cast<FuncGraphPtr>();
  MS_EXCEPTION_IF_NULL(anf_graph);
  DfGraphConvertor converter(anf_graph, phase_prefix_);
  converter.use_inputs_ = true;

  const auto &params = anf_graph->parameters();
  converter.inputs_ = params;

  converter.graph_type_ = GraphType::kAfter;
  converter.prev_after_cond_map_ = after_cond_map_;
  converter.prev_while_node_ = cur_while_node_;
  converter.prev_while_node_out_size_ = cur_while_node_out_size_;
  converter.bypass_node_prev_handle_cache_ = bypass_node_handle_cache_;
  converter.prev_while_used_input_index_ = while_used_input_index_;
  converter.prev_while_const_input_index_ = while_const_input_index_;
  converter.const_op_to_value_ = const_op_to_value_;

  auto while_op = Convert(converter.prev_while_node_);
  converter.SetWhileOutputHandle(while_op);
  converter.ConvertAllNode().BuildWhileAfterSubGraph();
  std::string name = graph_node->ToString() + "_ge_graph.dot";
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kFully)) {
    converter.DrawComputeGraph(name);
  }
  MS_LOG(DEBUG) << "add while after graph " << converter.graph_const_inputs_.size()
                << " const inputs to main graph const inputs";
  (void)std::transform(converter.graph_const_inputs_.begin(), converter.graph_const_inputs_.end(),
                       std::back_inserter(graph_const_inputs_), [](OperatorPtr x) { return x; });

  graph_outputs_ = converter.graph_outputs_;
  MS_LOG(DEBUG) << "convert while node after graph end";
  return;
}

void DfGraphConvertor::ConvertWhileNode(const CNodePtr &node) {
  if (IsSubGraph()) {
    return;
  }

  auto while_graph = while_graph_cache_[node];
  cur_while_node_ = node;

  auto &while_inputs = *(call_input_handle_cache_[node]);
  cur_while_node_out_size_ = while_inputs.size();
  while_dfgraph_cache_[node] = std::make_shared<std::vector<DfGraph>>();
  // convert cond graph
  auto cond_graph_node = while_graph[0];
  ConvertWhileCond(cond_graph_node);

  // convert body graph
  auto body_graph_node = while_graph[1];
  ConvertWhileBody(body_graph_node);

  OpAdapterPtr adpt = FindAdapter(node, training_);
  if (adpt == nullptr) {
    MS_LOG(DEBUG) << "Not found adapter";
    return;
  }

  OperatorPtr op = Convert(node);
  auto graphs = while_dfgraph_cache_[node];
  adpt->setSubgraph(op, graphs);

  // convert after graph
  auto after_graph_node = while_graph[kAfterIndexInCache];
  ConvertWhileAfter(after_graph_node);
  return;
}

std::shared_ptr<std::vector<DfGraph>> DfGraphConvertor::BuildBranchGraphs(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  bool is_case = IsCaseNode(cnode);
  std::shared_ptr<std::vector<DfGraph>> df_branches = std::make_shared<std::vector<DfGraph>>();
  MS_EXCEPTION_IF_NULL(df_branches);
  if (IsNormalGraph() || IsBodyGraph() || IsBranchGraph()) {
    size_t branch_call_input_size = 0;
    size_t node_input_index = 0;
    if (!is_kernel_graph_) {
      for (size_t i = 1; i < cnode->size(); i++) {
        auto pred = cnode->input(i);
        if (!IsDataInput(cnode, pred, 0)) {
          continue;
        }
        node_input_index++;
        branch_call_input_size++;
      }
    }
    MS_EXCEPTION_IF_NULL(cnode->input(0));
    CNodePtr input_node = is_kernel_graph_ ? cnode : cnode->input(0)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(input_node);
    MS_EXCEPTION_IF_NULL(input_node->input(kInputOffset));
    auto bnode = is_case ? input_node->input(kInputOffset)->cast<CNodePtr>() : input_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(bnode);
    const size_t init_i = is_case ? 1 : 2;

    for (size_t i = init_i; i < bnode->size(); i++) {
      ParamIndexMap branch_to_parent_node_map;
      size_t branch_index = 0;  //  branch graph input's index
      if (bnode->input(i)->isa<CNode>()) {
        auto branch_node = bnode->input(i)->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(branch_node);
        for (size_t j = kInputOffset; j < branch_node->size(); j++) {
          auto pred = branch_node->input(j);
          if (!IsDataInput(cnode, pred, 0)) {
            continue;
          }
          branch_to_parent_node_map[branch_index] = node_input_index;
          node_input_index++;
          branch_index++;
        }
      }
      if (!is_kernel_graph_) {
        for (size_t k = 0; k < branch_call_input_size; k++) {
          branch_to_parent_node_map[branch_index] = k;
          branch_index++;
        }
      }
      ProcessSubgraph(cnode, bnode->input(i), branch_to_parent_node_map);
      (void)(df_branches->emplace_back(branches_map_[bnode->input(i).get()]));
    }
  }
  return df_branches;
}

void DfGraphConvertor::BuildCallSubGraphs(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(cnode->input(1));
  ValueNodePtr graph_node = cnode->input(1)->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(graph_node);
  MS_EXCEPTION_IF_NULL(graph_node->value());
  auto anf_graph = graph_node->value()->cast<AnfGraphPtr>();
  MS_EXCEPTION_IF_NULL(anf_graph);
  DfGraphConvertor converter(anf_graph, phase_prefix_);
  converter.graph_type_ = GraphType::kNormal;
  converter.use_inputs_ = true;
  converter.inputs_ = anf_graph->parameters();
  std::string graph_name = anf_graph->ToString();
  auto iter = call_subgraphs_repeat_times.find(graph_name);
  if (iter == call_subgraphs_repeat_times.end()) {
    call_subgraphs_repeat_times[graph_name] = 1;
  } else {
    iter->second += 1;
    graph_name = graph_name + "_call_" + std::to_string(iter->second);
  }
  (void)converter.ConvertAllNode().BuildGraph(graph_name);

  call_dfgraph_cache_[node] = std::make_shared<std::vector<DfGraph>>();
  MS_EXCEPTION_IF_NULL(call_dfgraph_cache_[node]);
  call_dfgraph_cache_[node]->emplace_back(*(converter.df_graph_));
  MS_LOG(INFO) << "build call subgraph end.";
}

void DfGraphConvertor::SetSubgraph(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return;
  }
  auto cnode = node->cast<CNodePtr>();
  if (IsWhileNode(cnode)) {
    MS_LOG(DEBUG) << "Start to set while's sub graph.";
    CacheWhileGraph(cnode);
    ConvertWhileNode(cnode);
    MS_LOG(DEBUG) << "Set while's sub graph end.";
    return;
  }

  if (IsBranchNode(cnode)) {
    MS_LOG(DEBUG) << "Start to set if/case's sub graph.";
    std::shared_ptr<std::vector<DfGraph>> df_branches = BuildBranchGraphs(cnode);
    if (op_cache_.find(node.get()) == op_cache_.end()) {
      return;
    }

    OpAdapterPtr adpt = FindAdapter(node, training_);
    if (adpt == nullptr) {
      MS_LOG(DEBUG) << "Not found adapter";
      return;
    }

    OperatorPtr op = Convert(node);
    bool is_case = IsCaseNode(node);
    if (is_case) {
      adpt->setSubgraph(op, 0, df_branches);
    } else {
      adpt->setSubgraph(op, df_branches);
    }
    MS_LOG(DEBUG) << "Set if/case's sub graph end.";
    return;
  }

  if (IsCallNode(cnode)) {
    MS_LOG(DEBUG) << "Start to set call's sub graph.";
    BuildCallSubGraphs(node);
    if (op_cache_.find(node.get()) == op_cache_.end()) {
      return;
    }
    OpAdapterPtr adpt = FindAdapter(node, training_);
    if (adpt == nullptr) {
      MS_LOG(EXCEPTION) << "Not found adapter";
      return;
    }
    OperatorPtr op = Convert(node);
    auto df_graphs = call_dfgraph_cache_[node];
    adpt->setSubgraph(op, df_graphs);
    MS_LOG(DEBUG) << "Set call's sub graph end.";
  }
  return;
}

void DfGraphConvertor::GetBranchNodeInput(const CNodePtr node) {
  if (branch_input_handle_cache_.find(node.get()) != branch_input_handle_cache_.end()) {
    return;
  }
  bool is_case = IsCaseNode(node);
  std::vector<AnfNodePtr> branch_inputs;
  const size_t branch_index = 1;

  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(node->input(0));
  CNodePtr sw_node = is_kernel_graph_ ? node : node->input(0)->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(sw_node);
  AnfNodePtr branch_index_iter = sw_node->input(branch_index);
  AnfNodePtr branch_dyn_input_node = nullptr;
  const size_t make_tuple_index = 2;
  AnfNodePtr make_tuple_iter = sw_node->input(make_tuple_index);
  branch_dyn_input_node = make_tuple_iter;  // switch node's 2nd input as dyn input

  std::shared_ptr<std::vector<OutHandler>> tuple_items = std::make_shared<std::vector<OutHandler>>();
  MS_EXCEPTION_IF_NULL(tuple_items);

  CNodePtr input_node = node;
  if (!is_kernel_graph_) {
    for (size_t i = 1; i < node->size(); i++) {
      auto pred = node->input(i);
      (void)(branch_inputs.emplace_back(pred));
    }
    input_node = node->input(0)->cast<CNodePtr>();
  }
  MS_EXCEPTION_IF_NULL(input_node);
  auto bnode = is_case ? input_node->input(make_tuple_index)->cast<CNodePtr>() : input_node;
  MS_EXCEPTION_IF_NULL(bnode);
  const size_t init_i = is_case ? 1 : 2;
  for (size_t i = init_i; i < bnode->size(); ++i) {
    const auto &bnode_input = bnode->input(i);
    MS_EXCEPTION_IF_NULL(bnode_input);
    if (!bnode_input->isa<CNode>()) {
      continue;
    }
    auto branch_node = bnode_input->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(branch_node);
    for (size_t j = 2; j < branch_node->size(); ++j) {
      auto pred = branch_node->input(j);
      (void)(branch_inputs.emplace_back(pred));
    }
  }
  std::vector<AnfNodePtr> branch_control_input;
  for (size_t i = 0; i < branch_inputs.size(); i++) {
    auto item = branch_inputs[i];
    if (!IsDataInput(node, item, 0)) {
      branch_control_input.emplace_back(item);
      continue;
    }
    if (IsBodyGraph() && item->isa<Parameter>()) {
      auto idx = std::find(inputs_.begin(), inputs_.end(), item) - inputs_.begin();
      (void)(tuple_items->emplace_back(subgraph_input_cache_[idx], "", item));
    } else {
      auto hd = GetHandler(item);
      tuple_items->emplace_back(hd);
    }
  }
  tuple_out_handle_cache_[branch_dyn_input_node.get()] = tuple_items;

  std::shared_ptr<std::vector<AnfNodePtr>> branch_input_items = std::make_shared<std::vector<AnfNodePtr>>();
  MS_EXCEPTION_IF_NULL(branch_input_items);
  (void)branch_input_items->emplace_back(branch_index_iter);
  (void)branch_input_items->emplace_back(branch_dyn_input_node);

  (void)std::copy(branch_control_input.begin(), branch_control_input.end(), std::back_inserter(*branch_input_items));
  branch_input_handle_cache_[node.get()] = branch_input_items;
  return;
}

void DfGraphConvertor::GetCallNodeInputs(const CNodePtr &node) {
  if (node == nullptr) {
    return;
  }
  if (call_input_handle_cache_.find(node) != call_input_handle_cache_.end()) {
    return;
  }

  auto call_input_items = std::make_shared<std::vector<OutHandler>>();
  MS_EXCEPTION_IF_NULL(call_input_items);
  std::vector<AnfNodePtr> inputs = GetAnfCallInputs(is_kernel_graph_, node);

  auto &params = anf_graph_->parameters();
  auto while_op = Convert(node);

  while_const_input_index_.clear();
  std::set<size_t> while_input_node_index;
  for (auto iter = while_used_input_index_.begin(); iter != while_used_input_index_.end(); ++iter) {
    auto n = inputs[*iter];
    MS_EXCEPTION_IF_NULL(n);
    OutHandler out_handler;
    if (IsAfterGraph() && n->isa<Parameter>()) {
      auto idx = std::find(params.begin(), params.end(), n) - params.begin();
      auto idx_cond = prev_after_cond_map_[idx];
      if (bypass_node_prev_handle_cache_.find(idx_cond) != bypass_node_prev_handle_cache_.end()) {
        out_handler = bypass_node_prev_handle_cache_[idx_cond];
      } else {
        auto idx_out = prev_cond_to_while_out_index_[idx_cond];
        out_handler = while_output_handle_cache_[prev_while_node_]->at(idx_out);
      }
    } else {
      out_handler = GetHandler(inputs[*iter]);
    }
    MS_EXCEPTION_IF_NULL(out_handler.op);
    if ((out_handler.op->GetOpType() == "Const" || out_handler.op->GetOpType() == "Constant") &&
        const_op_to_value_.find(out_handler.op) != const_op_to_value_.end()) {
      while_const_input_index_[*iter] = out_handler;
    } else {
      (void)while_input_node_index.insert(*iter);
      call_input_items->emplace_back(out_handler);
    }
  }
  cur_while_node_out_size_ = call_input_items->size();
  bypass_node_handle_cache_.clear();

  for (size_t i = 0; i < inputs.size(); i++) {
    if (while_input_node_index.find(i) == while_input_node_index.end()) {
      auto n = inputs[i];
      MS_EXCEPTION_IF_NULL(n);
      if (HasAbstractMonad(n)) {
        continue;
      }
      if (IsAfterGraph() && n->isa<Parameter>()) {
        auto idx = std::find(params.begin(), params.end(), n) - params.begin();
        auto idx_cond = prev_after_cond_map_[idx];
        if (bypass_node_prev_handle_cache_.find(idx_cond) != bypass_node_prev_handle_cache_.end()) {
          bypass_node_handle_cache_[i] = bypass_node_prev_handle_cache_[idx_cond];
        } else {
          auto idx_out = prev_cond_to_while_out_index_[idx_cond];
          bypass_node_handle_cache_[i] = while_output_handle_cache_[prev_while_node_]->at(idx_out);
        }
      } else {
        bypass_node_handle_cache_[i] = GetHandler(n);
      }
    }
  }

  auto op = Convert(node);
  auto adpt = FindAdapter(node, training_);
  MS_EXCEPTION_IF_NULL(adpt);
  adpt->setDynamicOutputNum(op, cur_while_node_out_size_);
  call_input_handle_cache_[node] = call_input_items;
  return;
}

void DfGraphConvertor::SetGraphInputs(std::vector<Operator> *inputs) {
  if (IsNormalGraph() && ConfigManager::GetInstance().dataset_mode() == DS_SINK_MODE) {
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    std::vector<PrimitivePtr> input_prims;
    if (ms_context->get_param<bool>(MS_CTX_ENABLE_GE_HETEROGENOUS)) {
      input_prims = {prim::kPrimQueueData};
    } else {
      input_prims = {prim::kPrimGetNext, prim::kPrimDynamicGetNextV2};
    }

    OperatorPtr input;
    auto nodes = GetOrderedCNodes(anf_graph_);
    for (auto &it : nodes) {
      if (std::any_of(input_prims.begin(), input_prims.end(),
                      [&it](const PrimitivePtr &prim) { return IsPrimitiveCNode(it, prim); })) {
        auto it_op = op_cache_.find(it.get());
        if (it_op != op_cache_.end()) {
          input = it_op->second;
          break;
        } else {
          MS_LOG(EXCEPTION) << "Can not find the operator of node: " << it->fullname_with_scope();
        }
      }
    }
    if (input == nullptr) {
      MS_LOG(EXCEPTION) << "Can not find the GetNext node in graph in sink_mode, please check.";
    }
    inputs->emplace_back(*input);

    MS_EXCEPTION_IF_NULL(anf_graph_);
    anf_graph_->set_flag(kGraphFlagHasGetNext, true);
  } else {
    auto params = anf_graph_->parameters();
    int index = 0;
    for (auto &it : params) {
      auto param = it->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(param);
      auto name = param->name();
      if (std::find(init_data_names_.begin(), init_data_names_.end(), name) == init_data_names_.end()) {
        const auto &param_shape = param->Shape();
        MS_EXCEPTION_IF_NULL(param_shape);
        const auto &shape = param_shape->cast<abstract::ShapePtr>();
        if (shape != nullptr) {
          const auto &sv = shape->shape();
          if (IsDynamic(sv)) {
            dynamic_shape_inputs_ = true;
          }
          input_shapes_.emplace_back(sv);
        }
      }
      //  the parameters which has not been converted to var
      if (vars_.find(name) == vars_.end()) {
        if (HasAbstractMonad(it)) {
          MS_LOG(INFO) << it->DebugString() << " is a monad parameter, skip.";
          continue;
        }
        auto op = Convert(it);
        MS_EXCEPTION_IF_NULL(op);
        MS_LOG(INFO) << "add not var input " << it->ToString() << ", index " << index;
        if (op == nullptr) {
          MS_LOG(ERROR) << "Convert graph failed!";
          return;
        }
        UpdateDataOpDesc(it, op);

        if (IsNormalGraph()) {
          MS_LOG(INFO) << "add input " << it->ToString() << ", index " << index;
          SetXDataIndex(op, index);
          index++;
        }
        inputs->emplace_back(*op);
      } else if (vars_[name] != nullptr) {
        MS_LOG(INFO) << "add var input " << it->ToString();
        auto op = Convert(it);
        MS_EXCEPTION_IF_NULL(op);
        UpdateConstOpDesc(it, vars_[name]);
        inputs->emplace_back(*op);
      }
    }
  }
}

bool DfGraphConvertor::IsConstantOp(const OperatorPtr &op) const {
  if (op == nullptr) {
    return false;
  }
  return (op->GetOpType() == "Constant" || op->GetOpType() == "Const");
}

OperatorPtr DfGraphConvertor::SetGraphInputsForNotVar(const AnfNodePtr &it, int64_t *index,
                                                      std::vector<Operator> *inputs) {
  MS_EXCEPTION_IF_NULL(index);
  MS_EXCEPTION_IF_NULL(inputs);
  auto op = Convert(it);
  if (op == nullptr) {
    MS_LOG(EXCEPTION) << "Convert graph failed!";
  }
  UpdateDataOpDesc(it, op);
  if (IsNormalGraph()) {
    MS_LOG(INFO) << "add input " << it->ToString() << ", index " << *index;
    auto op_type = op->GetOpType();
    if (op_type == kTypeData || op_type == kTypeRefData) {
      SetXDataIndex(op, (*index));
      (*index)++;
    } else {
      auto name = std::static_pointer_cast<Parameter>(it)->name();
      MS_LOG(EXCEPTION) << "Op " << name << " is invalid type " << op->GetOpType() << " as graph input.";
    }
  }
  inputs->push_back(*op);
  return op;
}

void DfGraphConvertor::SetGraphInputs(std::vector<Operator> *inputs, AnfNodeWeakPtrList *ge_inputs) {
  MS_EXCEPTION_IF_NULL(inputs);
  MS_EXCEPTION_IF_NULL(ge_inputs);
  MS_LOG(INFO) << "IsNormalGraph=" << IsNormalGraph() << ", dataset_mode"
               << ConfigManager::GetInstance().dataset_mode();
  AddInputInDataSink(inputs);
  auto params = anf_graph_->parameters();
  MS_LOG(INFO) << "Parameters size " << params.size();
  int64_t index = 0;
  std::set<std::string> name_records = {};
  for (auto &it : params) {
    auto name = std::static_pointer_cast<Parameter>(it)->name();
    OperatorPtr op;
    //  the parameters which has not been converted to var
    if (vars_.find(name) == vars_.end()) {
      auto abs = it->abstract();
      MS_EXCEPTION_IF_NULL(abs);
      if (HasAbstractMonad(it) || abs->isa<abstract::AbstractSequence>()) {
        MS_LOG(INFO) << it->DebugString() << " is a monad or tuple/list parameter, skip.";
        continue;
      }
      op = SetGraphInputsForNotVar(it, &index, inputs);
    } else if (vars_[name] != nullptr) {
      MS_LOG(INFO) << "add var input " << it->ToString() << ", index " << index;
      op = Convert(it);
      MS_EXCEPTION_IF_NULL(op);
      if (name_records.count(name) != 0) {
        // two parameters have same ref_key
        MS_LOG(INFO) << "var input " << it->ToString() << " is already added";
        continue;
      }
      (void)name_records.insert(name);
      UpdateConstOpDesc(it, vars_[name]);
      auto op_type = op->GetOpType();
      if (op_type == kTypeRefData) {
        SetXDataIndex(op, index);
        index++;
      } else if (IsConstantOp(op)) {
        continue;
      } else {
        MS_LOG(EXCEPTION) << "Op " << name << " is invalid type " << op->GetOpType() << " as graph input.";
      }
      inputs->push_back(*op);
    }
    (void)ge_inputs->emplace_back(AnfNodeWeakPtr(it));
  }
  MS_LOG(INFO) << "Input size " << inputs->size();
}

void DfGraphConvertor::AddInputInDataSink(vector<Operator> *inputs) {
  MS_EXCEPTION_IF_NULL(inputs);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::vector<PrimitivePtr> input_prims;
  if (ms_context->get_param<bool>(MS_CTX_ENABLE_GE_HETEROGENOUS)) {
    input_prims = {prim::kPrimQueueData};
  } else {
    input_prims = {prim::kPrimGetNext, prim::kPrimDynamicGetNextV2};
  }
  OperatorPtr input = nullptr;
  auto nodes = GetOrderedCNodes(anf_graph_);
  for (auto &it : nodes) {
    if (std::any_of(input_prims.begin(), input_prims.end(),
                    [&it](const PrimitivePtr &prim) { return IsPrimitiveCNode(it, prim); })) {
      auto it_op = op_cache_.find(it.get());
      if (it_op != op_cache_.end()) {
        input = it_op->second;
        break;
      } else {
        MS_LOG(EXCEPTION) << "Can not find the operator of node: " << it->fullname_with_scope();
      }
    }
  }
  if (IsNormalGraph() && ConfigManager::GetInstance().dataset_mode() == DS_SINK_MODE && input != nullptr) {
    (void)inputs->emplace_back(*input);
    MS_EXCEPTION_IF_NULL(anf_graph_);
    anf_graph_->set_flag(kGraphFlagHasGetNext, true);
  }
}

void DfGraphConvertor::BuildInitDataGraph(const std::string &name) {
  MS_LOG(INFO) << "Start BuildInitDataGraph.";

  // If MS_CTX_ENABLE_GE_HETEROGENOUS is true, no need InitData graph
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<bool>(MS_CTX_ENABLE_GE_HETEROGENOUS)) {
    df_graph_ = nullptr;
    return;
  }

  AnfNodePtr init_dataset_queue_node = nullptr;
  auto nodes = GetOrderedCNodes(anf_graph_);
  for (auto &it : nodes) {
    if (IsInitDataSetQueueNode(it)) {
      init_dataset_queue_node = it;
      break;
    }
  }
  OperatorPtr init_data_op = Convert(init_dataset_queue_node);
  MS_EXCEPTION_IF_NULL(init_data_op);
  if (error_ != SUCCESS) {
    return;
  }
  std::vector<::ge::Operator> inputs{*init_data_op};
  std::vector<::ge::Operator> outputs{*init_data_op};
  df_graph_ = make_shared<DfGraph>(name);
  (void)df_graph_->SetInputs(inputs);
  (void)df_graph_->SetOutputs(outputs);
  MS_LOG(INFO) << "End BuildInitDataGraph.";
}

void DfGraphConvertor::FillEmptyInputsWithNoInputOp(std::vector<Operator> *inputs) {
  MS_EXCEPTION_IF_NULL(inputs);
  MS_LOG(INFO) << "Fill empty graph inputs with cnode whose inputs are empty.";
  auto nodes = GetOrderedCNodes(anf_graph_);
  for (auto &it : nodes) {
    if (!it->isa<CNode>()) {
      continue;
    }
    std::string name = common::AnfAlgo::GetCNodeName(it);
    if (name == prim::kPrimSwitch->name() || name == prim::kPrimSwitchLayer->name() ||
        name == prim::kPrimPartial->name()) {
      continue;
    }
    auto adpt = FindAdapter(it, training_);
    if (adpt == nullptr) {
      continue;
    }
    if (adpt->getInputMap().empty() && adpt->getAttrInputMap().empty()) {
      auto cnode_op = op_cache_.find(it.get());
      if (cnode_op != op_cache_.end()) {
        (void)inputs->emplace_back(*(cnode_op->second));
        break;
      } else {
        MS_LOG(EXCEPTION) << "Can not find the operator of node: " << it->fullname_with_scope();
      }
    }
  }
}

void DfGraphConvertor::SetupInputFormat(const FuncGraphManagerPtr &manager, const AnfNodePtr &node) {
  if (!node->isa<Parameter>()) {
    return;
  }
  auto para = node->cast<ParameterPtr>();
  std::vector<int64_t> shape;
  TypeId type;
  std::string format = kOpFormat_DEFAULT;
  if (para->has_default()) {
    auto value = para->default_param();
    MS_EXCEPTION_IF_NULL(value);
    auto tensor = value->cast<std::shared_ptr<tensor::Tensor>>();
    MS_EXCEPTION_IF_NULL(tensor);
    shape = tensor->shape_c();
    type = tensor->data_type();
    format = SelectParamOriFormat(manager, para);
  } else {
    if (auto normal_shape_ptr = dyn_cast<abstract::Shape>(para->Shape()); normal_shape_ptr != nullptr) {
      shape = normal_shape_ptr->shape();
    } else if (!dyn_cast<abstract::NoShape>(para->Shape())) {
      MS_LOG(INFO) << "Invalid shape.";
      return;
    }
    if (para->Type()) {
      type = para->Type()->type_id();
      if (type == kObjectTypeTensorType) {
        type = dyn_cast<TensorType>(para->Type())->element()->type_id();
      }
    } else {
      MS_LOG(INFO) << "Invalid shape.";
      return;
    }
  }
  std::string param_debug_info = para->DebugString();
  auto param_format = param_format_.find(param_debug_info);
  if (param_format != param_format_.end()) {
    format = param_format->second;
    MS_LOG(DEBUG) << "Parameter debug info: " << param_debug_info << ", format is " << format;
  }
  auto desc = TransformUtil::GetGeTensorDesc(shape, type, format);
  StorageFormatConvertor::SetupStorageFormat(anf_graph_, node, desc);
}

void DfGraphConvertor::GenFakeGraphInRefMode() {
  const auto &nodes = GetOrderedCNodes(anf_graph_);
  for (const auto &node : nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    SaveParamFormat(node->cast<CNodePtr>());
  }
  auto manager = Manage(anf_graph_, true);
  MS_EXCEPTION_IF_NULL(manager);
  std::vector<AnfNodeWeakPtr> ge_input_nodes = {};
  const auto &params = anf_graph_->parameters();
  for (auto &node : params) {
    MS_EXCEPTION_IF_NULL(node);
    auto abs = node->abstract();
    MS_EXCEPTION_IF_NULL(abs);
    if (HasAbstractMonad(node) || abs->isa<abstract::AbstractSequence>()) {
      continue;
    }
    SetupInputFormat(manager, node);
    (void)ge_input_nodes.emplace_back(AnfNodeWeakPtr(node));
  }
  auto input_name_list = std::make_shared<GEInputList>();
  input_name_list->ge_inputs = ge_input_nodes;
  anf_graph_->set_user_data(input_name_list);
  for (auto &anf_node : params) {
    MS_EXCEPTION_IF_NULL(anf_node);
    auto para = anf_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(para);
    auto name = para->name();
    if (std::find(init_data_names_.begin(), init_data_names_.end(), name) == init_data_names_.end()) {
      const auto &param_shape = para->Shape();
      MS_EXCEPTION_IF_NULL(param_shape);
      const auto &shape = param_shape->cast<abstract::ShapePtr>();
      if (shape != nullptr) {
        const auto &sv = shape->shape();
        if (IsDynamic(sv)) {
          dynamic_shape_inputs_ = true;
        }
        input_shapes_.push_back(sv);
      }
    }
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  // set up init sub graph
  static bool is_inited = false;
  init_graph_ = nullptr;
  bool sink_mode = ConfigManager::GetInstance().dataset_mode() == DS_SINK_MODE;
  if (training_ && sink_mode && ms_context->get_param<bool>(MS_CTX_ENABLE_LOOP_SINK) && !is_inited) {
    init_graph_ = GenExampleGraph(kInit);
    is_inited = true;
  }
}

void DfGraphConvertor::GenFakeGraph(const std::string &name) {
  MS_LOG(INFO) << "Gen fake compute graph " << name;
  df_graph_ = GenExampleGraph(name);
  MS_EXCEPTION_IF_NULL(df_graph_);
  bool sink_mode = ConfigManager::GetInstance().dataset_mode() == DS_SINK_MODE;
  if (IsNormalGraph() && sink_mode) {
    MS_EXCEPTION_IF_NULL(anf_graph_);
    anf_graph_->set_flag(kGraphFlagHasGetNext, true);
  }
  const auto &params = anf_graph_->parameters();
  bool has_weight = std::any_of(params.begin(), params.end(), [](const auto &para) {
    auto parameter = para->template cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(parameter);
    return parameter->has_default();
  });
  if (distribute_ && has_weight) {
    this->broadcast_graph_ = GenExampleGraph(kBroadcast);
  }
  if (!ref_mode_) {
    return;
  }
  GenFakeGraphInRefMode();
}

DfGraphConvertor &DfGraphConvertor::BuildGraph(const std::string &name) {
  MS_LOG(INFO) << "Start BuildGraph, graph: " << anf_graph_->ToString();

  if (error_ != SUCCESS) {
    return *this;
  }

  GetCallNodeInputs(cur_while_node_);
  // branch node set input.
  bool is_initdata_graph = false;
  std::vector<AnfNodePtr> nodes = GetOrderedCNodes(anf_graph_);
  for (auto &it : nodes) {
    if (IsBranchNode(it)) {
      auto node = it->cast<CNodePtr>();
      GetBranchNodeInput(node);
    }
    if (IsInitDataSetQueueNode(it)) {
      is_initdata_graph = true;
    }
  }
  auto manager = anf_graph_->manager();
  if (manager == nullptr) {
    auto new_manager = MakeManager({anf_graph_});
    MS_EXCEPTION_IF_NULL(new_manager);
    new_manager->AddFuncGraph(anf_graph_);
    anf_graph_->set_manager(new_manager);
  }

  if (is_initdata_graph) {
    BuildInitDataGraph(name);
    return *this;
  }
  nodes = GetOrderedCNodes(anf_graph_);
  for (auto &it : nodes) {
    SetNodeInput(it);
    SetSubgraph(it);
    UpdateOpDesc(it);
  }

  if (error_ == SUCCESS) {
    df_graph_ = make_shared<DfGraph>(name);
  } else {
    return *this;
  }

  // set graph input according to the order from anf graph
  std::vector<Operator> inputs;
  std::vector<AnfNodeWeakPtr> ge_input_nodes = {};
  if (ref_mode_ && !export_air_) {
    SetGraphInputs(&inputs, &ge_input_nodes);
  } else {
    SetGraphInputs(&inputs);
  }

  // Add const nodes as graph input for some operator work with constant
  MS_LOG(INFO) << "Graph const input size: " << graph_const_inputs_.size();
  auto fv_names = GetFvNames(anf_graph_);
  for (auto &input : graph_const_inputs_) {
    if (fv_names.find(input->GetName()) == fv_names.end()) {
      inputs.emplace_back(*input);
    }
  }

  FillEmptyInputsWithNoInputOp(&inputs);

  MS_LOG(INFO) << "Set graph input num: " << inputs.size();
  (void)df_graph_->SetInputs(inputs);

  SetGraphOutputs(true);
  (void)df_graph_->SetOutputs(graph_outputs_);

  IdentityOptimization();
  NoOpOptimization();
  if (has_es_node_) {
    ESOptimization();
  }

  compute_sout_ << "}" << endl;
  // For the graph(e.g. eval_subgraph) whose IterNum is 1, do not set NeedIteration flag.
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ConfigManager::GetInstance().iter_num() > 1 && ms_context->get_param<bool>(MS_CTX_ENABLE_LOOP_SINK)) {
    df_graph_->SetNeedIteration(true);
    anf_graph_->set_flag(kGraphNeedIteration, true);
  }
  if (ref_mode_) {
    std::sort(ref_datas_.begin(), ref_datas_.end(), [](const OperatorPtr &left, const OperatorPtr &right) -> bool {
      int64_t left_idx;
      int64_t right_idx;
      left->GetAttr(kTypeIndex, left_idx);
      right->GetAttr(kTypeIndex, right_idx);
      return left_idx < right_idx;
    });
    auto input_name_list = std::make_shared<GEInputList>();
    MS_EXCEPTION_IF_NULL(input_name_list);
    input_name_list->ge_inputs = ge_input_nodes;
    anf_graph_->set_user_data(input_name_list);
  }
  MS_LOG(INFO) << "End BuildGraph, graph: " << anf_graph_->ToString();
  return *this;
}

void DfGraphConvertor::SetGraphOutputs(bool is_main_graph) {
  if (cur_while_node_ == nullptr) {
    graph_outputs_.clear();
    std::vector<AnfNodePtr> return_nodes;
    auto ret_node = anf_graph_->get_return();
    MS_EXCEPTION_IF_NULL(ret_node);
    auto output_nodes = ret_node->inputs();
    if (has_es_node_) {
      return_nodes = GetEmbeddingApplyAdamOutput(ret_node);
    } else if (((!HasSubgraph(anf_graph_) && is_main_graph)) ||
               (output_nodes.size() > 1 && IsESNodeWithNoOutput(output_nodes[1]))) {
      // replace return node with graph output node.
      return_nodes.insert(return_nodes.end(), output_nodes.begin() + 1, output_nodes.end());
    } else {
      return_nodes.emplace_back(ret_node);
    }
    for (const auto &output_node : return_nodes) {
      MS_EXCEPTION_IF_NULL(output_node);
      auto adpt = FindAdapter(output_node, training_);
      MS_EXCEPTION_IF_NULL(adpt);
      auto op_ptr = Convert(output_node);
      std::vector<OutHandler> handles;
      if (op_ptr != nullptr) {
        handles = adpt->getOutputs(op_ptr);
      } else if (tuple_out_handle_cache_.count(output_node.get()) > 0) {
        handles = *tuple_out_handle_cache_[output_node.get()];
      } else {
        MS_LOG(EXCEPTION) << "Can not find matched handles for node " << output_node->ToString();
      }

      for (const auto &handle : handles) {
        (void)graph_outputs_.emplace_back(std::make_pair(*handle.op, handle.out));
      }
    }
  }

  MS_LOG(INFO) << "Set graph " << anf_graph_->ToString() << " output, num: " << graph_outputs_.size();
  for (size_t i = 0; i < graph_outputs_.size(); i++) {
    MS_LOG(INFO) << "Graph output " << i << ": node: " << graph_outputs_[i].first.GetName()
                 << ", out: " << graph_outputs_[i].second;
  }
}

void DfGraphConvertor::UpdateConstOpDesc(const AnfNodePtr &it, const OperatorPtr &op) const {
  if (!it->isa<Parameter>()) {
    MS_LOG(DEBUG) << "It is not parameter, name: " << it->DebugString();
    return;
  }
  auto para = it->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(para);
  std::string format = SelectParamOriFormat(graph_manager_, it);
  std::string param_debug_info = para->DebugString();
  auto param_format = param_format_.find(param_debug_info);
  if (param_format != param_format_.end()) {
    format = param_format->second;
    MS_LOG(DEBUG) << "Parameter debug info: " << param_debug_info << ", format is " << format;
  }
  if (format == kOpFormat_DEFAULT || format == kOpFormat_NCHW) {
    MS_LOG(DEBUG) << "Format is not changed, no need to update op desc, name: " << param_debug_info;
    return;
  }
  if (!para->has_default()) {
    MS_LOG(DEBUG) << "Parameter has no default, no need to update op desc, name: " << param_debug_info;
    return;
  }
  auto value = para->default_param();
  MS_EXCEPTION_IF_NULL(value);
  auto tensor = value->cast<std::shared_ptr<tensor::Tensor>>();
  MS_EXCEPTION_IF_NULL(tensor);
  auto const_op_desc = TransformUtil::GetGeTensorDesc(tensor->shape_c(), tensor->data_type(), format);
  StorageFormatConvertor::SetupStorageFormat(anf_graph_, it, const_op_desc, format);
  if (const_op_desc == nullptr) {
    MS_LOG(WARNING) << "Create parameter " << para->name() << " output descriptor failed!";
    return;
  }
  (void)op->UpdateOutputDesc(kTypeY, *const_op_desc);
}

void DfGraphConvertor::UpdateDataOpDesc(const AnfNodePtr &it, const OperatorPtr &op) const {
  auto node = std::static_pointer_cast<AnfNode>(it);
  MS_EXCEPTION_IF_NULL(node);
  if (node == nullptr) {
    MS_LOG(ERROR) << "Update data op descriptor failed! Invalid node.";
    return;
  }
  std::vector<int64_t> shape;
  if (auto normal_shape_ptr = dyn_cast<abstract::Shape>(node->Shape()); normal_shape_ptr != nullptr) {
    shape = normal_shape_ptr->shape();
  } else if (auto no_shape_ptr = dyn_cast<abstract::NoShape>(node->Shape()); no_shape_ptr != nullptr) {
    shape = {};
  } else {
    MS_LOG(INFO) << "Invalid shape to update data op descriptor.";
    return;
  }
  if (node->Type() == nullptr) {
    MS_LOG(INFO) << "Invalid type to update data op descriptor.";
    return;
  }
  TypeId me_type = node->Type()->type_id();
  if (kObjectTypeTensorType == me_type) {
    me_type = dyn_cast<TensorType>(node->Type())->element()->type_id();
  }
  std::ostringstream buf;
  buf << "[" << shape << "]";
  MS_LOG(INFO) << "input shape is " << buf.str() << ", type is " << me_type;
  std::string format = SelectParamOriFormat(graph_manager_, it);
  if (it->isa<Parameter>()) {
    auto param = it->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param);
    std::string param_name = param->DebugString();
    auto param_format = param_format_.find(param_name);
    if (param_format != param_format_.end()) {
      format = param_format->second;
      MS_LOG(DEBUG) << "parameter: " << param_name << ", format is " << format;
    }
  }
  auto desc = TransformUtil::GetGeTensorDesc(shape, me_type, format);
  StorageFormatConvertor::SetupStorageFormat(anf_graph_, it, desc, format);
  if (desc == nullptr) {
    MS_LOG(ERROR) << "Update data op descriptor failed! TensorDesc is null.";
  } else {
    (void)op->UpdateInputDesc(kTypeX, *desc);
    (void)op->UpdateOutputDesc(kTypeY, *desc);
  }
}

DfGraphPtr DfGraphConvertor::GetComputeGraph() { return df_graph_; }

DfGraphPtr DfGraphConvertor::GetInitGraph() { return init_graph_; }

DfGraphPtr DfGraphConvertor::GetSaveCheckpointGraph() { return save_ckp_graph_; }

DfGraphPtr DfGraphConvertor::GetBroadcastGraph() { return broadcast_graph_; }

const std::vector<std::string> trans_var_list = {string(kNameAssign), string(kNameAssignAdd), string(kNameAssignSub)};

AnfNodePtr DfGraphConvertor::ParseLoadInput(const CNodePtr &cnode) const {
  MS_EXCEPTION_IF_NULL(cnode);
  size_t min_inputs_size = 3;
  if (cnode->size() < min_inputs_size) {
    MS_LOG(EXCEPTION) << "input size error, " << cnode->ToString();
  }
  const size_t para_index = 1;
  return cnode->input(para_index);
}

void DfGraphConvertor::TransformConstOp(const CNodePtr &node, const AnfNodePtr &pred) {
  // transform "Const" op to "Variable" op when the next node is "Assign" op.
  std::string c_name = GetCNodeTargetFuncName(node);
  auto pos = std::find(trans_var_list.begin(), trans_var_list.end(), c_name);
  if (!training_ && !IsSubGraph() && pos != trans_var_list.end() && pred->isa<Parameter>()) {
    std::string name = std::static_pointer_cast<Parameter>(pred)->name();
    auto op_itor = op_cache_.find(pred.get());
    if (op_itor == op_cache_.end()) {
      MS_LOG(EXCEPTION) << "Can not find op for node " << pred->ToString() << ".";
    }
    if (op_itor->second != nullptr &&
        (op_itor->second->GetOpType() == "Constant" || op_itor->second->GetOpType() == "Const") &&
        vars_.find(name) != vars_.end()) {
      MS_EXCEPTION_IF_NULL(vars_[name]);
      if (ref_mode_) {
        auto variable = std::make_shared<RefData>(name);
        MS_EXCEPTION_IF_NULL(variable);
        auto desc = vars_[name]->GetOutputDesc(kTypeY);
        (void)variable->update_output_desc_y(desc);
        (void)variable->update_input_desc_x(desc);
        (void)variable->set_attr_index(ref_datas_.size());
        (void)ref_datas_.emplace_back(variable);
        MS_LOG(DEBUG) << "Trans to variable, var = " << variable->GetName() << ".";
        op_itor->second = variable;  // replace parameter with variable
        vars_[name] = variable;
      } else {
        auto variable = std::make_shared<Variable>(name);
        MS_EXCEPTION_IF_NULL(variable);
        auto desc = vars_[name]->GetOutputDesc(kTypeY);
        (void)variable->update_output_desc_y(desc);
        MS_LOG(DEBUG) << "Trans to variable, var = " << variable->GetName() << ".";
        op_itor->second = variable;  // replace parameter with variable
        vars_[name] = variable;
      }
    }
  }
}

AnfNodePtr DfGraphConvertor::GetRealInputNode(const CNodePtr &node, const AnfNodePtr &input) {
  if (input == nullptr || node == nullptr) {
    return nullptr;
  }
  AnfNodePtr pred = input;
  while (pred->isa<CNode>() && GetCNodeTargetFuncName(pred->cast<CNodePtr>()) == prim::kPrimDepend->name()) {
    pred = pred->cast<CNodePtr>()->input(1);
  }

  // skip input of UMonad, IOMonad
  if (IsValueNode<UMonad>(pred) || IsValueNode<IOMonad>(pred)) {
    return nullptr;
  }
  if (HasAbstractMonad(pred)) {
    return nullptr;
  }

  // skip input of the None, UpdateState
  if (IsValueNode<None>(pred) || IsPrimitiveCNode(pred, prim::kPrimUpdateState)) {
    return nullptr;
  }

  if (IsPrimitiveCNode(pred, prim::kPrimLoad)) {
    pred = ParseLoadInput(pred->cast<CNodePtr>());
    // for scenario like: Depend->Load->TensorMove
    if (IsPrimitiveCNode(pred, prim::kPrimDepend)) {
      return GetRealInputNode(node, pred);
    }
  }
  TransformConstOp(node, pred);
  return pred;
}

bool DfGraphConvertor::IsDataInput(const AnfNodePtr &node, const AnfNodePtr &input, size_t input_index) {
  if (node == nullptr || input == nullptr) {
    MS_LOG(ERROR) << "Node or input is null.";
    return false;
  }
  // Ignore the null ValueTupe in MakeTuple
  if (IsMakeTupleWithNullValue(node, input)) {
    return false;
  }

  // skip NoOp
  auto op = Convert(node);
  if (op != nullptr && op->GetOpType() == kTypeNoOp) {
    return false;
  }

  // skip input of UMonad, IOMonad
  if (IsMonad(input)) {
    return false;
  }

  // skip input of the None, UpdateState
  if (IsValueNode<None>(input) || IsPrimitiveCNode(input, prim::kPrimUpdateState)) {
    return false;
  }

  const PrimitiveSet has_control_node = {prim::kPrimLoad, prim::kPrimDepend, prim::kPrimTupleGetItem};
  if (input_index != kDataInputIndex && IsOneOfPrimitiveCNode(node, has_control_node)) {
    return false;
  }

  // Ge Operator of HcomReceive has no input.
  if (IsPrimitiveCNode(node, prim::kPrimReceive)) {
    return false;
  }

  // The NPUClearFloatStatusV2 of GE has no input and output, and the NPUGetFloatStatusV2 has no input.
  // The extra data edges of MindSpore need to be converted to control edges of GE.
  if (IsOverFlowNode(node, input)) {
    return false;
  }

  if (IsESNodeWithNoOutput(input)) {
    return false;
  }

  return true;
}

OutHandler DfGraphConvertor::GetNormalOpInput(const AnfNodePtr &node, const AnfNodePtr &pred) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(pred);
  OutHandler out_handler;
  if (IsSubGraph() && pred->isa<Parameter>()) {
    auto idx = std::find(inputs_.begin(), inputs_.end(), pred) - inputs_.begin();
    OperatorPtr op = subgraph_input_cache_[idx];
    out_handler.op = op;
    return out_handler;
  }

  if (IsAfterGraph() && pred->isa<Parameter>()) {
    auto idx = std::find(inputs_.begin(), inputs_.end(), pred) - inputs_.begin();
    auto idx_cond = prev_after_cond_map_[idx];
    if (bypass_node_prev_handle_cache_.find(idx_cond) != bypass_node_prev_handle_cache_.end()) {
      out_handler = bypass_node_prev_handle_cache_[idx_cond];
    } else {
      auto idx_out = prev_cond_to_while_out_index_[idx_cond];
      MS_EXCEPTION_IF_NULL(while_output_handle_cache_[prev_while_node_]);
      out_handler = while_output_handle_cache_[prev_while_node_]->at(idx_out);
    }
    return out_handler;
  }

  if (out_handle_cache_.find(pred.get()) != out_handle_cache_.end()) {
    return out_handle_cache_[pred.get()];
  }
  auto op = Convert(pred);
  if (op == nullptr) {
    MS_LOG(WARNING) << "Convert input node failed, input node: " << pred->fullname_with_scope()
                    << ", node: " << node->fullname_with_scope() << ", graph: " << anf_graph_->ToString()
                    << ". Please check whether the node is Partial node or successor node of Partial in sub-graph.";
  }
  out_handler.op = op;
  out_handler.node = pred;
  return out_handler;
}

void DfGraphConvertor::DrawOpInput(const AnfNodePtr &node, const AnfNodePtr &pred, size_t i) {
  MS_EXCEPTION_IF_NULL(pred);
  if (pred->isa<CNode>() && GetCNodeTargetFuncName(pred->cast<CNodePtr>()) == mindspore::kTupleGetItemOpName) {
    MS_EXCEPTION_IF_NULL(pred->cast<CNodePtr>());
    MS_EXCEPTION_IF_NULL(pred->cast<CNodePtr>()->input(1));
    compute_sout_ << op_draw_name_[pred->cast<CNodePtr>()->input(1).get()] << " -> " << op_draw_name_[node.get()] << ":"
                  << i << endl;
  } else if (pred->isa<Parameter>()) {
    compute_sout_ << op_draw_name_[pred.get()] << " -> " << op_draw_name_[node.get()] << ":" << i << endl;
  } else {
    // don't draw anything.
    MS_LOG(INFO) << "DRAW_GE_GRAPH: Shouldn't have this case.";
  }
  return;
}

std::vector<OutHandler> DfGraphConvertor::GetInputHandles(const AnfNodePtr &node, const AnfNodePtr &input) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(input);
  std::vector<OutHandler> handles;
  auto cache_ret = tuple_out_handle_cache_.find(input.get());
  if (cache_ret != tuple_out_handle_cache_.end()) {
    handles = *(cache_ret->second);
  } else if (IsWhileNode(input)) {
    // While node in subgraph does not convert.
    // Output handle of While node is inconsistent with MS.
    MS_LOG(WARNING) << "Input node is while node, input node: " << input->fullname_with_scope()
                    << ", node: " << node->fullname_with_scope() << ", graph: " << anf_graph_->ToString();
    std::transform(graph_outputs_.begin(), graph_outputs_.end(), std::back_inserter(handles), [](const auto output) {
      return OutHandler(std::make_shared<::ge::Operator>(output.first), output.second);
    });
  } else {
    auto pred_adpt = FindAdapter(input, training_);
    MS_EXCEPTION_IF_NULL(pred_adpt);
    // When node's output is dynamic or node has multiple output, it need to get all handles.
    // TupleGetItem's input is dynamic output(eg:MakeTuple), but it only need to get one handle.
    if ((pred_adpt->IsDyOutputOp(0) || pred_adpt->IsMultipleOutputOp(input))) {
      MS_EXCEPTION_IF_NULL(Convert(input));
      handles = pred_adpt->getOutputs(Convert(input));
    } else {
      auto handle = GetNormalOpInput(node, input);
      if (handle.op != nullptr) {
        handles.emplace_back(handle);
      }
    }
  }

  if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
    std::vector<OutHandler> return_handles;
    CNodePtr cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    size_t tuplegetitem_idx = common::AnfAlgo::GetTupleGetItemOutIndex(cnode);
    if (tuplegetitem_idx >= handles.size()) {
      MS_LOG(EXCEPTION) << "Node output index " << tuplegetitem_idx << " is out of range [0," << handles.size()
                        << "), node: " << node->fullname_with_scope()
                        << ", input node: " << input->fullname_with_scope();
    } else {
      return_handles.emplace_back(handles[tuplegetitem_idx]);
      return return_handles;
    }
  }

  return handles;
}

void DfGraphConvertor::SetDynamicInputHandleByMultiInput(const OpAdapterPtr &adpt, const CNodePtr &node,
                                                         const CNodePtr &from_node_input) {
  MS_EXCEPTION_IF_NULL(adpt);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(from_node_input);
  auto inputs = from_node_input->inputs();
  std::vector<OutHandler> handles;
  for (size_t i = 1; i < inputs.size(); i++) {
    auto input = inputs[i];
    if (!IsDataInput(from_node_input, input, i)) {
      SetNodeControlInput(node, input);
      continue;
    }
    TransformConstOp(from_node_input, input);
    auto input_handles = GetInputHandles(from_node_input, input);
    handles.insert(handles.end(), input_handles.begin(), input_handles.end());
    if (input_handles.empty()) {
      MS_LOG(INFO) << "input handles is empty, node: " << from_node_input->fullname_with_scope()
                   << ", input node: " << input->fullname_with_scope();
      continue;
    }
    AddGraphConstInput(input_handles[0].op);
    DrawOpInput(node, input, i);
  }

  auto ret = adpt->setInput(Convert(node), 1, std::make_shared<std::vector<OutHandler>>(handles));
  if (ret != SUCCESS) {
    MS_LOG(EXCEPTION) << "Set node input handle failed, node:" << node->fullname_with_scope();
  }
}

bool DfGraphConvertor::IsMergeOrSwitchLayerInput(const CNodePtr &node) const {
  auto manager = anf_graph_->manager();
  if (manager == nullptr) {
    auto new_manager = MakeManager({anf_graph_});
    MS_EXCEPTION_IF_NULL(new_manager);
    new_manager->AddFuncGraph(anf_graph_);
    anf_graph_->set_manager(new_manager);
    manager = new_manager;
  }
  auto node_users = manager->node_users()[node];

  return (node_users.size() == 1 && std::find_if(node_users.begin(), node_users.end(), [](const auto &node_user) {
                                      return IsPrimitiveCNode(node_user.first, prim::kPrimMerge) ||
                                             IsPrimitiveCNode(node_user.first, prim::kPrimSwitchLayer);
                                    }) != node_users.end());
}

void DfGraphConvertor::SetMakeTupleInput(const OpAdapterPtr &adpt, const CNodePtr &make_tuple_node) {
  MS_EXCEPTION_IF_NULL(adpt);
  MS_EXCEPTION_IF_NULL(make_tuple_node);
  MS_LOG(DEBUG) << "Set MakeTuple input handle: " << make_tuple_node->fullname_with_scope();
  // Skip MakeTuple make_tuple_node before Merge. Two branches(true/false) should not be merged before Merge, which
  // will lead to assign stream error in GE. Skip MakeTuple node before switch_layer, switch_layer's inputs will be
  // set in control flow process
  if (IsMergeOrSwitchLayerInput(make_tuple_node)) {
    MS_LOG(INFO) << "Skip make_tuple_node " << make_tuple_node->fullname_with_scope() << ", not set input handle.";
    return;
  }
  SetDynamicInputHandleByMultiInput(adpt, make_tuple_node, make_tuple_node);
}

void DfGraphConvertor::SetMergeInput(const OpAdapterPtr &adpt, const CNodePtr &merge_node) {
  MS_EXCEPTION_IF_NULL(adpt);
  MS_EXCEPTION_IF_NULL(merge_node);
  auto inputs = merge_node->inputs();
  if (inputs.size() != kMergeInputSize) {
    MS_LOG(EXCEPTION) << "Merge input size should be " << kMergeInputSize << ", but is " << inputs.size()
                      << ", node: " << merge_node->fullname_with_scope();
  }
  auto make_tuple = inputs[1];
  MS_EXCEPTION_IF_NULL(make_tuple);
  if (!IsPrimitiveCNode(make_tuple, prim::kPrimMakeTuple)) {
    MS_LOG(EXCEPTION) << "Merge input is not MakeTuple, but is " << make_tuple->fullname_with_scope()
                      << ", node: " << merge_node->fullname_with_scope();
  }
  SetDynamicInputHandleByMultiInput(adpt, merge_node, make_tuple->cast<CNodePtr>());
}

void DfGraphConvertor::SetNodeControlInput(const AnfNodePtr &node, const AnfNodePtr &input) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(input);
  if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem) && input->isa<ValueNode>()) {
    return;
  }
  if (input->isa<Parameter>() && HasAbstractMonad(input)) {
    MS_LOG(DEBUG) << "Node input is monad node, do not add control edge. node:" << node->fullname_with_scope()
                  << ", input: " << input->ToString();
    return;
  }
  auto dst = Convert(node);
  MS_EXCEPTION_IF_NULL(dst);
  auto src = Convert(input);
  if (src != nullptr) {
    dst->AddControlInput(*src);
  }
}

bool DfGraphConvertor::IsDynamicInputBeforeNormalInput(const OpAdapterPtr &adpt, int *ge_input_size,
                                                       mindspore::HashMap<int, int> *ge_input_to_ms_input) {
  MS_EXCEPTION_IF_NULL(adpt);
  const auto &input_map = adpt->getInputMap();
  const auto &dyn_input_map = adpt->getDynInputMap();

  // If adpt has no dynamic input, return false.
  if (dyn_input_map.empty()) {
    return false;
  }

  // If dynamic input is behind the normal input, return false
  int min_dynamic_idx = std::numeric_limits<int>::max();
  int max_normal_idx = -1;
  for (const auto &iter : dyn_input_map) {
    int ms_order = iter.first - kIndex1;
    int ge_order = iter.second.index;
    min_dynamic_idx = std::min(min_dynamic_idx, ge_order);
    *ge_input_size = std::max(*ge_input_size, ge_order + 1);
    (*ge_input_to_ms_input)[ge_order] = ms_order;
  }
  for (const auto &iter : input_map) {
    int ms_order = iter.first - kIndex1;
    int ge_order = iter.second.index;
    max_normal_idx = std::max(max_normal_idx, ge_order);
    *ge_input_size = std::max(*ge_input_size, ge_order + 1);
    (*ge_input_to_ms_input)[ge_order] = ms_order;
  }
  if (min_dynamic_idx == std::numeric_limits<int>::max() || max_normal_idx == -1 || min_dynamic_idx > max_normal_idx) {
    return false;
  }
  return true;
}

void DfGraphConvertor::SetDynamicInputBeforeNormalInput(const OpAdapterPtr &adpt, const CNodePtr &node,
                                                        const std::vector<AnfNodePtr> &inputs, const int &ge_input_size,
                                                        const mindspore::HashMap<int, int> &ge_input_to_ms_input,
                                                        std::vector<int64_t> *dyn_input_sizes) {
  //  If dynamic input is ahead of the normal input, use 'create_dynamic_input_by_index_name' to create dynamic input,
  //  and this func must be called before set normal input.
  OperatorPtr src = Convert(node);
  MS_EXCEPTION_IF_NULL(adpt);
  const auto &dyn_input_map = adpt->getDynInputMap();
  MS_EXCEPTION_IF_NULL(dyn_input_sizes);
  if (dyn_input_sizes->empty()) {
    *dyn_input_sizes = std::vector<int64_t>(ge_input_size, -1);
    for (const auto &iter : dyn_input_map) {
      dyn_input_sizes->at(iter.first - kIndex1) = 1;
    }
  }
  std::vector<int64_t> new_dyn_input_sizes(ge_input_size, -1);
  std::vector<int> ge_tensor_orders =
    GetGeTensorOrders(ge_input_to_ms_input, *dyn_input_sizes, ge_input_size, &new_dyn_input_sizes);

  std::vector<size_t> ms_control_inputs;
  for (size_t i = 1; i < inputs.size(); ++i) {
    if (HasAbstractMonad(inputs[i])) {
      ms_control_inputs.emplace_back(i);
    }
  }

  MS_LOG(INFO) << "Adjust the dyn input order and use create_dynamic_input_byindex_name for node: "
               << node->fullname_with_scope();
  // ge_input_idx: the real ge input order
  // ge_tensor_orders: the tensor input order
  // ge_input_to_ms_input: the relationship between ge input order and ms input order
  // new_dyn_input_sizes:  tensor size of dynamic input
  for (int ge_input_idx = 0; ge_input_idx < ge_input_size; ++ge_input_idx) {
    int ms_input_idx = ge_input_to_ms_input.at(ge_input_idx) + kIndex1;
    // ge_tensor_idx: the ge input idx of unfold mindspore inputs
    int ge_tensor_idx = ge_tensor_orders[ge_input_idx] + kIndex1;
    if (ge_tensor_idx >= static_cast<int>(inputs.size())) {
      MS_LOG(INFO) << "ge tensor index is more than ms inputs size, ge_tensor_idx:" << ge_tensor_idx
                   << ", input size: " << inputs.size();
      continue;
    }
    AnfNodePtr pred = inputs[ge_tensor_idx];
    MS_EXCEPTION_IF_NULL(pred);
    if (!IsDataInput(node, pred, ge_input_idx)) {
      SetNodeControlInput(node, pred);
      continue;
    }
    auto handles = GetInputHandles(node, pred);
    if (handles.empty()) {
      MS_LOG(INFO) << "Input handles is empty, input node: " << pred->fullname_with_scope()
                   << ", node: " << node->fullname_with_scope() << ", index: " << ms_input_idx;
      continue;
    }
    int ret;
    int64_t dyn_input_num = new_dyn_input_sizes[ge_input_idx];
    if (dyn_input_num != -1) {
      for (size_t dyn_input_idx = 1; dyn_input_idx < LongToSize(dyn_input_num); ++dyn_input_idx) {
        auto dyn_input_handle = GetInputHandles(node, inputs[ge_tensor_idx + dyn_input_idx]);
        handles.insert(handles.end(), dyn_input_handle.begin(), dyn_input_handle.end());
      }
      size_t dyn_input_begin_idx = 0;
      for (size_t i = 0; i < IntToSize(ge_input_idx); ++i) {
        dyn_input_begin_idx += new_dyn_input_sizes[i] == -1 ? 1 : LongToSize(new_dyn_input_sizes[i]);
      }
      ret = adpt->setInput(src, SizeToInt(ms_input_idx), std::make_shared<std::vector<OutHandler>>(handles), true,
                           dyn_input_begin_idx);
    } else {
      if (handles.size() != 1 && pred->isa<ValueNode>()) {
        handles.clear();
        auto handle = GetNormalOpInput(node, pred);
        handles.emplace_back(handle);
      }
      if (handles.size() != 1) {
        MS_LOG(EXCEPTION) << "Input handles size " << handles.size() << " is not equal to 1, "
                          << node->fullname_with_scope() << ", input node: " << pred->fullname_with_scope()
                          << ", index: " << ms_input_idx;
      }
      ret = adpt->setInput(src, SizeToInt(ms_input_idx), handles[0]);
    }
    if (ret != SUCCESS) {
      MS_LOG(DEBUG) << "Set node input handle failed, node:" << node->fullname_with_scope()
                    << ", input node: " << pred->fullname_with_scope() << ", index: " << ms_input_idx;
    } else {
      DrawOpInput(node, pred, ge_input_idx);
      AddGraphConstInput(handles[0].op);
    }
  }

  for (size_t ms_control_input : ms_control_inputs) {
    AnfNodePtr pred = inputs[ms_control_input];
    SetNodeControlInput(node, pred);
  }

  // Set input from attr.
  SetOpAttrToInput(adpt, node);
  return;
}

void DfGraphConvertor::AddInputAttrsForESNode(const CNodePtr &node, const AnfNodePtr &input) {
  const mindspore::HashSet<PrimitivePtr, PrimitiveHasher, PrimitiveEqual> es_need_add_attr = {
    prim::kPrimInitPartitionMap,     prim::kPrimInitEmbeddingHashmap,      prim::kPrimEmbeddingTableImport,
    prim::kPrimEmbeddingTableExport, prim::kPrimEmbeddingComputeVarImport, prim::kPrimEmbeddingComputeVarExport,
    prim::kPrimEmbeddingApplyAdam,   prim::kPrimEmbeddingApplyAdamW,       prim::kPrimEmbeddingApplyAdaGrad,
    prim::kPrimEmbeddingApplyFtrl,
  };
  if (!IsOneOfPrimitiveCNode(node, es_need_add_attr)) {
    return;
  }
  auto real = GetRealInputNode(node, input);
  MS_EXCEPTION_IF_NULL(real);
  auto op = Convert(real);
  MS_EXCEPTION_IF_NULL(real);
  if (!real->isa<ValueNode>()) {
    return;
  }
  (void)op->SetAttr(kProcessNodeEngineID, "PS");
}

void DfGraphConvertor::SetOpInput(const OpAdapterPtr &adpt, const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(adpt);
  MS_EXCEPTION_IF_NULL(node);
  OperatorPtr src = Convert(node);
  bool branch_flag = false;
  auto &inputs = node->inputs();
  size_t input_size = inputs.size();
  if (branch_input_handle_cache_.find(node.get()) != branch_input_handle_cache_.end()) {
    branch_flag = true;
    MS_EXCEPTION_IF_NULL(branch_input_handle_cache_[node.get()]);
    input_size = branch_input_handle_cache_[node.get()]->size() + 1;
  } else if (!IsSubGraph() && call_input_handle_cache_.find(node) != call_input_handle_cache_.end()) {
    auto &handles = call_input_handle_cache_[node];
    MS_EXCEPTION_IF_NULL(handles);
    MS_LOG(DEBUG) << "call node input size: " << handles->size();
    adpt->setInput(src, 1, handles);
    return;
  }

  MS_LOG(INFO) << "Set op input for node: " << node->fullname_with_scope();
  if (IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
    SetMakeTupleInput(adpt, node);
    return;
  }

  if (IsPrimitiveCNode(node, prim::kPrimMerge)) {
    SetMergeInput(adpt, node);
    return;
  }
  bool is_call = IsCallNode(node);
  std::vector<int64_t> dyn_input_sizes;
  if (common::AnfAlgo::HasNodeAttr(kAttrDynInputSizes, node)) {
    dyn_input_sizes = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(node, kAttrDynInputSizes);
  }

  int ge_input_size = 1;
  mindspore::HashMap<int, int> ge_input_to_ms_input;
  if (IsDynamicInputBeforeNormalInput(adpt, &ge_input_size, &ge_input_to_ms_input)) {
    SetDynamicInputBeforeNormalInput(adpt, node, inputs, ge_input_size, ge_input_to_ms_input, &dyn_input_sizes);
    return;
  }
  // For call node, the first input is kernel_graph, which should not be added to input args.
  size_t input_idx = is_call ? 2 : 1;
  size_t real_input_idx = 1;
  while (input_idx < input_size) {
    AnfNodePtr pred = branch_flag ? branch_input_handle_cache_[node.get()]->at(input_idx - 1) : inputs[input_idx];
    MS_EXCEPTION_IF_NULL(pred);
    if (!IsDataInput(node, pred, real_input_idx)) {
      SetNodeControlInput(node, pred);
      input_idx += 1;
      real_input_idx += 1;
      continue;
    }
    TransformConstOp(node, pred);
    auto handles = GetInputHandles(node, pred);
    if (handles.empty()) {
      MS_LOG(INFO) << "Input handles is empty, input node: " << pred->fullname_with_scope()
                   << ", node: " << node->fullname_with_scope() << ", index: " << real_input_idx;
      input_idx += 1;
      real_input_idx += 1;
      continue;
    }

    int ret;
    int64_t dyn_input_num = GetDynInputNum(adpt, is_call, dyn_input_sizes, real_input_idx, input_size, node);
    if (dyn_input_num != -1) {
      for (size_t dyn_input_idx = 1; dyn_input_idx < LongToSize(dyn_input_num); ++dyn_input_idx) {
        auto dyn_input_handle = GetInputHandles(node, inputs[input_idx + dyn_input_idx]);
        handles.insert(handles.end(), dyn_input_handle.begin(), dyn_input_handle.end());
      }
      ret = adpt->setInput(src, SizeToInt(real_input_idx), std::make_shared<std::vector<OutHandler>>(handles));
      input_idx += LongToSize(dyn_input_num);
    } else {
      if (handles.size() != 1 && pred->isa<ValueNode>()) {
        handles.clear();
        auto handle = GetNormalOpInput(node, pred);
        handles.emplace_back(handle);
      }
      if (handles.size() != 1) {
        MS_LOG(EXCEPTION) << "Input handles size " << handles.size() << " is not equal to 1, "
                          << node->fullname_with_scope() << ", input node: " << pred->fullname_with_scope()
                          << ", index: " << real_input_idx;
      }
      ret = adpt->setInput(src, SizeToInt(real_input_idx), handles[0]);
      input_idx += 1;
    }
    if (ret != SUCCESS) {
      MS_LOG(DEBUG) << "Set node input handle failed, node:" << node->fullname_with_scope()
                    << ", input node: " << pred->fullname_with_scope() << ", index: " << real_input_idx;
    } else {
      DrawOpInput(node, pred, real_input_idx);
      AddGraphConstInput(handles[0].op);
    }
    AddInputAttrsForESNode(node, pred);
    real_input_idx += 1;
  }
  // Set input from attr.
  SetOpAttrToInput(adpt, node);
}

void DfGraphConvertor::SetOpAttrToInput(const OpAdapterPtr &adpt, const CNodePtr &node) {
  OperatorPtr src = Convert(node);
  auto &inputs = node->inputs();
  size_t input_size = inputs.size();
  const auto &primitive = GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(primitive);
  const auto monad_size = std::count_if(inputs.begin() + kIndex1, inputs.end(), [](const AnfNodePtr &input) {
    return input->isa<ValueNode>() && HasAbstractMonad(input);
  });
  const auto &attr_input_map = adpt->getAttrInputMap();
  const auto &input_map = adpt->getInputMap();
  if (input_map.size() != attr_input_map.size() + input_size - monad_size - kIndex1) {
    MS_LOG(DEBUG) << "For node: " << node->DebugString()
                  << ", the size of real input:" << input_size - monad_size - kIndex1
                  << " + the size of attr_input_map: " << attr_input_map.size()
                  << " != the size of input_map:" << input_map.size()
                  << ", so do not convert input from attr any more.";
    return;
  }
  MS_EXCEPTION_IF_NULL(anf_graph_);
  for (auto &it : attr_input_map) {
    // Get attr from node.
    auto value = primitive->GetAttr(it.first);
    if (value == nullptr) {
      MS_LOG(INFO) << "Node: " << node->DebugString() << " has no attr: " << it.first;
      continue;
    }
    // Create input node for attr value.
    auto input_node = NewValueNode(value);
    input_node->set_abstract(value->ToAbstract());
    anf_graph_->manager()->AddEdge(node, input_node);
    auto new_input_op = Convert(input_node);
    // Get input desc.
    auto input_name = it.second;
    auto input_desc = std::find_if(input_map.begin(), input_map.end(),
                                   [input_name](const auto &item) { return item.second.name == input_name; });
    if (input_desc == input_map.end()) {
      MS_LOG(WARNING) << "Node: " << node->DebugString() << " has no input :" << input_name;
      continue;
    }
    MS_LOG(INFO) << "Set input from attr:" << it.first << " for node: " << node->DebugString()
                 << ", new value node:" << input_node->DebugString();
    input_desc->second.set_op(src, new_input_op);
    // Input idx may be wrong.
    DrawOpInput(node, input_node, static_cast<size_t>(input_desc->first));
    AddGraphConstInput(new_input_op);
  }
}

void DfGraphConvertor::AddGraphConstInput(const OperatorPtr &op) {
  if (op == nullptr) {
    return;
  }
  if (IsSubGraph()) {
    return;
  }

  if (op->GetOpType() == "Constant" || op->GetOpType() == "Const") {
    graph_const_inputs_.emplace_back(op);
  }
}

void DfGraphConvertor::SetNodeInput(const AnfNodePtr node) {
  if (!node->isa<CNode>()) {
    return;
  }
  if (op_cache_.find(node.get()) == op_cache_.end()) {
    return;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  OpAdapterPtr adpt = FindAdapter(cnode, training_);
  if (adpt == nullptr) {
    error_ = NOT_FOUND;
    return;
  }

  // get Operator from op_cache_, use adapter to set Inputs
  DfGraphConvertor::SetOpInput(adpt, cnode);
}

std::string DfGraphConvertor::GetGNodeName(const ::ge::GNode &node) const {
  ::ge::AscendString name;
  auto ret = node.GetName(name);
  if (ret == ::ge::GRAPH_SUCCESS) {
    return std::string(name.GetString());
  } else {
    MS_LOG(WARNING) << "Get GNode name failed, ret: " << ret;
    return std::string();
  }
}

std::string DfGraphConvertor::GetGNodeType(const ::ge::GNode &node) const {
  ::ge::AscendString node_type;
  auto ret = node.GetType(node_type);
  if (ret == ::ge::GRAPH_SUCCESS) {
    return std::string(node_type.GetString());
  } else {
    MS_LOG(WARNING) << "Get GNode type failed, ret: " << ret;
    return std::string();
  }
}

// 1) Identity or IdentityN is the input of Merge, not delete
// 2) Identity or IdentityN is the subgraph(If) input, not delete
// 3) Identity or IdentityN it the output, not delete
// 4) Identity or IdentityN has multiple users, not delete
// 5) Nodes with control edges, temporarily not delete
bool DfGraphConvertor::IsIdentityRedundant(const ::ge::GNode &node) const {
  auto node_type = GetGNodeType(node);
  if (node_type != kTypeIdentityN && node_type != kTypeIdentity) {
    MS_LOG(DEBUG) << "Node is not Identity or IdentityN, but is " << node_type << ", node name: " << GetGNodeName(node);
    return false;
  }

  auto node_name = GetGNodeName(node);
  auto ret = std::find_if(graph_outputs_.begin(), graph_outputs_.end(),
                          [&node_name](const auto &output) { return output.first.GetName() == node_name; });
  if (ret != graph_outputs_.end()) {
    return false;
  }

  for (size_t output_index = 0; output_index < node.GetOutputsSize(); output_index++) {
    auto output_nodes = node.GetOutDataNodesAndPortIndexs(static_cast<int32_t>(output_index));
    if (!output_nodes.empty() && has_es_node_) {
      return true;
    }
    if (output_nodes.size() != 1) {
      return false;
    }

    auto output_node_type = GetGNodeType(*(output_nodes.begin()->first));
    if (output_node_type == kTypeMerge || output_node_type == kTypeIf) {
      return false;
    }
  }

  if (!node.GetOutControlNodes().empty()) {
    return false;
  }

  return true;
}

void DfGraphConvertor::RemoveIdentity(::ge::GNode identity_node) {
  MS_LOG(INFO) << "Start Remove Identity or IdentityN, identity_node: " << GetGNodeName(identity_node);
  auto node_type = GetGNodeType(identity_node);
  if (node_type != kTypeIdentity && node_type != kTypeIdentityN) {
    MS_LOG(EXCEPTION) << "Node is not Identity or IdentityN, but is " << node_type
                      << ", identity_node name: " << GetGNodeName(identity_node);
    return;
  }
  if (identity_node.GetInputsSize() != identity_node.GetOutputsSize()) {
    MS_LOG(EXCEPTION) << "Node output size " << identity_node.GetOutputsSize() << " is not equal to input size "
                      << identity_node.GetInputsSize() << ", identity_node: " << GetGNodeName(identity_node);
    return;
  }

  ::ge::graphStatus ret;
  for (size_t output_index = 0; output_index < identity_node.GetOutputsSize(); output_index++) {
    auto output_nodes = identity_node.GetOutDataNodesAndPortIndexs(static_cast<int>(output_index));
    if (output_nodes.size() != 1 && !has_es_node_) {
      return;
    }

    // 1. Set identity_node data edge
    for (size_t i = 0; i < output_nodes.size(); i++) {
      auto node_output = output_nodes[i];
      auto input_index = output_index;
      auto node_input = identity_node.GetInDataNodesAndPortIndexs(static_cast<int32_t>(input_index));
      ret = df_graph_->RemoveEdge(identity_node, static_cast<int32_t>(output_index), *node_output.first,
                                  node_output.second);
      if (ret != ::ge::GRAPH_SUCCESS) {
        MS_LOG(EXCEPTION) << "Remove edge failed, src identity_node: " << GetGNodeName(identity_node)
                          << ", index: " << output_index << ", dst identity_node: " << GetGNodeName(*node_output.first)
                          << ", index: " << node_output.second << ", ret: " << ret;
        return;
      }
      ret = df_graph_->AddDataEdge(*node_input.first, node_input.second, *node_output.first, node_output.second);
      if (ret != ::ge::GRAPH_SUCCESS) {
        MS_LOG(EXCEPTION) << "Add data edge failed, src identity_node: " << GetGNodeName(*node_input.first)
                          << ", index: "
                          << ", dst identity_node: " << GetGNodeName(*node_output.first)
                          << ", index: " << node_output.second << ", ret: " << ret;
        return;
      }

      // 2. Set identity_node control edge
      auto node_control = identity_node.GetInControlNodes();
      for (const auto &item : node_control) {
        ret = df_graph_->AddControlEdge(*item, *node_output.first);
        if (ret != ::ge::GRAPH_SUCCESS) {
          MS_LOG(EXCEPTION) << "Add control edge failed, src identity_node: " << GetGNodeName(*item)
                            << ", dst identity_node: " << GetGNodeName(*node_output.first) << ", ret: " << ret;
          return;
        }
      }
    }
  }

  // 3. Remove identity
  ret = df_graph_->RemoveNode(identity_node);
  if (ret != ::ge::GRAPH_SUCCESS) {
    MS_LOG(EXCEPTION) << "Remove identity_node failed, identity_node: " << GetGNodeName(identity_node)
                      << ", ret: " << ret;
    return;
  }
}

void DfGraphConvertor::IdentityOptimization() {
  MS_LOG(INFO) << "Start IdentityOptimization, graph: " << anf_graph_->ToString();
  MS_EXCEPTION_IF_NULL(df_graph_);
  auto all_nodes = df_graph_->GetDirectNode();
  for (const auto &node : all_nodes) {
    if (IsIdentityRedundant(node)) {
      RemoveIdentity(node);
    }
  }
  MS_LOG(INFO) << "End IdentityOptimization, graph: " << anf_graph_->ToString();
}

void DfGraphConvertor::NoOpOptimization() {
  MS_LOG(INFO) << "Start NoOpOptimization, graph:" << anf_graph_->ToString();
  MS_EXCEPTION_IF_NULL(df_graph_);
  auto all_nodes = df_graph_->GetDirectNode();
  for (const auto &node : all_nodes) {
    if (IsNoOpRedundant(node)) {
      RemoveNoOp(node);
    }
  }
  MS_LOG(INFO) << "End NoopOptimization, graph:" << anf_graph_->ToString();
}

void DfGraphConvertor::ESOptimization() {
  MS_LOG(INFO) << "Start ESOptimization, graph:" << anf_graph_->ToString();
  MS_EXCEPTION_IF_NULL(df_graph_);
  auto all_nodes = df_graph_->GetDirectNode();
  ::ge::GNode no_op;
  bool not_remove = false;
  for (const auto &node : all_nodes) {
    node.GetAttr(kAttrNotRemove, not_remove);
    if (not_remove) {
      no_op = node;
      break;
    }
  }
  if (not_remove) {
    auto output_control_node = no_op.GetOutControlNodes();
    if (output_control_node.empty()) {
      return;
    }
    RemoveIdentityForES(*output_control_node[0]);
  }
}

void DfGraphConvertor::RemoveIdentityForES(::ge::GNode node) {
  ::ge::graphStatus ret;
  auto out_control_node = node.GetOutControlNodes();
  for (size_t input_index = 0; input_index < node.GetInputsSize(); input_index++) {
    auto node_input = node.GetInDataNodesAndPortIndexs(static_cast<int32_t>(input_index));
    ret = df_graph_->RemoveEdge(*node_input.first, node_input.second, node, input_index);
    if (ret != ::ge::GRAPH_SUCCESS) {
      MS_LOG(EXCEPTION) << "Remove edge failed, src node: " << GetGNodeName(*node_input.first)
                        << ", index: " << node_input.second << ", dst identity_node: " << GetGNodeName(node)
                        << ", index: " << input_index << ", ret: " << ret;
      return;
    }
  }
  ret = df_graph_->RemoveNode(node);
  if (ret != ::ge::GRAPH_SUCCESS) {
    MS_LOG(EXCEPTION) << "Remove node failed, node: " << GetGNodeName(node);
  }
  if (out_control_node.empty()) {
    return;
  }
  auto output_node = out_control_node[0];
  MS_EXCEPTION_IF_NULL(output_node);
  RemoveIdentityForES(*output_node);
}

bool DfGraphConvertor::IsNoOpRedundant(const ::ge::GNode &node) const {
  auto node_type = GetGNodeType(node);
  if (node_type != kTypeNoOp) {
    return false;
  }
  if (!training_) {
    return true;
  }

  bool not_remove = false;
  node.GetAttr(kAttrNotRemove, not_remove);
  if (not_remove) {
    return false;
  }

  auto out_control_node = node.GetOutControlNodes();
  auto in_control_node = node.GetInControlNodes();
  if (out_control_node.size() == 1 || in_control_node.size() == 1) {
    return true;
  }
  if (out_control_node.size() > kNoOpOptThreshold || in_control_node.size() > kNoOpOptThreshold) {
    return false;
  }
  return true;
}
void DfGraphConvertor::RemoveNoOp(::ge::GNode noop) {
  MS_LOG(INFO) << "Start Remove NoOp, node:" << GetGNodeName(noop);
  auto node_type = GetGNodeType(noop);
  if (node_type != kTypeNoOp) {
    MS_LOG(EXCEPTION) << "Node is not NoOp, but is: " << GetGNodeName(noop);
  }

  auto in_control_nodes = noop.GetInControlNodes();
  auto out_control_nodes = noop.GetOutControlNodes();
  auto ret = df_graph_->RemoveNode(noop);
  if (ret != ::ge::GRAPH_SUCCESS) {
    MS_LOG(EXCEPTION) << "Remove node failed, node: " << GetGNodeName(noop);
  }
  for (auto src_node : in_control_nodes) {
    for (auto dst_node : out_control_nodes) {
      ret = df_graph_->AddControlEdge(*src_node, *dst_node);
      if (ret != ::ge::GRAPH_SUCCESS) {
        MS_LOG(EXCEPTION) << "Add control edge failed, src node: " << GetGNodeName(*src_node)
                          << ", dst node:" << GetGNodeName(*dst_node);
      }
    }
  }
  MS_LOG(INFO) << "End Remove Noop, node: " << GetGNodeName(noop);
}

void DfGraphConvertor::ProcessSubgraph(const AnfNodePtr &node, const AnfNodePtr &branch_node,
                                       ParamIndexMap &branch_to_parent_node_map) {
  MS_LOG(INFO) << "ProcessSubgraph begin.";
  ValueNodePtr graph_node = nullptr;
  if (branch_node->isa<CNode>()) {
    graph_node = branch_node->cast<CNodePtr>()->input(1)->cast<ValueNodePtr>();
  } else if (branch_node->isa<ValueNode>()) {
    graph_node = branch_node->cast<ValueNodePtr>();
  } else {
    return;
  }

  MS_EXCEPTION_IF_NULL(graph_node);
  auto anf_graph = graph_node->value()->cast<AnfGraphPtr>();
  MS_EXCEPTION_IF_NULL(anf_graph);
  DfGraphConvertor converter(anf_graph, phase_prefix_);
  converter.graph_type_ = GraphType::kBranch;

  auto &params = anf_graph->parameters();
  if (ref_mode_) {
    for (size_t i = 0; i < params.size(); i++) {
      auto &param = params[i];
      if (branch_to_parent_node_map.find(i) != branch_to_parent_node_map.end()) {
        size_t parent_index = branch_to_parent_node_map[i];
        OperatorPtr op = nullptr;
        op = std::make_shared<Data>();
        MS_EXCEPTION_IF_NULL(op);
        SetXDataIndex(op, parent_index);
        converter.op_cache_[param.get()] = op;
      } else if (!HasAbstractMonad(param)) {
        MS_LOG(EXCEPTION) << "Branch graph input index to parent node dyn input index error, "
                          << "branch graph: " << anf_graph->ToString() << "'s " << i << "(st/nd/rd/st)"
                          << " input can not find the corresponding parent node input index.";
      }
    }
  } else {
    auto &dyn_input = branch_input_handle_cache_[node.get()];
    MS_EXCEPTION_IF_NULL(dyn_input);
    auto &inputs = tuple_out_handle_cache_[dyn_input->at(1).get()];
    MS_EXCEPTION_IF_NULL(inputs);
    for (size_t i = 0; i < params.size(); i++) {
      auto &param = params[i];
      if (branch_to_parent_node_map.find(i) != branch_to_parent_node_map.end()) {
        size_t parent_index = branch_to_parent_node_map[i];
        auto &parent_handle = inputs->at(parent_index);
        OperatorPtr op = nullptr;
        MS_EXCEPTION_IF_NULL(parent_handle.op);
        if (parent_handle.op->GetOpType() == kTypeVariable) {
          auto name = parent_handle.op->GetName();
          op = std::make_shared<Variable>(name);
          MS_EXCEPTION_IF_NULL(op);
          SetXDataIndex(op, parent_index);
        } else {
          op = std::make_shared<Data>();
          MS_EXCEPTION_IF_NULL(op);
          SetXDataIndex(op, parent_index);
        }
        converter.op_cache_[param.get()] = op;
      } else if (!HasAbstractMonad(param)) {
        MS_LOG(EXCEPTION) << "Branch graph input index to parent node dyn input index error, "
                          << "branch graph: " << anf_graph->ToString() << "'s " << i << "(st/nd/rd/st)"
                          << " input can not find the corresponding parent node input index.";
      }
    }
  }

  std::string graph_name = anf_graph->ToString();
  auto iter = branches_repeat_times.find(graph_name);
  if (iter == branches_repeat_times.end()) {
    branches_repeat_times[graph_name] = 1;
  } else {
    iter->second += 1;
    graph_name = graph_name + "_" + std::to_string(iter->second);
  }
  (void)converter.ConvertAllNode().BuildGraph(graph_name);
#ifdef ENABLE_DUMP_IR
  std::string name = graph_node->ToString() + "_ge_graph.dot";
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kFully)) {
    converter.DrawComputeGraph(name);
  }
#endif
  branches_map_[branch_node.get()] = *(converter.df_graph_);
  MS_LOG(INFO) << "ProcessSubgraph end.";
}

// Update GE op's shape and type info
void DfGraphConvertor::UpdateOpDesc(const AnfNodePtr node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node == nullptr || !node->isa<CNode>()) {
    return;
  }

  if (op_cache_.find(node.get()) == op_cache_.end()) {
    return;
  }

  OpAdapterPtr adpt = FindAdapter(node, training_);
  if (adpt == nullptr) {
    error_ = NOT_FOUND;
    return;
  }

  // get Operator from op_cache_
  OperatorPtr op = Convert(node);
  MS_EXCEPTION_IF_NULL(op);
  std::string op_type = op->GetOpType();
  if (!IsNeedToUpdateTensorDesc(op_type, node)) {
    MS_LOG(INFO) << "No need to set the opDesc of node: " << node->fullname_with_scope() << ", op type is " << op_type;
    return;
  }

  adpt->updateOutputDesc(op, node->Shape(), node->Type(), node);
}

OperatorPtr DfGraphConvertor::Convert(const AnfNodePtr node) {
  if (node == nullptr) {
    MS_LOG(ERROR) << "node is nullptr";
    error_ = NOT_FOUND;
    return nullptr;
  }
  // find in cache
  if (op_cache_.count(node.get()) != 0) {
    MS_LOG(DEBUG) << "Get op from cache: " << op_cache_[node.get()]->GetName();
    return op_cache_[node.get()];
  }

  // do not convert primitive node
  if (IsValueNode<Primitive>(node)) {
    return nullptr;
  }
  // convert a new one
  if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    if (IsSubGraph() && IsWhileNode(cnode)) {
      return nullptr;
    }
    if (!IsSubGraph() && IsWhileNode(cnode)) {
      CacheWhileGraph(cnode);
      auto &graphs = while_graph_cache_[cnode];
      GetWhileUsedInputIndex(graphs);
      SetParamIndexMap(graphs);
      cur_while_node_ = cnode;
    }
    return ConvertCNode(cnode);
  }

  if (node->isa<Parameter>() && IsSubGraph()) {
    return nullptr;
  }

  if (node->isa<Parameter>()) {
    return ConvertParameter(node);
  }
  if (node->isa<ValueNode>()) {
    if (IsValueNode<Monad>(node)) {
      return nullptr;
    }
    return ConvertValueNode(node->cast<ValueNodePtr>());
  }

  MS_LOG(ERROR) << "Invalid AnfNode";
  error_ = INVALID_ARGUMENT;
  return nullptr;
}

void DfGraphConvertor::ConvertTopK(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto value_ptr = node->input(kIndex2)->cast<ValueNodePtr>();
  if (value_ptr == nullptr) {
    // input is not const valuenode, cannot convert to int32, throw exception when input k is int64 since cann
    // has precision problem, can be deleted after cann support int64 for input k
    if (common::AnfAlgo::GetPrevNodeOutputInferDataType(node, kIndex1) == kNumberTypeInt64) {
      MS_LOG(EXCEPTION) << "Op TopK(" << node->fullname_with_scope() << ")'s second input k is an int64 mutable "
                        << "tensor/scalar, which is not supported in ascend, please use int32.";
    }
    return;
  }
  MS_LOG(INFO) << "Convert TopK second input's type from int64 to int32.";
  auto input_value = value_ptr->value();
  MS_EXCEPTION_IF_NULL(input_value);
  std::ostringstream ss;
  ss << "op" << value_ptr.get();
  op_draw_name_[value_ptr.get()] = ss.str();
  compute_sout_ << ss.str() << "[label= \"" << value_ptr->value()->ToString() << "\" shape=ellipse]" << endl;
  int32_t k_value;
  if (input_value->isa<tensor::Tensor>()) {
    auto input_tensor = input_value->cast<tensor::TensorPtr>();
    if (input_tensor->data_type() == kNumberTypeInt32) {
      k_value = *static_cast<int32_t *>(input_tensor->data_c());
    } else {
      k_value = LongToInt(*static_cast<int64_t *>(input_tensor->data_c()));
    }
  } else {
    k_value = LongToInt(GetValue<int64_t>(input_value));
  }
  OpAdapterPtr adpt = FindAdapter(value_ptr, training_);
  MS_EXCEPTION_IF_NULL(adpt);
  auto op = adpt->generate(value_ptr);
  (void)adpt->setAttr(op, "value", k_value);
  op_cache_[value_ptr.get()] = op;
}

AnfNodePtr DfGraphConvertor::CreateCast(const AnfNodePtr &input, const TypePtr &dst_type) const {
  auto func_graph = input->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  AnfNodePtrList inputs = {NewValueNode(prim::kPrimCast), input,
                           NewValueNode(static_cast<int64_t>(dst_type->type_id()))};
  auto cnode = func_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(cnode);
  auto abs_tensor = std::make_shared<abstract::AbstractTensor>(dst_type, input->Shape());
  cnode->set_abstract(abs_tensor);
  return cnode;
}

std::vector<int64_t> DfGraphConvertor::CastToInt(const ValuePtr &value) const {
  if (value == nullptr) {
    return {};
  }
  std::vector<int64_t> cur_value = {};
  if (utils::isa<ValueSequencePtr>(value)) {
    auto val_seq_ptr = value->cast<ValueSequencePtr>();
    MS_EXCEPTION_IF_NULL(val_seq_ptr);
    if (!val_seq_ptr->value().empty()) {
      auto first_val = val_seq_ptr->value().front();
      MS_EXCEPTION_IF_NULL(first_val);
      MS_EXCEPTION_IF_NULL(first_val->type());
      if (first_val->type()->number_type() == kNumberTypeInt64) {
        cur_value = GetValue<std::vector<int64_t>>(value);
      } else {
        auto origin_value = GetValue<std::vector<int>>(value);
        (void)std::transform(origin_value.begin(), origin_value.end(), std::back_inserter(cur_value),
                             [](int index) { return static_cast<int64_t>(index); });
      }
    }
  } else {
    MS_EXCEPTION_IF_NULL(value->type());
    if (value->type()->number_type() == kNumberTypeInt64) {
      cur_value.emplace_back(GetValue<int64_t>(value));
    } else {
      cur_value.emplace_back(static_cast<int64_t>(GetValue<int>(value)));
    }
  }
  return cur_value;
}

void DfGraphConvertor::TransInputDataType(const CNodePtr &node, const std::string &node_name) const {
  auto iter = kTransInputDTypeMap.find(node_name);
  if (iter == kTransInputDTypeMap.end()) {
    return;
  }
  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(DEBUG) << "Trans input data type of node:" << node->DebugString();
  for (auto &item : iter->second) {
    auto input_node = node->input(item.first);
    TypeId dst_type = item.second;
    MS_EXCEPTION_IF_NULL(input_node);
    if (input_node->isa<CNode>() || input_node->isa<Parameter>()) {
      auto src_type = input_node->Type()->type_id();
      if (kObjectTypeTensorType == src_type) {
        src_type = dyn_cast<TensorType>(input_node->Type())->element()->type_id();
      }
      if (!IsValidConversion(src_type, dst_type)) {
        continue;
      }
      auto new_cast = CreateCast(input_node, TypeIdToType(dst_type));
      node->set_input(item.first, new_cast);
    } else if (input_node->isa<ValueNode>()) {
      auto input_value_node = input_node->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(input_value_node);
      auto value = input_value_node->value();
      ValuePtr new_value = CastDstValue(value, dst_type);
      if (new_value == nullptr) {
        continue;
      }
      auto new_value_node = std::make_shared<ValueNode>(new_value);
      MS_EXCEPTION_IF_NULL(new_value_node);
      new_value_node->set_abstract(new_value->ToAbstract());
      node->set_input(item.first, new_value_node);
    }
  }
  MS_LOG(DEBUG) << "Finish to trans input data type of node:" << node->DebugString();
}

void DfGraphConvertor::TransAttrDataType(const CNodePtr &node, const std::string &node_name) const {
  auto iter = kTransAttrDTypeMap.find(node_name);
  if (iter == kTransAttrDTypeMap.end()) {
    return;
  }
  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(DEBUG) << "Trans attr data type of node:" << node->DebugString();
  auto prim = common::AnfAlgo::GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(prim);
  for (auto &item : iter->second) {
    std::string attr_name = item.first;
    TypeId dst_type = item.second;
    if (!prim->HasAttr(attr_name)) {
      MS_LOG(EXCEPTION) << "Please check kTransAttrDTypeMap, node:" << node->DebugString()
                        << " has no attr:" << attr_name;
    }
    auto attr_value = prim->GetAttr(attr_name);
    auto new_attr_value = CastDstValue(attr_value, dst_type);
    if (new_attr_value == nullptr) {
      continue;
    }
    prim->set_attr(attr_name, new_attr_value);
  }
  MS_LOG(DEBUG) << "Finish to trans attr data type of node:" << node->DebugString();
}

void DfGraphConvertor::TransDataType(const FuncGraphPtr &anf_graph) const {
  MS_EXCEPTION_IF_NULL(anf_graph);
  MS_LOG(DEBUG) << "TransDataType begin. graph:" << anf_graph->ToString();
  std::vector<AnfNodePtr> nodes = GetOrderedCNodes(anf_graph);
  for (auto &it : nodes) {
    if (it->isa<CNode>()) {
      auto node = it->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(node);
      std::string name = GetCNodeTargetFuncName(node);
      TransInputDataType(node, name);
      TransAttrDataType(node, name);
    }
  }
  MS_LOG(DEBUG) << "TransDataType end. graph:" << anf_graph->ToString();
}

void DfGraphConvertor::ConvertReshape(const CNodePtr &node) {
  MS_LOG(INFO) << "Convert the second input of reshape to op attr.";
  const auto kInputNum = 3;
  if (node->size() < kInputNum) {
    MS_LOG(WARNING) << "Reshape must have two inputs.";
    return;
  }
  OpAdapterPtr adpt = FindAdapter(node, training_);
  if (adpt == nullptr) {
    return;
  }
  auto op = adpt->generate(node);
  MS_EXCEPTION_IF_NULL(op);
  // get shape form attr
  auto primitive = GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(primitive);
  if (primitive->HasAttr("shape")) {
    auto value = primitive->GetAttr("shape");
    auto list = CastToInt(value);
    (void)op->SetAttr("shape", list);
  }
  if (primitive->HasAttr("allowzero")) {
    auto value = primitive->GetAttr("allowzero");
    auto list = CastToInt(value);
    if (list.size() == 1) {
      (void)op->SetAttr("allowzero", list[0]);
    }
  }
  op_cache_[node.get()] = op;
}

void DfGraphConvertor::ConvertDynamicStitch(const CNodePtr &node) {
  MS_LOG(INFO) << "Convert and set 'N' attr of DynamicStitch.";
  OpAdapterPtr adpt = FindAdapter(node, training_);
  if (adpt == nullptr) {
    return;
  }
  auto op = adpt->generate(node);
  MS_EXCEPTION_IF_NULL(op);
  int64_t input_length = 0;
  auto indices = node->input(1);
  MS_EXCEPTION_IF_NULL(indices);
  if (indices->isa<CNode>()) {
    input_length = SizeToLong(indices->cast<CNodePtr>()->size()) - 1;
  } else if (IsValueNode<ValueSequence>(indices)) {
    const auto tuple = GetValueNode<ValueSequencePtr>(indices);
    MS_EXCEPTION_IF_NULL(tuple);
    input_length = SizeToLong(tuple->size());
  } else {
    MS_LOG(EXCEPTION) << "Input 1 of DynamicStitch is neither CNode nor ValueNode contains ValueSequence, but "
                      << indices->ToString() << ", can not set 'N' attr.";
  }

  (void)op->SetAttr("N", input_length);
  MS_LOG(INFO) << "Set 'N' attr of DynamicStitch to " << input_length;
  op_cache_[node.get()] = op;
}

void DfGraphConvertor::ConvertParallelGroupToHcom(const CNodePtr &node) {
  auto group_name = common::AnfAlgo::GetNodeAttr<std::string>(node, kParallelGroup);
  OpAdapterPtr adpt = FindAdapter(node, training_);
  if (adpt == nullptr) {
    return;
  }

  // get operator
  OperatorPtr op = nullptr;
  auto it_op = op_cache_.find(node.get());
  if (it_op != op_cache_.end()) {
    op = it_op->second;
  } else {
    op = adpt->generate(node);
  }
  MS_EXCEPTION_IF_NULL(op);
  (void)op->SetAttr(kParallelGroup, group_name);
  op_cache_[node.get()] = op;
}

void DfGraphConvertor::ConvertParallelGroupIdToHcom(const CNodePtr &node) {
  auto parallel_group_id_value = node->GetAttr(kParallelGroupId);
  auto parallel_group_id = GetValue<uint32_t>(parallel_group_id_value);
  OpAdapterPtr adpt = FindAdapter(node, training_);
  if (adpt == nullptr) {
    return;
  }

  // get operator
  OperatorPtr op = nullptr;
  auto it_op = op_cache_.find(node.get());
  if (it_op != op_cache_.end()) {
    op = it_op->second;
  } else {
    op = adpt->generate(node);
    op_cache_[node.get()] = op;
  }
  MS_EXCEPTION_IF_NULL(op);
  (void)op->SetAttr(kParallelGroupId, parallel_group_id);
  MS_LOG(INFO) << "Successfully convert _parallel_group_id: " << parallel_group_id << " to ge op: " << op->GetName();
}

void DfGraphConvertor::ConvertHcomFusionId(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(INFO) << "Add Hcom fusion_id";
  OpAdapterPtr adpt = FindAdapter(node, training_);
  if (adpt == nullptr) {
    return;
  }
  auto op = adpt->generate(node);
  MS_EXCEPTION_IF_NULL(op);
  // get shape form attr
  auto primitive = GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(primitive);
  auto fusion_value = primitive->GetAttr("fusion");
  if (fusion_value == nullptr) {
    MS_LOG(WARNING) << "Failed to get attr fusion for gather node " << node->fullname_with_scope();
    return;
  }
  int64_t fusion = 0;
  if (fusion_value->isa<Int64Imm>()) {
    fusion = GetValue<int64_t>(fusion_value);
  } else if (fusion_value->isa<Int32Imm>()) {
    fusion = GetValue<int32_t>(fusion_value);
  } else {
    MS_LOG(WARNING) << "Attr fusion is not int64/int32 type, real type " << fusion_value->type_name()
                    << ", gather node " << node->fullname_with_scope();
    return;
  }
  int64_t fusion_id = -1;

  // fusion 0: no fusion; 1(default): fusion; 2: fusion the ops by fusion id.
  if (fusion >= 1) {
    fusion_id = fusion;
    fusion = kHcclFusionByFusionID;
  } else if (fusion < 0) {
    fusion = kHcclFusionDefault;
  }

  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CellReuseLevel() != CellReuseLevel::kNoCellReuse) {
    MS_LOG(INFO) << "cell reuse not support all fusion";
    fusion = 0;
  }
  MS_EXCEPTION_IF_NULL(parallel::ParallelContext::GetInstance());
  auto parallel_mode = parallel::ParallelContext::GetInstance()->parallel_mode();
  if (!MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_TASK_OPT) &&
      (parallel_mode == parallel::kSemiAutoParallel || parallel_mode == parallel::kAutoParallel)) {
    fusion_id = 0;
    fusion = 0;
  }
  (void)op->SetAttr("fusion_id", fusion_id);
  (void)op->SetAttr("fusion", fusion);
  AddCommAttrForHcclNode(node, op);
  op_cache_[node.get()] = op;
}

void DfGraphConvertor::ConvertAllToAllv(const CNodePtr &node) {
  OpAdapterPtr adpt = FindAdapter(node, training_);
  if (adpt == nullptr) {
    return;
  }
  auto op = adpt->generate(node);
  MS_EXCEPTION_IF_NULL(op);
  op_cache_[node.get()] = op;
  AddCommAttrForHcclNode(node, op);
  // set _is_inserted_by_ge attr to avoid mistaken delete
  auto primitive = GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(primitive);
  auto is_inserted_value = primitive->GetAttr("is_inserted_by_ge");
  if (is_inserted_value == nullptr) {
    return;
  }
  auto is_inserted = GetValue<bool>(is_inserted_value);
  (void)op->SetAttr("_is_inserted_by_ge", is_inserted);
}

void DfGraphConvertor::ConvertUniformReal(const CNodePtr &node) {
  OpAdapterPtr adpt = FindAdapter(node, training_);
  if (adpt == nullptr) {
    return;
  }
  auto op = adpt->generate(node);
  MS_EXCEPTION_IF_NULL(op);
  op_cache_[node.get()] = op;
  (void)op->SetAttr("dtype", ::ge::DataType::DT_FLOAT);
}

void DfGraphConvertor::ConvertUpdateState(const CNodePtr &node) {
  OpAdapterPtr adpt = FindAdapter(node, training_);
  if (adpt == nullptr) {
    return;
  }
  auto op = adpt->generate(node);
  MS_EXCEPTION_IF_NULL(op);
  op_cache_[node.get()] = op;
  if (common::AnfAlgo::HasNodeAttr(kAttrNotRemove, node)) {
    bool not_remove = common::AnfAlgo::GetNodeAttr<bool>(node, kAttrNotRemove);
    (void)op->SetAttr(kProcessNodeEngineID, "PS");
    (void)op->SetAttr(kAttrNotRemove, not_remove);
    has_es_node_ = true;
  }
}

void DfGraphConvertor::ConvertHcclNode(const CNodePtr &node) {
  OpAdapterPtr adpt = FindAdapter(node, training_);
  if (adpt == nullptr) {
    return;
  }
  auto op = adpt->generate(node);
  MS_EXCEPTION_IF_NULL(op);
  AddCommAttrForHcclNode(node, op);
  op_cache_[node.get()] = op;
}

void DfGraphConvertor::AddCommAttrForHcclNode(const CNodePtr &node, const OperatorPtr &converted_op) const {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(converted_op);
  if (!common::AnfAlgo::HasNodeAttr(kAttrGroup, node)) {
    MS_LOG(WARNING) << "Node " << node->fullname_with_scope() << " does not have attr " << kAttrGroup << " skip.";
    return;
  }
  std::string group = common::AnfAlgo::GetNodeAttr<std::string>(node, kAttrGroup);
  (void)converted_op->SetAttr("group", group);
#ifdef ENABLE_D
  if (!common::GetEnv(kSimulationLevel).empty()) {
    auto hccl_inner_comm_name = device::DummyAscendCollectiveCommLib::GetInstance().HcclInnerCommName(group);
    MS_LOG(INFO) << "Set comm handle and comm group name of the hccl node: " << node->fullname_with_scope()
                 << "comm name:" << hccl_inner_comm_name;
    (void)converted_op->SetAttr("group", hccl_inner_comm_name);
    return;
  }
  if (common::GetEnv(kSimulationLevel).empty() && !common::IsNeedProfileMemory()) {
    if (common::UseHostCollective() && !hccl::HcclAdapter::GetInstance().UseHcclCM()) {
      // For HcclCommInitRootInfo manner, set 'group' and 'comm' attrs. 'group' attr value should be hccl's inner comm
      // name.
      auto comm = device::ascend::AscendCollectiveCommLib::GetInstance().HcclCommunicator(group);
      auto hccl_inner_comm_name = device::ascend::AscendCollectiveCommLib::GetInstance().HcclInnerCommName(group);
      MS_LOG(INFO) << "Set comm handle and comm group name of the hccl node: " << node->fullname_with_scope()
                   << ". Comm handle: " << comm << ", comm name:" << hccl_inner_comm_name;
      MS_EXCEPTION_IF_NULL(comm);
      (void)converted_op->SetAttr("comm", reinterpret_cast<int64_t>(comm));
      (void)converted_op->SetAttr("group", hccl_inner_comm_name);
    } else {
      // For rank_table manner, 'group' attr should be user set group name.
      MS_LOG(INFO) << "Set group name for ranktable manner: " << group;
      (void)converted_op->SetAttr("group", group);
    }
  }
#endif
}

void DfGraphConvertor::ConvertConv2D(const CNodePtr &node) {
  MS_LOG(INFO) << "Convert and set 'padding' attr for Conv2D-like op.";
  MS_EXCEPTION_IF_NULL(node);
  OpAdapterPtr adpt = FindAdapter(node, training_);
  if (adpt == nullptr) {
    return;
  }
  auto op = adpt->generate(node);
  MS_EXCEPTION_IF_NULL(op);
  op_cache_[node.get()] = op;
  auto primitive = GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(primitive);
  std::string pad_mode;
  if (auto pad_value = primitive->GetAttr("padding"); pad_value != nullptr) {
    pad_mode = GetValue<std::string>(pad_value);
  } else if (auto value = primitive->GetAttr("pad_mode"); value != nullptr) {
    // Get 'pad_mode' attr and set it to 'padding' attr for ge
    const mindspore::HashMap<int64_t, std::string> pad_mode_map{{1, "SAME"}, {2, "VALID"}};
    if (value->isa<StringImm>()) {
      pad_mode = GetValue<std::string>(value);
      (void)std::transform(pad_mode.cbegin(), pad_mode.cend(), pad_mode.begin(), toupper);
      if (pad_mode != "SAME" && pad_mode != "VALID") {
        return;
      }
    } else if (auto it = pad_mode_map.find(GetValue<int64_t>(value)); it != pad_mode_map.cend()) {
      // 'pad_mode' attr could be an enumeration
      pad_mode = it->second;
    } else {
      return;
    }
  } else {
    MS_LOG(INFO) << "Node: " << node->fullname_with_scope() << " has no 'padding' or 'pad_mode' attr";
    return;
  }
  MS_LOG(INFO) << "Set 'padding' attr of node: " << node->fullname_with_scope() << " to " << pad_mode;
  (void)op->SetAttr("padding", pad_mode);
}

void DfGraphConvertor::ConvertOCRRecPreHandle(const CNodePtr &node) {
  MS_LOG(INFO) << "Add OCRRecognitionPreHandle _op_max_shape attr";
  OpAdapterPtr adpt = FindAdapter(node, training_);
  if (adpt == nullptr) {
    return;
  }
  auto op = adpt->generate(node);
  MS_EXCEPTION_IF_NULL(op);
  // get shape form attr
  auto primitive = GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(primitive);
  auto value = primitive->GetAttr("_op_max_shape");
  if (value == nullptr) {
    return;
  }
  auto op_max_shape = GetValue<std::string>(value);
  (void)op->SetAttr("_op_max_shape", op_max_shape);
  op_cache_[node.get()] = op;
}

OutHandler DfGraphConvertor::GetHandler(const AnfNodePtr &node) {
  if (node == nullptr) {
    MS_LOG(ERROR) << "Get nullptr while getting handler from node";
    return OutHandler(nullptr, "");
  }
  if (out_handle_cache_.find(node.get()) != out_handle_cache_.end()) {
    return out_handle_cache_[node.get()];
  }
  auto op = Convert(node);
  if (op != nullptr) {
    auto name = op->GetName();
    if ((vars_.count(name) != 0) && vars_[name] != nullptr) {
      op = vars_[name];
      MS_LOG(DEBUG) << "update tuple_out_handle_cache_ " << name;
    }
    return OutHandler(op, "", node);
  } else {
    MS_LOG(DEBUG) << "Add an empty out handler: " << node->ToString();
    return OutHandler();
  }
}

bool DfGraphConvertor::CheckCNode(const std::string &name, const CNodePtr node) {
  // ignore apply node of return
  if (name == "" || name == prim::kPrimSwitch->name() || name == prim::kPrimSwitchLayer->name() ||
      name == prim::kPrimPartial->name()) {
    return false;
  }

  const mindspore::HashMap<std::string, std::function<void(decltype(this), const CNodePtr &)>>
    auxiliary_node_converters{
      // Convert TopK second input from int64 to int32.
      {prim::kPrimTopK->name(), &DfGraphConvertor::ConvertTopK},
      // Convert Reshape add const input to attr(shape)
      {prim::kPrimReshape->name(), &DfGraphConvertor::ConvertReshape},
      {prim::kPrimOCRRecognitionPreHandle->name(), &DfGraphConvertor::ConvertOCRRecPreHandle},
      // Add attr 'pad_mode' to Conv2D-like op
      {prim::kPrimConv2D->name(), &DfGraphConvertor::ConvertConv2D},
      {prim::kPrimDepthwiseConv2dNative->name(), &DfGraphConvertor::ConvertConv2D},
      {kNameConv2DBackpropInputV2, &DfGraphConvertor::ConvertConv2D},
      {prim::kPrimConv2DBackpropInput->name(), &DfGraphConvertor::ConvertConv2D},
      {prim::kPrimConv2DBackpropFilter->name(), &DfGraphConvertor::ConvertConv2D},
      // Add attr 'N' to DynamicStitch
      {prim::kPrimDynamicStitch->name(), &DfGraphConvertor::ConvertDynamicStitch},
      // Convert hccl op for comm handle
      {prim::kPrimAllReduce->name(), &DfGraphConvertor::ConvertHcomFusionId},
      {prim::kPrimAllGather->name(), &DfGraphConvertor::ConvertHcomFusionId},
      {prim::kPrimReduceScatter->name(), &DfGraphConvertor::ConvertHcomFusionId},
      {prim::kPrimBroadcast->name(), &DfGraphConvertor::ConvertHcclNode},
      {prim::kPrimReduceScatter->name(), &DfGraphConvertor::ConvertHcclNode},
      {prim::kPrimSend->name(), &DfGraphConvertor::ConvertHcclNode},
      {prim::kPrimReceive->name(), &DfGraphConvertor::ConvertHcclNode},
      {prim::kPrimAllToAllv->name(), &DfGraphConvertor::ConvertAllToAllv},
      {prim::kPrimUniformReal->name(), &DfGraphConvertor::ConvertUniformReal},
      {prim::kPrimMatmulReduceScatter->name(), &DfGraphConvertor::ConvertHcclNode},
      {prim::kPrimAllGatherMatmul->name(), &DfGraphConvertor::ConvertHcclNode},
      {prim::kPrimUpdateState->name(), &DfGraphConvertor::ConvertUpdateState},
    };

  if (const auto it = auxiliary_node_converters.find(name); it != auxiliary_node_converters.cend()) {
    it->second(this, node);
  }
  if (common::AnfAlgo::HasNodeAttr(kParallelGroup, node)) {
    ConvertParallelGroupToHcom(node);
  }
  if (node->HasAttr(kParallelGroupId)) {
    ConvertParallelGroupIdToHcom(node);
  }

  return true;
}

void CheckAndAddScopeAttrInt(const OperatorPtr op, const PrimitivePtr primitive, const std::string &attr_name) {
  auto attr_value = primitive->GetAttr(attr_name);
  if (attr_value != nullptr) {
    auto value = GetValue<int64_t>(attr_value);
    (void)op->SetAttr(attr_name, value);
  }
}

void CheckAndAddScopeAttrString(const OperatorPtr op, const PrimitivePtr primitive, const std::string &attr_name) {
  auto attr_value = primitive->GetAttr(attr_name);
  if (attr_value != nullptr) {
    auto value = GetValue<std::string>(attr_value);
    (void)op->SetAttr(attr_name, value);
  }
}

// If node does not have abstract, it will fail when the node is generated to operator.
void DfGraphConvertor::SetNodeAbstract(const CNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(node);
  if (node->abstract() != nullptr) {
    return;
  }
  if (IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
    auto inputs = node->inputs();
    AbstractBasePtrList elem;
    std::transform(inputs.begin() + 1, inputs.end(), std::back_inserter(elem),
                   [](const AnfNodePtr &node) { return node->abstract(); });
    node->set_abstract(std::make_shared<abstract::AbstractTuple>(elem));
    return;
  }
  if (IsPrimitiveCNode(node, prim::kPrimReturn) || IsPrimitiveCNode(node, prim::kPrimDepend)) {
    auto inputs = node->inputs();
    if (inputs.size() < kInputSize2) {
      MS_LOG(EXCEPTION) << "node input size " << inputs.size() << " less than 2, node: " << node->fullname_with_scope();
    }
    auto input = inputs[1];
    MS_EXCEPTION_IF_NULL(input);
    node->set_abstract(input->abstract());
    return;
  }
  MS_LOG(WARNING) << "Node has not abstract:" << node->fullname_with_scope() << ", DebugString: " << node->ToString();
}

OperatorPtr DfGraphConvertor::ConvertCNode(const CNodePtr node) {
  SaveParamFormat(node);
  std::string name = GetCNodeTargetFuncName(node);
  if (!CheckCNode(name, node)) {
    return nullptr;
  }

  // get corresponding OpAdapter
  OpAdapterPtr adpt = FindAdapter(node, training_);
  if (adpt == nullptr) {
    MS_LOG(ERROR) << "Cannot get adapter for " << node->fullname_with_scope();
    unsupported_ops_names_.insert(name);
    error_ = NOT_FOUND;
    return nullptr;
  }
  SetNodeAbstract(node);
  // get operator
  OperatorPtr op = nullptr;
  auto it_op = op_cache_.find(node.get());
  if (it_op != op_cache_.end()) {
    op = it_op->second;
  } else {
    if (cur_while_node_ == node) {
      op = adpt->generateDynOutputOp(node);
    } else {
      op = adpt->generate(node);
    }
  }

  // set attribute for primitive
  (void)adpt->setAttr(op, node);
  auto value_node = node->input(0)->cast<ValueNodePtr>();
  if (value_node != nullptr && value_node->value()->cast<PrimitivePtr>() != nullptr) {
    MS_LOG(DEBUG) << "Set attr for subgraph multi dims";
    auto primitive = value_node->value()->cast<PrimitivePtr>();
    CheckAndAddScopeAttrInt(op, primitive, "_subgraph_multi_dims_index");
    CheckAndAddScopeAttrString(op, primitive, "_subgraph_multi_dims_input_dims");
    CheckAndAddScopeAttrString(op, primitive, "_subgraph_multi_dims_input_shape");
  }

  // add into cache
  (void)op_cache_.emplace(node.get(), op);

  DrawCNode(node, adpt);

  return op_cache_[node.get()];
}

OperatorPtr DfGraphConvertor::ConvertParameter(const AnfNodePtr node) {
  // convert Parameter in ANF to variable in DataFlow
  auto adpt = FindAdapter(node, training_);
  if (adpt == nullptr) {
    MS_LOG(EXCEPTION) << "Can not find adapter for Parameter";
  }
  auto op = adpt->generate(node);
  op_cache_[node.get()] = op;

  // build index for parameter using name
  std::string name = std::static_pointer_cast<Parameter>(node)->name();
  params_[name] = node;
  std::ostringstream ss;
  ss << "op" << node.get();
  op_draw_name_[node.get()] = ss.str();
  compute_sout_ << ss.str() << "[shape=octagon, label=\"" << name << "\"]" << endl;
  return op_cache_[node.get()];
}

void DfGraphConvertor::SaveParamFormat(const CNodePtr node) {
  AnfNodePtr op = node->input(0);
  if (IsValueNode<Primitive>(op)) {
    auto prim = GetValueNode<PrimitivePtr>(op);
    std::string format;
    auto op_def = ops::GetOpDef(prim->name());
    if (op_def) {
      for (size_t index = 0; index < op_def->args_.size() && index < node->size() - 1; index++) {
        auto arg = op_def->args_[index];
        if (arg.as_init_arg_ && (arg.arg_name_ == ops::kFormat || arg.arg_name_ == ops::kDataFormat)) {
          auto value_ptr = node->input(index + 1)->cast<ValueNodePtr>();
          if (value_ptr == nullptr) {
            break;
          }
          auto input_value = value_ptr->value();
          MS_EXCEPTION_IF_NULL(input_value);
          auto format_id = GetValue<int64_t>(input_value);
          format = FormatEnumToString(static_cast<Format>(format_id));
        }
      }
    }
    auto value_ptr = prim->GetAttr(ops::kFormat);
    if (value_ptr) {
      if (value_ptr->isa<Int64Imm>()) {
        bool converted = CheckAndConvertUtils::ConvertAttrValueToString(prim->name(), "format", &value_ptr);
        if (converted) {
          format = value_ptr->ToString();
        } else {
          CheckAndConvertUtils::GetFormatStringVal(prim, &format);
        }
      } else if (value_ptr->isa<StringImm>()) {
        format = value_ptr->ToString();
      }
    }

    if (format == "NCDHW" || format == "NHWC") {
      for (size_t i = 1; i < node->size(); i++) {
        auto input = node->input(i);
        if (input->isa<Parameter>()) {
          param_format_[input->DebugString()] = format;
          MS_LOG(DEBUG) << "Save Param " << input->DebugString() << " format: " << format;
        }
      }
    }
  }
}

Status DfGraphConvertor::TryConvertValueNodeToMultiConst(const ValueNodePtr node) {
  MS_EXCEPTION_IF_NULL(node);
  ValuePtr value = node->value();
  MS_EXCEPTION_IF_NULL(value);
  if (!value->isa<ValueList>() && !value->isa<ValueTuple>()) {
    return FAILED;
  }

  auto vec = value->isa<ValueTuple>() ? value->cast<ValueTuplePtr>()->value() : value->cast<ValueListPtr>()->value();
  if (vec.empty()) {
    return FAILED;
  }

  std::shared_ptr<std::vector<OutHandler>> tuple_items = std::make_shared<std::vector<OutHandler>>();
  // if the the sequence has only one element which is a scalar, it should be convert to a 1-D Tensor rather than a
  // 0-D Scalar.
  if (vec.size() == 1 && !vec[0]->isa<MeTensor>()) {
    return FAILED;
  }
  for (size_t i = 0; i < vec.size(); i++) {
    MS_EXCEPTION_IF_NULL(vec[i]);
    GeTensorPtr ge_tensor = nullptr;
    if (vec[i]->isa<MeTensor>()) {
      ge_tensor = transform::TransformUtil::ConvertTensor(vec[i]->cast<MeTensorPtr>(), kOpFormat_DEFAULT);
      MS_EXCEPTION_IF_NULL(ge_tensor);
    } else {
      ge_tensor = transform::TransformUtil::ConvertScalar(vec[i]);
      if (ge_tensor == nullptr) {
        return FAILED;
      }
    }
    auto const_op = std::make_shared<Constant>(node->fullname_with_scope() + "/const/inputs/" + std::to_string(i));
    AddGraphConstInput(const_op);
    (void)const_op->set_attr_value(*ge_tensor);
    (void)const_op->update_output_desc_y(ge_tensor->GetTensorDesc());
    (void)tuple_items->emplace_back(OutHandler(const_op, ""));
  }
  if (tuple_items->empty()) {
    return FAILED;
  }

  tuple_out_handle_cache_[node.get()] = tuple_items;
  if (!vec[0]->isa<MeTensor>()) {
    return FAILED;
  }
  return SUCCESS;
}

OperatorPtr DfGraphConvertor::ConvertValueNode(const ValueNodePtr node) {
  // convert valuenode in ANF to Const in DataFlow
  // find paramerte referenced by SymbolicKeyInstance of valuenode
  std::ostringstream ss;
  ss << "op" << node.get();
  op_draw_name_[node.get()] = ss.str();
  compute_sout_ << ss.str() << "[label= \"" << node->value()->ToString() << "\" shape=ellipse]" << endl;

  if (TryConvertValueNodeToMultiConst(node) == SUCCESS) {
    MS_LOG(INFO) << "Convert value node to multi Constant OP success";
    return nullptr;
  }

  OpAdapterPtr adpt = FindAdapter(node, training_);
  if (adpt == nullptr) {
    error_ = NOT_FOUND;
    return nullptr;
  }
  auto op = adpt->generate(node);
  // set const's attrs
  if (adpt->setAttr(op, "value", node->value()) != 0) {
    MS_LOG(WARNING) << "set attr value for const failed";
  }

  if (op->GetOpType() != "Constant" && op->GetOpType() != "Const") {
    MS_LOG(ERROR) << "Get Constant operator failed, ge node type: " << op->GetOpType()
                  << ", ms node info: " << node->ToString() << ", is train: " << training_;
    return nullptr;
  }
  ::ge::Tensor ge_tensor;
  (void)op->GetAttr("value", ge_tensor);
  auto ge_desc = ge_tensor.GetTensorDesc();
  (void)op->UpdateOutputDesc(kTypeY, ge_desc);

  op_cache_[node.get()] = op;
  return op_cache_[node.get()];
}

void DfGraphConvertor::DrawCNode(const CNodePtr node, const OpAdapterPtr adpt) {
  if (adpt == nullptr || node == nullptr) {
    MS_LOG(ERROR) << "Failed to draw apply node as adpt or node is nullptr!";
    return;
  }
  std::ostringstream ss;
  ss << "op" << node.get();
  op_draw_name_[node.get()] = ss.str();

  compute_sout_ << ss.str() << "[label=<";
  compute_sout_ << "<table border='1' cellborder='1'>" << endl;

  auto input_map = adpt->getInputMap();
  auto dyn_input_map = adpt->getDynInputMap();
  if (input_map.size() + dyn_input_map.size() > 0) {
    compute_sout_ << "<tr>";
    for (auto &it : input_map) {
      compute_sout_ << "<td port='" << it.first << "'>" << it.second.name << "</td>";
    }
    for (auto &it : dyn_input_map) {
      compute_sout_ << "<td port='" << it.first << "'>" << it.second.name << "</td>";
    }
    compute_sout_ << "</tr>" << endl;
  }

  compute_sout_ << "<tr><td colspan=\"" << (input_map.size() + dyn_input_map.size()) << "\">\"" << node->ToString()
                << ":" << GetCNodeTargetFuncName(node) << "\"</td></tr>" << endl;

  // print attrs' values
  auto atts = adpt->GetAttrsFromDrawGraph();
  for (auto &it : atts) {
    compute_sout_ << "<tr><td colspan=\"" << (input_map.size() + dyn_input_map.size()) << "\">\"" << it
                  << "\"</td></tr>";
  }

  adpt->clearAttrVect();

  compute_sout_ << "</table>> shape=plaintext]" << endl;
}
void DfGraphConvertor::RegisterAdapter(const std::string &name, OpAdapterPtr adpt) {
  OpAdapterMap::get()[name] = std::make_shared<OpAdapterDesc>(adpt);
}
void DfGraphConvertor::RegisterAdapter(const std::string &name, OpAdapterPtr train_adpt, OpAdapterPtr infer_adpt) {
  OpAdapterMap::get()[name] = std::make_shared<OpAdapterDesc>(train_adpt, infer_adpt);
}

std::map<std::string, ValuePtr> GeOpConvertor::GetAttrAndValue(const AnfNodePtr &node, const bool training = true) {
  MS_EXCEPTION_IF_NULL(node);
  std::map<std::string, ValuePtr> attr_list;
  if (!node->isa<CNode>()) {
    MS_LOG(INFO) << "Current node isn't a cnode! node info:" << node->DebugString();
    return attr_list;
  }

  OpAdapterPtr adpt = FindAdapter(node, training);
  if (adpt == nullptr) {
    MS_LOG(INFO) << "Current node can't find adpt! node info:" << node->DebugString();
    return attr_list;
  }

  attr_list = adpt->GetNormalOpAttrList(node);
  return attr_list;
}

std::string GeOpConvertor::GetOpType(const AnfNodePtr &node, const bool training = true) {
  MS_EXCEPTION_IF_NULL(node);
  OpAdapterPtr adpt = FindAdapter(node, training);
  if (adpt == nullptr) {
    MS_LOG(INFO) << "Current node can't find adpt! node info:" << node->DebugString();
    return "";
  }
  return adpt->getOpType();
}

std::shared_ptr<GeTensorDesc> GeOpConvertor::GetTensorDesc(const ShapeVector &dev_shape, const TypeId &dev_type,
                                                           const std::string &dev_format, const ShapeVector &ori_shape,
                                                           const std::string &ori_format) {
  auto tensor_desc = transform::TransformUtil::GetGeTensorDesc(dev_shape, dev_type, dev_format, ori_shape, ori_format);
  MS_EXCEPTION_IF_NULL(tensor_desc);
  return tensor_desc;
}

mindspore::HashMap<std::string, std::string> GeOpConvertor::GetNeedAddInput(const AnfNodePtr &node,
                                                                            const bool training) {
  MS_EXCEPTION_IF_NULL(node);
  OpAdapterPtr adpt = FindAdapter(node, training);
  if (adpt == nullptr) {
    MS_LOG(INFO) << "Current node can't find adpt! node info:" << node->DebugString();
    return {};
  }

  return adpt->getAttrInputMap();
}

bool GeOpConvertor::IsDynamicInput(const AnfNodePtr &node, const size_t idx) {
  MS_EXCEPTION_IF_NULL(node);
  OpAdapterPtr adapterPtr = FindAdapter(node, true);
  if (adapterPtr == nullptr) {
    MS_LOG(INFO) << "Can't find a adapter for op:" << node->DebugString();
    return false;
  }
  return adapterPtr->IsDynInputOp(idx);
}

std::map<int, std::string> GeOpConvertor::GetAclInputNames(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  OpAdapterPtr adapterPtr = FindAdapter(node, true);
  if (adapterPtr == nullptr) {
    MS_LOG(EXCEPTION) << "Can't find a adapter for op:" << node->DebugString();
  }

  std::map<int, std::string> input_names;
  for (const auto &[k, v] : adapterPtr->getInputMap()) {
    input_names.emplace(k, v.name);
  }
  // dynamic input
  for (const auto &[k, v] : adapterPtr->getDynInputMap()) {
    input_names.emplace(k, v.name);
  }
  return input_names;
}

std::map<int, std::string> GeOpConvertor::GetAclOutputNames(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  OpAdapterPtr adapterPtr = FindAdapter(node, true);
  if (adapterPtr == nullptr) {
    MS_LOG(EXCEPTION) << "Can't find a adapter for op:" << node->DebugString();
  }

  std::map<int, std::string> output_names;
  for (const auto &[k, v] : adapterPtr->getOutputMap()) {
    output_names.emplace(k, v.name);
  }

  // dynamic output
  for (const auto &[k, v] : adapterPtr->getDynOutputMap()) {
    output_names.emplace(k, v.name);
  }
  return output_names;
}

std::map<int, std::string> GeOpConvertor::GetAclDynamicInputNames(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  OpAdapterPtr adapterPtr = FindAdapter(node, true);
  if (adapterPtr == nullptr) {
    MS_LOG(EXCEPTION) << "Can't find a adapter for op:" << node->DebugString();
  }
  std::map<int, std::string> dyn_input_names;
  for (const auto &[k, v] : adapterPtr->getDynInputMap()) {
    dyn_input_names.emplace(k, v.name);
  }
  return dyn_input_names;
}

std::map<int, std::string> GeOpConvertor::GetAclDynamicOutputNames(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  OpAdapterPtr adapterPtr = FindAdapter(node, true);
  if (adapterPtr == nullptr) {
    MS_LOG(EXCEPTION) << "Can't find a adapter for op:" << node->DebugString();
  }
  std::map<int, std::string> dyn_output_names;
  for (const auto &[k, v] : adapterPtr->getDynOutputMap()) {
    dyn_output_names.emplace(k, v.name);
  }
  return dyn_output_names;
}
}  // namespace mindspore::transform
