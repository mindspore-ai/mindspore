/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#define USE_DEPRECATED_API

#include "mindspore/lite/tools/converter/quantizer/quantize_util.h"
#include <cmath>
#include <string>
#include <memory>
#include <vector>
#include <set>
#include <functional>
#include <deque>
#include "abstract/abstract_value.h"
#include "tools/common/graph_util.h"
#include "tools/lite_exporter/anf_exporter.h"
#include "tools/converter/graphdef_transform.h"
#include "tools/common/tensor_util.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "ops/fusion/mat_mul_fusion.h"
#include "ops/fusion/conv2d_transpose_fusion.h"
#include "ops/gather.h"
#include "ops/op_utils.h"
#include "src/common/utils.h"
#include "src/litert/cxx_api/tensor/tensor_impl.h"
#include "ir/anf.h"

using std::string;
using std::vector;

namespace mindspore::lite::quant {
namespace {
constexpr int kLstmInputWeightIndex = 1;
constexpr int kLstmStateWeightIndex = 2;
constexpr int kLstmWeightShapeSize = 3;
constexpr int kSingleDirBiasTensorSize = 4;
constexpr int kLstmBiasShapeSize = 2;
constexpr int kLstmBiasIndex = 3;
constexpr size_t kGatherAxisIndex = 3;
constexpr size_t kAnfPrimitiveIndex = 0;
}  // namespace

QuantParamHolderPtr GetCNodeQuantHolder(const CNodePtr &cnode) {
  MS_CHECK_TRUE_RET(cnode != nullptr, nullptr);
  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  if (primitive == nullptr) {
    MS_LOG(INFO) << "primitive is nullptr";
    return nullptr;
  }
  return GetCNodeQuantHolder(primitive);
}

QuantParamHolderPtr GetCNodeQuantHolder(const PrimitivePtr &primitive) {
  MS_CHECK_TRUE_RET(primitive != nullptr, nullptr);
  QuantParamHolderPtr quant_params_holder = nullptr;
  auto quant_params_valueptr = primitive->GetAttr("quant_params");
  if (quant_params_valueptr == nullptr) {
    quant_params_holder = std::make_shared<QuantParamHolder>(0, 0);
    MS_CHECK_TRUE_MSG(quant_params_holder != nullptr, nullptr, "quant_params_holder is nullptr.");
    primitive->AddAttr("quant_params", quant_params_holder);
  } else {
    quant_params_holder = quant_params_valueptr->cast<QuantParamHolderPtr>();
    if (quant_params_holder == nullptr) {
      quant_params_holder = std::make_shared<QuantParamHolder>(0, 0);
      MS_CHECK_TRUE_MSG(quant_params_holder != nullptr, nullptr, "quant_params_holder is nullptr.");
      primitive->AddAttr("quant_params", quant_params_holder);
    }
  }
  return quant_params_holder;
}

int GetQuantType(const CNodePtr &cnode, schema::QuantType *quant_type) {
  CHECK_NULL_RETURN(cnode);
  auto quant_param_holder = GetCNodeQuantHolder(cnode);
  if (quant_param_holder == nullptr) {
    *quant_type = schema::QuantType_QUANT_NONE;
    return RET_OK;
  }
  *quant_type = quant_param_holder->quant_type();
  return RET_OK;
}

void GetFuncGraphs(const FuncGraphPtr &func_graph, std::set<FuncGraphPtr> *all_func_graphs) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(all_func_graphs != nullptr);
  all_func_graphs->insert(func_graph);
  auto nodes = func_graph->GetOrderedCnodes();
  std::deque<CNodePtr> to_process{};
  to_process.insert(to_process.end(), nodes.begin(), nodes.end());
  while (!to_process.empty()) {
    auto &cur_cnode = to_process.front();
    for (auto &input : cur_cnode->inputs()) {
      if (!IsValueNode<FuncGraph>(input)) {
        continue;
      }
      auto new_fg = GetValueNode<FuncGraphPtr>(input);
      if (all_func_graphs->find(new_fg) != all_func_graphs->end()) {
        continue;
      }
      all_func_graphs->insert(new_fg);
      auto new_nodes = new_fg->GetOrderedCnodes();
      to_process.insert(to_process.end(), new_nodes.begin(), new_nodes.end());
    }
    to_process.pop_front();
  }
}

int UpdateDataType(const AnfNodePtr &cnode, TypeId new_data_type) {
  auto abstract_base = cnode->abstract();
  if (abstract_base == nullptr) {
    MS_LOG(ERROR) << "Abstract of node is nullptr, " << cnode->fullname_with_scope();
    return RET_NULL_PTR;
  }

  std::vector<AbstractBasePtr> abstracts;
  if (utils::isa<abstract::AbstractTuple>(abstract_base)) {
    auto abstract_tuple = utils::cast<abstract::AbstractTuplePtr>(abstract_base);
    abstracts = abstract_tuple->elements();
  } else {
    abstracts.push_back(abstract_base);
  }
  for (auto &abstract : abstracts) {
    auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(abstract);
    CHECK_NULL_RETURN(abstract_tensor);
    CHECK_NULL_RETURN(abstract_tensor->element());
    abstract_tensor->element()->set_type(TypeIdToType(new_data_type));
  }
  return RET_OK;
}

ValueNodePtr NewQuantCastPrimitive(int src_type, int dst_type,
                                   const std::vector<schema::QuantParamT> &input_quant_params,
                                   const std::vector<schema::QuantParamT> &output_quant_params) {
  auto prim_c = std::make_shared<ops::QuantDTypeCast>();
  MS_CHECK_TRUE_MSG(prim_c != nullptr, nullptr, "prim_c is nullptr.");
  prim_c->Init(src_type, dst_type);
  auto quant_params_holder = std::make_shared<QuantParamHolder>(input_quant_params.size(), output_quant_params.size());
  MS_CHECK_TRUE_MSG(quant_params_holder != nullptr, nullptr, "quant_params_holder is nullptr.");
  quant_params_holder->set_quant_type(schema::QuantType_QUANT_ALL);
  quant_params_holder->set_input_quant_param(0, input_quant_params);
  quant_params_holder->set_output_quant_param(0, output_quant_params);
  auto prim = prim_c->GetPrim();
  MS_CHECK_TRUE_MSG(prim != nullptr, nullptr, "prim is nullptr");
  prim->AddAttr("quant_params", quant_params_holder);
  return NewValueNode(prim);
}

bool IsGraphInDTypeCast(const CNodePtr &cnode) {
  if (!opt::CheckPrimitiveType(cnode, prim::kPrimQuantDTypeCast)) {
    return false;
  }
  auto input_node = cnode->input(1);
  MS_CHECK_FALSE(input_node == nullptr, false);
  return IsGraphInput(input_node);
}

bool IsGraphOutDTypeCast(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  if (!opt::CheckPrimitiveType(cnode, prim::kPrimQuantDTypeCast)) {
    return false;
  }
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    manager = Manage(func_graph, true);
  }
  CHECK_NULL_RETURN(manager);
  auto node_users = manager->node_users()[cnode];
  MS_CHECK_TRUE_RET(!node_users.empty(), RET_NULL_PTR);
  for (auto &node_user : node_users) {
    auto output_cnode = node_user.first->cast<CNodePtr>();
    CHECK_NULL_RETURN(output_cnode);
    if (!opt::CheckPrimitiveType(output_cnode, prim::kPrimReturn)) {
      return false;
    }
  }
  return true;
}

int GetCastNodeType(const FuncGraphPtr &func_graph, const CNodePtr &cnode, CastNodeType *cast_node_type) {
  if (!opt::CheckPrimitiveType(cnode, prim::kPrimQuantDTypeCast)) {
    MS_LOG(DEBUG) << "Not QuantDtypeCastNode, cnode name: " << cnode->fullname_with_scope();
    return RET_NOT_SUPPORT;
  }
  auto input_node = cnode->input(1);
  MS_CHECK_FALSE(input_node == nullptr, RET_ERROR);

  // input node
  TypeId pre_node_dtype = kTypeUnknown;
  if (opt::GetDataTypeFromAnfNode(input_node, &pre_node_dtype) != RET_OK) {
    MS_LOG(ERROR) << "Get data type failed, cnode name: " << input_node->fullname_with_scope();
    return RET_ERROR;
  }

  // output node
  TypeId post_node_dtype = kTypeUnknown;
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    manager = Manage(func_graph, true);
  }
  CHECK_NULL_RETURN(manager);
  auto node_users = manager->node_users()[cnode];
  MS_CHECK_TRUE_RET(!node_users.empty(), RET_NULL_PTR);
  auto output_cnode = node_users.begin()->first->cast<CNodePtr>();
  CHECK_NULL_RETURN(output_cnode);

  if (!opt::CheckPrimitiveType(output_cnode, prim::kPrimReturn)) {
    if (opt::GetDataTypeFromAnfNode(output_cnode, &post_node_dtype) != RET_OK) {
      MS_LOG(ERROR) << "Get data type failed, cnode name: " << output_cnode->fullname_with_scope();
      return RET_ERROR;
    }
    if (pre_node_dtype == kNumberTypeFloat32 &&
        (post_node_dtype == kNumberTypeInt8 || post_node_dtype == kNumberTypeUInt8)) {
      *cast_node_type = kQuant;
    } else if ((pre_node_dtype == kNumberTypeInt8 || pre_node_dtype == kNumberTypeUInt8) &&
               post_node_dtype == kNumberTypeFloat32) {
      *cast_node_type = kDeQuant;
    } else {
      MS_LOG(ERROR) << "Not support QuantDTypeCastNode, cnode name: " << cnode->fullname_with_scope();
    }
  } else {
    if (pre_node_dtype == kNumberTypeFloat32) {
      *cast_node_type = kQuant;
    } else if (pre_node_dtype == kNumberTypeInt8 || pre_node_dtype == kNumberTypeUInt8) {
      *cast_node_type = kDeQuant;
    } else {
      MS_LOG(ERROR) << "Not support QuantDTypeCastNode, cnode name: " << cnode->fullname_with_scope();
    }
  }
  return RET_OK;
}

bool TensorQuantParamsInited(const schema::TensorT &tensor) {
  if (tensor.quantParams.empty()) {
    return false;
  }

  bool is_quant_params_inited =
    std::all_of(tensor.quantParams.cbegin(), tensor.quantParams.cend(),
                [](const std::unique_ptr<mindspore::schema::QuantParamT> &quant_param) { return quant_param->inited; });
  return is_quant_params_inited;
}

std::string NodePrimitiveType(const CNodePtr &cnode) {
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "cnode is null";
    return "";
  }
  auto primitive_c = GetValueNode<std::shared_ptr<ops::PrimitiveC>>(cnode->input(0));
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "primitive_c is null";
    return "";
  }
  return primitive_c->name();
}

Status BuildModelByFuncGraph(const std::shared_ptr<mindspore::Model> &model, const FuncGraphPtr &func_graph,
                             const std::shared_ptr<ConverterPara> &param, size_t *size) {
  auto meta_graph = Export(func_graph, true, true);
  if (meta_graph == nullptr) {
    MS_LOG(ERROR) << "Export to meta_graph failed";
    return kLiteNullptr;
  }

  // transform
  GraphDefTransform fb_transform;
  fb_transform.SetGraphDef(meta_graph);
  auto status = fb_transform.Transform(param);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "FBTransform model failed";
    delete meta_graph;
    return kLiteError;
  }
  meta_graph->version = Version();

  status = UpdateGraphOutputName(meta_graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "UpdateGraphOutputName failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
    delete meta_graph;
    return kLiteError;
  }

  flatbuffers::FlatBufferBuilder builder(kMaxNum1024);
  auto offset = schema::MetaGraph::Pack(builder, meta_graph);
  builder.Finish(offset);
  schema::FinishMetaGraphBuffer(builder, offset);
  *size = builder.GetSize();
  auto *content = reinterpret_cast<const char *>(builder.GetBufferPointer());
  if (content == nullptr) {
    MS_LOG(ERROR) << "GetBufferPointer return null";
    delete meta_graph;
    return kLiteNullptr;
  }
  auto context = std::make_shared<mindspore::Context>();
  context->SetThreadAffinity(kCpuBindMode);
  if (context == nullptr) {
    MS_LOG(ERROR) << "New context failed while running.";
    delete meta_graph;
    return kLiteNullptr;
  }
  std::shared_ptr<CPUDeviceInfo> device_info = std::make_shared<CPUDeviceInfo>();
  auto &device_list = context->MutableDeviceInfo();
  device_list.push_back(device_info);
  auto ret = model->Build(content, *size, kMindIR, context);
  delete meta_graph;
  return ret;
}

mindspore::lite::Tensor *MSTensorToLiteTensor(const MSTensor &tensor) {
  if (tensor.impl() == nullptr) {
    MS_LOG(ERROR) << "Tensor " << tensor.Name() << " is nullptr.";
    return static_cast<lite::Tensor *>(nullptr);
  }
  auto lite_impl = std::static_pointer_cast<LiteTensorImpl>(tensor.impl());
  return static_cast<mindspore::lite::Tensor *>(lite_impl->lite_tensor());
}

std::vector<mindspore::lite::Tensor *> MSTensorToLiteTensors(const std::vector<mindspore::MSTensor> &src_tensors) {
  std::vector<mindspore::lite::Tensor *> dst_tensors(src_tensors.size());
  for (const auto &src_tensor : src_tensors) {
    auto tensor = MSTensorToLiteTensor(src_tensor);
    if (tensor == nullptr) {
      return {};
    }
    dst_tensors.emplace_back(tensor);
  }
  return dst_tensors;
}

void GetLiteParameter(const AnfNodePtr &node, ParameterPtr *param_node, tensor::TensorPtr *tensor_info) {
  if (node == nullptr) {
    MS_LOG(ERROR) << "node is nullptr";
    return;
  }
  auto op_name = node->fullname_with_scope();

  *param_node = node->cast<ParameterPtr>();
  if (*param_node == nullptr) {
    MS_LOG(INFO) << op_name << " can not cast to ParameterPtr";
    return;
  }
  if (!(*param_node)->has_default()) {
    MS_LOG(INFO) << op_name << " not has_default";
    return;
  }

  *tensor_info = std::static_pointer_cast<tensor::Tensor>((*param_node)->default_param());
  if (*tensor_info == nullptr) {
    MS_LOG(INFO) << "default_param can not cast to tensor::Tensor";
    return;
  }
}

int UpdateTensorDataAndSize(const AnfNodePtr &node, const tensor::TensorPtr &weight, void *quant_datas, int new_size,
                            TypeId new_data_type) {
  MS_CHECK_TRUE_RET(weight != nullptr, RET_NULL_PTR);
  MS_CHECK_TRUE_RET(new_size > 0, RET_NULL_PTR);
  weight->set_data_type(new_data_type);
  if (new_size != weight->data().nbytes()) {
    MS_LOG(ERROR) << "Data size of tensor info is error.";
    return RET_ERROR;
  }
  if (memcpy_s(weight->data_c(), new_size, quant_datas, new_size) != EOK) {
    MS_LOG(ERROR) << "memcpy data failed.";
    return RET_ERROR;
  }
  // set dtype
  auto ret = UpdateDataType(node, new_data_type);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << node->fullname_with_scope() << " set new dtype failed.";
    return ret;
  }
  auto abstract_base = node->abstract();
  if (abstract_base == nullptr) {
    MS_LOG(ERROR) << "Abstract of node is nullptr, " << node->fullname_with_scope();
    return RET_NULL_PTR;
  }
  if (!utils::isa<abstract::AbstractTensorPtr>(abstract_base)) {
    MS_LOG(ERROR) << "Abstract of node should be anstract tensor, " << node->fullname_with_scope();
    return RET_ERROR;
  }
  auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(abstract_base);
  CHECK_NULL_RETURN(abstract_tensor);
  CHECK_NULL_RETURN(abstract_tensor->element());
  abstract_tensor->element()->set_type(TypeIdToType(new_data_type));
  return RET_OK;
}

int GetMatMulPreferredDim(const PrimitivePtr &primitive, int input_index, const std::vector<int> &dims) {
  size_t last_first_index = dims.size() - 1;
  size_t last_second_index = dims.size() - 2;
  auto matmul_prim = api::MakeShared<ops::MatMul>(primitive);
  MS_ASSERT(matmul_prim != nullptr);
  // For MatMul A
  if (input_index == 0) {
    if (matmul_prim->GetAttr(ops::kTransposeA) != nullptr && matmul_prim->get_transpose_a()) {
      return last_first_index;
    } else {
      return last_second_index;
    }
  }
  // For MatMul B
  if (input_index == 1) {
    if (matmul_prim->GetAttr(ops::kTransposeB) != nullptr && matmul_prim->get_transpose_b()) {
      return last_second_index;
    } else {
      return last_first_index;
    }
  }
  return 0;
}

int GetDeConvPreferredDim(const PrimitivePtr &primitive, const std::vector<int> &dims) {
  auto prim = api::MakeShared<ops::Conv2DTranspose>(primitive);
  MS_ASSERT(prim != nullptr);
  if (prim->get_in_channel() == prim->get_group() && prim->get_out_channel() == prim->get_group()) {
    // DepthWise-DeConv (CO\CI) KH KW 1
    return 0;
  } else {
    // DeConv:CI KH KW CO
    return dims.size() - 1;
  }
}

int GetGatherPreferredDim(const CNodePtr &cnode) {
  if (cnode->size() < kGatherAxisIndex + 1) {
    MS_LOG(WARNING) << "gather cnode size < 4.";
    return 0;
  }
  DataInfo data_info;
  auto output_type_node = cnode->input(kGatherAxisIndex);
  if (utils::isa<ParameterPtr>(output_type_node)) {
    if (FetchDataFromParameterNode(cnode, kGatherAxisIndex, converter::kFmkTypeMs, &data_info, true) != lite::RET_OK) {
      MS_LOG(WARNING) << "Fetch data from parameter node failed.";
      return 0;
    }
  } else if (utils::isa<ValueNodePtr>(output_type_node)) {
    if (FetchDataFromValueNode(cnode, kGatherAxisIndex, converter::kFmkTypeMs, false, &data_info, true) !=
        lite::RET_OK) {
      MS_LOG(WARNING) << "Fetch data from value node failed.";
      return 0;
    }
  } else {
    MS_LOG(WARNING) << "The data type is not a const.";
    return 0;
  }

  auto axis_data = reinterpret_cast<const int *>(data_info.data_.data());
  CHECK_NULL_RETURN(axis_data);
  return axis_data[0];
}

int CalChannels(const std::vector<int> &dims, int channel_cnt, bool *channel_at_first) {
  auto channels = dims[0];
  if (!(*channel_at_first)) {
    if (dims.size() != DIMENSION_2D) {
      MS_LOG(WARNING) << "unexpected dims size: " << dims.size();
      *channel_at_first = true;
    } else {
      channels = dims[1];
    }
  } else {
    channels = channel_cnt == -1 ? channels : channel_cnt;
  }
  return channels;
}

int GetPreferredDim(const CNodePtr &cnode, int input_index, const std::vector<int> &dims) {
  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  CHECK_NULL_RETURN(primitive);
  if (primitive->name() == ops::kNameMatMulFusion) {
    return GetMatMulPreferredDim(primitive, input_index, dims);
  } else if (primitive->name() == ops::kNameConv2dTransposeFusion) {
    return GetDeConvPreferredDim(primitive, dims);
  } else if (primitive->name() == ops::kNameGather) {
    return GetGatherPreferredDim(cnode);
  }
  // The first index.
  return 0;
}

std::vector<int> ConvertShapeVectorToInt32(const ShapeVector &dims) {
  std::vector<int> shape;
  for (auto dim : dims) {
    if (dim > INT32_MAX || dim < INT32_MIN) {
      MS_LOG(ERROR) << dim << " over int32 range.";
      shape.push_back(-1);
    } else {
      shape.push_back(dim);
    }
  }
  return shape;
}

void CalQuantAssitInfo(const schema::PrimitiveT &primitive, const std::vector<int> &shapes, int index,
                       bool *channel_at_first, int *channel_cnt) {
  MS_ASSERT(primitive != nullptr);
  if (shapes.empty()) {
    MS_LOG(ERROR) << " shape vector is empty.";
    return;
  }
  if (primitive.value.type == schema::PrimitiveType_MatMulFusion && static_cast<int>(shapes.size()) == DIMENSION_2D) {
    auto matmul_prim = primitive.value.AsMatMulFusion();
    MS_ASSERT(matmul_prim != nullptr);
    *channel_at_first = index != 1 || matmul_prim->transpose_b;
  } else if (primitive.value.type == schema::PrimitiveType_LSTM) {
    if (index == kLstmInputWeightIndex || index == kLstmStateWeightIndex) {
      if (shapes.size() != kLstmWeightShapeSize) {
        MS_LOG(WARNING) << "unexpected lstm shape size: " << shapes.size();
      } else {
        *channel_cnt = shapes[0] * shapes[1];
      }
    } else if (index == kLstmBiasIndex) {
      if (shapes.size() != kLstmBiasShapeSize) {
        MS_LOG(WARNING) << "unexpected lstm shape size: " << shapes.size();
      } else {
        auto tensor_elem_cnt = shapes[0] * shapes[1];
        if (tensor_elem_cnt % kSingleDirBiasTensorSize == 0) {
          *channel_cnt = kSingleDirBiasTensorSize;
        }
      }
    } else {
      MS_LOG(WARNING) << "unexpected index of lstm: " << index;
    }
  }
}

bool CheckNodeInSet(const CNodePtr &cnode, const std::set<PrimitivePtr> &support_primitive_types) {
  for (const auto &type : support_primitive_types) {
    if (opt::CheckPrimitiveType(cnode, type)) {
      return true;
    }
  }
  return false;
}

int DeQuantData(const mindspore::MSTensor *tensor, std::vector<double> *dequant_data, int preferred_dim) {
  return DeQuantData(reinterpret_cast<const int8_t *>(tensor->Data().get()), tensor->ElementNum(),
                     tensor->QuantParams(), dequant_data, preferred_dim);
}

int GetElementNumFromShape(const std::vector<int> &dims, int *total_size) {
  CHECK_NULL_RETURN(total_size);
  *total_size = 1;
  for (auto dim : dims) {
    MS_CHECK_FALSE_MSG(INT_MUL_OVERFLOW(*total_size, dim), RET_ERROR, "Int mul overflow.");
    *total_size *= dim;
  }
  return RET_OK;
}

int GetBucketAllIndex(const std::vector<int> &dims, int preferred_dim,
                      std::vector<std::vector<int>> *buckets_data_index) {
  int outer = 1;
  for (int i = 0; i < preferred_dim; i++) {
    outer *= dims[i];
  }
  int bucket_count = dims[preferred_dim];
  int inner = 1;
  for (size_t i = preferred_dim + 1; i < dims.size(); i++) {
    inner *= dims[i];
  }
  if (inner <= 0 || outer <= 0 || bucket_count <= 0) {
    return RET_ERROR;
  }
  for (int i = 0; i < bucket_count; i++) {
    auto index = i * inner;
    std::vector<int> bucket_index(inner * outer);
    for (int j = 0; j < outer; j++) {
      for (int k = 0; k < inner; k++) {
        bucket_index[j * inner + k] = index + k;
      }
      index += bucket_count * inner;
    }
    buckets_data_index->push_back(bucket_index);
  }
  return RET_OK;
}

bool CheckControlFlowType(const AnfNodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  std::map<std::string, PrimitivePtr> control_flow_ops = {{"PartialFusion", prim::kPrimPartialFusion},
                                                          {"Switch", prim::kPrimSwitch},
                                                          {"switch_layer", prim::kPrimSwitchLayer},
                                                          {"call", prim::kPrimCall}};

  if (node->isa<mindspore::CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    // control flow call
    if (!IsValueNode<mindspore::Primitive>(cnode->input(kAnfPrimitiveIndex))) {
      return true;
    }
    auto prim = GetValuePtr<mindspore::Primitive>(cnode->input(kAnfPrimitiveIndex));
    if (control_flow_ops.find(prim->name()) != control_flow_ops.end()) {
      return true;
    }
  } else if (node->isa<ValueNode>()) {
    auto prim = GetValuePtr<mindspore::Primitive>(node);
    if (control_flow_ops.find(prim->name()) != control_flow_ops.end()) {
      return true;
    }
  }
  return false;
}

bool CanTensorWeightQuantized(const CNodePtr &cnode, const AnfNodePtr &input_node, ShapeVector *weight_shape) {
  if (input_node == nullptr) {
    MS_LOG(INFO) << "CanTensorQuantized input is nullptr!";
    return false;
  }
  ParameterPtr param_node = nullptr;
  if (input_node->isa<Parameter>()) {
    param_node = input_node->cast<ParameterPtr>();
  }
  if (param_node == nullptr) {
    MS_LOG(INFO) << "CanTensorQuantized invalid param_node!";
    return false;
  }
  if (!param_node->has_default()) {
    MS_LOG(INFO) << "param_node don't has default.";
    return false;
  }
  auto abstract_base = param_node->abstract();
  if (abstract_base == nullptr) {
    MS_LOG(INFO) << "abstract is nullptr";
    return false;
  }
  if (!utils::isa<abstract::ShapePtr>(abstract_base->GetShapeTrack())) {
    MS_LOG(INFO) << "Shape of Abstract of parameter should be ShapePtr " << param_node->name();
    return false;
  }
  MS_CHECK_TRUE_RET(weight_shape != nullptr, false);
  *weight_shape = utils::cast<abstract::ShapePtr>(abstract_base->GetShapeTrack())->shape();
  if (weight_shape->size() < DIMENSION_2D) {  // do not quant single dim tensors
    return false;
  }
  return true;
}

bool CanTensorWeightQuantized(const CNodePtr &cnode, const AnfNodePtr &input_node, int preferred_dim,
                              int min_quant_weight_size, int min_quant_weight_channel) {
  ShapeVector weight_shape;
  if (!CanTensorWeightQuantized(cnode, input_node, &weight_shape)) {
    return false;
  }
  MS_CHECK_TRUE_RET(!weight_shape.empty(), false);
  int total_shape_size = 1;
  auto ret = GetElementNumFromShape(ConvertShapeVectorToInt32(weight_shape), &total_shape_size);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Get element num from shape failed.";
    return ret;
  }
  if (total_shape_size < 0 || total_shape_size <= min_quant_weight_size) {
    MS_LOG(INFO) << "shape_size " << total_shape_size << " less min_quant_weight_size " << min_quant_weight_size;
    return false;
  }

  static const std::set<PrimitivePtr> check_channel_ops = {prim::kPrimConv2DFusion, prim::kPrimConv2dTransposeFusion};

  if (CheckNodeInSet(cnode, check_channel_ops) && weight_shape.size() >= DIMENSION_2D &&
      weight_shape[preferred_dim] <= min_quant_weight_channel) {
    MS_LOG(INFO) << "preferred_dim shape:" << weight_shape[preferred_dim] << " less min_quant_weight_channel_ "
                 << min_quant_weight_channel;
    return false;
  }
  return true;
}
}  // namespace mindspore::lite::quant
