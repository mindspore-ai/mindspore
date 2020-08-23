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

#include "tools/anf_importer/import_from_protobuf.h"

#include <fcntl.h>
#include <unistd.h>

#include <fstream>
#include <map>
#include <memory>
#include <stack>
#include <unordered_map>
#include <vector>
#include "src/ops/primitive_c.h"
#include "frontend/operator/ops.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "include/errorcode.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "schema/inner/model_generated.h"
#include "securec/include/securec.h"
#include "src/ir/tensor.h"
#include "src/param_value_lite.h"
#include "tools/converter/parser/onnx/onnx.pb.h"
#include "utils/log_adapter.h"

using string = std::string;
using int32 = int32_t;
using int64 = int64_t;
using uint64 = uint64_t;

namespace mindspore::lite {

static constexpr char kConstantValueNode[] = "Constant";
static constexpr char kCNodeShapeAttr[] = "shape";
static constexpr char kCNodeShape1Attr[] = "shape1";
static constexpr char kCNodeShape2Attr[] = "shape2";

enum ParseForm : int {
  FORM_PARSE_TYPE = 0,
  FORM_PARSE_SCALAR = 1,
  FORM_PARSE_TENSOR = 2,
};

static std::map<std::string, ParseForm> kParseTypeSwitchMap{
  {"type", FORM_PARSE_TYPE}, {"scalar", FORM_PARSE_SCALAR}, {"tensor", FORM_PARSE_TENSOR}};

static std::unordered_map<int, TypeId> kDefaultValueSwitchMap{
  {onnx::TensorProto_DataType_BOOL, kNumberTypeBool},     {onnx::TensorProto_DataType_INT8, kNumberTypeInt8},
  {onnx::TensorProto_DataType_INT16, kNumberTypeInt16},   {onnx::TensorProto_DataType_INT32, kNumberTypeInt32},
  {onnx::TensorProto_DataType_INT64, kNumberTypeInt64},   {onnx::TensorProto_DataType_UINT8, kNumberTypeUInt8},
  {onnx::TensorProto_DataType_UINT16, kNumberTypeUInt16}, {onnx::TensorProto_DataType_UINT32, kNumberTypeUInt32},
  {onnx::TensorProto_DataType_UINT64, kNumberTypeUInt64}, {onnx::TensorProto_DataType_FLOAT16, kNumberTypeFloat16},
  {onnx::TensorProto_DataType_FLOAT, kNumberTypeFloat32}, {onnx::TensorProto_DataType_DOUBLE, kNumberTypeFloat64},
  {onnx::TensorProto_DataType_STRING, kObjectTypeString},
};

#define PARSE_ONNXATTR_IN_SCALAR_FORM(type, valuetype)                                                \
  void ParseAttrInScalar_##type##_##valuetype(const PrimitivePtr &prim, const std::string &attr_name, \
                                              const onnx::TensorProto &attr_tensor) {                 \
    MS_EXCEPTION_IF_NULL(prim);                                                                       \
    std::vector<ValuePtr> attr_value_vec;                                                             \
    for (int i = 0; i < attr_tensor.type##_data_size(); ++i) {                                        \
      auto value = static_cast<valuetype>(attr_tensor.type##_data(i));                                \
      attr_value_vec.push_back(MakeValue<valuetype>(value));                                          \
    }                                                                                                 \
    if (attr_value_vec.size() == 1) {                                                                 \
      prim->AddAttr(attr_name, attr_value_vec[0]);                                                    \
    } else {                                                                                          \
      prim->AddAttr(attr_name, std::make_shared<ValueList>(attr_value_vec));                          \
    }                                                                                                 \
  }

PARSE_ONNXATTR_IN_SCALAR_FORM(double, double)
PARSE_ONNXATTR_IN_SCALAR_FORM(float, float)
PARSE_ONNXATTR_IN_SCALAR_FORM(string, string)
PARSE_ONNXATTR_IN_SCALAR_FORM(int32, int32)
PARSE_ONNXATTR_IN_SCALAR_FORM(int32, bool)
PARSE_ONNXATTR_IN_SCALAR_FORM(int64, int64)
PARSE_ONNXATTR_IN_SCALAR_FORM(uint64, uint64)

bool AnfImporterFromProtobuf::BuildParameterForFuncGraph(const ParameterPtr &node,
                                                         const onnx::ValueInfoProto &value_proto) {
  MS_EXCEPTION_IF_NULL(node);
  if (!value_proto.has_type() || !value_proto.has_name()) {
    MS_LOG(ERROR) << "onnx ValueInfoProto has no type or name! ";
    return false;
  }
  node->set_name(value_proto.name());
  const auto &type_proto = value_proto.type();
  if (!type_proto.has_tensor_type()) {
    MS_LOG(ERROR) << "onnx TypeProto has no tesor_type! ";
    return false;
  }
  const onnx::TypeProto_Tensor &tensor_typeproto = type_proto.tensor_type();
  if (!tensor_typeproto.has_elem_type() || !tensor_typeproto.has_shape()) {
    MS_LOG(ERROR) << "onnx TypeProto_Tensor has no elem_type or shape! ";
    return false;
  }
  const onnx::TensorShapeProto &tensor_shape = tensor_typeproto.shape();
  std::vector<int> shape;
  for (int i = 0; i < tensor_shape.dim_size(); ++i) {
    shape.push_back(tensor_shape.dim(i).dim_value());
  }

  if (kDefaultValueSwitchMap.find(tensor_typeproto.elem_type()) == kDefaultValueSwitchMap.end()) {
    MS_LOG(ERROR) << "onnx TypeProto_Tensor  elem_type is not support yet!";
    return false;
  }

  auto type_ptr = TypeIdToType(kDefaultValueSwitchMap[tensor_typeproto.elem_type()]);
  auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, shape);
  node->set_abstract(abstract_tensor);

  if (default_para_map_.find(value_proto.name()) != default_para_map_.end()) {
    tensor::Tensor *tensor_info = new tensor::Tensor(kDefaultValueSwitchMap[tensor_typeproto.elem_type()], shape);
    MS_EXCEPTION_IF_NULL(tensor_info);
    tensor_info->MallocData();
    const onnx::TensorProto initialize_proto = default_para_map_[value_proto.name()];
    std::string initial_data = initialize_proto.raw_data();
    auto *tensor_data_buf = reinterpret_cast<uint8_t *>(tensor_info->Data());
    MS_EXCEPTION_IF_NULL(tensor_data_buf);
    tensor_info->SetData(nullptr);
    auto ret = memcpy_s(tensor_data_buf, tensor_info->Size(), initial_data.data(), initial_data.size());
    if (EOK != ret) {
      MS_LOG(ERROR) << "memcpy_s error";
      delete tensor_data_buf;
      delete tensor_info;
      return false;
    }

    ParamValueLitePtr param_value = std::make_shared<ParamValueLite>();
    MS_EXCEPTION_IF_NULL(param_value);
    param_value->set_tensor_addr(tensor_data_buf);
    param_value->set_tensor_size(tensor_info->Size());
    param_value->set_tensor_type(tensor_info->data_type());
    param_value->set_tensor_shape(tensor_info->shape());
    node->set_default_param(param_value);
    delete tensor_info;
  }
  anfnode_build_map_[value_proto.name()] = node;
  return true;
}

bool AnfImporterFromProtobuf::ImportParametersForGraph(const FuncGraphPtr &outputFuncGraph,
                                                       const onnx::GraphProto &importProto) {
  MS_EXCEPTION_IF_NULL(outputFuncGraph);
  MS_LOG(INFO) << "Parameters had default paramerer size is: " << importProto.initializer_size();

  for (int i = 0; i < importProto.initializer_size(); ++i) {
    const onnx::TensorProto &initializer_proto = importProto.initializer(i);
    if (!initializer_proto.has_name()) {
      MS_LOG(ERROR) << "initializer vector of onnx GraphProto has no name at index: " << i;
      return false;
    }
    default_para_map_[initializer_proto.name()] = initializer_proto;
  }

  MS_LOG(INFO) << "all parameters size: " << importProto.input_size();
  for (int i = 0; i < importProto.input_size(); ++i) {
    const onnx::ValueInfoProto &input_proto = importProto.input(i);
    if (!BuildParameterForFuncGraph(outputFuncGraph->add_parameter(), input_proto)) {
      MS_LOG(ERROR) << "Build parameter for funcgraph fail at index: " << i;
      return false;
    }
  }
  return true;
}

bool AnfImporterFromProtobuf::ObtainCNodeAttrInTypeForm(const PrimitivePtr &prim, const std::string &attr_name,
                                                        const onnx::TensorProto &attr_tensor) {
  MS_EXCEPTION_IF_NULL(prim);
  const int attr_tensor_type = attr_tensor.data_type();
  if (kDefaultValueSwitchMap.find(attr_tensor_type) == kDefaultValueSwitchMap.end()) {
    MS_LOG(ERROR) << "Obtain attr in type-form has not support input type:" << attr_tensor_type;
    return false;
  }
  prim->AddAttr(attr_name, TypeIdToType(kDefaultValueSwitchMap[attr_tensor_type]));
  return true;
}

bool AnfImporterFromProtobuf::ObtainCNodeAttrInScalarForm(const PrimitivePtr &prim, const std::string &attr_name,
                                                          const onnx::TensorProto &attr_tensor) {
  MS_EXCEPTION_IF_NULL(prim);
  const int attr_tensor_type = attr_tensor.data_type();
  switch (attr_tensor_type) {
    case onnx::TensorProto_DataType_STRING: {
      ParseAttrInScalar_string_string(prim, attr_name, attr_tensor);
      break;
    }
    case onnx::TensorProto_DataType_INT32: {
      ParseAttrInScalar_int32_int32(prim, attr_name, attr_tensor);
      break;
    }
    case onnx::TensorProto_DataType_INT64: {
      ParseAttrInScalar_int64_int64(prim, attr_name, attr_tensor);
      break;
    }
    case onnx::TensorProto_DataType_UINT64: {
      ParseAttrInScalar_uint64_uint64(prim, attr_name, attr_tensor);
      break;
    }
    case onnx::TensorProto_DataType_FLOAT: {
      ParseAttrInScalar_float_float(prim, attr_name, attr_tensor);
      break;
    }
    case onnx::TensorProto_DataType_DOUBLE: {
      ParseAttrInScalar_double_double(prim, attr_name, attr_tensor);
      break;
    }
    case onnx::TensorProto_DataType_BOOL: {
      ParseAttrInScalar_int32_bool(prim, attr_name, attr_tensor);
      auto value = prim->GetAttr(attr_name);
      break;
    }
    default:
      MS_LOG(ERROR) << "Obtain attr in scalar-form has not support input type: " << attr_tensor_type;
      return false;
  }
  return true;
}

bool AnfImporterFromProtobuf::ObtainCNodeAttrInTensorForm(const PrimitivePtr &prim, const std::string &attr_name,
                                                          const onnx::TensorProto &attr_tensor) {
  MS_EXCEPTION_IF_NULL(prim);
  const int attr_tensor_type = attr_tensor.data_type();
  const std::string &tensor_buf = attr_tensor.raw_data();
  std::vector<int> shape;
  auto ret = EOK;
  if (attr_tensor.dims_size() != 0) {
    for (int i = 0; i < attr_tensor.dims_size(); ++i) {
      shape.push_back(attr_tensor.dims(i));
    }
    tensor::TensorPtr tensor_info = std::make_shared<tensor::Tensor>(kDefaultValueSwitchMap[attr_tensor_type], shape);
    tensor_info->MallocData();
    auto *tensor_data_buf = reinterpret_cast<uint8_t *>(tensor_info->Data());
    ret = memcpy_s(tensor_data_buf, tensor_info->Size(), tensor_buf.data(), tensor_buf.size());
    prim->set_attr(attr_name, MakeValue(tensor_info));
  } else {
    if (attr_tensor_type == onnx::TensorProto_DataType_DOUBLE) {
      size_t data_size = sizeof(double);
      double attr_value = 0.0;
      ret = memcpy_s(&attr_value, data_size, tensor_buf.data(), tensor_buf.size());
      prim->set_attr(attr_name, MakeValue<double>(attr_value));
    } else if (attr_tensor_type == onnx::TensorProto_DataType_INT64) {
      size_t data_size = sizeof(int64_t);
      int32_t attr_value = 0;
      ret = memcpy_s(&attr_value, data_size, tensor_buf.data(), tensor_buf.size());
      prim->set_attr(attr_name, MakeValue<int32_t>(attr_value));
    } else if (attr_tensor_type == onnx::TensorProto_DataType_BOOL) {
      size_t data_size = sizeof(bool);
      bool attr_value = false;
      ret = memcpy_s(&attr_value, data_size, tensor_buf.data(), tensor_buf.size());
      prim->set_attr(attr_name, MakeValue<bool>(attr_value));
    }
  }

  return ret == EOK;
}

bool AnfImporterFromProtobuf::GetAttrValueForCNode(const PrimitivePtr &prim, const onnx::AttributeProto &attr_proto) {
  MS_EXCEPTION_IF_NULL(prim);
  const std::string &attr_name = attr_proto.name();
  if (!attr_proto.has_ref_attr_name()) {
    MS_LOG(ERROR) << "CNode parse attr type has no ref_attr_name";
    return false;
  }
  const std::string &ref_attr_name = attr_proto.ref_attr_name();
  const onnx::TensorProto &attr_tensor = attr_proto.t();
  switch (kParseTypeSwitchMap[ref_attr_name]) {
    case FORM_PARSE_TYPE: {
      return ObtainCNodeAttrInTypeForm(prim, attr_name, attr_tensor);
    }
    case FORM_PARSE_SCALAR: {
      return ObtainCNodeAttrInScalarForm(prim, attr_name, attr_tensor);
    }
    case FORM_PARSE_TENSOR: {
      return ObtainCNodeAttrInTensorForm(prim, attr_name, attr_tensor);
    }
    default:
      MS_LOG(ERROR) << "parse attr type don't support input of ref_attr_name";
      return false;
  }
}
bool AnfImporterFromProtobuf::ObtainValueNodeInTensorForm(const std::string &value_node_name,
                                                          const onnx::TensorProto &attr_tensor) {
  const int attr_tensor_type = attr_tensor.data_type();
  std::vector<int> shape;
  for (int i = 0; i < attr_tensor.dims_size(); ++i) {
    shape.push_back(attr_tensor.dims(i));
  }
  tensor::TensorPtr tensor_info = std::make_shared<tensor::Tensor>(kDefaultValueSwitchMap[attr_tensor_type], shape);
  tensor_info->MallocData();
  const std::string &tensor_buf = attr_tensor.raw_data();
  auto *tensor_data_buf = reinterpret_cast<uint8_t *>(tensor_info->Data());
  auto ret = memcpy_s(tensor_data_buf, tensor_info->Size(), tensor_buf.data(), tensor_buf.size());
  if (EOK != ret) {
    MS_LOG(ERROR) << "memcpy_s error";
    return false;
  }
  auto new_value_node = NewValueNode(MakeValue(tensor_info));
  MS_EXCEPTION_IF_NULL(new_value_node);
  auto type_ptr = TypeIdToType(kDefaultValueSwitchMap[attr_tensor_type]);
  auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, shape);
  new_value_node->set_abstract(abstract_tensor);
  anfnode_build_map_[value_node_name] = new_value_node;
  return true;
}

bool AnfImporterFromProtobuf::ObtainValueNodeInScalarForm(const std::string &value_node_name,
                                                          const onnx::TensorProto &attr_tensor) {
  const int attr_tensor_type = attr_tensor.data_type();
  ValuePtr value_ptr = nullptr;
  switch (attr_tensor_type) {
    case onnx::TensorProto_DataType_INT32: {
      std::vector<int32> add_data;
      for (int i = 0; i < attr_tensor.int32_data_size(); ++i) {
        add_data.push_back(attr_tensor.int32_data(i));
      }
      if (add_data.size() == 1) {
        value_ptr = MakeValue(add_data[0]);
      } else if (!add_data.empty()) {
        value_ptr = MakeValue<std::vector<int32> >(add_data);
      }
      break;
    }
    case onnx::TensorProto_DataType_FLOAT: {
      std::vector<float> add_data;
      for (int i = 0; i < attr_tensor.float_data_size(); ++i) {
        add_data.push_back(attr_tensor.float_data(i));
      }

      if (add_data.size() == 1) {
        value_ptr = MakeValue(add_data[0]);
      } else if (!add_data.empty()) {
        value_ptr = MakeValue<std::vector<float> >(add_data);
      }
      break;
    }
    case onnx::TensorProto_DataType_UNDEFINED: {
      std::vector<ValuePtr> elems;
      value_ptr = std::make_shared<ValueTuple>(elems);
      break;
    }
    default:
      MS_LOG(ERROR) << "Obtain attr in scalar-form has not support input type: " << attr_tensor_type;
      return false;
  }
  auto new_value_node = NewValueNode(value_ptr);
  MS_EXCEPTION_IF_NULL(new_value_node);
  new_value_node->set_abstract(value_ptr->ToAbstract());
  anfnode_build_map_[value_node_name] = new_value_node;

  return true;
}

bool AnfImporterFromProtobuf::ObtainValueNodeInTypeForm(const std::string &value_node_name,
                                                        const onnx::TensorProto &attr_tensor) {
  const int attr_tensor_type = attr_tensor.data_type();
  if (kDefaultValueSwitchMap.find(attr_tensor_type) == kDefaultValueSwitchMap.end()) {
    MS_LOG(ERROR) << "Obtain ValueNode attr in type-form has not support input type: " << attr_tensor_type;
    return false;
  }
  auto new_value_node = NewValueNode(TypeIdToType(kDefaultValueSwitchMap[attr_tensor_type]));
  abstract::AbstractTypePtr abs_type = std::make_shared<abstract::AbstractType>(std::make_shared<TypeType>());
  new_value_node->set_abstract(abs_type);
  anfnode_build_map_[value_node_name] = new_value_node;
  return true;
}

bool AnfImporterFromProtobuf::GetAttrValueForValueNode(const std::string &ref_attr_name,
                                                       const std::string &value_node_name,
                                                       const onnx::TensorProto &attr_tensor) {
  switch (kParseTypeSwitchMap[ref_attr_name]) {
    case FORM_PARSE_SCALAR: {
      return ObtainValueNodeInScalarForm(value_node_name, attr_tensor);
    }
    case FORM_PARSE_TENSOR: {
      return ObtainValueNodeInTensorForm(value_node_name, attr_tensor);
    }
    case FORM_PARSE_TYPE: {
      return ObtainValueNodeInTypeForm(value_node_name, attr_tensor);
    }
    default:
      MS_LOG(ERROR) << "parse ValueNode value don't support input of ref_attr_name";
      return false;
  }
}

bool AnfImporterFromProtobuf::BuildValueNodeForFuncGraph(const onnx::NodeProto &node_proto) {
  const std::string &value_node_name = node_proto.output(0);
  const onnx::AttributeProto &attr_proto = node_proto.attribute(0);
  if (!attr_proto.has_ref_attr_name()) {
    MS_LOG(ERROR) << "parse ValueNode  don't have ref_attr_name";
    return false;
  }
  const std::string &ref_attr_name = attr_proto.ref_attr_name();
  const onnx::TensorProto &attr_tensor = attr_proto.t();

  return GetAttrValueForValueNode(ref_attr_name, value_node_name, attr_tensor);
}

abstract::AbstractTensorPtr AnfImporterFromProtobuf::GetAbstractForCNode(const onnx::AttributeProto &attr_proto) {
  std::vector<int> shape_vec;
  const onnx::TensorProto &attr_tensor = attr_proto.t();
  for (int i = 0; i < attr_tensor.dims_size(); ++i) {
    shape_vec.push_back(attr_tensor.dims(i));
  }
  auto type_ptr = TypeIdToType(kDefaultValueSwitchMap[attr_tensor.data_type()]);
  auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vec);
  MS_EXCEPTION_IF_NULL(abstract_tensor);
  return abstract_tensor;
}

CNodePtr AnfImporterFromProtobuf::BuildCNodeForFuncGraph(const FuncGraphPtr &outputFuncGraph,
                                                         const onnx::NodeProto &node_proto,
                                                         const schema::QuantType &quantType) {
  MS_EXCEPTION_IF_NULL(outputFuncGraph);
  if (!node_proto.has_op_type()) {
    MS_LOG(ERROR) << "Get CNode op_type failed!";
    return nullptr;
  }
  const std::string &node_name = node_proto.output(0);
  const std::string &fullname_with_scope = node_proto.domain();
  const std::string &node_type = node_proto.op_type();
  PrimitivePtr prim = std::make_shared<Primitive>(node_type);
  MS_EXCEPTION_IF_NULL(prim);
  prim->set_instance_name(node_type);

  abstract::AbstractTensorPtr abstract = nullptr;
  abstract::AbstractTensorPtr abstract_first = nullptr;
  abstract::AbstractTensorPtr abstract_second = nullptr;
  for (int i = 0; i < node_proto.attribute_size(); ++i) {
    const onnx::AttributeProto &attr_proto = node_proto.attribute(i);
    if (attr_proto.name() == kCNodeShapeAttr) {
      abstract = GetAbstractForCNode(attr_proto);
      continue;
    }
    if (attr_proto.name() == kCNodeShape1Attr) {
      abstract_first = GetAbstractForCNode(attr_proto);
      continue;
    }
    if (attr_proto.name() == kCNodeShape2Attr) {
      abstract_second = GetAbstractForCNode(attr_proto);
      continue;
    }
    if (!GetAttrValueForCNode(prim, attr_proto)) {
      MS_LOG(ERROR) << "Get CNode attr failed!";
      return nullptr;
    }
  }
  std::vector<AnfNodePtr> inputs;
  inputs.clear();
  for (int i = 0; i < node_proto.input_size(); ++i) {
    const std::string &input_name = node_proto.input(i);
    if (anfnode_build_map_.find(input_name) == anfnode_build_map_.end()) {
      MS_LOG(ERROR) << node_name << " input " << i << input_name << "can't find in nodes have parsed";
      return nullptr;
    }
    inputs.push_back(anfnode_build_map_[input_name]);
  }
  auto primitivec_ptr = PrimitiveC::UnPackFromPrimitive(*prim, inputs);
  if (primitivec_ptr == nullptr) {
    MS_LOG(ERROR) << "Create PrimitiveC return nullptr, " << prim->name();
    return nullptr;
  }
  inputs.insert(inputs.begin(), NewValueNode(primitivec_ptr));
  CNodePtr cnode_ptr = outputFuncGraph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(cnode_ptr);
  if (node_type == "LayerNorm") {
    AbstractBasePtrList elem;
    elem.push_back(abstract);
    elem.push_back(abstract_first);
    elem.push_back(abstract_second);
    cnode_ptr->set_abstract(std::make_shared<abstract::AbstractTuple>(elem));
  } else if (node_type == "ArgMaxWithValue") {
    AbstractBasePtrList elem;
    elem.push_back(abstract);
    elem.push_back(abstract_first);
    cnode_ptr->set_abstract(std::make_shared<abstract::AbstractTuple>(elem));
  } else if (nullptr == abstract) {
    AbstractBasePtrList elem;
    for (size_t index = 1; index < cnode_ptr->inputs().size(); ++index) {
      elem.push_back(cnode_ptr->input(index)->abstract());
    }
    cnode_ptr->set_abstract(std::make_shared<abstract::AbstractTuple>(elem));
  } else {
    cnode_ptr->set_abstract(abstract);
  }
  cnode_ptr->set_fullname_with_scope(fullname_with_scope);
  anfnode_build_map_[node_name] = cnode_ptr;
  return cnode_ptr;
}

bool AnfImporterFromProtobuf::BuildReturnForFuncGraph(const FuncGraphPtr &outputFuncGraph,
                                                      const onnx::GraphProto &importProto, const CNodePtr &cnode_ptr) {
  MS_EXCEPTION_IF_NULL(outputFuncGraph);
  MS_EXCEPTION_IF_NULL(cnode_ptr);
  std::vector<AnfNodePtr> inputs;
  if (importProto.output_size() > 1) {
    inputs.clear();
    auto primitiveT = std::make_unique<schema::PrimitiveT>();
    MS_ASSERT(primitiveT != nullptr);
    primitiveT->value.type = schema::PrimitiveType_MakeTuple;
    std::shared_ptr<PrimitiveC> primitivec_ptr = std::make_shared<PrimitiveC>(primitiveT.release());
    MS_ASSERT(primitivec_ptr != nullptr);
    inputs.push_back(NewValueNode(primitivec_ptr));
    AbstractBasePtrList elem;
    for (int out_size = 0; out_size < importProto.output_size(); ++out_size) {
      const onnx::ValueInfoProto &output_node = importProto.output(out_size);
      const std::string &out_tuple = output_node.name();
      inputs.push_back(anfnode_build_map_[out_tuple]);
      elem.push_back(anfnode_build_map_[out_tuple]->abstract());
    }
    auto maketuple_ptr = outputFuncGraph->NewCNode(inputs);
    maketuple_ptr->set_abstract(std::make_shared<abstract::AbstractTuple>(elem));
    inputs.clear();
    auto primReturn = std::make_unique<schema::PrimitiveT>();
    MS_ASSERT(primReturn != nullptr);
    primReturn->value.type = schema::PrimitiveType_Return;
    std::shared_ptr<PrimitiveC> primitive_return_value_ptr = std::make_shared<PrimitiveC>(primReturn.release());
    MS_ASSERT(primitive_return_value_ptr != nullptr);
    inputs.push_back(NewValueNode(primitive_return_value_ptr));
    inputs.push_back(maketuple_ptr);
    auto return_node = outputFuncGraph->NewCNode(inputs);
    MS_EXCEPTION_IF_NULL(return_node);
    outputFuncGraph->set_return(return_node);
    MS_LOG(INFO) << "Construct funcgraph finined, all success.";
  } else {
    const onnx::ValueInfoProto &output_node = importProto.output(0);
    const onnx::TypeProto &output_typeproto = output_node.type();
    int output_type = output_typeproto.tensor_type().elem_type();
    std::vector<int> output_shape;
    for (int i = 0; i < output_typeproto.tensor_type().shape().dim_size(); ++i) {
      output_shape.push_back(output_typeproto.tensor_type().shape().dim(i).dim_value());
    }
    auto type_ptr = TypeIdToType(kDefaultValueSwitchMap[output_type]);
    auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, output_shape);

    inputs.clear();
    auto primReturn = std::make_unique<schema::PrimitiveT>();
    MS_ASSERT(primReturn != nullptr);
    primReturn->value.type = schema::PrimitiveType_Return;
    std::shared_ptr<PrimitiveC> primitiveTReturnValuePtr = std::make_shared<PrimitiveC>(primReturn.release());
    MS_ASSERT(primitiveTReturnValuePtr != nullptr);
    inputs.push_back(NewValueNode(primitiveTReturnValuePtr));
    inputs.push_back(cnode_ptr);
    auto return_node = outputFuncGraph->NewCNode(inputs);
    MS_EXCEPTION_IF_NULL(return_node);
    return_node->set_abstract(abstract_tensor);
    outputFuncGraph->set_return(return_node);
    MS_LOG(INFO) << "Construct funcgraph finined, all success!";
  }
  return true;
}

bool AnfImporterFromProtobuf::ImportNodesForGraph(const FuncGraphPtr &outputFuncGraph,
                                                  const onnx::GraphProto &importProto,
                                                  const schema::QuantType &quantType) {
  MS_EXCEPTION_IF_NULL(outputFuncGraph);
  MS_LOG(INFO) << "The CNdoe size : " << importProto.node_size();
  CNodePtr cnode_ptr = nullptr;
  for (int i = 0; i < importProto.node_size(); ++i) {
    const onnx::NodeProto &node_proto = importProto.node(i);
    const std::string &node_type = node_proto.op_type();
    if (node_type == kConstantValueNode) {
      if (!BuildValueNodeForFuncGraph(node_proto)) {
        MS_LOG(ERROR) << "Build ValueNode for funcgraph fail at index: : " << i;
        return false;
      }
      continue;
    }
    cnode_ptr = BuildCNodeForFuncGraph(outputFuncGraph, node_proto, quantType);
    if (cnode_ptr == nullptr) {
      MS_LOG(ERROR) << "Build CNode for funcgraph fail at index: : " << i;
      return false;
    }
  }

  BuildReturnForFuncGraph(outputFuncGraph, importProto, cnode_ptr);
  return true;
}

bool AnfImporterFromProtobuf::BuildFuncGraph(const FuncGraphPtr &outputFuncGraph, const onnx::GraphProto &importProto,
                                             const schema::QuantType &quantType) {
  MS_EXCEPTION_IF_NULL(outputFuncGraph);
  GraphDebugInfoPtr debug_info_ptr = outputFuncGraph->debug_info();
  MS_EXCEPTION_IF_NULL(debug_info_ptr);
  if (importProto.has_name()) {
    debug_info_ptr->set_name(importProto.name());
  } else {
    MS_LOG(ERROR) << "FuncGraph under converting has not name!";
  }

  if (!ImportParametersForGraph(outputFuncGraph, importProto)) {
    return false;
  }
  return ImportNodesForGraph(outputFuncGraph, importProto, quantType);
}

bool AnfImporterFromProtobuf::ParseModelConfigureInfo(const onnx::ModelProto &model_proto) {
  if (!model_proto.has_producer_name()) {
    MS_LOG(ERROR) << "Parse model producer name from pb file failed!";
    return false;
  }
  producer_name_ = model_proto.producer_name();

  if (!model_proto.has_model_version()) {
    MS_LOG(ERROR) << "Parse model producer version from pb file failed!";
    return false;
  }
  model_version_ = model_proto.model_version();

  if (!model_proto.has_ir_version()) {
    MS_LOG(ERROR) << "Parse model version from pb file failed!";
    return false;
  }
  ir_version_ = model_proto.ir_version();
  return true;
}

int AnfImporterFromProtobuf::Import(const schema::QuantType &quantType) {
  FuncGraphPtr dstGraph = std::make_shared<mindspore::FuncGraph>();
  MS_EXCEPTION_IF_NULL(dstGraph);
  if (!ParseModelConfigureInfo(*onnx_model_)) {
    MS_LOG(ERROR) << "Parse configuration info for pb file failed!";
  }
  const onnx::GraphProto &graphBuild = onnx_model_->graph();
  if (!BuildFuncGraph(dstGraph, graphBuild, quantType)) {
    MS_LOG(ERROR) << "Build funcgraph failed!";
    func_graph_ = nullptr;
    return RET_ERROR;
  }
  func_graph_ = dstGraph;
  MS_LOG(INFO) << "Parse pb to build FuncGraph Success!";
  return RET_OK;
}

onnx::ModelProto *AnfImporterFromProtobuf::ReadOnnxFromBinary(const std::string &model_path) {
  std::unique_ptr<char[]> onnx_file(new (std::nothrow) char[PATH_MAX]{0});
#ifdef _WIN32
  if (_fullpath(onnx_file.get(), model_path.c_str(), 1024) == nullptr) {
    MS_LOG(ERROR) << "open file failed.";
    return nullptr;
  }
#else
  if (realpath(model_path.c_str(), onnx_file.get()) == nullptr) {
    MS_LOG(ERROR) << "open file failed.";
    return nullptr;
  }
#endif
  int fd = open(onnx_file.get(), O_RDONLY);
  google::protobuf::io::FileInputStream input(fd);
  google::protobuf::io::CodedInputStream code_input(&input);
  code_input.SetTotalBytesLimit(INT_MAX, 536870912);
  auto onnx_model = new onnx::ModelProto;
  bool ret = onnx_model->ParseFromCodedStream(&code_input);
  if (!ret) {
    MS_LOG(ERROR) << "load onnx file failed";
    delete onnx_model;
    return nullptr;
  }
  (void)close(fd);
  MS_LOG(INFO) << "enter ReadProtoFromBinary success!" << std::endl;
  return onnx_model;
}

FuncGraphPtr AnfImporterFromProtobuf::GetResult() { return this->func_graph_; }
}  // namespace mindspore::lite
