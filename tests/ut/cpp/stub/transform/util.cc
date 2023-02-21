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

#include "include/transform/graph_ir/utils.h"

#include <string>
#include <utility>
#include <map>

#include "securec/include/securec.h"
#include "include/common/utils/convert_utils.h"
#include "include/common/utils/utils.h"
#include "transform/graph_ir/df_graph_manager.h"
#include "transform/graph_ir/op_adapter_map.h"
#include "transform/graph_ir/op_adapter.h"
#include "transform/graph_ir/op_adapter_desc.h"
#include "transform/graph_ir/op_adapter_util.h"
#include "graph/operator.h"

namespace ge {
void Operator::InputRegister(char const *, char const *) {}
void Operator::OutputRegister(char const *, char const *) {}
void Operator::OptionalInputRegister(char const *, char const *) {}
void Operator::DynamicInputRegister(char const *, unsigned int, char const *, bool) {}
void Operator::DynamicOutputRegister(char const *, unsigned int, char const *, bool) {}
std::string Operator::GetOpType() const { return ""; }
}  // namespace ge

namespace mindspore {
namespace transform {
namespace {
const size_t kErrorSize = 0;
static std::map<MeDataType, size_t> datatype_size_map = {
  {MeDataType::kNumberTypeFloat16, sizeof(float) / 2}, {MeDataType::kNumberTypeFloat32, sizeof(float)},  // 1/2 of float
  {MeDataType::kNumberTypeFloat64, sizeof(double)},    {MeDataType::kNumberTypeInt8, sizeof(int8_t)},
  {MeDataType::kNumberTypeInt16, sizeof(int16_t)},     {MeDataType::kNumberTypeInt32, sizeof(int32_t)},
  {MeDataType::kNumberTypeInt64, sizeof(int64_t)},     {MeDataType::kNumberTypeUInt8, sizeof(uint8_t)},
  {MeDataType::kNumberTypeUInt16, sizeof(uint16_t)},   {MeDataType::kNumberTypeUInt32, sizeof(uint32_t)},
  {MeDataType::kNumberTypeUInt64, sizeof(uint64_t)},   {MeDataType::kNumberTypeBool, sizeof(bool)}};

mindspore::HashMap<std::string, OpAdapterDescPtr> adpt_map_ = {
  {kNameCustomOp, std::make_shared<OpAdapterDesc>(std::make_shared<OpAdapter<Operator>>())}};
}  // namespace

size_t TransformUtil::GetDataTypeSize(const MeDataType &type) {
  if (datatype_size_map.find(type) != datatype_size_map.end()) {
    return datatype_size_map[type];
  } else {
    MS_LOG(ERROR) << "Illegal tensor data type!";
    return kErrorSize;
  }
}

AnfGraphPtr GetAnfGraph(uint32_t graph_id) { return nullptr; }
MeTensorPtr ConvertGeTensor(const GeTensorPtr ge_tensor, const ShapeVector &request_dims) { return nullptr; }
MeTensorPtr ConvertGeTensor(const GeTensorPtr &ge_tensor) { return nullptr; }
MeTensorPtr ConvertGeTensor(const GeTensorPtr &tensor, const TypeId &me_type) { return nullptr; }
OpAdapterPtr FindAdapter(const std::string &op_name, bool train) { return nullptr; }

OperatorPtr OpAdapterImpl::GenerateCustomOp(const AnfNodePtr anf) { return nullptr; }
int OpAdapterImpl::setAttr(const OperatorPtr &op, const std::string &attr_key, const ValuePtr &attr_value) { return 0; }
int OpAdapterImpl::setAttr(const OperatorPtr &op, const PrimitivePtr &prim) { return 0; }
int OpAdapterImpl::setAttr(const OperatorPtr &op, const AnfNodePtr &node) { return 0; }
int OpAdapterImpl::setInput(const OperatorPtr &op, int index, const OutHandler &handle) { return 0; }
int OpAdapterImpl::setInput(const OperatorPtr &op, int index, const OperatorPtr &input) { return 0; }
int OpAdapterImpl::setInput(const OperatorPtr &op, int index,
                            const std::shared_ptr<std::vector<OutHandler>> &handler_vec) {
  return 0;
}
void OpAdapterImpl::updateOutputDesc(const OperatorPtr &op, const abstract::BaseShapePtr &shp, const TypePtr &type,
                                     const AnfNodePtr &node) {}
std::map<std::string, ValuePtr> OpAdapterImpl::GetNormalOpAttrList(const OperatorPtr &op,
                                                                   const AnfNodePtr &node) const {
  return {};
}
OutHandler OpAdapterImpl::getOutput(const OperatorPtr &op, int index) {
  OutHandler handler;
  return handler;
}

std::vector<OutHandler> OpAdapterImpl::getOutputs(const OperatorPtr &op) const { return std::vector<OutHandler>(); }

Status OpAdapterImpl::SetOpSubgraphFunc(const OperatorPtr &op, int index,
                                        const std::shared_ptr<std::vector<DfGraph>> &branches) {
  return SUCCESS;
}

Status OpAdapterImpl::SetOpSubgraphFunc(const OperatorPtr &op, const std::shared_ptr<std::vector<DfGraph>> &subgraphs) {
  return SUCCESS;
}

bool IsCustomCNode(const mindspore::AnfNodePtr &node) { return true; }
std::string TransformUtil::NormOpName(const std::string &anf_name) { return ""; }
}  // namespace transform
}  // namespace mindspore