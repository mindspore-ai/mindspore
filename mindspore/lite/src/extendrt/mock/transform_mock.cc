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
#ifndef TRANSFORM_MOCK_H
#define TRANSFORM_MOCK_H
#include <memory>
#include "transform/graph_ir/op_adapter_map.h"
#include "graph/operator.h"
#include "transform/graph_ir/op_adapter_desc.h"
#include "transform/graph_ir/op_adapter.h"

namespace mindspore {
namespace transform {
namespace {
mindspore::HashMap<std::string, OpAdapterDescPtr> adpt_map_ = {
  {kNameCustomOp, std::make_shared<OpAdapterDesc>(std::make_shared<OpAdapter<Operator>>())}};
}  // namespace

mindspore::HashMap<std::string, OpAdapterDescPtr> &OpAdapterMap::get() { return adpt_map_; }

void OpAdapterImpl::updateOutputDesc(const OperatorPtr &op, const abstract::BaseShapePtr &shp, const TypePtr &type,
                                     const AnfNodePtr &node) {}

int OpAdapterImpl::setAttr(const OperatorPtr &op, const std::string &attr_key, const ValuePtr &attr_value) { return 0; }
int OpAdapterImpl::setAttr(const OperatorPtr &op, const PrimitivePtr &prim) { return 0; }
int OpAdapterImpl::setAttr(const OperatorPtr &op, const AnfNodePtr &node) { return 0; }
int OpAdapterImpl::setAttr(const OperatorPtr &op, const uint32_t &input_idx, const ValuePtr &attr_value) { return 0; }
int OpAdapterImpl::setInput(const OperatorPtr &op, int index, const OutHandler &handle) { return 0; }
int OpAdapterImpl::setInput(const OperatorPtr &op, int index, const OperatorPtr &input) { return 0; }
int OpAdapterImpl::setInput(const OperatorPtr &op, int index,
                            const std::shared_ptr<std::vector<OutHandler>> &handler_vec, bool use_create_byindex_func,
                            size_t dyn_index) {
  return 0;
}
OutHandler OpAdapterImpl::getOutput(const OperatorPtr &op, int index) {
  OutHandler handler;
  return handler;
}
int OpAdapterImpl::getAttr(const OperatorPtr &op, const std::string &attr_key, ValuePtr *attr_value) { return 0; }
int OpAdapterImpl::getAttr(const OperatorPtr &op, uint32_t input_idx, ValuePtr *attr_value) { return 0; }
std::string OpAdapterImpl::GetCustomOpType(const PrimitivePtr &prim) const { return ""; }
std::map<std::string, ValuePtr> OpAdapterImpl::GetOpAttrList(const OperatorPtr &) const { return {}; }
Status OpAdapterImpl::SetOpSubgraphFunc(const OperatorPtr &op, int index,
                                        const std::shared_ptr<std::vector<DfGraph>> &branches) {
  return SUCCESS;
}

Status OpAdapterImpl::SetOpSubgraphFunc(const OperatorPtr &op, const std::shared_ptr<std::vector<DfGraph>> &subgraphs) {
  return SUCCESS;
}
std::vector<OutHandler> OpAdapterImpl::getOutputs(const OperatorPtr &op) const { return std::vector<OutHandler>(); }
OperatorPtr OpAdapterImpl::GenerateCustomOp(const AnfNodePtr anf) { return nullptr; }
std::map<std::string, ValuePtr> OpAdapterImpl::GetNormalOpAttrList(const OperatorPtr &op,
                                                                   const AnfNodePtr &node) const {
  return {};
}

GeDataType TransformUtil::ConvertDataType(const MeDataType &type) { return GeDataType::DT_UNDEFINED; }
GeDataType ConvertDataType(const MeDataType &type) { return GeDataType::DT_UNDEFINED; }
bool IsCustomCNode(const AnfNodePtr &anf) { return false; }
}  // namespace transform
}  // namespace mindspore

#endif
