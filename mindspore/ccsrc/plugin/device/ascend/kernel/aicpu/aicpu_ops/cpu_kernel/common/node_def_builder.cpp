/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * Description: tensorflow's kernel info
 */
#include "cpu_kernel/common/node_def_builder.h"
#include <memory>
#include <vector>
#include "cpu_kernel/common/cpu_kernel_utils.h"

namespace aicpu {
std::shared_ptr<NodeDef> NodeDefBuilder::CreateNodeDef() {
    return CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
}

NodeDefBuilder::NodeDefBuilder(NodeDef *nodeDef, std::string name, std::string opName) {
    nodeDef_ = nodeDef;
    name_ = name;
    nodeDef_->SetOpType(opName);
}

void NodeDefBuilder::BuildNodeFromInputOutputNode(const InputOutputNode& node, bool isInput) {
    std::shared_ptr<Tensor> tensor;
    if (isInput) {
        tensor = nodeDef_->AddInputs();
    } else {
        tensor = nodeDef_->AddOutputs();
    }
    aicpu::CpuKernelUtils::SetTensorName(node.node, tensor);
    tensor->SetDataType(node.dType);
    auto shape = tensor->GetTensorShape();
    shape->SetDimSizes(node.dims);
    shape->SetFormat(node.format);
    int64_t dataSize = 1;
    for (size_t i = 0; i < node.dims.size(); i++) {
        dataSize = dataSize * node.dims[i];
    }
    dataSize = dataSize * GetSizeByDataType(node.dType);
    if (node.dims.empty()) {
        dataSize = GetSizeByDataType(node.dType);
    }
    if (node.data == nullptr) {
        dataSize = 0;
    }
    tensor->SetDataSize(dataSize);
    tensor->SetData(node.data);
}

NodeDefBuilder& NodeDefBuilder::Input(const InputOutputNode& input) {
    BuildNodeFromInputOutputNode(input, true);
    return *this;
}

NodeDefBuilder& NodeDefBuilder::Output(const InputOutputNode& output) {
    BuildNodeFromInputOutputNode(output, false);
    return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, int32_t value) {
    auto attr = CpuKernelUtils::CreateAttrValue();
    attr->SetInt(value);
    nodeDef_->AddAttrs(name, attr.get());
    return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, int64_t value) {
    auto attr = CpuKernelUtils::CreateAttrValue();
    attr->SetInt(value);
    nodeDef_->AddAttrs(name, attr.get());
    return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, float value) {
    auto attr = CpuKernelUtils::CreateAttrValue();
    attr->SetFloat(value);
    nodeDef_->AddAttrs(name, attr.get());
    return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, double value) {
    auto attr = CpuKernelUtils::CreateAttrValue();
    attr->SetFloat(value);
    nodeDef_->AddAttrs(name, attr.get());
    return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, bool value) {
    auto attr = CpuKernelUtils::CreateAttrValue();
    attr->SetBool(value);
    nodeDef_->AddAttrs(name, attr.get());
    return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, aicpu::DataType value) {
    auto attr = CpuKernelUtils::CreateAttrValue();
    attr->SetDataType(value);
    nodeDef_->AddAttrs(name, attr.get());
    return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, const std::vector<bool> &value) {
    auto attr = CpuKernelUtils::CreateAttrValue();
    attr->SetListBool(value);
    nodeDef_->AddAttrs(name, attr.get());
    return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, const std::string &value) {
    auto attr = CpuKernelUtils::CreateAttrValue();
    attr->SetString(value);
    nodeDef_->AddAttrs(name, attr.get());
    return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, const std::vector<std::string> &value) {
    auto attr = CpuKernelUtils::CreateAttrValue();
    attr->SetListString(value);
    nodeDef_->AddAttrs(name, attr.get());
    return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, const std::vector<int64_t> &value) {
    auto attr = CpuKernelUtils::CreateAttrValue();
    attr->SetListInt(value);
    nodeDef_->AddAttrs(name, attr.get());
    return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, const std::vector<std::vector<int64_t>> &value) {
    auto attr = CpuKernelUtils::CreateAttrValue();
    attr->SetListListInt(value);
    nodeDef_->AddAttrs(name, attr.get());
    return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, const std::vector<float> &value) {
    auto attr = CpuKernelUtils::CreateAttrValue();
    attr->SetListFloat(value);
    nodeDef_->AddAttrs(name, attr.get());
    return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, const std::vector<aicpu::DataType> &value) {
    auto attr = CpuKernelUtils::CreateAttrValue();
    attr->SetListDataType(value);
    nodeDef_->AddAttrs(name, attr.get());
    return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, const std::vector<int64_t> &dims, std::string type) {
    if (type == "shape") {
        auto shape = CpuKernelUtils::CreateAttrValue();
        auto value = CpuKernelUtils::CreateTensorShape();
        value->SetDimSizes(dims);
        shape->SetTensorShape(value.get());
        nodeDef_->AddAttrs(name, shape.get());
    }
    return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, const std::vector<std::vector<int64_t>> &shapeLists,
                                     std::string type) {
    if (type == "shape_list") {
        auto shapeItems = CpuKernelUtils::CreateAttrValue();
        for (size_t i = 0; i < shapeLists.size(); i++) {
            auto value = shapeItems->AddListTensorShape();
            value->SetDimSizes(shapeLists[i]);
        }
        nodeDef_->AddAttrs(name, shapeItems.get());
    }
    return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, aicpu::Tensor *tensor) {
    auto attr = CpuKernelUtils::CreateAttrValue();
    attr->SetTensor(tensor);
    nodeDef_->AddAttrs(name, attr.get());
    return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, std::vector<aicpu::Tensor *> &tensors) {
    auto attr = CpuKernelUtils::CreateAttrValue();
    attr->SetListTensor(tensors);
    nodeDef_->AddAttrs(name, attr.get());
    return *this;
}
}
