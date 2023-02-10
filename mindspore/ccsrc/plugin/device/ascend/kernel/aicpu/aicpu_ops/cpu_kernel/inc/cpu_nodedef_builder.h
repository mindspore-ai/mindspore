/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 * Description: api of the nodedef builder
 */

#ifndef CPU_NODEDEF_BUILDER_H
#define CPU_NODEDEF_BUILDER_H

#include <memory>
#include <string>
#include <vector>
#include "cpu_kernel/inc/cpu_ops_kernel.h"

namespace aicpu {
class AICPU_VISIBILITY NodeDefBuilder {
 public:
  struct InputOutputNode {
    std::string node;
    aicpu::DataType dType;
    std::vector<int64_t> dims;
    void *data;
    aicpu::Format format;
  };

  static std::shared_ptr<NodeDef> CreateNodeDef();

  NodeDefBuilder(NodeDef *nodeDef, std::string name, std::string opName);

  NodeDefBuilder &Input(const InputOutputNode &input);

  NodeDefBuilder &Output(const InputOutputNode &output);

  NodeDefBuilder &Attr(std::string name, int32_t value);

  NodeDefBuilder &Attr(std::string name, int64_t value);

  NodeDefBuilder &Attr(std::string name, float value);

  NodeDefBuilder &Attr(std::string name, double value);

  NodeDefBuilder &Attr(std::string name, bool value);

  NodeDefBuilder &Attr(std::string name, aicpu::DataType value);

  NodeDefBuilder &Attr(std::string name, const std::vector<bool> &value);

  NodeDefBuilder &Attr(std::string name, const std::string &value);

  NodeDefBuilder &Attr(std::string name, const std::vector<std::string> &value);

  NodeDefBuilder &Attr(std::string name, const std::vector<int64_t> &value);

  NodeDefBuilder &Attr(std::string name, const std::vector<float> &value);

  NodeDefBuilder &Attr(std::string name, const std::vector<aicpu::DataType> &value);

  NodeDefBuilder &Attr(std::string name, const std::vector<int64_t> &shape, std::string type);

  NodeDefBuilder &Attr(std::string name, const std::vector<std::vector<int64_t>> &shapeLists, std::string type);

  NodeDefBuilder &Attr(std::string name, aicpu::Tensor *tensor);

  NodeDefBuilder &Attr(std::string name, std::vector<aicpu::Tensor *> &tensors);

 private:
  void BuildNodeFromInputOutputNode(const InputOutputNode &node, bool isInput);

  NodeDef *nodeDef_;

  std::string name_;

  std::string opName_;
};
}  // namespace aicpu

#endif
