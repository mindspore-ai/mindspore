/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_EXPANDERS_UTILS_H_
#define MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_EXPANDERS_UTILS_H_

#include <string>
#include <memory>
#include <vector>
#include <utility>

#include "backend/common/graph_kernel/model/lite_graph.h"
#include "backend/common/graph_kernel/model/node.h"
#include "backend/common/graph_kernel/model/graph_builder.h"

namespace mindspore::graphkernel::expanders {
using inner::NodePtr;
using inner::NodePtrList;
using BaseInfoList = std::vector<inner::NodeBase>;
class Validator;

class OpDesc {
 public:
  inner::LiteGraphPtr Run(const BaseInfoList &inputs, const BaseInfoList &outputs, const inner::DAttrs &attrs,
                          const std::string &processor);
  const std::string &Name() const { return name_; }
  const BaseInfoList &InputsInfo() const { return inputs_info_; }
  const BaseInfoList &OutputsInfo() const { return outputs_info_; }
  const inner::DAttrs &Attrs() const { return attrs_; }
  const std::string &Processor() const { return processor_; }
  virtual ~OpDesc() = default;

 protected:
  virtual void Init() {}
  virtual bool CheckInputs() { return true; }
  virtual NodePtrList Expand(const NodePtrList &inputs) = 0;
  bool CheckOutputs();

  inner::GraphBuilder gb;
  std::string name_;
  BaseInfoList inputs_info_;
  BaseInfoList outputs_info_;
  inner::DAttrs attrs_;
  std::string processor_;
  std::vector<std::unique_ptr<Validator>> validators_;

  friend class OpDescFactory;
};

class Validator {
 public:
  virtual bool Check(const OpDesc &e) = 0;
  virtual ~Validator() = default;
};

class CheckAllFormatsSame : public Validator {
 public:
  bool Check(const OpDesc &e) override {
    const auto &inputs_info = e.InputsInfo();
    if (inputs_info.empty()) {
      return true;
    }
    const auto &fmt_0 = inputs_info[0].format;
    for (size_t i = 1; i < inputs_info.size(); i++) {
      if (inputs_info[i].format != fmt_0) {
        MS_LOG(INFO) << "Unmatched format for op " << e.Name();
        return false;
      }
    }
    return true;
  }
};

class CheckAttr : public Validator {
 public:
  CheckAttr(std::initializer_list<std::string> l) : attrs_(std::move(l)) {}
  virtual ~CheckAttr() = default;
  bool Check(const OpDesc &e) override {
    for (auto &a : attrs_) {
      if (e.Attrs().count(a) == 0) {
        MS_LOG(INFO) << "attr " << a << " does not exist. op " << e.Name();
        return false;
      }
    }
    return true;
  }

 private:
  std::vector<std::string> attrs_;
};

class SupportFormat : public Validator {
 public:
  void AddFormat(std::initializer_list<std::string> l) { (void)formats_.emplace_back(l); }
  bool Check(const OpDesc &e) override {
    for (auto &formats : formats_) {
      if (formats.size() != e.InputsInfo().size()) {
        continue;
      }
      bool match = true;
      for (size_t i = 0; i < formats.size(); i++) {
        if (e.InputsInfo()[i].format != formats[i]) {
          match = false;
          break;
        }
      }
      if (match) {
        return true;
      }
    }
    MS_LOG(INFO) << "unsupported format for op " << e.Name();
    return false;
  }
  virtual ~SupportFormat() = default;

 private:
  std::vector<std::vector<std::string>> formats_;
};

std::vector<int64_t> GetAxisList(const ValuePtr &value);
ShapeVector ExpandDimsInferShape(const ShapeVector &shape, const std::vector<int64_t> &axis);
NodePtr ReluExpand(const inner::GraphBuilder &gb, const NodePtrList &inputs);
NodePtr SigmoidExpand(const inner::GraphBuilder &gb, const NodePtrList &inputs);
NodePtr GeluExpand(const inner::GraphBuilder &gb, const NodePtrList &inputs);
std::vector<int64_t> InferShapeFromFractalnz(const std::vector<int64_t> &fractal);
std::vector<int64_t> GetReducedOriShape(const std::vector<int64_t> &shape, const std::vector<int64_t> &axis);
std::vector<int64_t> ToFracZAxis(const std::vector<int64_t> &ori_shape, const std::vector<int64_t> &ori_axis);
}  // namespace mindspore::graphkernel::expanders
#endif  // MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_EXPANDERS_UTILS_H_
