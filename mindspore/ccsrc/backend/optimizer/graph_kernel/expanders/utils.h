/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_EXPANDERS_UTILS_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_EXPANDERS_UTILS_H_

#include <string>
#include <memory>
#include <vector>

#include "backend/optimizer/graph_kernel/model/lite_graph.h"
#include "backend/optimizer/graph_kernel/model/node.h"

namespace mindspore {
namespace opt {
namespace expanders {
using graphkernel::NodePtrList;
using BaseInfoList = std::vector<graphkernel::NodeBase>;
class Validator;

class OpExpander {
 public:
  graphkernel::LiteGraphPtr Run(const BaseInfoList &inputs, const BaseInfoList &outputs,
                                const graphkernel::DAttrs &attrs, const std::string &processor);
  virtual ~OpExpander() = default;

 protected:
  virtual bool CheckInputs() { return true; }
  virtual NodePtrList Expand() = 0;
  bool CheckOutputs();

  graphkernel::LiteGraph::GraphBuilder gb;
  std::string op_;
  BaseInfoList inputs_info_;
  BaseInfoList outputs_info_;
  graphkernel::DAttrs attrs_;
  std::string processor_;
  std::vector<std::unique_ptr<Validator>> validators_;

  friend class OpExpanderFactory;
  friend class CheckAllFormatsSame;
  friend class CheckAttr;
  friend class SupportFormat;
};

class Validator {
 public:
  virtual bool Check(const OpExpander &e) = 0;
};

class CheckAllFormatsSame : public Validator {
 public:
  bool Check(const OpExpander &e) override {
    if (e.inputs_info_.empty()) return true;
    const auto &fmt_0 = e.inputs_info_[0].format;
    for (size_t i = 1; i < e.inputs_info_.size(); i++) {
      if (e.inputs_info_[i].format != fmt_0) {
        MS_LOG(INFO) << "Unmatched format for op " << e.op_;
        return false;
      }
    }
    return true;
  }
};

class CheckAttr : public Validator {
 public:
  CheckAttr(std::initializer_list<std::string> l) : attrs_(l) {}
  ~CheckAttr() = default;
  bool Check(const OpExpander &e) override {
    for (auto &a : attrs_) {
      if (e.attrs_.count(a) == 0) {
        MS_LOG(INFO) << "attr " << a << " does not exist. op " << e.op_;
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
  void AddFormat(std::initializer_list<std::string> l) { formats_.emplace_back(l); }
  bool Check(const OpExpander &e) override {
    for (auto &formats : formats_) {
      if (formats.size() != e.inputs_info_.size()) {
        continue;
      }
      bool match = true;
      for (size_t i = 0; i < formats.size(); i++) {
        if (e.inputs_info_[i].format != formats[i]) {
          match = false;
          break;
        }
      }
      if (match) {
        return true;
      }
    }
    MS_LOG(INFO) << "unsupported format for op " << e.op_;
    return false;
  }

 private:
  std::vector<std::vector<std::string>> formats_;
};

std::vector<int64_t> GetAxisList(const ValuePtr &value);
ShapeVector ExpandDimsInferShape(const ShapeVector &shape, const std::vector<int64_t> &axis);
}  // namespace expanders
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_EXPANDERS_UTILS_H_
