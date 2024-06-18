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

#include "src/common/draw/graphviz_graph.h"
#include <set>
#include <algorithm>
#include <sstream>
#include <vector>

namespace mindspore::lite {
std::string Edge::From() const { return from_->name(); }
std::string Edge::name() const { return this->name_; }

void Edge::AppendOutput(const GVNode *to, size_t port) {
  tos_.emplace_back(to);
  to_ports_.emplace_back(port);
}

std::string Edge::Code() const {
  std::ostringstream oss;
  if (from_->type() == kNodeTypeCNode) {
    oss << from_->prefix() << from_->name() << ":O" << from_port_ << " -> ";
  } else {
    oss << from_->prefix() << from_->name() << " -> ";
  }
  auto from_str = oss.str();
  oss.str("");
  for (size_t i = 0; i < tos_.size(); i++) {
    auto to = tos_[i];
    if (to->type() == kNodeTypeCNode) {
      oss << from_str << to->prefix() << to->name() << ":I" << to_ports_[i] << " [label=\"" << info_ << "\"];";
    } else {
      oss << from_str << to->prefix() << to->name() << " [label=\"" << info_ << "\"];";
    }
  }
  return oss.str();
}

GVNode *GVNode::CreateCNode(const std::string &id, const std::string &label, size_t input_size,
                            const std::vector<std::string> &output_names, const std::vector<std::string> &output_infos,
                            bool highlight) {
  auto node = new (std::nothrow) GVNode(id, label, kNodeTypeCNode, input_size, output_names.size(), highlight);
  if (node == nullptr) {
    MS_LOG(ERROR) << "new GVNode failed!";
    return nullptr;
  }
  node->prefix_ = "Node_";
  node->shape_ = "plaintext";
  node->color_ = "cornsilk";
  node->Init(output_names, output_infos);
  return node;
}

GVNode *GVNode::CreateInput(const std::string &id, const std::vector<std::string> &output_names,
                            const std::vector<std::string> &output_infos, bool highlight) {
  auto node = new (std::nothrow) GVNode(id, id, kNodeTypeInput, 0, output_names.size(), highlight);
  if (node == nullptr) {
    MS_LOG(ERROR) << "new GVNode failed!";
    return nullptr;
  }
  node->prefix_ = "Input_";
  node->shape_ = "egg";
  node->Init(output_names, output_infos);
  return node;
}

GVNode *GVNode::CreateOutput(const std::string &id, size_t input_size, bool highlight) {
  auto node = new (std::nothrow) GVNode(id, id, kNodeTypeOutput, input_size, 0, highlight);
  if (node == nullptr) {
    MS_LOG(ERROR) << "new GVNode failed!";
    return nullptr;
  }
  node->prefix_ = "Output_";
  node->shape_ = "egg";
  node->Init({}, {});
  return node;
}

GVNode *GVNode::CreateWeight(const std::string &id, const std::string &label,
                             const std::vector<std::string> &output_names, const std::vector<std::string> &output_infos,
                             bool highlight) {
  auto node = new (std::nothrow) GVNode(id, label, kNodeTypeWeight, 0, output_names.size(), highlight);
  if (node == nullptr) {
    MS_LOG(ERROR) << "new GVNode failed!";
    return nullptr;
  }
  node->prefix_ = "Weight_";
  node->shape_ = "octagon";
  node->color_ = "paleturquoise";
  node->Init(output_names, output_infos);
  return node;
}

GVNode::~GVNode() {
  for (auto output : outputs_) {
    delete output;
  }
  outputs_.clear();
}

void GVNode::Init(const std::vector<std::string> &output_names, const std::vector<std::string> &output_infos) {
  inputs_.reserve(input_size_);
  outputs_.reserve(output_size_);
  if (output_names.size() != output_size_) {
    MS_LOG(ERROR) << "GVNode init failed! output_names size " << output_names.size()
                  << ", output_size_ = " << output_size_;
    return;
  }
  for (size_t i = 0; i < output_size_; i++) {
    auto edge = new (std::nothrow) Edge(output_names[i], this, i, output_infos[i]);
    if (edge == nullptr) {
      MS_LOG(ERROR) << "GVNode init failed! New Edge failed, please check whether memory is enough!";
      return;
    }

    this->outputs_.emplace_back(edge);
  }
}

size_t GVNode::FindCols() const {
  auto max = std::max(input_size_, output_size_);
  auto min = std::min(input_size_, output_size_);
  if (min == 0 || max == 0) {
    return 1;
  }
  size_t ret = max;
  while (ret <= input_size_ * output_size_) {
    if (ret % min == 0) {
      break;
    }
    ret++;
  }
  while (ret <= input_size_ * output_size_) {
    if (ret % max == 0) {
      break;
    }
    ret += min;
  }
  return ret;
}

std::string GVNode::Code() const {
  std::ostringstream oss;
  if (type_ == kNodeTypeCNode) {
    auto bgcolor = highlight_ ? "red" : color_;
    oss << "\t"
        << "\t"
        << "\t"
        << "\t";
    auto indent = oss.str();
    oss.str("");
    auto cols = FindCols();
    oss << "<<table port='core'>" << std::endl;
    oss << indent << "<tr>";
    auto input_cols = input_size_ == 0 ? 0 : cols / input_size_;
    for (size_t i = 0; i < input_size_; i++) {
      oss << "<td align='center' colspan='" << input_cols << "' port='I" << i << "'>I" << i << "</td>";
    }
    oss << "</tr>" << std::endl;
    oss << indent << "<tr><td align='center' colspan='" << cols << "' bgcolor='" << bgcolor << "'>" << label_
        << "</td></tr>" << std::endl;
    oss << indent << "<tr>";
    auto output_cols = output_size_ == 0 ? 0 : cols / output_size_;
    for (size_t i = 0; i < output_size_; i++) {
      oss << "<td align='center' colspan='" << output_cols << "' port='O" << i << "'>O" << i << "</td>";
    }
    oss << "</tr>" << std::endl;
    oss << indent << "</table>>";
  } else {
    oss << "\"" << label_ << "\"";
  }
  auto label = oss.str();
  oss.str("");
  oss << prefix_ << id_ << " [shape=" << shape_;
  oss << ", label=" << label;
  if (type_ != kNodeTypeCNode) {
    oss << ", style=filled, fillcolor=" << color_;
  }
  oss << "];";
  return oss.str();
}

GVGraph::~GVGraph() {
  for (auto *node : nodes_) {
    delete node;
  }
  nodes_.clear();
}

void GVGraph::AppendNode(GVNode *node) {
  if (node == nullptr) {
    return;
  }
  nodes_.emplace_back(node);
  node_map_[node->name()] = node;
}

int GVGraph::Link(const std::string &from_name, size_t from_port, const std::string &to_name, size_t to_port) {
  auto from = node_map_.find(from_name);
  if (from == node_map_.end()) {
    MS_LOG(ERROR) << "Node " << from_name << " is not belong to this graph.";
    return RET_ERROR;
  }

  if (from->second == nullptr) {
    MS_LOG(ERROR) << "from node is null!";
    return RET_ERROR;
  }
  if (from_port >= from->second->output_size()) {
    MS_LOG(ERROR) << "`from_port`(" << from_port << ") out of range of node(" << from_name
                  << ")'s output ports number: " << from->second->output_size();
    return RET_ERROR;
  }
  auto to = node_map_.find(to_name);
  if (to == node_map_.end()) {
    MS_LOG(ERROR) << "Node " << to_name << " is not belong to this graph.";
    return RET_ERROR;
  }
  if (to->second == nullptr) {
    MS_LOG(ERROR) << "to node is null!";
    return RET_ERROR;
  }
  if (to_port >= to->second->input_size()) {
    MS_LOG(ERROR) << "`to_port`(" << to_port << ") out of range of node(" << to_name
                  << ")'s input ports number: " << to->second->input_size();
    return RET_ERROR;
  }
  if (to_port < to->second->size()) {
    MS_LOG(ERROR) << "node(" << to_name << ")'s " << to_port << "th input port already link to "
                  << to->second->inputs()[to_port]->From();
    return RET_ERROR;
  }
  auto edge = from->second->outputs()[from_port];
  edge->AppendOutput(to->second, to_port);
  to->second->AppendInput(edge);
  return RET_OK;
}

std::string GVGraph::Code() const {
  std::ostringstream oss;
  oss << "digraph " << name_ << " {" << std::endl;
  for (auto node : nodes_) {
    oss << node->Code() << std::endl;
  }
  for (auto node : nodes_) {
    for (auto output : node->outputs()) {
      oss << output->Code() << std::endl;
    }
  }
  oss << "}";
  return oss.str();
}
}  // namespace mindspore::lite
