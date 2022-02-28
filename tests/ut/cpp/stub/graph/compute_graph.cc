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

#include "graph/compute_graph.h"
#include "graph/detail/attributes_holder.h"
#include "graph/attr_store.h"

ge::ComputeGraph::ComputeGraph(const std::string &name) {}

ge::ComputeGraph::~ComputeGraph() {}

ge::ProtoAttrMap &ge::ComputeGraph::MutableAttrMap() {
    std::shared_ptr<ge::ProtoAttrMap> attrs = std::make_shared<ge::ProtoAttrMap>();
    return *attrs;
}

ge::ConstProtoAttrMap &ge::ComputeGraph::GetAttrMap() const {
    std::shared_ptr<ge::ConstProtoAttrMap> attrs = std::make_shared<ge::ConstProtoAttrMap>();
    return *attrs;
}

ge::NodePtr ge::ComputeGraph::AddNode(ge::OpDescPtr op) {
    ge::NodePtr nodePtr;
    return nodePtr;
}
