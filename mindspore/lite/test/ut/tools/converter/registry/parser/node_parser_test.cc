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

#include "ut/tools/converter/registry/parser/node_parser_test.h"
#include <map>
#include <string>
#include <vector>
#include "ops/fusion/add_fusion.h"
#include "ops/split.h"
#include "ops/concat.h"
#include "ops/custom.h"

namespace mindspore {
class AddNodeParserTest : public NodeParserTest {
 public:
  AddNodeParserTest() = default;
  ~AddNodeParserTest() = default;
  ops::PrimitiveC *Parse() override {
    auto primc = std::make_unique<ops::AddFusion>();
    return primc.release();
  }
};

class SplitNodeParserTest : public NodeParserTest {
 public:
  SplitNodeParserTest() = default;
  ~SplitNodeParserTest() = default;
  ops::PrimitiveC *Parse() override {
    auto primc = std::make_unique<ops::Split>();
    primc->set_axis(0);
    primc->set_output_num(2);
    return primc.release();
  }
};

class ConcatNodeParserTest : public NodeParserTest {
 public:
  ConcatNodeParserTest() = default;
  ~ConcatNodeParserTest() = default;
  ops::PrimitiveC *Parse() override {
    auto primc = std::make_unique<ops::Concat>();
    primc->set_axis(0);
    return primc.release();
  }
};

// hypothesize custom op called proposal has these attrs : ["image_height", "image_width"].
class CustomProposalNodeParserTest : public NodeParserTest {
 public:
  CustomProposalNodeParserTest() = default;
  ~CustomProposalNodeParserTest() = default;
  ops::PrimitiveC *Parse() override {
    auto primc = std::make_unique<ops::Custom>();
    primc->set_type("Proposal");
    std::map<std::string, std::vector<uint8_t>> custom_attrs;
    std::string height = std::to_string(100);
    std::vector<uint8_t> height_attr(height.begin(), height.end());
    custom_attrs["image_height"] = height_attr;
    std::string width = std::to_string(200);
    std::vector<uint8_t> width_attr(width.begin(), width.end());
    custom_attrs["image_width"] = width_attr;
    primc->set_attr(custom_attrs);
    return primc.release();
  }
};

constexpr auto kAdd = "add";
constexpr auto kSplit = "split";
constexpr auto kConcat = "concat";
constexpr auto kProposal = "proposal";
REGISTER_NODE_PARSER_TEST(kAdd, std::make_shared<AddNodeParserTest>())
REGISTER_NODE_PARSER_TEST(kSplit, std::make_shared<SplitNodeParserTest>())
REGISTER_NODE_PARSER_TEST(kConcat, std::make_shared<ConcatNodeParserTest>())
REGISTER_NODE_PARSER_TEST(kProposal, std::make_shared<CustomProposalNodeParserTest>())
}  // namespace mindspore
