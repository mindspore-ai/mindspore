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

#include <memory>
#include <string>
#include "common/common.h"
#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"
#include "minddata/dataset/engine/ir/datasetops/skip_node.h"
#include "minddata/dataset/engine/ir/datasetops/take_node.h"
#include "minddata/dataset/engine/ir/datasetops/repeat_node.h"
#include "minddata/dataset/include/datasets.h"

using namespace mindspore::dataset;

class MindDataTestTreeModifying : public UT::DatasetOpTesting {
 public:
  MindDataTestTreeModifying() = default;
};

TEST_F(MindDataTestTreeModifying, AppendChild) {
  MS_LOG(INFO) << "Doing MindDataTestTreeModifying-AppendChild";
  /*
  * Input tree:
  *      ds4
  *     /   \
  *   ds3   ds2
  *     |
  *    ds1
  *
  * ds4->AppendChild(ds6) yields this tree
  *
  *      _ ds4 _
  *     /   |   \
  *   ds3  ds2  ds6
  *    |
  *   ds1
  */
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds1 = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));
  std::shared_ptr<Dataset> ds2 = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));
  std::shared_ptr<Dataset> ds6 = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));
  std::shared_ptr<Dataset> ds3 = ds1->Take(10);
  std::shared_ptr<Dataset> ds4 = ds3->Concat({ds2});
  Status rc;

  std::shared_ptr<DatasetNode> root = ds4->IRNode();
  auto ir_tree = std::make_shared<TreeAdapter>();
  rc = ir_tree->Compile(root);  // Compile adds a new RootNode to the top of the tree
  EXPECT_EQ(rc, Status::OK());
  // Descend two levels as Compile adds the root node and the epochctrl node on top of ds4
  std::shared_ptr<DatasetNode> ds4_node = ir_tree->RootIRNode()->Children()[0]->Children()[0];
  // You can inspect the plan by sending *ir_tree->RootIRNode() to std::cout
  std::shared_ptr<DatasetNode> node_to_insert = ds6->IRNode();
  rc = ds4_node->AppendChild(node_to_insert);
  EXPECT_EQ(rc, Status::OK());
  EXPECT_TRUE( ds4_node->Children()[2] == node_to_insert);
}

TEST_F(MindDataTestTreeModifying, InsertChildAt01) {
  MS_LOG(INFO) << "Doing MindDataTestTreeModifying-InsertChildAt01";
  /*
   * Input tree:
   *      ds4
   *     /   \
   *   ds3   ds2
   *    |     |
   *   ds1   ds5
   *
   * Case 1: ds4->InsertChildAt(1, ds6) yields this tree
   *
   *      _ ds4 _
   *     /   |   \
   *   ds3  ds6  ds2
   *    |         |
   *   ds1       ds5
   *
   * Case 2: ds4->InsertChildAt(0, ds6) yields this tree
   *
   *      _ ds4 _
   *     /   |   \
   *   ds6  ds3  ds2
   *         |    |
   *        ds1  ds5
   *
   * Case 3: ds4->InsertChildAt(2, ds6) yields this tree
   *
   *      _ ds4 _
   *     /   |   \
   *   ds3  ds2  ds6
   *    |    |
   *   ds1  ds5
   *
   */
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds1 = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));
  std::shared_ptr<Dataset> ds3 = ds1->Take(10);
  std::shared_ptr<Dataset> ds5 = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));
  std::shared_ptr<Dataset> ds2 = ds5->Repeat(4);
  std::shared_ptr<Dataset> ds4 = ds3->Concat({ds2});
  Status rc;
  std::shared_ptr<DatasetNode> root = ds4->IRNode();
  auto ir_tree = std::make_shared<TreeAdapter>();

  // Case 1:
  rc = ir_tree->Compile(root);  // Compile adds a new RootNode to the top of the tree
  EXPECT_EQ(rc, Status::OK());
  // Descend two levels as Compile adds the root node and the epochctrl node on top of ds4
  std::shared_ptr<DatasetNode> ds4_node = ir_tree->RootIRNode()->Children()[0]->Children()[0];
  std::shared_ptr<Dataset> ds6 = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));
  std::shared_ptr<DatasetNode> ds6_to_insert = ds6->IRNode();
  std::shared_ptr<DatasetNode> ds2_node = ds4_node->Children()[1];
  rc = ds4_node->InsertChildAt(1, ds6_to_insert);
  EXPECT_EQ(rc, Status::OK());
  EXPECT_TRUE( ds4_node->Children()[1] == ds6_to_insert);
  EXPECT_TRUE( ds4_node->Children()[2] == ds2_node);

  // Case 2:
  rc = ir_tree->Compile(root);  // Compile adds a new RootNode to the top of the tree
  EXPECT_EQ(rc, Status::OK());
  // Descend two levels as Compile adds the root node and the epochctrl node on top of ds4
  ds4_node = ir_tree->RootIRNode()->Children()[0]->Children()[0];
  ds6 = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));
  ds6_to_insert = ds6->IRNode();
  std::shared_ptr<DatasetNode> ds3_node = ds4_node->Children()[0];
  rc = ds4_node->InsertChildAt(0, ds6_to_insert);
  EXPECT_EQ(rc, Status::OK());
  EXPECT_TRUE( ds4_node->Children()[0] == ds6_to_insert);
  EXPECT_TRUE( ds4_node->Children()[1] == ds3_node);

  // Case 3:
  rc = ir_tree->Compile(root);  // Compile adds a new RootNode to the top of the tree
  EXPECT_EQ(rc, Status::OK());
  // Descend two levels as Compile adds the root node and the epochctrl node on top of ds4
  ds4_node = ir_tree->RootIRNode()->Children()[0]->Children()[0];
  ds6 = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));
  ds6_to_insert = ds6->IRNode();
  rc = ds4_node->InsertChildAt(2, ds6_to_insert);
  EXPECT_EQ(rc, Status::OK());
  EXPECT_TRUE( ds4_node->Children()[2] == ds6_to_insert);
}

TEST_F(MindDataTestTreeModifying, InsertChildAt04) {
  MS_LOG(INFO) << "Doing MindDataTestTreeModifying-InsertChildAt04";

  /*
   * Input tree:
   *      ds4
   *     /   \
   *   ds3   ds2
   *    |     |
   *   ds1   ds5
   *
   */
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds1 = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));
  std::shared_ptr<Dataset> ds3 = ds1->Take(10);
  std::shared_ptr<Dataset> ds5 = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));
  std::shared_ptr<Dataset> ds2 = ds5->Repeat(4);
  std::shared_ptr<Dataset> ds4 = ds3->Concat({ds2});
  Status rc;
  std::shared_ptr<DatasetNode> root = ds4->IRNode();
  auto ir_tree = std::make_shared<TreeAdapter>();

  // Case 4: ds4->InsertChildAt(3, ds6) raises an error
  rc = ir_tree->Compile(root);  // Compile adds a new RootNode to the top of the tree
  EXPECT_EQ(rc, Status::OK());
  // Descend two levels as Compile adds the root node and the epochctrl node on top of ds4
  std::shared_ptr<DatasetNode> ds4_node = ir_tree->RootIRNode()->Children()[0]->Children()[0];
  std::shared_ptr<Dataset> ds6 = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));
  std::shared_ptr<DatasetNode> ds6_to_insert = ds6->IRNode();
  std::shared_ptr<DatasetNode> ds3_node = ds4_node->Children()[0];
  std::shared_ptr<DatasetNode> ds2_node = ds4_node->Children()[1];
  rc = ds4_node->InsertChildAt(3, ds6_to_insert);
  EXPECT_NE(rc, Status::OK());
  EXPECT_TRUE( ds4_node->Children()[0] == ds3_node);
  EXPECT_TRUE( ds4_node->Children()[1] == ds2_node);

  // Case 5: ds4->InsertChildAt(-1, ds6) raises an error
  rc = ir_tree->Compile(root);  // Compile adds a new RootNode to the top of the tree
  EXPECT_EQ(rc, Status::OK());
  // Descend two levels as Compile adds the root node and the epochctrl node on top of ds4
  ds4_node = ir_tree->RootIRNode()->Children()[0]->Children()[0];
  ds6 = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));
  ds6_to_insert = ds6->IRNode();
  ds3_node = ds4_node->Children()[0];
  ds2_node = ds4_node->Children()[1];
  rc = ds4_node->InsertChildAt(-1, ds6_to_insert);
  EXPECT_NE(rc, Status::OK());
  EXPECT_TRUE( ds4_node->Children()[0] == ds3_node);
  EXPECT_TRUE( ds4_node->Children()[1] == ds2_node);
}

TEST_F(MindDataTestTreeModifying, InsertAbove01) {
  MS_LOG(INFO) << "Doing MindDataTestTreeModifying-InsertAbove01";
  /*
   * Insert the input <node> above this node
   * Input tree:
   *       ds4
   *      /   \
   *     ds3  ds2
   *      |
   *     ds1
   *
   * Case 1: If we want to insert a new node ds5 between ds4 and ds3, use
   *           ds3->InsertAbove(ds5)
   *
   *       ds4
   *      /   \
   *     ds5  ds2
   *      |
   *     ds3
   *      |
   *     ds1
   *
   * Case 2: Likewise, ds2->InsertAbove(ds6) yields
   *
   *       ds4
   *      /   \
   *     ds3  ds6
   *      |    |
   *     ds1  ds2
   *
   * Case 3: We can insert a new node between ds3 and ds1 by ds1->InsertAbove(ds7)
   *
   *       ds4
   *      /   \
   *     ds3  ds2
   *      |
   *     ds7
   *      |
   *     ds1
   *
   */
  // Case 1
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds1 = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));
  std::shared_ptr<Dataset> ds2 = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));
  std::shared_ptr<Dataset> ds3 = ds1->Take(10);
  std::shared_ptr<Dataset> ds4 = ds3->Concat({ds2});
  Status rc;

  std::shared_ptr<DatasetNode> root = ds4->IRNode();
  auto ir_tree = std::make_shared<TreeAdapter>();
  rc = ir_tree->Compile(root);  // Compile adds a new RootNode to the top of the tree
  EXPECT_EQ(rc, Status::OK());
  // Descend two levels as Compile adds the root node and the epochctrl node on top of ds4
  std::shared_ptr<DatasetNode> ds4_node = ir_tree->RootIRNode()->Children()[0]->Children()[0];
  std::shared_ptr<DatasetNode> ds3_node = ds4_node->Children()[0];
  std::shared_ptr<SkipNode> ds5_to_insert = std::make_shared<SkipNode>(nullptr, 1);
  rc = ds3_node->InsertAbove(ds5_to_insert);
  EXPECT_EQ(rc, Status::OK());
  EXPECT_TRUE(ds5_to_insert->Children()[0] == ds3_node);
  EXPECT_TRUE( ds4_node->Children()[0] == ds5_to_insert);
}

TEST_F(MindDataTestTreeModifying, InsertAbove02) {
  MS_LOG(INFO) << "Doing MindDataTestTreeModifying-InsertAbove02";

  // Case 2
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds1 = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));
  std::shared_ptr<Dataset> ds2 = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));
  std::shared_ptr<Dataset> ds3 = ds1->Take(10);
  std::shared_ptr<Dataset> ds4 = ds3 + ds2;
  Status rc;

  std::shared_ptr<DatasetNode> root = ds4->IRNode();
  auto ir_tree = std::make_shared<TreeAdapter>();
  rc = ir_tree->Compile(root);  // Compile adds a new RootNode to the top of the tree
  EXPECT_EQ(rc, Status::OK());
  // Descend two levels as Compile adds the root node and the epochctrl node on top of ds4
  std::shared_ptr<DatasetNode> ds4_node = ir_tree->RootIRNode()->Children()[0]->Children()[0];
  std::shared_ptr<DatasetNode> ds2_node = ds4_node->Children()[1];
  std::shared_ptr<TakeNode> ds6_to_insert = std::make_shared<TakeNode>(nullptr, 12);
  rc = ds2_node->InsertAbove(ds6_to_insert);
  EXPECT_EQ(rc, Status::OK());
  EXPECT_TRUE(ds6_to_insert->Children()[0] == ds2_node);
  EXPECT_TRUE( ds4_node->Children()[1] == ds6_to_insert);
}

TEST_F(MindDataTestTreeModifying, InsertAbove03) {
  MS_LOG(INFO) << "Doing MindDataTestTreeModifying-InsertAbove03";

  // Case 3
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds1 = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));
  std::shared_ptr<Dataset> ds2 = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));
  std::shared_ptr<Dataset> ds3 = ds1->Take(10);
  std::shared_ptr<Dataset> ds4 = ds3->Concat({ds2});
  Status rc;

  std::shared_ptr<DatasetNode> root = ds4->IRNode();
  auto ir_tree = std::make_shared<TreeAdapter>();
  rc = ir_tree->Compile(root);  // Compile adds a new RootNode to the top of the tree
  EXPECT_EQ(rc, Status::OK());
  // Descend two levels as Compile adds the root node and the epochctrl node on top of ds4
  std::shared_ptr<DatasetNode> ds4_node = ir_tree->RootIRNode()->Children()[0]->Children()[0];
  std::shared_ptr<DatasetNode> ds3_node = ds4_node->Children()[0];
  std::shared_ptr<DatasetNode> ds1_node = ds3_node->Children()[0];
  std::shared_ptr<RepeatNode> ds7_to_insert = std::make_shared<RepeatNode>(nullptr, 3);
  rc = ds1_node->InsertAbove(ds7_to_insert);
  EXPECT_TRUE(ds7_to_insert->Children()[0] == ds1_node);
  EXPECT_TRUE( ds3_node->Children()[0] == ds7_to_insert);
}

TEST_F(MindDataTestTreeModifying, Drop01) {
  MS_LOG(INFO) << "Doing MindDataTestTreeModifying-Drop01";
  /*
   * Drop() detaches this node from the tree it is in. Calling Drop() from a standalone node is a no-op.
   *
   * Input tree:
   *       ds10
   *      /    \
   *    ds9    ds6
   *     |   /  |  \
   *    ds8 ds5 ds4 ds1
   *     |     /  \
   *    ds7  ds3  ds2
   *
   * Case 1: When the node has no child and no sibling, Drop() detaches the node from its tree.
   *
   *   ds7->Drop() yields the tree below:
   *
   *       ds10
   *      /    \
   *    ds9    ds6
   *     |   /  |  \
   *    ds8 ds5 ds4 ds1
   *           /  \
   *         ds3  ds2
   *
   * Case 2: When the node has one child and no sibling, Drop() detaches the node from its tree and the node's child
   *         becomes its parent's child.
   *
   *   ds8->Drop() yields the tree below:
   *
   *       ds10
   *      /    \
   *    ds9    ds6
   *     |   /  |  \
   *    ds7 ds5 ds4 ds1
   *           /  \
   *         ds3  ds2
   *
   */
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds7 = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));
  std::shared_ptr<Dataset> ds8 = ds7->Take(20);
  std::shared_ptr<Dataset> ds9 = ds8->Skip(1);
  std::shared_ptr<Dataset> ds3 = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));
  std::shared_ptr<Dataset> ds2 = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));
  std::shared_ptr<Dataset> ds4 = ds3->Concat({ds2});
  std::shared_ptr<Dataset> ds6 = ds4->Take(13);
  std::shared_ptr<Dataset> ds10 = ds9 + ds6;
  Status rc;

  std::shared_ptr<DatasetNode> root = ds10->IRNode();
  auto ir_tree = std::make_shared<TreeAdapter>();

  // Case 1
  rc = ir_tree->Compile(root);  // Compile adds a new RootNode to the top of the tree
  EXPECT_EQ(rc, Status::OK());
  // Descend two levels as Compile adds the root node and the epochctrl node on top of ds4
  std::shared_ptr<DatasetNode> ds10_node = ir_tree->RootIRNode()->Children()[0]->Children()[0];
  std::shared_ptr<DatasetNode> ds9_node = ds10_node->Children()[0];
  std::shared_ptr<DatasetNode> ds8_node = ds9_node->Children()[0];
  std::shared_ptr<DatasetNode> ds7_node = ds8_node->Children()[0];
  rc = ds7_node->Drop();
  EXPECT_EQ(rc, Status::OK());
  // ds8 becomes a childless node
  EXPECT_TRUE(ds8_node->Children().empty());
  EXPECT_TRUE(ds7_node->Children().empty());

  // Case 2
  rc = ir_tree->Compile(root);  // Compile adds a new RootNode to the top of the tree
  EXPECT_EQ(rc, Status::OK());
  // Descend two levels as Compile adds the root node and the epochctrl node on top of ds4
  ds10_node = ir_tree->RootIRNode()->Children()[0]->Children()[0];
  ds9_node = ds10_node->Children()[0];
  ds8_node = ds9_node->Children()[0];
  ds7_node = ds8_node->Children()[0];
  rc = ds8_node->Drop();
  EXPECT_EQ(rc, Status::OK());
  // ds7 becomes a child of ds9
  EXPECT_TRUE(ds9_node->Children()[0] == ds7_node);
  EXPECT_TRUE(ds8_node->Children().empty());
}

TEST_F(MindDataTestTreeModifying, Drop03) {
  MS_LOG(INFO) << "Doing MindDataTestTreeModifying-Drop03";
  /* Case 3: When the node has more than one child and no sibling, Drop() detaches the node from its tree and the node's
   *         children become its parent's children.
   *
   *   When the input tree is
   *       ds10
   *      /    \
   *    ds9    ds6
   *     |      |
   *    ds8    ds4
   *     |    /   \
   *    ds7  ds3  ds2
   *
   *
   *   ds4->Drop() will raise an error because we cannot add the children of an n-ary operator (ds4) to a unary operator
   *   (ds6).
   *
   */
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds7 = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));
  std::shared_ptr<Dataset> ds8 = ds7->Take(20);
  std::shared_ptr<Dataset> ds9 = ds8->Skip(1);
  std::shared_ptr<Dataset> ds3 = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));
  std::shared_ptr<Dataset> ds2 = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));
  std::shared_ptr<Dataset> ds4 = ds3->Concat({ds2});
  std::shared_ptr<Dataset> ds6 = ds4->Take(13);
  std::shared_ptr<Dataset> ds10 = ds9 + ds6;
  Status rc;

  std::shared_ptr<DatasetNode> root = ds10->IRNode();
  auto ir_tree = std::make_shared<TreeAdapter>();
  rc = ir_tree->Compile(root);  // Compile adds a new RootNode to the top of the tree
  EXPECT_EQ(rc, Status::OK());
  // Descend two levels as Compile adds the root node and the epochctrl node on top of ds4
  std::shared_ptr<DatasetNode> ds10_node = ir_tree->RootIRNode()->Children()[0]->Children()[0];
  std::shared_ptr<DatasetNode> ds6_node = ds10_node->Children()[1];
  std::shared_ptr<DatasetNode> ds4_node = ds6_node->Children()[0];
  std::shared_ptr<DatasetNode> ds3_node = ds4_node->Children()[0];
  std::shared_ptr<DatasetNode> ds2_node = ds4_node->Children()[1];
  rc = ds4_node->Drop();
  EXPECT_NE(rc, Status::OK());
}

TEST_F(MindDataTestTreeModifying, Drop04) {
  MS_LOG(INFO) << "Doing MindDataTestTreeModifying-Drop04";
  /* Case 4: When the node has no child but has siblings, Drop() detaches the node from its tree and its siblings will be
   *         squeezed left.
   *
   * Input tree:
   *       ds10
   *      /    \
   *    ds9    ds6
   *     |   /  |  \
   *    ds8 ds5 ds4 ds1
   *     |     /  \
   *    ds7  ds3  ds2
   *
   *   ds5->Drop() yields the tree below:
   *
   *       ds10
   *      /    \
   *    ds9    ds6
   *     |     /  \
   *    ds8   ds4 ds1
   *     |    /  \
   *    ds7 ds3  ds2
   *
   */
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds7 = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));
  std::shared_ptr<Dataset> ds8 = ds7->Take(20);
  std::shared_ptr<Dataset> ds9 = ds8->Skip(1);
  std::shared_ptr<Dataset> ds3 = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));
  std::shared_ptr<Dataset> ds2 = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));
  std::shared_ptr<Dataset> ds4 = ds3->Concat({ds2});
  std::shared_ptr<Dataset> ds5 = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));
  std::shared_ptr<Dataset> ds1 = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));
  std::shared_ptr<Dataset> ds6 = ds5->Concat({ds4, ds1});
  std::shared_ptr<Dataset> ds10 = ds9 + ds6;
  Status rc;

  std::shared_ptr<DatasetNode> root = ds10->IRNode();
  auto ir_tree = std::make_shared<TreeAdapter>();
  rc = ir_tree->Compile(root);  // Compile adds a new RootNode to the top of the tree
  EXPECT_EQ(rc, Status::OK());
  // Descend two levels as Compile adds the root node and the epochctrl node on top of ds4
  std::shared_ptr<DatasetNode> ds10_node = ir_tree->RootIRNode()->Children()[0]->Children()[0];
  std::shared_ptr<DatasetNode> ds6_node = ds10_node->Children()[1];
  std::shared_ptr<DatasetNode> ds5_node = ds6_node->Children()[0];
  std::shared_ptr<DatasetNode> ds4_node = ds6_node->Children()[1];
  EXPECT_TRUE(ds5_node->IsDataSource());
  EXPECT_TRUE(ds6_node->IsNaryOperator());
  rc = ds5_node->Drop();
  EXPECT_EQ(rc, Status::OK());
  EXPECT_TRUE(ds6_node->Children().size() == 2);
  EXPECT_TRUE(ds6_node->Children()[0] == ds4_node);
  EXPECT_TRUE(ds5_node->Children().empty());
}

TEST_F(MindDataTestTreeModifying, Drop05) {
  MS_LOG(INFO) << "Doing MindDataTestTreeModifying-Drop05";
  /*
   * Case 5: When the node has only one child but has siblings, Drop() detaches the node from its tree and the node's
   *         children become its parent's children.
   *
   * Input tree:
   *       ds10
   *      /    \
   *    ds9    ds6
   *     |   /  |  \
   *    ds8 ds5 ds4 ds1
   *     |      |
   *    ds7    ds3
   *
   *   ds4->Drop() yields the tree below:
   *
   *       ds10
   *      /    \
   *    ds9    ds6
   *     |   /  |  \
   *    ds8 ds5 ds3 ds1
   *     |
   *    ds7
   *
   */
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds7 = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));
  std::shared_ptr<Dataset> ds8 = ds7->Take(20);
  std::shared_ptr<Dataset> ds9 = ds8->Skip(1);
  std::shared_ptr<Dataset> ds3 = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));
  std::shared_ptr<Dataset> ds4 = ds3->Skip(1);
  std::shared_ptr<Dataset> ds5 = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));
  std::shared_ptr<Dataset> ds1 = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));
  std::shared_ptr<Dataset> ds6 = ds5->Concat({ds4, ds1});
  std::shared_ptr<Dataset> ds10 = ds9 + ds6;
  Status rc;

  std::shared_ptr<DatasetNode> root = ds10->IRNode();
  auto ir_tree = std::make_shared<TreeAdapter>();
  rc = ir_tree->Compile(root);  // Compile adds a new RootNode to the top of the tree
  EXPECT_EQ(rc, Status::OK());
  // Descend two levels as Compile adds the root node and the epochctrl node on top of ds4
  std::shared_ptr<DatasetNode> ds10_node = ir_tree->RootIRNode()->Children()[0]->Children()[0];
  std::shared_ptr<DatasetNode> ds6_node = ds10_node->Children()[1];
  std::shared_ptr<DatasetNode> ds4_node = ds6_node->Children()[1];
  std::shared_ptr<DatasetNode> ds3_node = ds4_node->Children()[0];
  rc = ds4_node->Drop();
  EXPECT_EQ(rc, Status::OK());
  EXPECT_TRUE(ds6_node->Children().size() == 3);
  EXPECT_TRUE(ds6_node->Children()[1] == ds3_node);
  EXPECT_TRUE(ds4_node->Children().empty());
}

TEST_F(MindDataTestTreeModifying, Drop06) {
  MS_LOG(INFO) << "Doing MindDataTestTreeModifying-Drop06";
  /*
   * Case 6: When the node has more than one child and more than one sibling, Drop() will raise an error.
   *         If we want to drop ds4 from the input tree, ds4->Drop() will not work. We will have to do it
   *         with a combination of Drop(), InsertChildAt()
   *
   * Input tree:
   *       ds10
   *      /    \
   *    ds9    ds6
   *     |   /  |  \
   *    ds8 ds5 ds4 ds1
   *     |     /  \
   *    ds7  ds3  ds2
   *
   * If we want to form this tree below:
   *
   *       ds10
   *      /    \
   *    ds9    ds6_____
   *     |   /  |   |  \
   *    ds8 ds5 ds3 ds2 ds1
   *     |
   *    ds7
   *
   */
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds7 = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));
  std::shared_ptr<Dataset> ds8 = ds7->Take(20);
  std::shared_ptr<Dataset> ds9 = ds8->Skip(1);
  std::shared_ptr<Dataset> ds3 = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));
  std::shared_ptr<Dataset> ds2 = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));
  std::shared_ptr<Dataset> ds4 = ds3->Concat({ds2});
  std::shared_ptr<Dataset> ds5 = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));
  std::shared_ptr<Dataset> ds1 = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));
  std::shared_ptr<Dataset> ds6 = ds5->Concat({ds4, ds1});  // ds1 is put after (ds5, ds4)!!!
  std::shared_ptr<Dataset> ds10 = ds9 + ds6;
  Status rc;

  std::shared_ptr<DatasetNode> root = ds10->IRNode();
  auto ir_tree = std::make_shared<TreeAdapter>();
  rc = ir_tree->Compile(root);  // Compile adds a new RootNode to the top of the tree
  EXPECT_EQ(rc, Status::OK());
  // Descend two levels as Compile adds the root node and the epochctrl node on top of ds4
  std::shared_ptr<DatasetNode> ds10_node = ir_tree->RootIRNode()->Children()[0]->Children()[0];
  std::shared_ptr<DatasetNode> ds6_node = ds10_node->Children()[1];
  std::shared_ptr<DatasetNode> ds4_node = ds6_node->Children()[1];
  rc = ds4_node->Drop();
  EXPECT_NE(rc, Status::OK());
}
