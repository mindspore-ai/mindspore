/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/core/client.h"
#include "common/common.h"
#include "gtest/gtest.h"

#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"
#include "minddata/dataset/engine/opt/pre/getter_pass.h"

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::MsLogLevel::INFO;

class MindDataTestIRNodes : public UT::DatasetOpTesting {
 public:
  MindDataTestIRNodes() = default;
  void SetUp() override { GlobalInit(); }

  // compare the ptr of the nodes in two trees, used to test the deep copy of nodes, will return error code
  // if (ptr1 == ptr2) does not equal to flag or the two tree has different structures (or node names are not the same)
  Status CompareTwoTrees(std::shared_ptr<DatasetNode> root1, std::shared_ptr<DatasetNode> root2, bool flag) {
    CHECK_FAIL_RETURN_UNEXPECTED(root1 != nullptr && root2 != nullptr, "Error in Compare, nullptr.");
    if (((root1.get() == root2.get()) != flag) || (root1->Name() != root2->Name())) {
      std::string err_msg =
        "Expect node ptr " + root1->Name() + (flag ? "==" : "!=") + root2->Name() + " but they aren't!";
      RETURN_STATUS_UNEXPECTED(err_msg);
    }

    size_t num_child = root1->Children().size();

    CHECK_FAIL_RETURN_UNEXPECTED(num_child == root2->Children().size(),
                                 root1->Name() + " has " + std::to_string(num_child) + "child, node #2 has " +
                                   std::to_string(root2->Children().size()) + " child.");

    for (size_t ind = 0; ind < num_child; ind++) {
      RETURN_IF_NOT_OK(CompareTwoTrees(root1->Children()[ind], root2->Children()[ind], flag));
    }
    return Status::OK();
  }

  // print the node's name in post order
  Status PostOrderPrintTree(std::shared_ptr<DatasetNode> ir, std::string &names) {
    RETURN_UNEXPECTED_IF_NULL(ir);
    for (auto child : ir->Children()) {
      RETURN_IF_NOT_OK(PostOrderPrintTree(child, names));
    }
    names += (ir->Name() + "->");
    return Status::OK();
  }
};

TEST_F(MindDataTestIRNodes, MindDataTestSimpleDeepCopy) {
  MS_LOG(INFO) << "Doing MindDataTestIRNodes-MindDataTestSimpleDeepCopy.";

  auto tree1 = RandomData(44)->Repeat(2)->Project({"label"})->Shuffle(10)->Batch(2)->IRNode();

  auto tree2 = tree1->DeepCopy();
  std::string tree_1_names, tree_2_names;

  ASSERT_OK(PostOrderPrintTree(tree1, tree_1_names));
  ASSERT_OK(PostOrderPrintTree(tree2, tree_2_names));

  // expected output for the 2 names:
  // RandomDataset->Repeat->Project->Shuffle->Batch->
  EXPECT_EQ(tree_1_names, tree_2_names);

  ASSERT_OK(CompareTwoTrees(tree1, tree1, true));
  ASSERT_OK(CompareTwoTrees(tree1, tree2, false));

  // verify compare function is correct
  EXPECT_TRUE(CompareTwoTrees(tree2, tree2, false).IsError());
}

TEST_F(MindDataTestIRNodes, MindDataTestZipDeepCopy) {
  MS_LOG(INFO) << "Doing MindDataTestIRNodes-MindDataTestZipDeepCopy.";

  auto branch1 = RandomData(44)->Project({"label"});
  auto branch2 = RandomData(44)->Shuffle(10);

  auto tree1 = Zip({branch1, branch2})->Batch(2)->IRNode();

  auto tree2 = tree1->DeepCopy();
  std::string tree_1_names, tree_2_names;

  ASSERT_OK(PostOrderPrintTree(tree1, tree_1_names));
  ASSERT_OK(PostOrderPrintTree(tree2, tree_2_names));

  // expected output for the 2 names:
  // RandomDataset->Project->RandomDataset->Shuffle->Zip->Batch->
  EXPECT_EQ(tree_1_names, tree_2_names);

  // verify the pointer within the same tree are the same
  ASSERT_OK(CompareTwoTrees(tree1, tree1, true));
  // verify two trees
  ASSERT_OK(CompareTwoTrees(tree1, tree2, false));
}

TEST_F(MindDataTestIRNodes, MindDataTestNodeRemove) {
  MS_LOG(INFO) << "Doing MindDataTestIRNodes-MindDataTestNodeRemove.";

  auto branch1 = RandomData(44)->Project({"label"});
  auto branch2 = ImageFolder("path");
  auto tree = Zip({branch1, branch2})->IRNode();
  /***
   tree looks like this, we will remove node and test its functionalities
            Zip
           /   \
      Project  ImageFolder
        /
    RandomData
  ***/
  auto tree_copy_1 = tree->DeepCopy();
  ASSERT_EQ(tree_copy_1->Children().size(), 2);
  // remove the project in the tree and test
  ASSERT_OK(tree_copy_1->Children()[0]->Remove());  // remove Project from tree
  ASSERT_OK(CompareTwoTrees(tree_copy_1, Zip({RandomData(44), ImageFolder("path")})->IRNode(), false));
  // remove the ImageFolder, a leaf node from the tree
  std::string tree_1_names, tree_2_names;
  ASSERT_OK(PostOrderPrintTree(tree_copy_1, tree_1_names));
  EXPECT_EQ(tree_1_names, "RandomDataset->ImageFolderDataset->Zip->");
  auto tree_copy_2 = tree->DeepCopy();
  ASSERT_EQ(tree_copy_2->Children().size(), 2);
  tree_copy_2->Children()[1]->Remove();
  ASSERT_OK(PostOrderPrintTree(tree_copy_2, tree_2_names));
  EXPECT_EQ(tree_2_names, "RandomDataset->Project->Zip->");
}
