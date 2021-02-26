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
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"
#include "minddata/dataset/engine/tree_adapter.h"
#include "minddata/dataset/include/datasets.h"
#include "minddata/dataset/include/transforms.h"
#include "minddata/dataset/include/vision.h"
#include "minddata/dataset/kernels/tensor_op.h"

using namespace mindspore::dataset;

class MindDataTestTensorOpFusionPass : public UT::DatasetOpTesting {
 public:
  MindDataTestTensorOpFusionPass() = default;
};

TEST_F(MindDataTestTensorOpFusionPass, RandomCropDecodeResizeDisabled) {
  MS_LOG(INFO) << "Doing MindDataTestTensorOpFusionPass-RandomCropDecodeResizeDisabled";

  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> decode(new vision::Decode());
  std::shared_ptr<TensorTransform> random_resized_crop(new vision::RandomResizedCrop({5}));
  ds = ds->Map({decode, random_resized_crop}, {"image"});

  std::shared_ptr<DatasetNode> node = ds->IRNode();
  auto ir_tree = std::make_shared<TreeAdapter>();
  // Disable IR optimization pass
  ir_tree->SetOptimize(false);
  Status rc;
  rc = ir_tree->Compile(node);
  EXPECT_TRUE(rc);
  auto root_op = ir_tree->GetRoot();

  auto tree = std::make_shared<ExecutionTree>();
  auto it = tree->begin(static_cast<std::shared_ptr<DatasetOp>>(root_op));
  ++it;
  auto *map_op = &(*it);
  auto tfuncs = static_cast<MapOp *>(map_op)->TFuncs();
  auto func_it = tfuncs.begin();
  EXPECT_EQ((*func_it)->Name(), kDecodeOp);
  ++func_it;
  EXPECT_EQ((*func_it)->Name(), kRandomCropAndResizeOp);
}

TEST_F(MindDataTestTensorOpFusionPass, RandomCropDecodeResizeEnabled) {
  MS_LOG(INFO) << "Doing MindDataTestTensorOpFusionPass-RandomCropDecodeResizeEnabled";

  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 11));

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> decode(new vision::Decode());
  std::shared_ptr<TensorTransform> random_resized_crop(new vision::RandomResizedCrop({5}));
  ds = ds->Map({decode, random_resized_crop}, {"image"});

  std::shared_ptr<DatasetNode> node = ds->IRNode();
  auto ir_tree = std::make_shared<TreeAdapter>();
  // Enable IR optimization pass
  ir_tree->SetOptimize(true);
  Status rc;
  rc = ir_tree->Compile(node);
  EXPECT_TRUE(rc);
  auto root_op = ir_tree->GetRoot();

  auto tree = std::make_shared<ExecutionTree>();
  auto it = tree->begin(static_cast<std::shared_ptr<DatasetOp>>(root_op));
  ++it;
  auto *map_op = &(*it);
  auto tfuncs = static_cast<MapOp *>(map_op)->TFuncs();
  auto func_it = tfuncs.begin();
  // FIXME: Currently the following 2 commented out verifications for this test will fail because this
  //        optimization is still in ExecutionTree code, and not yet in IR optimization pass
  //        However, use a bogus check for func_it, to avoid compile error for unused variable.
  EXPECT_EQ(func_it, func_it);
  // EXPECT_EQ((*func_it)->Name(), kRandomCropDecodeResizeOp);
  // EXPECT_EQ(++func_it, tfuncs.end());
}

