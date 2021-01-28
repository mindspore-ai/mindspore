/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "common/common.h"
#include "utils/ms_utils.h"
#include "minddata/dataset/core/client.h"
#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/engine/datasetops/source/coco_op.h"
#include "minddata/dataset/engine/datasetops/source/sampler/distributed_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/pk_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/random_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/subset_random_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/weighted_random_sampler.h"
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/status.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"
#include "securec.h"

namespace common = mindspore::common;

using namespace mindspore::dataset;
using mindspore::MsLogLevel::ERROR;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

std::shared_ptr<BatchOp> Batch(int batch_size = 1, bool drop = false, int rows_per_buf = 2);

std::shared_ptr<ExecutionTree> Build(std::vector<std::shared_ptr<DatasetOp>> ops);

class MindDataTestCocoOp : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestCocoOp, TestCocoDetection) {
  // Start with an empty execution tree
  auto my_tree = std::make_shared<ExecutionTree>();
  std::string dataset_path, annotation_path;
  dataset_path = datasets_root_path_ + "/testCOCO/train/";
  annotation_path = datasets_root_path_ + "/testCOCO/annotations/train.json";

  std::string task("Detection");
  std::shared_ptr<CocoOp> my_coco_op;
  CocoOp::Builder builder;
  Status rc = builder.SetDir(dataset_path)
                     .SetFile(annotation_path)
                     .SetTask(task)
                     .Build(&my_coco_op);
  ASSERT_TRUE(rc.IsOk());

  rc = my_tree->AssociateNode(my_coco_op);
  ASSERT_TRUE(rc.IsOk());
  rc = my_tree->AssignRoot(my_coco_op);
  ASSERT_TRUE(rc.IsOk());

  MS_LOG(DEBUG) << "Launch tree and begin iteration.";
  rc = my_tree->Prepare();
  ASSERT_TRUE(rc.IsOk());

  rc = my_tree->Launch();
  ASSERT_TRUE(rc.IsOk());

  // Start the loop of reading tensors from our pipeline
  DatasetIterator di(my_tree);
  TensorRow tensor_list;
  rc = di.FetchNextTensorRow(&tensor_list);
  ASSERT_TRUE(rc.IsOk());

  int row_count = 0;
  while (!tensor_list.empty()) {
    MS_LOG(DEBUG) << "Row display for row #: " << row_count << ".";

    //Display the tensor by calling the printer on it
    for (int i = 0; i < tensor_list.size(); i++) {
      std::ostringstream ss;
      ss << "(" << tensor_list[i] << "): " << *tensor_list[i] << std::endl;
      MS_LOG(DEBUG) << "Tensor print: " << ss.str() << ".";
    }

    rc = di.FetchNextTensorRow(&tensor_list);
    ASSERT_TRUE(rc.IsOk());
    row_count++;
  }

  ASSERT_EQ(row_count, 6);
}

TEST_F(MindDataTestCocoOp, TestCocoStuff) {
  // Start with an empty execution tree
  auto my_tree = std::make_shared<ExecutionTree>();
  std::string dataset_path, annotation_path;
  dataset_path = datasets_root_path_ + "/testCOCO/train/";
  annotation_path = datasets_root_path_ + "/testCOCO/annotations/train.json";

  std::string task("Stuff");
  std::shared_ptr<CocoOp> my_coco_op;
  CocoOp::Builder builder;
  Status rc = builder.SetDir(dataset_path)
    .SetFile(annotation_path)
    .SetTask(task)
    .Build(&my_coco_op);
  ASSERT_TRUE(rc.IsOk());

  rc = my_tree->AssociateNode(my_coco_op);
  ASSERT_TRUE(rc.IsOk());
  rc = my_tree->AssignRoot(my_coco_op);
  ASSERT_TRUE(rc.IsOk());

  MS_LOG(DEBUG) << "Launch tree and begin iteration.";
  rc = my_tree->Prepare();
  ASSERT_TRUE(rc.IsOk());

  rc = my_tree->Launch();
  ASSERT_TRUE(rc.IsOk());

  // Start the loop of reading tensors from our pipeline
  DatasetIterator di(my_tree);
  TensorRow tensor_list;
  rc = di.FetchNextTensorRow(&tensor_list);
  ASSERT_TRUE(rc.IsOk());

  int row_count = 0;
  while (!tensor_list.empty()) {
    MS_LOG(DEBUG) << "Row display for row #: " << row_count << ".";

    //Display the tensor by calling the printer on it
    for (int i = 0; i < tensor_list.size(); i++) {
      std::ostringstream ss;
      ss << "(" << tensor_list[i] << "): " << *tensor_list[i] << std::endl;
      MS_LOG(DEBUG) << "Tensor print: " << ss.str() << ".";
    }

    rc = di.FetchNextTensorRow(&tensor_list);
    ASSERT_TRUE(rc.IsOk());
    row_count++;
  }

  ASSERT_EQ(row_count, 6);
}

TEST_F(MindDataTestCocoOp, TestCocoKeypoint) {
  // Start with an empty execution tree
  auto my_tree = std::make_shared<ExecutionTree>();
  std::string dataset_path, annotation_path;
  dataset_path = datasets_root_path_ + "/testCOCO/train/";
  annotation_path = datasets_root_path_ + "/testCOCO/annotations/key_point.json";

  std::string task("Keypoint");
  std::shared_ptr<CocoOp> my_coco_op;
  CocoOp::Builder builder;
  Status rc = builder.SetDir(dataset_path)
    .SetFile(annotation_path)
    .SetTask(task)
    .Build(&my_coco_op);
  ASSERT_TRUE(rc.IsOk());

  rc = my_tree->AssociateNode(my_coco_op);
  ASSERT_TRUE(rc.IsOk());
  rc = my_tree->AssignRoot(my_coco_op);
  ASSERT_TRUE(rc.IsOk());

  MS_LOG(DEBUG) << "Launch tree and begin iteration.";
  rc = my_tree->Prepare();
  ASSERT_TRUE(rc.IsOk());

  rc = my_tree->Launch();
  ASSERT_TRUE(rc.IsOk());

  // Start the loop of reading tensors from our pipeline
  DatasetIterator di(my_tree);
  TensorRow tensor_list;
  rc = di.FetchNextTensorRow(&tensor_list);
  ASSERT_TRUE(rc.IsOk());

  int row_count = 0;
  while (!tensor_list.empty()) {
    MS_LOG(DEBUG) << "Row display for row #: " << row_count << ".";

    //Display the tensor by calling the printer on it
    for (int i = 0; i < tensor_list.size(); i++) {
      std::ostringstream ss;
      ss << "(" << tensor_list[i] << "): " << *tensor_list[i] << std::endl;
      MS_LOG(DEBUG) << "Tensor print: " << ss.str() << ".";
    }
    rc = di.FetchNextTensorRow(&tensor_list);
    ASSERT_TRUE(rc.IsOk());
    row_count++;
  }

  ASSERT_EQ(row_count, 2);
}

TEST_F(MindDataTestCocoOp, TestCocoPanoptic) {
  // Start with an empty execution tree
  auto my_tree = std::make_shared<ExecutionTree>();
  std::string dataset_path, annotation_path;
  dataset_path = datasets_root_path_ + "/testCOCO/train/";
  annotation_path = datasets_root_path_ + "/testCOCO/annotations/panoptic.json";

  std::string task("Panoptic");
  std::shared_ptr<CocoOp> my_coco_op;
  CocoOp::Builder builder;
  Status rc = builder.SetDir(dataset_path)
    .SetFile(annotation_path)
    .SetTask(task)
    .Build(&my_coco_op);
  ASSERT_TRUE(rc.IsOk());

  rc = my_tree->AssociateNode(my_coco_op);
  ASSERT_TRUE(rc.IsOk());
  rc = my_tree->AssignRoot(my_coco_op);
  ASSERT_TRUE(rc.IsOk());

  MS_LOG(DEBUG) << "Launch tree and begin iteration.";
  rc = my_tree->Prepare();
  ASSERT_TRUE(rc.IsOk());

  rc = my_tree->Launch();
  ASSERT_TRUE(rc.IsOk());

  // Start the loop of reading tensors from our pipeline
  DatasetIterator di(my_tree);
  TensorRow tensor_list;
  rc = di.FetchNextTensorRow(&tensor_list);
  ASSERT_TRUE(rc.IsOk());

  int row_count = 0;
  while (!tensor_list.empty()) {
    MS_LOG(DEBUG) << "Row display for row #: " << row_count << ".";

    //Display the tensor by calling the printer on it
    for (int i = 0; i < tensor_list.size(); i++) {
      std::ostringstream ss;
      ss << "(" << tensor_list[i] << "): " << *tensor_list[i] << std::endl;
      MS_LOG(DEBUG) << "Tensor print: " << ss.str() << ".";
    }
    rc = di.FetchNextTensorRow(&tensor_list);
    ASSERT_TRUE(rc.IsOk());
    row_count++;
  }

  ASSERT_EQ(row_count, 2);
}
