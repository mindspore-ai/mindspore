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
#include "minddata/dataset/kernels/image/random_crop_and_resize_op.h"
#include "minddata/dataset/kernels/image/decode_op.h"
#include "minddata/dataset/engine/datasetops/source/image_folder_op.h"
#include "minddata/dataset/engine/execution_tree.h"


using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::MsLogLevel::INFO;

class MindDataTestTensorOpFusionPass : public UT::DatasetOpTesting {
 public:
  MindDataTestTensorOpFusionPass() = default;
  void SetUp() override { GlobalInit(); }
};

TEST_F(MindDataTestTensorOpFusionPass, RandomCropDecodeResize_fusion_disabled) {
  MS_LOG(INFO) << "Doing RandomCropDecodeResize_fusion";
  std::shared_ptr<ImageFolderOp> ImageFolder(int64_t num_works, int64_t rows, int64_t conns, std::string path,
                                             bool shuf = false, std::shared_ptr<Sampler> sampler = nullptr,
                                             std::map<std::string, int32_t> map = {}, bool decode = false);
  std::shared_ptr<ExecutionTree> Build(std::vector<std::shared_ptr<DatasetOp>> ops);
  auto rcar_op = std::make_shared<RandomCropAndResizeOp>();
  auto decode_op = std::make_shared<DecodeOp>();
  Status rc;
  std::vector<std::shared_ptr<TensorOp>> func_list;
  func_list.push_back(decode_op);
  func_list.push_back(rcar_op);
  std::shared_ptr<MapOp> map_op;
  MapOp::Builder map_decode_builder;
  map_decode_builder.SetInColNames({}).SetOutColNames({}).SetTensorFuncs(func_list).SetNumWorkers(4);
  rc = map_decode_builder.Build(&map_op);
  EXPECT_TRUE(rc.IsOk());
  auto tree = std::make_shared<ExecutionTree>();
  tree = Build({ImageFolder(16, 2, 32, "./", false), map_op});
  rc = tree->SetOptimize(false);
  EXPECT_TRUE(rc);
  rc = tree->Prepare();
  EXPECT_TRUE(rc.IsOk());
  rc = tree->SetOptimize(false);
  EXPECT_TRUE(rc.IsError());
  auto it = tree->begin();
  ++it;
  auto *m_op = &(*it);
  auto tfuncs = static_cast<MapOp *>(m_op)->TFuncs();
  auto func_it = tfuncs.begin();
  EXPECT_EQ((*func_it)->Name(), kDecodeOp);
  ++func_it;
  EXPECT_EQ((*func_it)->Name(), kRandomCropAndResizeOp);
}

TEST_F(MindDataTestTensorOpFusionPass, RandomCropDecodeResize_fusion_enabled) {
  MS_LOG(INFO) << "Doing RandomCropDecodeResize_fusion";
  std::shared_ptr<ImageFolderOp> ImageFolder(int64_t num_works, int64_t rows, int64_t conns, std::string path,
                                             bool shuf = false, std::shared_ptr<Sampler> sampler = nullptr,
                                             std::map<std::string, int32_t> map = {}, bool decode = false);
  std::shared_ptr<ExecutionTree> Build(std::vector<std::shared_ptr<DatasetOp>> ops);
  auto rcar_op = std::make_shared<RandomCropAndResizeOp>();
  auto decode_op = std::make_shared<DecodeOp>();
  Status rc;
  std::vector<std::shared_ptr<TensorOp>> func_list;
  func_list.push_back(decode_op);
  func_list.push_back(rcar_op);
  std::shared_ptr<MapOp> map_op;
  MapOp::Builder map_decode_builder;
  map_decode_builder.SetInColNames({}).SetOutColNames({}).SetTensorFuncs(func_list).SetNumWorkers(4);
  rc = map_decode_builder.Build(&map_op);
  EXPECT_TRUE(rc.IsOk());
  auto tree = std::make_shared<ExecutionTree>();
  tree = Build({ImageFolder(16, 2, 32, "./", false), map_op});
  rc = tree->SetOptimize(true);
  EXPECT_TRUE(rc);
  rc = tree->Prepare();
  EXPECT_TRUE(rc.IsOk());
  rc = tree->SetOptimize(false);
  EXPECT_TRUE(rc.IsError());
  auto it = tree->begin();
  ++it;
  auto *m_op = &(*it);
  auto tfuncs = static_cast<MapOp *>(m_op)->TFuncs();
  auto func_it = tfuncs.begin();
  EXPECT_EQ((*func_it)->Name(), kRandomCropDecodeResizeOp);
  EXPECT_EQ(++func_it, tfuncs.end());
}