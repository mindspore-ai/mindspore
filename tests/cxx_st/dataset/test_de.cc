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
#include <string>
#include <vector>
#include <cmath>
#include "common/common_test.h"
#include "include/api/types.h"
#include "minddata/dataset/include/minddata_eager.h"
#include "minddata/dataset/include/vision.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "include/api/model.h"
#include "include/api/serialization.h"
#include "include/api/context.h"

using namespace mindspore::api;
using namespace mindspore::dataset::vision;

static void SaveFile(int idx, Buffer buffer, int seq) {
  std::string path = "mnt/disk1/yolo_dvpp_result/result_Files/output" + std::to_string(idx) +
                     "_in_YoloV3-DarkNet_coco_bs_dvpp_" + std::to_string(seq) + ".bin";
  FILE *output_file = fopen(path.c_str(), "wb");
  if (output_file == nullptr) {
    std::cout << "Write file" << path << "failed when fopen" << std::endl;
    return;
  }

  size_t wsize = fwrite(buffer.Data(), buffer.DataSize(), sizeof(int8_t), output_file);
  if (wsize == 0) {
    std::cout << "Write file" << path << " failed when fwrite." << std::endl;
    return;
  }
  fclose(output_file);
  std::cout << "Save file " << path << "length" << buffer.DataSize() << " success." << std::endl;
}

class TestDE : public ST::Common {
 public:
  TestDE() {}
};

TEST_F(TestDE, ResNetPreprocess) {
  std::vector<std::shared_ptr<Tensor>> images;
  MindDataEager::LoadImageFromDir("/home/workspace/mindspore_dataset/imagenet/imagenet_original/val/n01440764",
                                  &images);

  MindDataEager Compose({Decode(), Resize({224, 224}),
                         Normalize({0.485 * 255, 0.456 * 255, 0.406 * 255}, {0.229 * 255, 0.224 * 255, 0.225 * 255}),
                         HWC2CHW()});

  for (auto &img : images) {
    img = Compose(img);
  }

  ASSERT_EQ(images[0]->Shape().size(), 3);
  ASSERT_EQ(images[0]->Shape()[0], 3);
  ASSERT_EQ(images[0]->Shape()[1], 224);
  ASSERT_EQ(images[0]->Shape()[2], 224);
}

TEST_F(TestDE, TestDvpp) {
  std::vector<std::shared_ptr<Tensor>> images;
  MindDataEager::LoadImageFromDir("/root/Dvpp_Unit_Dev/val2014_test/", &images);
  std::vector<uint32_t> crop_size = {224, 224};
  std::vector<uint32_t> resize_size = {256, 256};
  MindDataEager Solo({DvppDecodeResizeCropJpeg(crop_size, resize_size)});
  for (auto &img : images) {
    img = Solo(img);
    ASSERT_EQ(images[0]->Shape().size(), 3);
    if (crop_size.size() == 1) {
      ASSERT_EQ(images[0]->Shape()[0], pow(crop_size[0], 2) 1.5);
    } else {
      ASSERT_EQ(images[0]->Shape()[0], crop_size[0] * crop_size[1] * 1.5);
    }
    ASSERT_EQ(images[0]->Shape()[1], 1);
    ASSERT_EQ(images[0]->Shape()[2], 1);
  }
}

TEST_F(TestDE, TestYoloV3_with_Dvpp) {
  std::vector<std::shared_ptr<Tensor>> images;
  MindDataEager::LoadImageFromDir("/home/lizhenglong/val2014", &images);
  MindDataEager SingleOp({DvppDecodeResizeCropJpeg({416, 416}, {416, 416})});
  constexpr auto yolo_mindir_file = "/home/zhoufeng/yolov3/yolov3_darknet53.mindir";
  Context::Instance().SetDeviceTarget(kDeviceTypeAscend310).SetDeviceID(1);
  auto graph = Serialization::LoadModel(yolo_mindir_file, ModelType::kMindIR);
  Model yolov3((GraphCell(graph)));
  Status ret = yolov3.Build({{kModelOptionInsertOpCfgPath, "/mnt/disk1/yolo_dvpp_result/aipp_resnet50.cfg"}});
  ASSERT_TRUE(ret == SUCCESS);

  std::vector<std::string> names;
  std::vector<std::vector<int64_t>> shapes;
  std::vector<DataType> data_types;
  std::vector<size_t> mem_sizes;
  yolov3.GetOutputsInfo(&names, &shapes, &data_types, &mem_sizes);
  std::vector<Buffer> outputs;
  std::vector<Buffer> inputs;

  int64_t seq = 0;
  for (auto &img : images) {
    img = SingleOp(img);
    std::vector<float> input_shape = {416, 416};
    inputs.clear();
    inputs.emplace_back(img->Data(), img->DataSize());
    inputs.emplace_back(input_shape.data(), input_shape.size() * sizeof(float));
    ret = yolov3.Predict(inputs, &outputs);
    for (size_t i = 0; i < outputs.size(); ++i) {
      SaveFile(i, outputs[i], seq);
    }
    seq++;
    ASSERT_TRUE(ret == SUCCESS);
  }
}
