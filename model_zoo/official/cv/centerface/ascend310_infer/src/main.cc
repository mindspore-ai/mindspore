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
#include <sys/time.h>
#include <math.h>
#include <gflags/gflags.h>
#include <dirent.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <iosfwd>
#include <vector>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>


#include "include/api/model.h"
#include "include/api/context.h"
#include "include/api/types.h"
#include "include/api/serialization.h"
#include "include/dataset/transforms.h"
#include "include/dataset/vision_ascend.h"
#include "include/dataset/execute.h"
#include "include/dataset/vision.h"
#include "inc/utils.h"

using mindspore::Context;
using mindspore::Serialization;
using mindspore::Model;
using mindspore::Status;
using mindspore::ModelType;
using mindspore::GraphCell;
using mindspore::kSuccess;
using mindspore::MSTensor;
using mindspore::DataType;
using mindspore::dataset::Execute;
using mindspore::dataset::TensorTransform;
using mindspore::dataset::vision::Resize;
using mindspore::dataset::vision::HWC2CHW;
using mindspore::dataset::vision::Normalize;
using mindspore::dataset::vision::Decode;
using mindspore::dataset::vision::SwapRedBlue;
using mindspore::dataset::vision::Rescale;



DEFINE_string(mindir_path, "", "mindir path");
DEFINE_string(dataset_path, ".", "dataset path");
DEFINE_int32(device_id, 0, "device id");
DEFINE_string(cpu_dvpp, "", "cpu or dvpp process");
DEFINE_int32(image_height, 832, "image height");
DEFINE_int32(image_width, 832, "image width");


int Resize_Affine(const MSTensor &input, MSTensor *output) {
  int new_height, new_width;
  const float scale = 0.999;
  auto imgResize = MSTensor();
  std::vector<int64_t> shape = input.Shape();
  new_height = static_cast<int>(shape[0] * scale);
  new_width = static_cast<int>(shape[1] * scale);

  auto resize = Resize({new_height, new_width});
  Execute composeResize(resize);
  composeResize(input, &imgResize);

  cv::Mat src(new_height, new_width, CV_8UC3, imgResize.MutableData());
  cv::Point2f srcPoint2f[3], dstPoint2f[3];
  int max_h_w = std::max(static_cast<int>(shape[0]), static_cast<int>(shape[1]));
  srcPoint2f[0] = cv::Point2f(static_cast<float>(new_width / 2.0), static_cast<float>(new_height / 2.0));
  srcPoint2f[1] = cv::Point2f(static_cast<float>(new_width / 2.0), static_cast<float>((new_height - max_h_w) / 2.0));
  srcPoint2f[2] = cv::Point2f(static_cast<float>((new_width - max_h_w) / 2.0),
                              static_cast<float>((new_height - max_h_w) / 2.0));
  dstPoint2f[0] = cv::Point2f(static_cast<float>(FLAGS_image_width) / 2.0,
                              static_cast<float>(FLAGS_image_height) / 2.0);
  dstPoint2f[1] = cv::Point2f(static_cast<float>(FLAGS_image_width) / 2.0, 0.0);
  dstPoint2f[2] = cv::Point2f(0.0, 0.0);

  cv::Mat warp_mat(2, 3, CV_32FC1);
  warp_mat = cv::getAffineTransform(srcPoint2f, dstPoint2f);
  cv::Mat warp_dst = cv::Mat::zeros(cv::Size(static_cast<int>(FLAGS_image_height), static_cast<int>(FLAGS_image_width)),
                                    src.type());
  cv::warpAffine(src, warp_dst, warp_mat, warp_dst.size());
  void *affine_output;
  unsigned char *affine_address;
  affine_output = output->MutableData();
  affine_address = static_cast<unsigned char *>(affine_output);

  for (int nrow = 0; nrow < warp_dst.rows; nrow++) {
    for (int ncol = 0; ncol < warp_dst.cols; ncol++) {
      *affine_address = warp_dst.at<cv::Vec3b>(nrow, ncol)[0];
      affine_address++;
      *affine_address = warp_dst.at<cv::Vec3b>(nrow, ncol)[1];
      affine_address++;
      *affine_address = warp_dst.at<cv::Vec3b>(nrow, ncol)[2];
      affine_address++;
    }
  }
  return 0;
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (RealPath(FLAGS_mindir_path).empty()) {
    std::cout << "Invalid mindir" << std::endl;
    return 1;
  }

  auto context = std::make_shared<Context>();
  auto ascend310 = std::make_shared<mindspore::Ascend310DeviceInfo>();
  ascend310->SetDeviceID(FLAGS_device_id);
  ascend310->SetPrecisionMode("allow_fp32_to_fp16");
  ascend310->SetOpSelectImplMode("high_precision");
  ascend310->SetBufferOptimizeMode("off_optimize");
  context->MutableDeviceInfo().push_back(ascend310);
  mindspore::Graph graph;
  Serialization::Load(FLAGS_mindir_path, ModelType::kMindIR, &graph);

  Model model;
  Status ret = model.Build(GraphCell(graph), context);
  if (ret != kSuccess) {
    std::cout << "ERROR: Build failed." << std::endl;
    return 1;
  }

  auto decode = Decode();
  auto swapRedBlue = SwapRedBlue();
  auto normalize = Normalize({104.04, 113.985, 119.85}, {73.695, 69.87, 70.89});
  auto hwc2chw = HWC2CHW();
  Execute Decode_BGR({decode, swapRedBlue});
  Execute transform({normalize, hwc2chw});

  auto all_files = GetAllFiles(FLAGS_dataset_path);
  std::map<double, double> costTime_map;
  size_t size = all_files.size();

  for (size_t i = 0; i < size; ++i) {
    struct timeval start = {0};
    struct timeval end = {0};
    double startTimeMs;
    double endTimeMs;
    std::vector<MSTensor> inputs;
    std::vector<MSTensor> outputs;
    std::cout << "Start predict input files:" << all_files[i] << std::endl;
    auto img = MSTensor();
    auto imgDecodeBGR = MSTensor();
    auto image = ReadFileToTensor(all_files[i]);
    Decode_BGR(image, &imgDecodeBGR);

    size_t size_buffer = FLAGS_image_height * FLAGS_image_width * 3;
    MSTensor buffer("affine", DataType::kNumberTypeUInt8,
                    {static_cast<int64_t>(FLAGS_image_height), static_cast<int64_t>(FLAGS_image_width),
                     static_cast<int64_t>(3)}, nullptr, size_buffer);
    Resize_Affine(imgDecodeBGR, &buffer);
    transform(buffer, &img);
    std::vector<MSTensor> model_inputs = model.GetInputs();
    inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(),
                        img.Data().get(), img.DataSize());

    gettimeofday(&start, nullptr);
    ret = model.Predict(inputs, &outputs);
    gettimeofday(&end, nullptr);
    if (ret != kSuccess) {
      std::cout << "Predict " << all_files[i] << " failed." << std::endl;
      return 1;
    }
    startTimeMs = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
    endTimeMs = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
    costTime_map.insert(std::pair<double, double>(startTimeMs, endTimeMs));
    int rst = WriteResult(all_files[i], outputs);
    if (rst != 0) {
        std::cout << "write result failed." << std::endl;
        return rst;
    }
  }
  double average = 0.0;
  int inferCount = 0;

  for (auto iter = costTime_map.begin(); iter != costTime_map.end(); iter++) {
    double diff = 0.0;
    diff = iter->second - iter->first;
    average += diff;
    inferCount++;
  }
  average = average / inferCount;
  std::stringstream timeCost;
  timeCost << "NN inference cost average time: "<< average << " ms of infer_count " << inferCount << std::endl;
  std::cout << "NN inference cost average time: "<< average << "ms of infer_count " << inferCount << std::endl;
  std::string fileName = "./time_Result" + std::string("/test_perform_static.txt");
  std::ofstream fileStream(fileName.c_str(), std::ios::trunc);
  fileStream << timeCost.str();
  fileStream.close();
  costTime_map.clear();
  return 0;
}
