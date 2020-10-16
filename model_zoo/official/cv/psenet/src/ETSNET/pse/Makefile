# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

mindspore_home = ${MINDSPORE_HOME}
CXXFLAGS = -I include -I ${mindspore_home}/model_zoo/official/cv/psenet -std=c++11 -O3
CXX_SOURCES = adaptor.cpp
opencv_home = ${OPENCV_HOME}
OPENCV = -I$(opencv_home)/include -L$(opencv_home)/lib64 -lopencv_superres -lopencv_ml -lopencv_objdetect \
-lopencv_highgui -lopencv_dnn -lopencv_stitching -lopencv_videostab -lopencv_calib3d -lopencv_videoio \
-lopencv_features2d -lopencv_photo -lopencv_flann -lopencv_shape -lopencv_video -lopencv_imgcodecs -lopencv_imgproc \
-lopencv_core
PYBIND11 = `python -m pybind11 --includes`

LIB_SO = adaptor.so

$(LIB_SO): $(CXX_SOURCES) $(DEPS)
	$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $(CXX_SOURCES) --shared -fPIC $(OPENCV) $(PYBIND11)

clean:
	rm -rf $(LIB_SO)
