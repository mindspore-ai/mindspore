# Copyright 2021 Huawei Technologies Co., Ltd
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
import os
import numpy as np

npy_dir = "" #the npy files provided by author, and the directory name is MOT16_POI_train.

for dirpath, dirnames, filenames in os.walk(npy_dir):
    for filename in filenames:
        load_dir = os.path.join(dirpath, filename)
        loadData = np.load(load_dir)
        dirname = "./det/" + filename[ : 8] + "/" + "det/"
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        f = open(dirname+"det.txt", 'a')
        for info in loadData:
            s = ""
            for i, num in enumerate(info):
                if i in (0, 1, 7, 8, 9):
                    s += str(int(num))
                    if i != 9:
                        s += ','
                elif i < 10:
                    s += str(num)
                    s += ','
                else:
                    break
            #print(s)
            f.write(s)
            f.write('\n')
