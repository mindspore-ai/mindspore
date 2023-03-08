/*
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

package com.mindspore;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import com.mindspore.config.DataType;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.Arrays;

/**
 * Model Test
 *
 * @since 2023.2
 */
@RunWith(JUnit4.class)
public class MSTensorTest {
    @Test
    public void testCreateTensor() {
        int[] tensorShape = {6, 5, 5, 1};
        float[] tensorData = new float[6 * 5 * 5];
        Arrays.fill(tensorData, 0);
        ByteBuffer byteBuf = ByteBuffer.allocateDirect(6 * 5 * 5 * 4);
        FloatBuffer floatBuf = byteBuf.asFloatBuffer();
        floatBuf.put(tensorData);
        MSTensor newTensor = MSTensor.createTensor("conv1.weight", DataType.kNumberTypeFloat32, tensorShape,
                byteBuf);
        assertNotNull(newTensor);
    }

    @Test
    public void testSetData() {
        int[] tensorShape = {6, 5, 5, 1};
        ByteBuffer byteBuf = ByteBuffer.allocateDirect(6 * 5 * 5 * 4);
        MSTensor newTensor = MSTensor.createTensor("conv1.weight", DataType.kNumberTypeFloat32, tensorShape,
                byteBuf);
        assertNotNull(newTensor);
        float[] tensorData = new float[6 * 5 * 5];
        Arrays.fill(tensorData, 0);
        assertTrue(newTensor.setData(tensorData));
    }
}