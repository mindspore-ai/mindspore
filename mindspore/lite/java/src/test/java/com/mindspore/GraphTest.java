package com.mindspore;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

@RunWith(JUnit4.class)
public class GraphTest {

    @Test
    public void testLoadFailed() {
        Graph g = new Graph();
        assertFalse(g.load("./1.ms"));
    }

    @Test
    public void testLoadSuccess() {
        Graph g = new Graph();
        String pro = System.getProperty("user.dir");
        System.out.println("output:"+pro);
        assertTrue(g.load("../test/ut/src/runtime/kernel/arm/test_data/nets/lenet_train.ms"));
        g.free();
    }
}
