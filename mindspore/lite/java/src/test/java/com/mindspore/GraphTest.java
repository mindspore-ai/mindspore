package com.mindspore;

import com.mindspore.lite.NativeLibrary;
import com.mindspore.lite.Version;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.logging.Level;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

@RunWith(JUnit4.class)
public class GraphTest {

    @Test
    public void testLoadFailed() {
        try {
            NativeLibrary.load();
        } catch (Exception e) {
            System.err.println("Failed to load MindSporLite native library.");
            e.printStackTrace();
        }
        System.out.println(Version.version());
        Graph g = new Graph();
        assertFalse(g.load("./1.ms"));
    }

    @Test
    public void testLoadSuccess() {
        try {
            NativeLibrary.load();
            System.err.println("System: NativeLibrary load success.");
        } catch (Exception e) {
            System.err.println("Failed to load MindSporLite native library.");
            e.printStackTrace();
        }
        System.out.println(Version.version());
        Graph g = new Graph();
        String pro = System.getProperty("user.dir");
        System.out.println("output:"+pro);
        assertTrue(g.load("../test/ut/src/runtime/kernel/arm/test_data/nets/lenet_train.ms"));
        g.free();
    }
}
