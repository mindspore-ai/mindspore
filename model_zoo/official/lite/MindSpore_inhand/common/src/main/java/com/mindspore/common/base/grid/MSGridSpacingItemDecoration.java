package com.mindspore.common.base.grid;

import android.graphics.Rect;
import android.view.View;

import androidx.recyclerview.widget.RecyclerView;

public class MSGridSpacingItemDecoration extends RecyclerView.ItemDecoration {
    private final int space;

    public MSGridSpacingItemDecoration( int space) {
        this.space = space;
    }

    @Override
    public void getItemOffsets(Rect outRect, View view, RecyclerView parent, RecyclerView.State state) {
        outRect.left = space;
        outRect.bottom = space;
        if (parent.getChildLayoutPosition(view) % 4 == 0) {
            outRect.left = 0;
        }
    }

}
