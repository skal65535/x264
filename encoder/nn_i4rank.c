/*****************************************************************************
 * nn_i4rank.c: Intra4 mode-ranking proxy
 *****************************************************************************
 * Copyright (C) 2026 x264 project
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02111, USA.
 *
 * This program is also available under a commercial proprietary license.
 * For more information, contact us at licensing@x264.com.
 *****************************************************************************/

#include "common/common.h"
#include "common/nn.h"
#include "macroblock.h"
#include "nn_i4rank.h"

/* Offline-trained weights, single linear layer (no hidden layers); do not
 * edit by hand. */

static const int8_t x264_nn_i4rank_w[X264_NN_I4RANK_NUM_MODES * X264_NN_I4RANK_NUM_FEATURES] = {
     0,  -43,   17,  -24,   -1,   13,    1,  -16,    3,   -1,    0,  -41,   55,    3,  -21,   -7,   24,   -4,   -8,
     0,   12,  -45,  -17,    3,    8,  -11,    2,   -4,    3,  -29,    7,   36,   -7,  -12,   16,   -6,   -1,   -4,
     0,    5,    9,  -55,    1,    9,   -4,   -6,   -1,   -3,  -14,  -24,   35,    1,  -14,    4,    7,    2,    2,
     5,   21,   12,  -23,  -36,   19,   -5,  -15,  -33,    4,  -54,  -41,   64,   19,  -42,   -9,   12,   83,  -32,
     0,   18,   22,  -56,    4,    0,  -22,  -31,   -3,    5,  -47,  -53,  127,  -10,  -54,   26,   44,   -9,  -23,
    -1,    8,   25,  -50,    2,   32,  -50,  -31,    0,    5,  -26,  -55,  113,   -5,  -44,    9,   47,  -10,  -29,
     0,   21,   10,  -46,    5,   29,  -22,  -56,   -4,    5,  -51,  -30,  104,  -10,  -39,   30,   21,   -8,  -16,
     2,    8,   15,  -26,    4,   17,   -2,  -15,  -55,    1,  -23,  -42,   64,    3,  -30,   -4,   19,   36,  -22,
     1,   12,    9,  -24,    4,   15,  -10,   -6,   -3,  -46,  -32,  -27,   57,   -2,  -23,   12,    7,   -4,   13
};
static const int32_t x264_nn_i4rank_b[X264_NN_I4RANK_NUM_MODES] = {55, 45, 20, 139, 128, 116, 101, 92, 61};
static const int16_t x264_nn_i4rank_m[X264_NN_I4RANK_NUM_MODES] = {27665, 27665, 27665, 27665, 27665, 27665, 27665, 27665, 27665};

static const x264_nn_layer_t x264_nn_i4rank_layers[1] =
{
    { x264_nn_i4rank_w, x264_nn_i4rank_b, x264_nn_i4rank_m, 10, X264_NN_I4RANK_NUM_FEATURES, X264_NN_I4RANK_NUM_MODES, 1 },
};

/* Mirrors the real per-mode Intra4 signaling cost in bits (CAVLC-style
 * approximation): 1 bit if this mode equals the block's predicted mode
 * (prev_intra4x4_pred_mode_flag=1), else a fixed 4 bits (flag=0 + 3-bit
 * rem_intra4x4_pred_mode). Must match whatever process produced the net's
 * weights. */
static int nn_i4rank_cost_bits( int mode, int predicted_mode )
{
    return mode == predicted_mode ? 1 : 4;
}

int x264_nn_i4rank_topk( x264_t *h, int idx, int qp, int lambda,
                         const int8_t *predict_mode, pixel *p_src_by, pixel *p_dst_by,
                         int topk, int8_t out_modes[X264_NN_I4RANK_NUM_MODES + 5] )
{
    int32_t satd[X264_NN_I4RANK_NUM_MODES];
    int legal[X264_NN_I4RANK_NUM_MODES];
    int n_legal = 0;
    int extra_modes[4]; /* DC_LEFT/DC_TOP/DC_128 fallbacks (index >= 9): always passed through, unranked */
    int n_extra = 0;

    for( const int8_t *pm = predict_mode; *pm >= 0; pm++ )
    {
        int mode = *pm;
        if( mode >= X264_NN_I4RANK_NUM_MODES )
        {
            if( n_extra < 4 )
                extra_modes[n_extra++] = mode;
            continue;
        }
        if( h->mb.b_lossless )
            x264_predict_lossless_4x4( h, p_dst_by, 0, idx, mode );
        else
            h->predict_4x4[mode]( p_dst_by );
        satd[mode] = h->pixf.mbcmp[PIXEL_4x4]( p_src_by, FENC_STRIDE, p_dst_by, FDEC_STRIDE );
        legal[n_legal++] = mode;
    }

    /* The ranker only ever saw all-9-canonical-legal training data; don't
     * extrapolate to partial neighbourhoods. */
    if( n_legal < X264_NN_I4RANK_NUM_MODES )
    {
        int n = 0;
        for( int i = 0; i < n_legal; i++ )
            out_modes[n++] = legal[i];
        for( int i = 0; i < n_extra; i++ )
            out_modes[n++] = extra_modes[i];
        out_modes[n] = -1;
        return n;
    }

    int predicted_mode = x264_mb_predict_intra4x4_mode( h, idx );
    int cost_bits[X264_NN_I4RANK_NUM_MODES];
    int hmin = 65535;
    int64_t p[X264_NN_I4RANK_NUM_MODES], pmin = INT64_MAX;
    for( int m = 0; m < X264_NN_I4RANK_NUM_MODES; m++ )
    {
        cost_bits[m] = nn_i4rank_cost_bits( m, predicted_mode );
        p[m] = satd[m] + (int64_t)lambda * cost_bits[m];
        pmin = X264_MIN( pmin, p[m] );
        hmin = X264_MIN( hmin, cost_bits[m] );
    }

    int8_t ctx[X264_NN_I4RANK_NUM_FEATURES];
    ctx[0] = (int8_t)x264_clip3( qp - 26, -128, 127 );
    for( int m = 0; m < X264_NN_I4RANK_NUM_MODES; m++ )
    {
        /* relative gap from the best mode, scaled by 32 (truncating
         * division, not rounding -- a small fixed bias the trained weights
         * already absorb). */
        int64_t dp = ((p[m] - pmin) * 32) / (pmin + 32);
        ctx[1 + m] = (int8_t)X264_MIN( dp, 127 );
        int dh = cost_bits[m] - hmin;
        ctx[1 + X264_NN_I4RANK_NUM_MODES + m] = (int8_t)X264_MIN( dh, 127 );
    }

    int32_t logits[X264_NN_I4RANK_NUM_MODES];
    x264_nn_forward( x264_nn_i4rank_layers, 1, ctx, logits );

    /* insertion sort (9 elements): ranked[] holds mode indices, best first */
    int8_t ranked[X264_NN_I4RANK_NUM_MODES];
    int32_t ranked_logit[X264_NN_I4RANK_NUM_MODES];
    int n_ranked = 0;
    for( int m = 0; m < X264_NN_I4RANK_NUM_MODES; m++ )
    {
        int32_t lg = logits[m];
        int pos = n_ranked;
        while( pos > 0 && ranked_logit[pos-1] < lg )
            pos--;
        for( int j = n_ranked; j > pos; j-- )
        {
            ranked_logit[j] = ranked_logit[j-1];
            ranked[j] = ranked[j-1];
        }
        ranked_logit[pos] = lg;
        ranked[pos] = (int8_t)m;
        n_ranked++;
    }

    int k = X264_MIN( topk, n_ranked );
    int n = 0;
    for( int i = 0; i < k; i++ )
        out_modes[n++] = ranked[i];
    for( int i = 0; i < n_extra; i++ )
        out_modes[n++] = extra_modes[i];
    out_modes[n] = -1;
    return n;
}
