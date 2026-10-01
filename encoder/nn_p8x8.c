/*****************************************************************************
 * nn_p8x8.c: NN-assisted sub-8x8 partition-depth pruning
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
#include "nn_p8x8.h"

/* Offline-trained weights; do not edit by hand. */

#define X264_NN_P8X8_H1 16
#define X264_NN_P8X8_H2 8
#define X264_NN_P8X8_NUM_LAYERS 3

static const int8_t  x264_nn_p8x8_w0[16 * 13] = {
    -4,   18,   -1,  -28,   -4,    4,   -2,    1,    2,   -3,    2,   -2,   -2,
    20,   15,    1,   87,    0,    0,   -1,    1,    1,    0,    1,    0,    1,
    -4,   19,   -1,  -42,   -1,    0,   -2,    1,    0,   -1,    0,   -1,   -1,
     3,    6,   -5,   19,   -1,   -1,   -3,    2,    0,   -2,    0,   -1,   -4,
    -5,   20,   -1,  -49,    0,    0,   -1,    0,    0,   -1,    0,    0,    0,
    -7,   20,   -1,  -26,    2,   -2,    2,   -2,   -2,    2,   -3,    3,   -2,
    -5,   21,   -1,  -58,    0,   -1,   -1,    0,   -1,    0,   -2,    1,    0,
    -5,   34,    8,  114,    3,    7,   14,   -3,   -1,   12,    4,    6,   26,
   -12,   24,   -1,   12,    6,   -5,   -2,    3,   -2,    2,   -2,    3,   -4,
    -5,   22,   -1,  -63,    0,    0,    0,   -1,   -1,    0,   -1,   -1,    0,
    -6,   20,   -1,  -20,   -2,    2,   10,  -10,   -5,    5,   -4,    2,   -4,
    -4,   17,   -2,  -31,   -5,    6,   -6,    5,   14,  -14,   12,  -13,   -4,
    -6,   30,    3, -127,    1,   -3,    0,   -2,   -2,   -1,   -3,    0,    4,
   -49,   53,    4,  -74,    1,   -2,   -1,    0,   -4,    1,   -4,    2,    1,
    -5,   19,   -1,  -38,   -1,    1,    0,    0,   -1,    1,    1,   -2,   -1,
    -5,   19,   -1,  -43,    1,   -1,    1,   -1,   -3,    2,   -1,    0,   -1
};
static const int32_t x264_nn_p8x8_b0[16] = {   -395,    -757,    -396,    -464,    -396,    -392,    -395,    -331,    -392,    -397,    -385,    -335,    -345,    -235,    -394,    -394};
static const int16_t x264_nn_p8x8_m0[16] = { 19304,  19304,  19304,  19304,  19304,  19304,  19304,  19304,  19304,  19304,  19304,  19304,  19304,  19304,  19304,  19304};

static const int8_t  x264_nn_p8x8_w1[8 * 16] = {
    23,  127,   23,   28,   22,   22,   22,   13,   22,   24,   21,   16,   20,   21,   23,   23,
     0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    23,  126,   23,   29,   22,   22,   22,   13,   22,   24,   21,   16,   20,   20,   23,   23,
     0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
     0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    23,  127,   23,   24,   23,   22,   22,   13,   22,   24,   21,   17,   21,   25,   23,   23,
    22,  126,   23,   27,   22,   22,   22,   12,   22,   24,   21,   16,   20,   22,   22,   23,
    23,  127,   23,   25,   23,   22,   23,   13,   23,   24,   21,   17,   21,   24,   23,   23
};
static const int32_t x264_nn_p8x8_b1[8] = {   -257,      -1,    -277,      -1,      -1,    -156,    -247,    -183};
static const int16_t x264_nn_p8x8_m1[8] = { 18895,  18895,  18895,  18895,  18895,  18895,  18895,  18895};

static const int8_t  x264_nn_p8x8_w2[2 * 8] = {
   126,    0,  127,    0,    0,  127,  127,  126,
     0,    0,    0,    0,    0,    0,   -1,    1
};
static const int32_t x264_nn_p8x8_b2[2] = {    147,       0};
static const int16_t x264_nn_p8x8_m2[2] = { 28776,  28776};

static const x264_nn_layer_t x264_nn_p8x8_layers[X264_NN_P8X8_NUM_LAYERS] =
{
    { x264_nn_p8x8_w0, x264_nn_p8x8_b0, x264_nn_p8x8_m0, 15, X264_NN_P8X8_CTX_DIM, X264_NN_P8X8_H1, 0 },
    { x264_nn_p8x8_w1, x264_nn_p8x8_b1, x264_nn_p8x8_m1, 17, X264_NN_P8X8_H1,      X264_NN_P8X8_H2, 0 },
    { x264_nn_p8x8_w2, x264_nn_p8x8_b2, x264_nn_p8x8_m2, 12, X264_NN_P8X8_H2,      X264_NN_P8X8_POLICY_COUNT, 1 },
};

/* Normalizes a raw pixel to 8-bit range regardless of BIT_DEPTH. */
#define NN_PX(p) ((int)(p) >> (BIT_DEPTH-8))

/* Gathers the X264_NN_P8X8_CTX_DIM-int8 context for one 8x8 block's
 * already-completed ME result. Mirrors nn_p16_gather_context() one
 * partition-depth level down (4x4 quadrants of the 8x8 block instead of
 * 8x8 quadrants of a 16x16 block):
 *   [0]       qp - 26
 *   [1]       log8(satd_8x8) -- pixel SATD vs. the winning 8x8 prediction,
 *             no lambda cost
 *   [2]       log8(cost_mv_8x8) -- mv-signaling cost
 *   [3]       ref_8x8, clipped to int8 (not centered)
 *   [4..11]   log8(horizontal), log8(vertical) gradient energy of the
 *             source, per 4x4 quadrant (order: top-left, top-right,
 *             bottom-left, bottom-right; gx then gy)
 *   [12]      log8(max quadrant energy - min quadrant energy)
 * A contract with the offline trainer: changing this invalidates the
 * weight data above. */
static void nn_p8x8_gather_context( x264_t *h, int idx8x8, int qp, int32_t satd_8x8,
                                     int32_t cost_mv_8x8, int ref_8x8,
                                     int8_t ctx[X264_NN_P8X8_CTX_DIM] )
{
    int bx = (idx8x8 & 1) * 8, by = (idx8x8 >> 1) * 8;
    pixel *src = h->mb.pic.p_fenc[0] + by * FENC_STRIDE + bx;
    int gx[2][2] = {{0}}, gy[2][2] = {{0}};

    ctx[0] = (int8_t)x264_clip3( qp - 26, -128, 127 );
    ctx[1] = (int8_t)(x264_nn_log8( satd_8x8 ) - 64);
    ctx[2] = (int8_t)(x264_nn_log8( cost_mv_8x8 ) - 64);
    ctx[3] = (int8_t)x264_clip3( ref_8x8, -128, 127 );

    for( int y = 0; y < 8; y++ )
        for( int x = 0; x < 8; x++ )
        {
            int qy = y >> 2, qx = x >> 2;
            int v = NN_PX( src[y*FENC_STRIDE + x] );
            if( (x & 3) < 3 )
                gx[qy][qx] += abs( NN_PX( src[y*FENC_STRIDE + x+1] ) - v );
            if( (y & 3) < 3 )
                gy[qy][qx] += abs( NN_PX( src[(y+1)*FENC_STRIDE + x] ) - v );
        }

    int emin = INT_MAX, emax = 0;
    for( int qy = 0; qy < 2; qy++ )
        for( int qx = 0; qx < 2; qx++ )
        {
            int i = 4 + 2*(2*qy+qx);
            ctx[i]   = (int8_t)(x264_nn_log8( gx[qy][qx] ) - 64);
            ctx[i+1] = (int8_t)(x264_nn_log8( gy[qy][qx] ) - 64);
            int e = gx[qy][qx] + gy[qy][qx];
            emin = X264_MIN( emin, e );
            emax = X264_MAX( emax, e );
        }
    ctx[12] = (int8_t)(x264_nn_log8( emax - emin ) - 64);
}

x264_nn_p8x8_policy_t x264_nn_p8x8_policy( x264_t *h, int idx8x8, int qp, int32_t satd_8x8,
                                            int32_t cost_mv_8x8, int ref_8x8, int32_t stop_bias )
{
    int8_t ctx[X264_NN_P8X8_CTX_DIM];
    int32_t regret[X264_NN_P8X8_POLICY_COUNT];

    nn_p8x8_gather_context( h, idx8x8, qp, satd_8x8, cost_mv_8x8, ref_8x8, ctx );
    x264_nn_forward( x264_nn_p8x8_layers, X264_NN_P8X8_NUM_LAYERS, ctx, regret );

    return (regret[X264_NN_P8X8_POLICY_STOP] - stop_bias <= regret[X264_NN_P8X8_POLICY_CONTINUE])
         ? X264_NN_P8X8_POLICY_STOP : X264_NN_P8X8_POLICY_CONTINUE;
}
