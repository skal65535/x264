/*****************************************************************************
 * nn_b16.c: NN-assisted B-slice bidirectional (BI) prediction pruning
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
#include "nn_b16.h"

/* Offline-trained weights; do not edit by hand. */

#define X264_NN_B16_H1 16
#define X264_NN_B16_H2 8
#define X264_NN_B16_NUM_LAYERS 3

static const int8_t  x264_nn_b16_w0[16 * 11] = {
   -15,   -1,   -2,    0,    0,    6,   -1,  -20,    1,    0,    0,
   -16,    3,   -5,    2,   -1,    2, -127,   47,    1,    0,    0,
   -17,   -6,   -3,   -1,    1,    5,   -9,  -57,    3,    1,    1,
     8,   -3,    3,   -2,    1,    0,   43,  -13,    0,    0,    0,
   -15,   -4,    2,   -1,    3,    3,   20,  -74,    1,    0,    0,
     3,   -9,    1,   -4,   13,   -1,    1,  -13,    2,    3,    2,
   -23,   -1,   -3,   23,   -2,    2, -108,   32,   -4,    4,    7,
     1,   -4,    2,   -4,    6,    0,  -10,   14,   -1,    1,    1,
     1,    1,    0,    7,   -2,   -1,    3,   -3,   -4,    1,    1,
     2,    3,   -2,    1,   -3,    1,  -95,   20,    0,    0,    0,
    -1,    0,    0,    1,    0,    0,    6,    4,   -1,    0,    0,
    17,    0,    0,    1,    0,    0,    1,   -3,   -4,    1,    1,
    11,   -5,   -4,    3,    4,    1,   44,   65,    3,    1,    2,
    -8,    4,    4,    2,    2,   -1,    8,   34,  -10,    3,    2,
     9,    4,   -4,    1,   -2,    0,  -11,   11,   -1,    0,    0,
    -8,    1,   -3,   10,   -1,    0,  -14,  -47,    1,    0,    0
};
static const int32_t x264_nn_b16_b0[16] = {   -147,     -85,      27,      30,     -94,      41,      32,      46,      45,      -9,      60,     -30,      37,     -34,      26,      58};
static const int16_t x264_nn_b16_m0[16] = { 19336,  19336,  19336,  19336,  19336,  19336,  19336,  19336,  19336,  19336,  19336,  19336,  19336,  19336,  19336,  19336};

static const int8_t  x264_nn_b16_w1[8 * 16] = {
   101,   22,    1,    1,   23,    2,   84,    1,    5,    1,    4,    2,    3,    5,    0,    7,
     0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
     0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
     0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
   101,   22,    1,    1,   23,    2,   84,    1,    5,    0,    4,    2,    3,    4,    0,    7,
   100,   22,    1,    1,   23,    2,   83,    1,    5,    1,    4,    2,    3,    4,    1,    7,
   127,   46,    7,    2,   38,   11,   98,    3,    7,    2,   11,    4,    9,   10,    1,   12,
     0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0
};
static const int32_t x264_nn_b16_b1[8] = {    -14,      -1,      -1,      -1,     -19,     -16,      34,      -1};
static const int16_t x264_nn_b16_m1[8] = { 28272,  28272,  28272,  28272,  28272,  28272,  28272,  28272};

static const int8_t  x264_nn_b16_w2[2 * 8] = {
   112,    0,    0,    0,  111,  110,  127,    0,
     0,    0,    0,    0,    0,    0,    0,    0
};
static const int32_t x264_nn_b16_b2[2] = {    -63,       0};
static const int16_t x264_nn_b16_m2[2] = { 23990,  23990};

static const x264_nn_layer_t x264_nn_b16_layers[X264_NN_B16_NUM_LAYERS] =
{
    { x264_nn_b16_w0, x264_nn_b16_b0, x264_nn_b16_m0, 11, X264_NN_B16_CTX_DIM, X264_NN_B16_H1, 0 },
    { x264_nn_b16_w1, x264_nn_b16_b1, x264_nn_b16_m1, 15, X264_NN_B16_H1,      X264_NN_B16_H2, 0 },
    { x264_nn_b16_w2, x264_nn_b16_b2, x264_nn_b16_m2, 12, X264_NN_B16_H2,      X264_NN_B16_POLICY_COUNT, 1 },
};

/* Normalizes a raw pixel to 8-bit range regardless of BIT_DEPTH. */
#define NN_PX(p) ((int)(p) >> (BIT_DEPTH-8))

/* Gathers the X264_NN_B16_CTX_DIM-int8 context for the current B-slice MB's
 * already-completed L0/L1 (and, if available, Direct) results. Layout:
 *   [0]   qp - 26
 *   [1]   log8(satd_l0) -- pixel SATD of the winning L0 prediction
 *   [2]   log8(satd_l1) -- same, L1
 *   [3]   log8(cost_mv_l0) -- mv-signaling cost, L0
 *   [4]   log8(cost_mv_l1) -- mv-signaling cost, L1
 *   [5]   log8(|satd_l0 - satd_l1|) -- closeness: BI tends to help most when
 *         L0 and L1 are both reasonably good, not when one dominates
 *   [6]   ref_l0, clipped to int8 (not centered)
 *   [7]   ref_l1, clipped to int8 (not centered)
 *   [8]   log8(cost_direct) -- saturates to 127 (ctx[8]=63) when direct
 *         isn't available (COST_MAX), a natural "infinitely expensive"
 *         signal with no special-casing needed
 *   [9]   log8(horizontal gradient) of the 16x16 source block
 *   [10]  log8(vertical gradient) of the 16x16 source block
 * A contract with the offline trainer: changing this invalidates the
 * weight data above. */
static void nn_b16_gather_context( x264_t *h, int qp, int32_t satd_l0, int32_t cost_mv_l0, int ref_l0,
                                    int32_t satd_l1, int32_t cost_mv_l1, int ref_l1,
                                    int32_t cost_direct, int8_t ctx[X264_NN_B16_CTX_DIM] )
{
    pixel *src = h->mb.pic.p_fenc[0];
    int gx = 0, gy = 0;

    ctx[0] = (int8_t)x264_clip3( qp - 26, -128, 127 );
    ctx[1] = (int8_t)(x264_nn_log8( satd_l0 ) - 64);
    ctx[2] = (int8_t)(x264_nn_log8( satd_l1 ) - 64);
    ctx[3] = (int8_t)(x264_nn_log8( cost_mv_l0 ) - 64);
    ctx[4] = (int8_t)(x264_nn_log8( cost_mv_l1 ) - 64);
    ctx[5] = (int8_t)(x264_nn_log8( abs( satd_l0 - satd_l1 ) ) - 64);
    ctx[6] = (int8_t)x264_clip3( ref_l0, -128, 127 );
    ctx[7] = (int8_t)x264_clip3( ref_l1, -128, 127 );
    ctx[8] = (int8_t)(x264_nn_log8( cost_direct ) - 64);

    for( int y = 0; y < 16; y++ )
        for( int x = 0; x < 16; x++ )
        {
            int v = NN_PX( src[y*FENC_STRIDE + x] );
            if( x < 15 ) gx += abs( NN_PX( src[y*FENC_STRIDE + x+1] ) - v );
            if( y < 15 ) gy += abs( NN_PX( src[(y+1)*FENC_STRIDE + x] ) - v );
        }
    ctx[9]  = (int8_t)(x264_nn_log8( gx ) - 64);
    ctx[10] = (int8_t)(x264_nn_log8( gy ) - 64);
}

x264_nn_b16_policy_t x264_nn_b16_policy( x264_t *h, int qp, int32_t satd_l0, int32_t cost_mv_l0, int ref_l0,
                                          int32_t satd_l1, int32_t cost_mv_l1, int ref_l1,
                                          int32_t cost_direct, int32_t stop_bias )
{
    int8_t ctx[X264_NN_B16_CTX_DIM];
    int32_t regret[X264_NN_B16_POLICY_COUNT];

    nn_b16_gather_context( h, qp, satd_l0, cost_mv_l0, ref_l0, satd_l1, cost_mv_l1, ref_l1, cost_direct, ctx );
    x264_nn_forward( x264_nn_b16_layers, X264_NN_B16_NUM_LAYERS, ctx, regret );

    return (regret[X264_NN_B16_POLICY_STOP] - stop_bias <= regret[X264_NN_B16_POLICY_CONTINUE])
         ? X264_NN_B16_POLICY_STOP : X264_NN_B16_POLICY_CONTINUE;
}
