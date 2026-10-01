/*****************************************************************************
 * nn_p16.c: NN-assisted P-slice partition-depth pruning
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
#include "nn_p16.h"

/* Offline-trained weights; do not edit by hand. */
#define X264_NN_P16_H1 16
#define X264_NN_P16_H2 8
#define X264_NN_P16_NUM_LAYERS 3

static const int8_t  x264_nn_p16_w0[16 * 13] = {
   -24,   14,    0,   12,    0,    0,   -1,    0,   -1,    0,    0,    0,    2,
   -23,   14,    0,   12,    0,    0,   -1,    0,    0,    0,    0,   -1,    1,
   -36,   14,    5,  -18,   -2,    3,    1,   -4,    6,   -7,    0,    4,    2,
   -96,    5,   50,    5,    2,   -2,   23,    7,   -7,   22,    3,   -8,    4,
  -107,    0,   59,   11,    6,   23,   18,  -11,   -5,   15,    7,   -4,   -4,
  -125,   -2,   71,   49,   -2,   23,   21,  -11,   -2,   27,   16,   -8,  -10,
   -52,   15,   17,  -14,    1,    0,    6,   -5,    1,   -1,    0,    2,    6,
   -43,   15,    0,  -22,    0,   -1,    0,   -1,    0,   -2,   -1,   -1,    2,
  -119,   -3,   68,   31,   -1,   25,   17,   -9,   -2,   24,   16,   -6,   -9,
  -126,  -12,   80,   51,   -4,   25,   16,  -12,   -7,   27,   13,  -11,  -13,
  -126,   -3,   72,   31,    1,   25,   21,  -10,   -1,   25,   11,   -7,   -7,
   -24,   14,    0,   11,    0,    0,    0,    0,    0,    0,    0,   -1,    2,
   -98,   12,   46,  -23,   -6,   21,   10,   -2,   -9,   21,   -9,    3,   -9,
   -36,   14,   -2,  -20,    2,   -3,   -1,    0,   -2,    1,    0,   -2,    1,
  -127,   -4,   73,   36,   -2,   27,   19,  -10,   -2,   25,   16,   -6,  -10,
  -115,    0,   60,   75,    8,   15,   26,  -12,   -3,   25,   19,  -10,  -14
};
static const int32_t x264_nn_p16_b0[16] = {   -620,    -610,    -580,    -470,    -443,    -423,    -563,    -588,    -422,    -409,    -424,    -620,    -480,    -597,    -422,    -437};
static const int16_t x264_nn_p16_m0[16] = { 25466,  25466,  25466,  25466,  25466,  25466,  25466,  25466,  25466,  25466,  25466,  25466,  25466,  25466,  25466,  25466};

static const int8_t  x264_nn_p16_w1[8 * 16] = {
   127,  123,   95,   69,   47,   45,   80,  105,   46,   47,   45,  124,   54,  120,   45,   49,
     0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
   127,  122,   94,   68,   47,   44,   79,  105,   45,   46,   44,  123,   53,  119,   45,   48,
     0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
     0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
   127,  123,   94,   69,   47,   45,   80,  105,   46,   47,   45,  124,   54,  120,   46,   48,
   126,  122,   94,   68,   46,   44,   79,  104,   45,   46,   44,  123,   53,  119,   45,   48,
   127,  123,   94,   69,   47,   45,   80,  105,   46,   47,   45,  124,   54,  120,   46,   48
};
static const int32_t x264_nn_p16_b1[8] = {  -4332,      -9,   -4308,     -10,      -9,   -4332,   -4308,   -4331};
static const int16_t x264_nn_p16_m1[8] = { 26348,  26348,  26348,  26348,  26348,  26348,  26348,  26348};

static const int8_t  x264_nn_p16_w2[2 * 8] = {
   127,    0,  126,    0,    0,  127,  127,  127,
     0,    0,    0,    0,    0,    0,    0,    0
};
static const int32_t x264_nn_p16_b2[2] = {     15,       0};
static const int16_t x264_nn_p16_m2[2] = { 31074,  31074};

static const x264_nn_layer_t x264_nn_p16_layers[X264_NN_P16_NUM_LAYERS] =
{
    { x264_nn_p16_w0, x264_nn_p16_b0, x264_nn_p16_m0, 12, X264_NN_P16_CTX_DIM, X264_NN_P16_H1, 0 },
    { x264_nn_p16_w1, x264_nn_p16_b1, x264_nn_p16_m1, 20, X264_NN_P16_H1,      X264_NN_P16_H2, 0 },
    { x264_nn_p16_w2, x264_nn_p16_b2, x264_nn_p16_m2, 8, X264_NN_P16_H2,      X264_NN_P16_POLICY_COUNT, 1 },
};

/* Normalizes a raw pixel to 8-bit range regardless of BIT_DEPTH. */
#define NN_PX(p) ((int)(p) >> (BIT_DEPTH-8))

/* Gathers the X264_NN_P16_CTX_DIM-int8 context for the current macroblock's
 * already-completed P16x16 motion search. Layout (log8 values centered by
 * -64):
 *   [0]       qp - 26
 *   [1]       log8(satd_p16x16) -- pixel SATD vs. the winning P16x16
 *             prediction, no lambda cost
 *   [2]       log8(cost_mv_p16x16) -- mv-signaling cost
 *   [3]       ref_p16x16, clipped to int8 (not centered)
 *   [4..11]   log8(horizontal), log8(vertical) gradient energy of the
 *             source, per 8x8 quadrant of the 16x16 block (order:
 *             top-left, top-right, bottom-left, bottom-right; gx then gy)
 *   [12]      log8(max quadrant energy - min quadrant energy)
 * A contract with the offline trainer: changing this invalidates the
 * weight data above. */
static void nn_p16_gather_context( x264_t *h, int qp, int32_t satd_p16x16,
                                    int32_t cost_mv_p16x16, int ref_p16x16,
                                    int8_t ctx[X264_NN_P16_CTX_DIM] )
{
    pixel *src = h->mb.pic.p_fenc[0];
    int gx[2][2] = {{0}}, gy[2][2] = {{0}};

    ctx[0] = (int8_t)x264_clip3( qp - 26, -128, 127 );
    ctx[1] = (int8_t)(x264_nn_log8( satd_p16x16 ) - 64);
    ctx[2] = (int8_t)(x264_nn_log8( cost_mv_p16x16 ) - 64);
    ctx[3] = (int8_t)x264_clip3( ref_p16x16, -128, 127 );

    for( int y = 0; y < 16; y++ )
        for( int x = 0; x < 16; x++ )
        {
            int qy = y >> 3, qx = x >> 3;
            int v = NN_PX( src[y*FENC_STRIDE + x] );
            /* don't accumulate across the quadrant boundary, so each
             * quadrant's energy reflects only its own texture */
            if( (x & 7) < 7 )
                gx[qy][qx] += abs( NN_PX( src[y*FENC_STRIDE + x+1] ) - v );
            if( (y & 7) < 7 )
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

x264_nn_p16_policy_t x264_nn_p16_policy( x264_t *h, int qp, int32_t satd_p16x16,
                                          int32_t cost_mv_p16x16, int ref_p16x16,
                                          int32_t stop_bias )
{
    int8_t ctx[X264_NN_P16_CTX_DIM];
    int32_t regret[X264_NN_P16_POLICY_COUNT];

    nn_p16_gather_context( h, qp, satd_p16x16, cost_mv_p16x16, ref_p16x16, ctx );
    x264_nn_forward( x264_nn_p16_layers, X264_NN_P16_NUM_LAYERS, ctx, regret );

    return (regret[X264_NN_P16_POLICY_STOP] - stop_bias <= regret[X264_NN_P16_POLICY_CONTINUE])
         ? X264_NN_P16_POLICY_STOP : X264_NN_P16_POLICY_CONTINUE;
}
