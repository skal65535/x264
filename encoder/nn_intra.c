/*****************************************************************************
 * nn_intra.c: NN-assisted Intra16/Intra4 mode-decision pruning
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
#include "nn_intra.h"

/* Offline-trained weights; do not edit by hand. */

#define X264_NN_I16_H1 32
#define X264_NN_I16_H2 16
#define X264_NN_I16_NUM_LAYERS 3

static const int8_t  x264_nn_i16_w0[32 * 29] = {
    52,    2,    1,    1,   -6,    0,   -1,    1,    2,    1,   -1,   -1,    4,   -5,   -1,    0,    5,    2,    2,   -1,   -4,    1,   -1,   -2,    0,    2,    1,   -1,  -16,
    18,    0,   -1,    0,    0,    0,    2,    1,   -1,    1,   -1,    0,    0,    0,    1,   -1,    1,   -1,    0,    0,    0,    1,   -1,    1,   -1,   -1,    2,    5,  -16,
    67,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   15,   16,  -15,
    49,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   12,    9,  -16,
    48,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   12,    9,   14,
    27,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    4,    9,  -16,
    54,   -3,   -2,   -1,    7,    1,    0,    0,    0,    0,    1,   -2,   -1,    4,    0,    2,   -8,   -3,    1,    0,    3,   -1,   -1,    1,    2,   -1,    0,   -1,  -16,
    30,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    4,   10,  -19,
    48,    1,   -2,   -1,    0,    1,    4,    0,   -6,   -4,    3,    4,    0,   -1,    0,    0,    1,    3,   -3,   -3,    5,    2,    2,   -3,    1,   -1,    2,    1,  -16,
   127,    0,    0,    0,    0,    0,    1,    0,    0,    0,   -1,    0,    0,    1,   -1,   -1,    1,    0,    0,   -1,    0,    0,    1,    0,    0,   -1,   33,   26,  -19,
    48,    3,   -1,   -3,    0,   -2,    3,    0,    4,   -6,    1,    3,    3,    3,   -1,   -3,   -4,    1,   -3,    2,    0,    1,   -5,   -2,    4,    1,    0,    3,  -20,
    43,   -3,    1,    2,    3,    1,   -7,    1,    2,    1,   -1,   -3,    3,    1,    3,   -1,   -1,    1,    2,   -3,   -1,   -3,    2,   -1,   -1,    1,    1,    3,  -18,
    30,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    5,    9,  -16,
    41,    1,    0,    2,    0,   -5,    4,    3,    1,    3,   -2,   -5,   -1,    0,    0,    0,    1,    2,   -2,    0,   -1,   -2,   -2,    3,   -1,    1,    0,    3,  -15,
   121,    0,    0,    0,    0,    0,   -1,    0,    0,    0,    0,    0,    1,    1,    0,    0,    1,    0,    1,    0,    0,    0,    1,    1,   -1,    0,   32,   24,  -21,
   121,    0,    0,    0,    0,   -1,    0,    0,    0,    1,   -1,    0,    2,    2,   -1,    0,    0,    0,    1,    0,    0,    0,    1,    0,    0,   -1,   33,   22,  -22,
    -5,   -1,   15,   16,  -16,    1,    4,    5,    3,   -4,  -10,   -8,   11,    5,  -19,  -21,   17,   -1,   11,    8,  -15,  -11,   -2,    1,   13,   -2,   -1,    0,  -28,
    36,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    7,    9,  -16,
    48,   -1,   -1,    3,   -2,    4,   -2,   -5,    0,    4,   -2,   -4,    1,   -2,    2,    1,    0,   -2,   -1,    3,    0,    0,    3,    0,   -1,   -1,    1,    2,  -15,
    45,    0,    2,   -2,    3,    0,    4,   -3,   -2,    1,    1,    4,   -7,    0,   -9,    2,    4,    0,    2,   -1,    0,   -3,    1,    1,    2,   -1,    1,    0,  -16,
    -4,  -17,  -11,    9,   27,    1,   -3,   -2,    1,   13,    6,   -8,  -18,   15,   13,   -4,  -26,  -11,   -6,    4,   16,   -8,    2,    7,    2,   -3,    1,    1,    0,
    41,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    9,    8,  -16,
    42,   -1,   -7,    5,   -1,    0,    2,   -3,    3,    2,    0,   -3,    2,    1,    2,   -4,    1,   -2,    3,    4,   -4,    3,   -2,   -1,    0,    0,    4,    0,  -16,
    47,    1,    5,   -1,   -3,   -2,   -1,    3,   -1,   -3,   -5,    5,    2,    4,   -1,   -6,    1,   -1,    2,    0,    1,   -2,    0,    1,    2,   -2,    1,    0,  -16,
    -5,   13,   -5,  -12,   -5,   14,    3,  -13,  -18,    1,    1,    7,   -4,  -29,    4,   20,   21,    1,   -4,    0,    3,    8,   13,   -4,  -18,    0,    1,    0,  -51,
    43,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   10,    8,  -16,
    63,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   15,   13,  -20,
    53,    0,    1,    1,    0,   -1,   -1,   -1,    0,    2,   -2,    1,    2,    2,   -1,    0,    0,    0,    2,    0,   -1,    0,    0,   -1,    0,   -1,   10,    5,  -30,
    59,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   15,   11,  -16,
    -2,   19,   -6,  -14,   -7,  -17,    6,   11,   14,  -17,    9,    9,    5,   -3,    3,    6,   -8,   17,  -12,  -13,   -4,    8,  -14,   -7,    2,   11,    1,    0,    0,
    50,    3,   -5,   -4,    3,    0,   -1,   -2,    2,    3,    2,    2,   -6,   -5,    5,    3,    2,   -1,   -1,    1,    0,    3,    1,    0,   -4,    1,    2,    0,  -16,
    28,    0,    2,    0,   -1,   -1,   -1,    0,    1,    0,    0,   -1,    1,    1,    0,    1,   -1,    0,    0,    0,    1,    0,    0,    0,    0,    0,    6,    6,  -21
};
static const int32_t x264_nn_i16_b0[32] = {   -512,    -463,    -343,    -423,    -332,    -455,    -534,    -474,    -514,    -143,    -511,    -495,    -449,    -476,     -75,      81,      34,    -444,    -521,    -492,      27,    -420,    -472,    -499,      36,    -438,    -413,     425,    -391,      18,    -509,    -476};
static const int16_t x264_nn_i16_m0[32] = { 23580,  23580,  23580,  23580,  23580,  23580,  23580,  23580,  23580,  23580,  23580,  23580,  23580,  23580,  23580,  23580,  23580,  23580,  23580,  23580,  23580,  23580,  23580,  23580,  23580,  23580,  23580,  23580,  23580,  23580,  23580,  23580};

static const int8_t  x264_nn_i16_w1[16 * 32] = {
   118,   54,   22,   45,    7,   55,  109,   87,  127,   10,   67,  109,   77,  105,    3,    2,    9,   52,   97,   92,   10,   38,   79,  100,    9,   56,   43,   22,   24,    9,  100,   79,
   118,   54,   22,   45,    6,   55,  110,   88,  127,   10,   67,  109,   77,  106,    3,    2,    9,   52,   98,   93,   11,   39,   80,  101,    9,   56,   43,   22,   24,    9,  101,   80,
     0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
     0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
     0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
   118,   54,   22,   45,    7,   54,  109,   87,  127,   10,   67,  109,   77,  105,    3,    2,    9,   52,   98,   93,   11,   39,   79,  100,    9,   56,   43,   22,   24,    9,  100,   79,
   118,   54,   22,   45,    7,   55,  109,   87,  127,   10,   67,  109,   77,  105,    3,    2,    9,   52,   98,   93,   10,   39,   79,  100,    9,   56,   43,   22,   24,    9,  100,   79,
   118,   54,   22,   45,    6,   54,  109,   87,  126,   10,   67,  109,   77,  105,    3,    2,    9,   52,   97,   92,   11,   38,   79,  100,    9,   56,   43,   22,   24,    8,  100,   79,
   118,   54,   21,   45,    7,   54,  109,   87,  126,   10,   67,  109,   77,  105,    3,    2,    9,   52,   97,   92,   10,   38,   79,  100,    9,   55,   43,   22,   24,    8,  100,   79,
     0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
   118,   54,   22,   45,    7,   54,  109,   87,  126,   10,   67,  108,   77,  105,    3,    2,    9,   51,   97,   92,   10,   38,   79,  100,    9,   56,   43,   22,   24,    8,  100,   79,
     0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
   118,   53,   21,   44,    6,   54,  109,   87,  126,   10,   67,  108,   77,  105,    3,    2,    8,   51,   97,   92,   10,   38,   79,  100,    9,   55,   43,   22,   24,    9,   99,   79,
   118,   54,   21,   45,    7,   55,  109,   87,  127,   10,   67,  109,   77,  105,    3,    2,    9,   52,   98,   92,   11,   39,   79,  100,    9,   56,   43,   22,   24,    9,  100,   79,
   118,   54,   22,   45,    7,   54,  109,   87,  127,   10,   67,  109,   77,  105,    3,    2,    9,   52,   97,   92,   11,   38,   79,  100,    9,   56,   43,   22,   24,    9,  100,   79,
     0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0
};
static const int32_t x264_nn_i16_b1[16] = {     46,      46,      -1,      -1,      -1,      46,      46,      46,      46,      -1,      46,      -1,      45,      46,      46,      -1};
static const int16_t x264_nn_i16_m1[16] = { 24433,  24433,  24433,  24433,  24433,  24433,  24433,  24433,  24433,  24433,  24433,  24433,  24433,  24433,  24433,  24433};

static const int8_t  x264_nn_i16_w2[2 * 16] = {
   127,  127,    0,    0,    0,  126,  127,  126,  127,    0,  127,    0,  126,  127,  126,    0,
     0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0
};
static const int32_t x264_nn_i16_b2[2] = {    -12,       0};
static const int16_t x264_nn_i16_m2[2] = { 20847,  20847};

static const x264_nn_layer_t x264_nn_i16_layers[X264_NN_I16_NUM_LAYERS] =
{
    { x264_nn_i16_w0, x264_nn_i16_b0, x264_nn_i16_m0, 15, X264_NN_I16_CTX_DIM, X264_NN_I16_H1, 0 },
    { x264_nn_i16_w1, x264_nn_i16_b1, x264_nn_i16_m1, 18, X264_NN_I16_H1,      X264_NN_I16_H2, 0 },
    { x264_nn_i16_w2, x264_nn_i16_b2, x264_nn_i16_m2, 14, X264_NN_I16_H2,      X264_NN_I16_POLICY_COUNT, 1 },
};

/* Normalizes a raw pixel to 8-bit range regardless of BIT_DEPTH. */
#define NN_PX(p) ((int)(p) >> (BIT_DEPTH-8))

static int8_t nn_pixel( int v8 )
{
    return (int8_t)(v8 - 128);
}

/* Gathers the X264_NN_I16_CTX_DIM-int8 context for the current macroblock.
 * Layout (all values already centered to int8, i.e. "real value - 128" for
 * pixel-derived ones):
 *   [0]      qp - 26
 *   [1..16]  means of the 16 4x4 blocks of the source (raster order)
 *   [17..20] means of the 4 groups of 4 samples of the top border
 *   [21..24] same, for the left border
 *   [25]     top-left sample
 *   [26,27]  log-energy of the horizontal / vertical gradient of the source
 *   [28]     64 if the source looks flat, 0 otherwise
 * This layout is a contract with the offline trainer: changing it
 * invalidates the weight data above. */
static void nn_i16_gather_context( x264_t *h, int qp, int8_t ctx[X264_NN_I16_CTX_DIM] )
{
    pixel *src = h->mb.pic.p_fenc[0];
    pixel *dst = h->mb.pic.p_fdec[0];
    int gx = 0, gy = 0;

    ctx[0] = (int8_t)x264_clip3( qp - 26, -128, 127 );

    for( int by = 0; by < 4; by++ )
        for( int bx = 0; bx < 4; bx++ )
        {
            int sum = 0;
            for( int y = 0; y < 4; y++ )
                for( int x = 0; x < 4; x++ )
                    sum += NN_PX( src[(by*4+y)*FENC_STRIDE + bx*4+x] );
            ctx[1 + by*4+bx] = nn_pixel( (sum + 8) >> 4 );
        }

    for( int i = 0; i < 4; i++ )
    {
        int top = 0, left = 0;
        for( int j = 0; j < 4; j++ )
        {
            top  += NN_PX( dst[4*i+j - FDEC_STRIDE] );
            left += NN_PX( dst[-1 + (4*i+j)*FDEC_STRIDE] );
        }
        ctx[17+i] = nn_pixel( (top  + 2) >> 2 );
        ctx[21+i] = nn_pixel( (left + 2) >> 2 );
    }
    ctx[25] = nn_pixel( NN_PX( dst[-1 - FDEC_STRIDE] ) );

    /* p_fdec's edge pixels for an unavailable neighbour aren't reconstructed
     * content, so zero those slots (neutral/no-info, since ctx is centered
     * by nn_pixel()) rather than feed the net scratch-memory noise. */
    if( !(h->mb.i_neighbour_intra & MB_TOP) )
        ctx[17] = ctx[18] = ctx[19] = ctx[20] = 0;
    if( !(h->mb.i_neighbour_intra & MB_LEFT) )
        ctx[21] = ctx[22] = ctx[23] = ctx[24] = 0;
    if( !(h->mb.i_neighbour_intra & MB_TOPLEFT) )
        ctx[25] = 0;

    for( int y = 0; y < 16; y++ )
        for( int x = 0; x < 16; x++ )
        {
            int v = NN_PX( src[y*FENC_STRIDE + x] );
            if( x < 15 ) gx += abs( NN_PX( src[y*FENC_STRIDE + x+1] ) - v );
            if( y < 15 ) gy += abs( NN_PX( src[(y+1)*FENC_STRIDE + x] ) - v );
        }
    ctx[26] = (int8_t)(x264_nn_log8( gx ) - 64);
    ctx[27] = (int8_t)(x264_nn_log8( gy ) - 64);
    ctx[28] = (gx + gy) < (1 << 8) ? 64 : 0;
}

x264_nn_i16_policy_t x264_nn_i16_policy( x264_t *h, int qp, int32_t stop_bias )
{
    int8_t ctx[X264_NN_I16_CTX_DIM];
    int32_t regret[X264_NN_I16_POLICY_COUNT];

    nn_i16_gather_context( h, qp, ctx );
    x264_nn_forward( x264_nn_i16_layers, X264_NN_I16_NUM_LAYERS, ctx, regret );

    return (regret[X264_NN_I16_POLICY_STOP] - stop_bias <= regret[X264_NN_I16_POLICY_CONTINUE])
         ? X264_NN_I16_POLICY_STOP : X264_NN_I16_POLICY_CONTINUE;
}
