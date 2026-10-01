/*****************************************************************************
 * nn.c: tiny int8 neural-net forward pass
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

#include "base.h"
#include "nn.h"

/* Requantizes a raw int32 accumulator: pre-shift with rounding (saturating to
 * int16), multiply by the channel's multiplier, shift down with rounding. */
static int32_t nn_requant( int32_t acc, int32_t mult, int shift )
{
    int32_t a = (acc + (1 << (X264_NN_ACC_SHIFT-1))) >> X264_NN_ACC_SHIFT;
    a = x264_clip3( a, -32768, 32767 );
    return (a * mult + (1 << (shift-1))) >> shift;
}

void x264_nn_forward( const x264_nn_layer_t *layers, int n_layers,
                       const int8_t *input, int32_t *logits )
{
    int8_t buf[2][X264_NN_MAX_WIDTH];
    int cur = 0;
    const int8_t *in = input;

    for( int l = 0; l < n_layers; l++ )
    {
        const x264_nn_layer_t *layer = &layers[l];
        for( int o = 0; o < layer->n_out; o++ )
        {
            int32_t acc = layer->bias[o];
            const int8_t *w = layer->weight + o * layer->n_in;
            for( int i = 0; i < layer->n_in; i++ )
                acc += w[i] * in[i];
            int32_t v = nn_requant( acc, layer->mult[o], layer->shift );
            if( layer->b_output )
                logits[o] = x264_clip3( v, -32768, 32767 );
            else
                buf[cur][o] = (int8_t)x264_clip3( v, 0, X264_NN_ACT_MAX );
        }
        if( !layer->b_output )
        {
            in = buf[cur];
            cur ^= 1;
        }
    }
}

int x264_nn_log8( int g )
{
    uint32_t v = (uint32_t)g + 1;
    int b = 0;
    while( (v >> (b+1)) != 0 )
        b++;
    int r = 8*b + (int)(((v << 3) >> b) & 7);
    return X264_MIN( r, 127 );
}
