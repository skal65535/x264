/*****************************************************************************
 * nn.h: tiny int8 neural-net forward pass, used to prune encoder mode-decision
 * candidates (see encoder/nn_intra.c for the first user).
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

#ifndef X264_COMMON_NN_H
#define X264_COMMON_NN_H

#include <stdint.h>

/* Fixed-point layout shared by every tiny MLP used for mode-decision
 * pruning. All-integer, so a forward pass is bit-exact across
 * compilers/platforms/archs.
 *
 *   activations: int8 (pixel-derived inputs are value-128).
 *   weights: int8, one row of 'n_in' values per output channel.
 *   accumulator: int32 = bias + sum(weight * activation).
 *   requantization (nn_requant() in nn.c): round down by
 *     X264_NN_ACC_SHIFT bits (saturating to int16), multiply by a
 *     per-channel int16 'mult', shift down (rounding) by 'shift'.
 *   hidden layers: ReLU'd and clamped to [0, X264_NN_ACT_MAX].
 *   output layers: kept signed, saturated to int16, no ReLU.
 *
 * A contract with whatever process produces the weight headers -- changing
 * this invalidates every trained weight table. */
#define X264_NN_ACC_SHIFT 4
#define X264_NN_ACT_MAX    127
#define X264_NN_MAX_WIDTH   32 /* largest supported layer width (n_in or n_out) */

typedef struct
{
    const int8_t  *weight; /* [n_out][n_in], row-major */
    const int32_t *bias;   /* [n_out] */
    const int16_t *mult;   /* [n_out], per-channel requant multiplier */
    int shift;              /* per-layer requant shift (>= 1) */
    int n_in;
    int n_out;
    int b_output;           /* 0: hidden layer (ReLU'd int8 out); 1: final layer (int32 logits) */
} x264_nn_layer_t;

/* Runs a forward pass through a stack of dense layers. 'input' holds
 * layers[0].n_in int8 values; every layer but the last has b_output == 0,
 * the last has b_output == 1. 'logits' receives the last layer's n_out
 * int32 values. Every layer's n_in/n_out must be <= X264_NN_MAX_WIDTH. */
void x264_nn_forward( const x264_nn_layer_t *layers, int n_layers,
                       const int8_t *input, int32_t *logits );

/* 8 * log2(1+g) for g >= 0, piecewise-linear between powers of two, capped
 * to 127 -- cheap log-compression for context features (gradient energy,
 * SATD, ...) that can span several orders of magnitude. Must match
 * whatever process produced the net's weights. */
int x264_nn_log8( int g );

#endif
