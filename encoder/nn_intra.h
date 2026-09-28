/*****************************************************************************
 * nn_intra.h: NN-assisted Intra16/Intra4 mode-decision pruning
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

#ifndef X264_ENCODER_NN_INTRA_H
#define X264_ENCODER_NN_INTRA_H

/* Number of --nn levels, including level 0 (off). See x264_nn_i16_bias[] in
 * analyse.c for what each level means. */
#define X264_NN_NUM_LEVELS 6

/* Whether nn_intra.c's trained weights are real (1) or all-zero placeholders
 * (0); validate_parameters() forces i_nn_level to 0 while this is 0, so
 * placeholders are never exercised. */
#define X264_NN_WEIGHTS_TRAINED 1

/* Width of the int8 context gathered by nn_i16_gather_context() in
 * nn_intra.c; see that function for the exact layout. A contract with the
 * weight data in nn_intra.c and with whatever offline trainer produces it. */
#define X264_NN_I16_CTX_DIM 29

typedef enum
{
    X264_NN_I16_POLICY_STOP = 0,     /* only try I_16x16, skip I_8x8/I_4x4 for this MB */
    X264_NN_I16_POLICY_CONTINUE = 1, /* proceed to I_8x8/I_4x4 as usual */
    X264_NN_I16_POLICY_COUNT
} x264_nn_i16_policy_t;

/* Gathers a cheap pixel/QP context for the current macroblock
 * (h->mb.pic.p_fenc/p_fdec), runs the policy net, and returns STOP if
 * skipping I_8x8/I_4x4 is predicted to cost less than 'stop_bias' worth of
 * RD regret. Higher 'stop_bias' (a higher --nn level) makes STOP more
 * likely.
 *
 * Only call when X264_NN_WEIGHTS_TRAINED is set; x264_macroblock_analyse()
 * already gates on h->param.analyse.i_nn_level. */
x264_nn_i16_policy_t x264_nn_i16_policy( x264_t *h, int qp, int32_t stop_bias );

#endif
