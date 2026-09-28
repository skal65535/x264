/*****************************************************************************
 * nn_p8x8.h: NN-assisted sub-8x8 partition-depth pruning
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

#ifndef X264_ENCODER_NN_P8X8_H
#define X264_ENCODER_NN_P8X8_H

/* Whether nn_p8x8.c's trained weights are real (1) or all-zero placeholders
 * (0) -- independent flag, same reasoning as X264_NN_P16_WEIGHTS_TRAINED
 * (nn_p16.h). */
#define X264_NN_P8X8_WEIGHTS_TRAINED 1

/* Width of the int8 context gathered by nn_p8x8_gather_context() in
 * nn_p8x8.c; see that function for the exact layout. A contract with the
 * weight data in nn_p8x8.c and with whatever offline trainer produces it. */
#define X264_NN_P8X8_CTX_DIM 13

typedef enum
{
    X264_NN_P8X8_POLICY_STOP = 0,     /* commit to this 8x8 block undivided, skip p4x4/p8x4/p4x8 */
    X264_NN_P8X8_POLICY_CONTINUE = 1, /* proceed to mb_analyse_inter_p4x4()/p8x4()/p4x8() as usual */
    X264_NN_P8X8_POLICY_COUNT
} x264_nn_p8x8_policy_t;

/* Per-8x8-block analogue of x264_nn_p16_policy() -- called once per block
 * (idx8x8 in 0..3) right after mb_analyse_inter_p8x8()/_mixed_ref() has
 * populated a->l0.me8x8[idx8x8]. Gathers a cheap context from that block's
 * ME result plus source-pixel texture, runs the policy net, and returns
 * STOP if skipping p4x4/p8x4/p4x8 for this block is predicted to cost less
 * than 'stop_bias' worth of RD regret.
 *
 * x264_mb_analysis_t is private to analyse.c, so its ME-search results are
 * passed explicitly (same convention as x264_nn_p16_policy()):
 *   satd_8x8    a->l0.me8x8[idx8x8].cost - .cost_mv - .i_ref_cost
 *   cost_mv_8x8 a->l0.me8x8[idx8x8].cost_mv
 *   ref_8x8     a->l0.me8x8[idx8x8].i_ref
 *
 * Only call when X264_NN_P8X8_WEIGHTS_TRAINED is set and after
 * mb_analyse_inter_p8x8()/_mixed_ref() has completed for this MB. */
x264_nn_p8x8_policy_t x264_nn_p8x8_policy( x264_t *h, int idx8x8, int qp, int32_t satd_8x8,
                                            int32_t cost_mv_8x8, int ref_8x8, int32_t stop_bias );

#endif
