/*****************************************************************************
 * nn_p16.h: NN-assisted P-slice partition-depth pruning
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

#ifndef X264_ENCODER_NN_P16_H
#define X264_ENCODER_NN_P16_H

/* Whether nn_p16.c's trained weights are real (1) or all-zero placeholders
 * (0) -- independent of X264_NN_WEIGHTS_TRAINED (nn_intra.h), since this
 * net trains and lands on its own schedule despite sharing the --nn
 * knob. */
#define X264_NN_P16_WEIGHTS_TRAINED 1

/* Width of the int8 context gathered by nn_p16_gather_context() in
 * nn_p16.c; see that function for the exact layout. A contract with the
 * weight data in nn_p16.c and with whatever offline trainer produces it. */
#define X264_NN_P16_CTX_DIM 13

typedef enum
{
    X264_NN_P16_POLICY_STOP = 0,     /* commit to P16x16, skip P8x8/P16x8/P8x16 search */
    X264_NN_P16_POLICY_CONTINUE = 1, /* proceed to mb_analyse_inter_p8x8() as usual */
    X264_NN_P16_POLICY_COUNT
} x264_nn_p16_policy_t;

/* Gathers a cheap context from the completed P16x16 motion search plus
 * source-pixel texture, runs the policy net, and returns STOP if skipping
 * the P8x8/P16x8/P8x16 sub-partition search is predicted to cost less than
 * 'stop_bias' worth of RD regret. Higher 'stop_bias' (a higher --nn level)
 * makes STOP more likely.
 *
 * x264_mb_analysis_t is private to analyse.c, so its P16x16 search results
 * are passed explicitly (same convention as x264_nn_i4rank_topk()):
 *   satd_p16x16    a->l0.me16x16.cost - .cost_mv - .i_ref_cost (pixel SATD
 *                  vs. the winning prediction, no lambda cost)
 *   cost_mv_p16x16 a->l0.me16x16.cost_mv (mv-signaling cost)
 *   ref_p16x16     a->l0.me16x16.i_ref
 *
 * Only call when X264_NN_P16_WEIGHTS_TRAINED is set and after
 * mb_analyse_inter_p16x16() has completed (a->l0.me16x16 must hold a real
 * result, not sentinel garbage). */
x264_nn_p16_policy_t x264_nn_p16_policy( x264_t *h, int qp, int32_t satd_p16x16,
                                          int32_t cost_mv_p16x16, int ref_p16x16,
                                          int32_t stop_bias );

#endif
