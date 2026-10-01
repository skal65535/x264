/*****************************************************************************
 * nn_b16.h: NN-assisted B-slice bidirectional (BI) prediction pruning
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

#ifndef X264_ENCODER_NN_B16_H
#define X264_ENCODER_NN_B16_H

/* Whether nn_b16.c's trained weights are real (1) or all-zero placeholders
 * (0) -- independent flag, same reasoning as X264_NN_P16_WEIGHTS_TRAINED/
 * X264_NN_P8X8_WEIGHTS_TRAINED. */
#define X264_NN_B16_WEIGHTS_TRAINED 1

/* Width of the int8 context gathered by nn_b16_gather_context() in
 * nn_b16.c; see that function for the exact layout. A contract with the
 * weight data in nn_b16.c and with whatever offline trainer produces it. */
#define X264_NN_B16_CTX_DIM 11

typedef enum
{
    X264_NN_B16_POLICY_STOP = 0,     /* skip computing BI; keep the best of L0/L1/Direct */
    X264_NN_B16_POLICY_CONTINUE = 1, /* compute BI too, as usual */
    X264_NN_B16_POLICY_COUNT
} x264_nn_b16_policy_t;

/* Called once per B-slice MB inside mb_analyse_inter_b16x16(), right after
 * the L0/L1 16x16 searches complete but before the BI compositing/
 * comparison block runs. Gathers a cheap context from the completed L0/L1
 * (and, if available, Direct) results plus source-pixel texture, runs the
 * policy net, and returns STOP if computing BI is predicted to cost less
 * than 'stop_bias' worth of RD regret. Higher 'stop_bias' (a higher --nn
 * level) makes STOP more likely.
 *
 * x264_mb_analysis_t is private to analyse.c, so the relevant L0/L1/Direct
 * results are passed explicitly (same convention as x264_nn_p16_policy()/
 * x264_nn_p8x8_policy()):
 *   satd_l0/l1    a->l0/l1.me16x16.cost - .cost_mv - .i_ref_cost (pixel
 *                 SATD, no lambda cost)
 *   cost_mv_l0/l1 a->l0/l1.me16x16.cost_mv
 *   ref_l0/l1     a->l0/l1.me16x16.i_ref
 *   cost_direct   a->i_cost16x16direct (COST_MAX if direct isn't available
 *                 -- see b_direct_available; the net still produces a
 *                 valid decision, it just sees a saturated "infinitely
 *                 bad" signal)
 *
 * Only call when X264_NN_B16_WEIGHTS_TRAINED is set and after the L0/L1
 * searches in mb_analyse_inter_b16x16() have completed (its internal
 * B_SKIP early return must not have been taken -- that path never reaches
 * the BI block, so this policy is moot there). */
x264_nn_b16_policy_t x264_nn_b16_policy( x264_t *h, int qp, int32_t satd_l0, int32_t cost_mv_l0, int ref_l0,
                                          int32_t satd_l1, int32_t cost_mv_l1, int ref_l1,
                                          int32_t cost_direct, int32_t stop_bias );

#endif
