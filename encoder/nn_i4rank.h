/*****************************************************************************
 * nn_i4rank.h: Intra4 mode-ranking proxy
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

#ifndef X264_ENCODER_NN_I4RANK_H
#define X264_ENCODER_NN_I4RANK_H

/* Ranks which of the 9 canonical Intra4 directions (common/predict.h
 * intra4x4_pred_e 0-8) are worth the expensive rd_cost_i4x4() evaluation
 * for one 4x4 sub-block, vs. the cheap SATD this ranker uses. Used by
 * intra_rd_refine(), which otherwise evaluates every legal mode with no
 * pruning; only called at analyse.i_mbrd >= 2.
 *
 * A single linear layer (19 features -> 9 logits, no hidden layers). */
#define X264_NN_I4RANK_NUM_MODES 9
#define X264_NN_I4RANK_NUM_FEATURES 19  /* qp(1) + dp[9] (relative SATD-cost gap) + dh[9] (mode-signal-cost gap) */

/* Ranks the legal candidates for 4x4 sub-block 'idx' (p_src_by/p_dst_by are
 * this block's luma origin) and writes up to X264_NN_I4RANK_NUM_MODES + 4
 * mode indices into out_modes, best-ranked first, terminated by -1
 * (drop-in for predict_4x4_mode_available()'s convention). 'topk' caps how
 * many of the 9 canonical modes are kept; index >= 9 (DC_LEFT/DC_TOP/
 * DC_128 edge fallbacks) always passes through unranked, as does
 * everything if fewer than all 9 are legal. Returns the entry count
 * (excluding the terminator). */
int x264_nn_i4rank_topk( x264_t *h, int idx, int qp, int lambda,
                         const int8_t *predict_mode, pixel *p_src_by, pixel *p_dst_by,
                         int topk, int8_t out_modes[X264_NN_I4RANK_NUM_MODES + 5] );

#endif
