#include "cpu_types.hpp"

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

namespace cpu_utils {

void eagle_prepare_inputs_padded_kernel_impl(
    const torch::Tensor& cu_num_draft_tokens,
    const torch::Tensor& valid_sampled_tokens_count,
    const torch::Tensor& query_start_loc_gpu,
    torch::Tensor& token_indices_to_sample,
    torch::Tensor& num_rejected_tokens_gpu, const int64_t num_reqs) {
  const int64_t* cu_draft_ptr = cu_num_draft_tokens.data_ptr<int64_t>();
  const int64_t* valid_count_ptr =
      valid_sampled_tokens_count.data_ptr<int64_t>();
  const int32_t* query_loc_ptr = query_start_loc_gpu.data_ptr<int32_t>();
  int32_t* indices_out_ptr = token_indices_to_sample.data_ptr<int32_t>();
  int64_t* rejected_out_ptr = num_rejected_tokens_gpu.data_ptr<int64_t>();

#pragma omp parallel for
  for (int64_t req_idx = 0; req_idx < num_reqs; ++req_idx) {
    int64_t start_idx = req_idx == 0 ? 0 : cu_draft_ptr[req_idx - 1];
    int64_t num_draft_tokens = cu_draft_ptr[req_idx] - start_idx;
    int64_t num_valid_tokens = valid_count_ptr[req_idx];

    int64_t num_rejected = 0;
    if (num_draft_tokens > 0) {
      num_rejected = num_draft_tokens + 1 - num_valid_tokens;
    }

    int32_t q_last_tok_idx = query_loc_ptr[req_idx + 1] - 1;
    int32_t index_to_sample = q_last_tok_idx - num_rejected;

    indices_out_ptr[req_idx] = index_to_sample;
    rejected_out_ptr[req_idx] = num_rejected;
  }
}

void eagle_prepare_next_token_padded_kernel_impl(
    const torch::Tensor& sampled_token_ids,
    const torch::Tensor& discard_request_mask,
    const torch::Tensor& backup_next_token_ids, torch::Tensor& next_token_ids,
    torch::Tensor& valid_sampled_tokens_count, const int64_t vocab_size,
    const int64_t num_sampled_tokens_per_req, const int64_t num_reqs) {
  const int64_t* sampled_ids_ptr = sampled_token_ids.data_ptr<int64_t>();
  const bool* discard_mask_ptr = discard_request_mask.data_ptr<bool>();
  const int64_t* backup_ids_ptr = backup_next_token_ids.data_ptr<int64_t>();
  int64_t* next_ids_out_ptr = next_token_ids.data_ptr<int64_t>();
  int64_t* valid_count_out_ptr = valid_sampled_tokens_count.data_ptr<int64_t>();

  const int64_t stride = sampled_token_ids.stride(0);

#pragma omp parallel for
  for (int64_t req_idx = 0; req_idx < num_reqs; ++req_idx) {
    const int64_t* row_ptr = sampled_ids_ptr + req_idx * stride;
    int64_t valid_count = 0;
    int64_t last_valid_token = -1;

    for (int64_t pos = 0; pos < num_sampled_tokens_per_req; ++pos) {
      int64_t token = row_ptr[pos];
      if (token != -1 && token < vocab_size) {
        valid_count++;
        last_valid_token = token;
      }
    }

    bool discard = discard_mask_ptr[req_idx];
    if (discard) {
      next_ids_out_ptr[req_idx] = backup_ids_ptr[req_idx];
      valid_count_out_ptr[req_idx] = 0;
    } else {
      next_ids_out_ptr[req_idx] =
          (valid_count > 0) ? last_valid_token : backup_ids_ptr[req_idx];
      valid_count_out_ptr[req_idx] = valid_count;
    }
  }
}

void eagle_step_slot_mapping_metadata_kernel_impl(
    const torch::Tensor& positions, const torch::Tensor& block_table,
    torch::Tensor& seq_lens, torch::Tensor& out_clamped_positions,
    torch::Tensor& out_slot_mapping, const int64_t block_size,
    const int64_t max_model_len, const int64_t PAD_ID) {
  const int64_t batch_size = positions.size(0);
  const int64_t input_batch_size = out_slot_mapping.size(0);

  const int64_t* pos_ptr = positions.data_ptr<int64_t>();
  const int32_t* bt_ptr = block_table.data_ptr<int32_t>();
  int32_t* seq_lens_ptr = seq_lens.data_ptr<int32_t>();
  int64_t* out_clamped_ptr = out_clamped_positions.data_ptr<int64_t>();
  int64_t* out_slot_ptr = out_slot_mapping.data_ptr<int64_t>();

  const int64_t bt_stride = block_table.stride(0);
  const int64_t n_blocks_per_req = block_table.size(1);

#pragma omp parallel for
  for (int64_t req_idx = 0; req_idx < input_batch_size; ++req_idx) {
    if (req_idx >= batch_size) {
      out_slot_ptr[req_idx] = PAD_ID;
      continue;
    }

    int64_t position = pos_ptr[req_idx];
    int64_t new_position = position + 1;
    bool exceeds_max = new_position >= max_model_len;
    int64_t clamped_position = exceeds_max ? 0 : new_position;

    out_clamped_ptr[req_idx] = clamped_position;

    int64_t block_number = clamped_position / block_size;
    block_number = std::min(block_number, n_blocks_per_req - 1);
    int32_t block_id = bt_ptr[req_idx * bt_stride + block_number];
    int64_t slot_id = block_id * block_size + (clamped_position % block_size);
    out_slot_ptr[req_idx] = exceeds_max ? PAD_ID : slot_id;

    int32_t seq_len = seq_lens_ptr[req_idx];
    int32_t new_seq_len = exceeds_max ? 1 : (seq_len + 1);
    new_seq_len = std::min(new_seq_len, static_cast<int32_t>(max_model_len));
    seq_lens_ptr[req_idx] = new_seq_len;
  }
}

void copy_and_expand_eagle_inputs_kernel_impl(
    const torch::Tensor& target_token_ids,
    const torch::Tensor& target_positions, const torch::Tensor& next_token_ids,
    torch::Tensor& out_input_ids, torch::Tensor& out_positions,
    torch::Tensor& out_is_rejected_token_mask,
    torch::Tensor& out_is_masked_token_mask,
    torch::Tensor& out_new_token_indices,
    torch::Tensor& out_hidden_state_mapping,
    const torch::Tensor& query_start_loc, const torch::Tensor& query_end_loc,
    const int64_t padding_token_id, const int64_t parallel_drafting_token_id,
    const int64_t total_input_tokens,
    const int64_t num_padding_slots_per_request, const bool shift_input_ids) {
  const int64_t num_reqs = query_end_loc.size(0);

  const int64_t* target_ids_ptr = target_token_ids.data_ptr<int64_t>();
  const int64_t* target_pos_ptr = target_positions.data_ptr<int64_t>();
  const int64_t* next_ids_ptr = next_token_ids.data_ptr<int64_t>();
  const int32_t* query_start_ptr = query_start_loc.data_ptr<int32_t>();
  const int32_t* query_end_ptr = query_end_loc.data_ptr<int32_t>();

  int64_t* out_ids_ptr = out_input_ids.data_ptr<int64_t>();
  int64_t* out_pos_ptr = out_positions.data_ptr<int64_t>();
  bool* out_rej_mask_ptr = out_is_rejected_token_mask.data_ptr<bool>();
  bool* out_mask_ptr = out_is_masked_token_mask.data_ptr<bool>();
  int32_t* out_new_idx_ptr = out_new_token_indices.data_ptr<int32_t>();
  int32_t* out_hidden_map_ptr = out_hidden_state_mapping.data_ptr<int32_t>();

#pragma omp parallel for
  for (int64_t req_idx = 0; req_idx < num_reqs; ++req_idx) {
    int32_t q_start = query_start_ptr[req_idx];
    int32_t next_q_start = query_start_ptr[req_idx + 1];
    int32_t q_end = query_end_ptr[req_idx];

    int64_t num_valid_tokens =
        shift_input_ids ? (q_end - q_start) : (q_end - q_start + 1);
    int64_t input_offset = shift_input_ids ? 1 : 0;

    int64_t out_start = q_start + req_idx * (num_padding_slots_per_request -
                                             (shift_input_ids ? 1 : 0));
    int64_t num_rejected = next_q_start - q_end - 1;
    int64_t total_output_tokens =
        num_valid_tokens + num_padding_slots_per_request + num_rejected;

    int64_t start_pos = target_pos_ptr[q_start];
    int64_t bonus_token = next_ids_ptr[req_idx];

    for (int64_t j = 0; j < total_output_tokens; ++j) {
      int64_t out_idx = out_start + j;
      bool is_valid = j < num_valid_tokens;
      bool is_bonus = j == num_valid_tokens;
      bool is_parallel = (j > num_valid_tokens) &&
                         (j < num_valid_tokens + num_padding_slots_per_request);
      bool is_rejected = j >= num_valid_tokens + num_padding_slots_per_request;

      int64_t in_idx =
          std::min(static_cast<int64_t>(q_start + input_offset + j),
                   total_input_tokens - 1);

      int64_t token_id = padding_token_id;
      if (is_valid)
        token_id = target_ids_ptr[in_idx];
      else if (is_bonus)
        token_id = bonus_token;
      else if (is_parallel)
        token_id = parallel_drafting_token_id;

      out_ids_ptr[out_idx] = token_id;
      out_pos_ptr[out_idx] = is_rejected ? 0 : (start_pos + j);
      out_rej_mask_ptr[out_idx] = is_rejected;
      out_mask_ptr[out_idx] = is_parallel;

      if (is_bonus || is_parallel) {
        int64_t new_token_local_idx = j - num_valid_tokens;
        int64_t new_token_out_idx =
            req_idx * num_padding_slots_per_request + new_token_local_idx;
        out_new_idx_ptr[new_token_out_idx] = out_idx;
      }
    }

    if (shift_input_ids) {
      int64_t n_input = next_q_start - q_start;
      for (int64_t j = 0; j < n_input; ++j) {
        out_hidden_map_ptr[q_start + j] = out_start + j;
      }
    }
  }
}

void copy_and_expand_dflash_inputs_kernel_impl(
    const torch::Tensor& next_token_ids, const torch::Tensor& target_positions,
    torch::Tensor& out_input_ids, torch::Tensor& out_context_positions,
    torch::Tensor& out_query_positions, torch::Tensor& out_context_slot_mapping,
    torch::Tensor& out_query_slot_mapping, torch::Tensor& out_token_indices,
    const torch::Tensor& block_table, const torch::Tensor& query_start_loc,
    const std::optional<torch::Tensor>& num_rejected_tokens,
    const int64_t parallel_drafting_token_id, const int64_t block_size,
    const int64_t num_query_per_req, const int64_t num_speculative_tokens,
    const int64_t total_input_tokens, const bool has_num_rejected) {
  const int64_t num_reqs = query_start_loc.size(0) - 1;

  const int64_t* next_ids_ptr = next_token_ids.data_ptr<int64_t>();
  const int64_t* target_pos_ptr = target_positions.data_ptr<int64_t>();
  const int32_t* block_table_ptr = block_table.data_ptr<int32_t>();
  const int32_t* query_start_ptr = query_start_loc.data_ptr<int32_t>();
  const int64_t* rejected_ptr =
      has_num_rejected && num_rejected_tokens.has_value()
          ? num_rejected_tokens.value().data_ptr<int64_t>()
          : nullptr;

  int64_t* out_ids_ptr = out_input_ids.data_ptr<int64_t>();
  int64_t* out_ctx_pos_ptr = out_context_positions.data_ptr<int64_t>();
  int64_t* out_query_pos_ptr = out_query_positions.data_ptr<int64_t>();
  int64_t* out_ctx_slot_ptr = out_context_slot_mapping.data_ptr<int64_t>();
  int64_t* out_query_slot_ptr = out_query_slot_mapping.data_ptr<int64_t>();
  int32_t* out_token_idx_ptr = out_token_indices.data_ptr<int32_t>();

  const int64_t block_table_stride = block_table.stride(0);

#pragma omp parallel for
  for (int64_t req_idx = 0; req_idx < num_reqs; ++req_idx) {
    int32_t ctx_start = query_start_ptr[req_idx];
    int32_t ctx_end = query_start_ptr[req_idx + 1];
    int64_t num_ctx = ctx_end - ctx_start;
    int64_t valid_ctx_end = ctx_end;
    if (rejected_ptr != nullptr) {
      valid_ctx_end -= rejected_ptr[req_idx];
    }
    // Guard against out-of-bounds: ensure valid_ctx_end > ctx_start so that
    // valid_ctx_end - 1 never reads before the request's context range.
    valid_ctx_end =
        std::max(valid_ctx_end, static_cast<int64_t>(ctx_start + 1));

    int64_t last_pos = target_pos_ptr[valid_ctx_end - 1];

    for (int64_t j = 0; j < num_ctx; ++j) {
      int64_t ctx_idx = ctx_start + j;
      int64_t ctx_pos_idx = std::min(ctx_idx, total_input_tokens - 1);
      int64_t position = target_pos_ptr[ctx_pos_idx];
      int64_t block_num = position / block_size;
      block_num = std::min(block_num, block_table_stride - 1);
      int32_t block_id =
          block_table_ptr[req_idx * block_table_stride + block_num];
      int64_t slot = block_id * block_size + (position % block_size);

      out_ctx_pos_ptr[ctx_idx] = position;
      out_ctx_slot_ptr[ctx_idx] = slot;
    }

    for (int64_t query_off = 0; query_off < num_query_per_req; ++query_off) {
      int64_t query_out = req_idx * num_query_per_req + query_off;
      int64_t position = last_pos + 1 + query_off;
      int64_t block_num = position / block_size;
      block_num = std::min(block_num, block_table_stride - 1);
      int32_t block_id =
          block_table_ptr[req_idx * block_table_stride + block_num];
      int64_t slot = block_id * block_size + (position % block_size);

      out_query_pos_ptr[query_out] = position;
      out_query_slot_ptr[query_out] = slot;
      out_ids_ptr[query_out] =
          query_off == 0 ? next_ids_ptr[req_idx] : parallel_drafting_token_id;

      if (query_off > 0) {
        int64_t sample_out_idx =
            req_idx * num_speculative_tokens + (query_off - 1);
        out_token_idx_ptr[sample_out_idx] = query_out;
      }
    }
  }
}

void rejection_greedy_sample_kernel_impl(
    torch::Tensor& output_token_ids, const torch::Tensor& cu_num_draft_tokens,
    const torch::Tensor& draft_token_ids, const torch::Tensor& target_argmax,
    const torch::Tensor& bonus_token_ids,
    const std::optional<torch::Tensor>& is_greedy, const int64_t max_spec_len) {
  const int64_t batch_size = cu_num_draft_tokens.size(0);

  int64_t* out_ptr = output_token_ids.data_ptr<int64_t>();
  const int64_t* cu_draft_ptr = cu_num_draft_tokens.data_ptr<int64_t>();
  const int64_t* draft_ids_ptr = draft_token_ids.data_ptr<int64_t>();
  const int64_t* target_argmax_ptr = target_argmax.data_ptr<int64_t>();
  const int64_t* bonus_ids_ptr = bonus_token_ids.data_ptr<int64_t>();
  const bool* greedy_ptr =
      is_greedy.has_value() ? is_greedy.value().data_ptr<bool>() : nullptr;

  const int64_t out_stride = output_token_ids.stride(0);
  const int64_t bonus_stride = bonus_token_ids.stride(0);

#pragma omp parallel for
  for (int64_t req_idx = 0; req_idx < batch_size; ++req_idx) {
    if (greedy_ptr && !greedy_ptr[req_idx]) continue;

    int64_t start_idx = req_idx == 0 ? 0 : cu_draft_ptr[req_idx - 1];
    int64_t end_idx = cu_draft_ptr[req_idx];
    int64_t num_draft_tokens = end_idx - start_idx;

    bool rejected = false;
    for (int64_t pos = 0; pos < num_draft_tokens; ++pos) {
      int64_t target_id = target_argmax_ptr[start_idx + pos];
      out_ptr[req_idx * out_stride + pos] = target_id;

      if (draft_ids_ptr[start_idx + pos] != target_id) {
        rejected = true;
        break;
      }
    }

    if (!rejected) {
      out_ptr[req_idx * out_stride + num_draft_tokens] =
          bonus_ids_ptr[req_idx * bonus_stride];
    }
  }
}

void rejection_random_sample_kernel_impl(
    torch::Tensor& output_token_ids, const torch::Tensor& cu_num_draft_tokens,
    const torch::Tensor& draft_token_ids,
    const std::optional<torch::Tensor>& draft_probs,
    const torch::Tensor& target_probs, const torch::Tensor& bonus_token_ids,
    const torch::Tensor& recovered_token_ids,
    const torch::Tensor& uniform_probs,
    const std::optional<torch::Tensor>& is_greedy, const int64_t max_spec_len,
    const int64_t vocab_size, const bool no_draft_probs) {
  const int64_t batch_size = cu_num_draft_tokens.size(0);

  int64_t* out_ptr = output_token_ids.data_ptr<int64_t>();
  const int64_t* cu_draft_ptr = cu_num_draft_tokens.data_ptr<int64_t>();
  const int64_t* draft_ids_ptr = draft_token_ids.data_ptr<int64_t>();
  const float* draft_probs_ptr =
      no_draft_probs ? nullptr : draft_probs.value().data_ptr<float>();
  const float* target_probs_ptr = target_probs.data_ptr<float>();
  const int64_t* bonus_ids_ptr = bonus_token_ids.data_ptr<int64_t>();
  const int64_t* recovered_ids_ptr = recovered_token_ids.data_ptr<int64_t>();
  const float* uniform_probs_ptr = uniform_probs.data_ptr<float>();
  const bool* greedy_ptr =
      is_greedy.has_value() ? is_greedy.value().data_ptr<bool>() : nullptr;

  const int64_t out_stride = output_token_ids.stride(0);
  const int64_t bonus_stride = bonus_token_ids.stride(0);
  const int64_t target_stride = target_probs.stride(0);
  const int64_t draft_probs_stride =
      no_draft_probs ? 0 : draft_probs.value().stride(0);

#pragma omp parallel for
  for (int64_t req_idx = 0; req_idx < batch_size; ++req_idx) {
    if (greedy_ptr && greedy_ptr[req_idx]) continue;

    int64_t start_idx = req_idx == 0 ? 0 : cu_draft_ptr[req_idx - 1];
    int64_t end_idx = cu_draft_ptr[req_idx];
    int64_t num_draft_tokens = end_idx - start_idx;

    bool rejected = false;
    for (int64_t pos = 0; pos < num_draft_tokens; ++pos) {
      int64_t token_idx = start_idx + pos;
      int64_t draft_id = draft_ids_ptr[token_idx];

      float p = target_probs_ptr[token_idx * target_stride + draft_id];
      float q =
          no_draft_probs
              ? 1.0f
              : draft_probs_ptr[token_idx * draft_probs_stride + draft_id];
      float uniform_p = uniform_probs_ptr[token_idx];

      float ratio = (q > 0.0f) ? (p / q) : 0.0f;

      if (ratio >= uniform_p) {
        out_ptr[req_idx * out_stride + pos] = draft_id;
      } else {
        out_ptr[req_idx * out_stride + pos] = recovered_ids_ptr[token_idx];
        rejected = true;
        break;
      }
    }

    if (!rejected) {
      out_ptr[req_idx * out_stride + num_draft_tokens] =
          bonus_ids_ptr[req_idx * bonus_stride];
    }
  }
}

void expand_kernel_impl(torch::Tensor& output, const torch::Tensor& input,
                        const torch::Tensor& cu_num_tokens,
                        const int64_t replace_from, const int64_t replace_to) {
  const int64_t batch_size = cu_num_tokens.size(0);
  const int64_t* cu_tokens_ptr = cu_num_tokens.data_ptr<int64_t>();

  int64_t* out_ptr = output.data_ptr<int64_t>();
  const int64_t* in_ptr = input.data_ptr<int64_t>();

#pragma omp parallel for
  for (int64_t req_idx = 0; req_idx < batch_size; ++req_idx) {
    int64_t start_idx = req_idx == 0 ? 0 : cu_tokens_ptr[req_idx - 1];
    int64_t end_idx = cu_tokens_ptr[req_idx];
    int64_t val = in_ptr[req_idx];

    if (val == replace_from) {
      val = replace_to;
    }

    for (int64_t i = start_idx; i < end_idx; ++i) {
      out_ptr[i] = val;
    }
  }
}

void sample_recovered_tokens_kernel_impl(
    torch::Tensor& output_token_ids, const torch::Tensor& cu_num_draft_tokens,
    const torch::Tensor& draft_token_ids,
    const std::optional<torch::Tensor>& draft_probs,
    const torch::Tensor& target_probs, const torch::Tensor& inv_q,
    const int64_t vocab_size, const bool no_draft_probs) {
  const int64_t batch_size = cu_num_draft_tokens.size(0);

  int64_t* out_ptr = output_token_ids.data_ptr<int64_t>();
  const int64_t* cu_draft_ptr = cu_num_draft_tokens.data_ptr<int64_t>();
  const int64_t* draft_ids_ptr = draft_token_ids.data_ptr<int64_t>();
  const float* draft_probs_ptr =
      no_draft_probs ? nullptr : draft_probs.value().data_ptr<float>();
  const float* target_probs_ptr = target_probs.data_ptr<float>();
  const float* inv_q_ptr = inv_q.data_ptr<float>();

  const int64_t target_stride = target_probs.stride(0);
  const int64_t draft_probs_stride =
      no_draft_probs ? 0 : draft_probs.value().stride(0);
  const int64_t inv_q_stride = inv_q.stride(0);

#pragma omp parallel for
  for (int64_t req_idx = 0; req_idx < batch_size; ++req_idx) {
    int64_t start_idx = req_idx == 0 ? 0 : cu_draft_ptr[req_idx - 1];
    int64_t end_idx = cu_draft_ptr[req_idx];
    int64_t num_draft_tokens = end_idx - start_idx;

    const float* req_inv_q = inv_q_ptr + req_idx * inv_q_stride;

    for (int64_t pos = 0; pos < num_draft_tokens; ++pos) {
      int64_t token_idx = start_idx + pos;
      int64_t draft_id = draft_ids_ptr[token_idx];

      const float* token_target_probs =
          target_probs_ptr + token_idx * target_stride;
      const float* token_draft_probs =
          no_draft_probs ? nullptr
                         : (draft_probs_ptr + token_idx * draft_probs_stride);

      int64_t best_id = 0;
      float best_val = -1.0f;

      for (int64_t v = 0; v < vocab_size; ++v) {
        float prob = token_target_probs[v];
        if (no_draft_probs) {
          if (v == draft_id) prob = 0.0f;
        } else {
          float diff = prob - token_draft_probs[v];
          prob = diff > 0.0f ? diff : 0.0f;
        }

        float val = prob * req_inv_q[v];
        if (val > best_val) {
          best_val = val;
          best_id = v;
        }
      }
      out_ptr[token_idx] = best_id;
    }
  }
}

namespace {

// Per-chunk argmax returning (max value as float, index of its first
// occurrence within the chunk). Strict `>` comparisons keep the first
// occurrence, matching aten::argmax on CPU for NaN-free input; NaN inputs
// are ignored rather than propagated (logits are NaN-free by construction).
template <typename scalar_t>
std::pair<float, int64_t> chunk_argmax_scalar(const scalar_t* p, int64_t n) {
  float best = -std::numeric_limits<float>::infinity();
  int64_t best_i = 0;
  for (int64_t i = 0; i < n; ++i) {
    float v = static_cast<float>(p[i]);
    if (v > best) {
      best = v;
      best_i = i;
    }
  }
  return {best, best_i};
}

#if defined(__AVX512F__)

// Horizontal finish for a lane-wise running (max, first-index) pair: the
// global first occurrence is the smallest recorded index among lanes that
// hold the global max.
inline std::pair<float, int64_t> hreduce_argmax(__m512 vmax, __m512i vidx) {
  float maxv = _mm512_reduce_max_ps(vmax);
  __mmask16 eq = _mm512_cmp_ps_mask(vmax, _mm512_set1_ps(maxv), _CMP_EQ_OQ);
  __m512i cand = _mm512_mask_mov_epi32(
      _mm512_set1_epi32(std::numeric_limits<int32_t>::max()), eq, vidx);
  return {maxv, _mm512_reduce_min_epi32(cand)};
}

inline void lane_update(__m512 v, __m512i cur, __m512& vmax, __m512i& vidx) {
  __mmask16 m = _mm512_cmp_ps_mask(v, vmax, _CMP_GT_OQ);
  vmax = _mm512_mask_mov_ps(vmax, m, v);
  vidx = _mm512_mask_mov_epi32(vidx, m, cur);
}

std::pair<float, int64_t> chunk_argmax(const float* p, int64_t n) {
  __m512 vmax = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
  __m512i vidx = _mm512_setzero_si512();
  __m512i cur = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                  13, 14, 15);
  const __m512i step = _mm512_set1_epi32(16);
  int64_t i = 0;
  for (; i + 16 <= n; i += 16) {
    lane_update(_mm512_loadu_ps(p + i), cur, vmax, vidx);
    cur = _mm512_add_epi32(cur, step);
  }
  auto [best, best_i] = hreduce_argmax(vmax, vidx);
  for (; i < n; ++i) {
    if (p[i] > best) {
      best = p[i];
      best_i = i;
    }
  }
  return {best, best_i};
}

// bf16 -> fp32 widening via zero-extend + shift; no AVX512-BF16 needed.
inline __m512 bf16_to_ps(__m256i x) {
  return _mm512_castsi512_ps(
      _mm512_slli_epi32(_mm512_cvtepu16_epi32(x), 16));
}

std::pair<float, int64_t> chunk_argmax(const at::BFloat16* p, int64_t n) {
  const uint16_t* u = reinterpret_cast<const uint16_t*>(p);
  __m512 vmax = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
  __m512i vidx = _mm512_setzero_si512();
  __m512i cur = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                  13, 14, 15);
  const __m512i step = _mm512_set1_epi32(16);
  int64_t i = 0;
  for (; i + 32 <= n; i += 32) {
    __m512i raw = _mm512_loadu_si512(u + i);
    lane_update(bf16_to_ps(_mm512_castsi512_si256(raw)), cur, vmax, vidx);
    cur = _mm512_add_epi32(cur, step);
    lane_update(bf16_to_ps(_mm512_extracti64x4_epi64(raw, 1)), cur, vmax,
                vidx);
    cur = _mm512_add_epi32(cur, step);
  }
  auto [best, best_i] = hreduce_argmax(vmax, vidx);
  for (; i < n; ++i) {
    float v = static_cast<float>(p[i]);
    if (v > best) {
      best = v;
      best_i = i;
    }
  }
  return {best, best_i};
}

#else

template <typename scalar_t>
std::pair<float, int64_t> chunk_argmax(const scalar_t* p, int64_t n) {
  return chunk_argmax_scalar(p, n);
}

#endif

template <typename scalar_t>
void greedy_argmax_typed(const scalar_t* logits, int64_t num_seqs,
                         int64_t vocab_size, int64_t* out) {
  // Two-level reduction: chunks expose num_seqs * num_chunks independent
  // tasks so small batches still fill the machine.
  constexpr int64_t CHUNK = 16384;
  const int64_t num_chunks = (vocab_size + CHUNK - 1) / CHUNK;
  std::vector<float> cmax(num_seqs * num_chunks);
  std::vector<int64_t> cidx(num_seqs * num_chunks);

#pragma omp parallel for collapse(2)
  for (int64_t s = 0; s < num_seqs; ++s) {
    for (int64_t c = 0; c < num_chunks; ++c) {
      const int64_t begin = c * CHUNK;
      const int64_t len = std::min(CHUNK, vocab_size - begin);
      auto [v, i] = chunk_argmax(logits + s * vocab_size + begin, len);
      cmax[s * num_chunks + c] = v;
      cidx[s * num_chunks + c] = begin + i;
    }
  }

  for (int64_t s = 0; s < num_seqs; ++s) {
    float best = cmax[s * num_chunks];
    int64_t best_i = cidx[s * num_chunks];
    for (int64_t c = 1; c < num_chunks; ++c) {
      if (cmax[s * num_chunks + c] > best) {
        best = cmax[s * num_chunks + c];
        best_i = cidx[s * num_chunks + c];
      }
    }
    out[s] = best_i;
  }
}

}  // namespace

torch::Tensor greedy_sample_argmax_kernel_impl(const torch::Tensor& logits) {
  TORCH_CHECK(logits.dim() == 2, "greedy_sample_argmax: logits must be 2D");
  TORCH_CHECK(logits.is_contiguous(),
              "greedy_sample_argmax: logits must be contiguous");
  const int64_t num_seqs = logits.size(0);
  const int64_t vocab_size = logits.size(1);
  auto out = torch::empty({num_seqs},
                          logits.options().dtype(torch::kInt64));
  int64_t* out_ptr = out.data_ptr<int64_t>();
  switch (logits.scalar_type()) {
    case torch::kFloat:
      greedy_argmax_typed(logits.data_ptr<float>(), num_seqs, vocab_size,
                          out_ptr);
      break;
    case torch::kBFloat16:
      greedy_argmax_typed(logits.data_ptr<at::BFloat16>(), num_seqs,
                          vocab_size, out_ptr);
      break;
    default:
      TORCH_CHECK(false, "greedy_sample_argmax: unsupported dtype ",
                  logits.scalar_type());
  }
  return out;
}

}  // namespace cpu_utils
