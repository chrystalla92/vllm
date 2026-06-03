#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/torch.h>

#include <cmath>

namespace {

inline float sigmoidf_stable(float x) {
  if (x >= 0.0f) {
    const float z = std::exp(-x);
    return 1.0f / (1.0f + z);
  }
  const float z = std::exp(x);
  return z / (1.0f + z);
}

inline float softplusf_stable(float x) {
  if (x > 20.0f) {
    return x;
  }
  return std::log1p(std::exp(x));
}

inline at::Tensor l2norm_lastdim(const at::Tensor& x) {
  return x * at::rsqrt((x * x).sum(-1, true) + 1.0e-6);
}

}  // namespace

void gdn_recurrent_decode_cpu(
    const torch::Tensor& query, const torch::Tensor& key,
    const torch::Tensor& value, const torch::Tensor& a,
    const torch::Tensor& b, const torch::Tensor& A_log,
    const torch::Tensor& dt_bias, torch::Tensor& ssm_state,
    const torch::Tensor& state_indices, torch::Tensor& out, double scale) {
  TORCH_CHECK(query.device().is_cpu(), "query must be CPU");
  const int64_t tokens = query.size(0);
  const int64_t heads = value.size(1);
  const int64_t k_dim = query.size(2);
  const int64_t v_dim = value.size(2);

  auto q = query.contiguous();
  auto k = key.contiguous();
  auto v = value.contiguous();
  auto a_f = a.to(at::kFloat).contiguous();
  auto b_f = b.to(at::kFloat).contiguous();
  auto A_f = A_log.to(at::kFloat).contiguous();
  auto dt_f = dt_bias.to(at::kFloat).contiguous();
  auto state_f = ssm_state.to(at::kFloat);
  auto idx = state_indices.to(at::kLong).contiguous();
  auto out_f = at::empty({tokens, heads, v_dim}, q.options().dtype(at::kFloat));

  const float* q_ptr = q.data_ptr<float>();
  const float* k_ptr = k.data_ptr<float>();
  const float* v_ptr = v.data_ptr<float>();
  const float* a_ptr = a_f.data_ptr<float>();
  const float* b_ptr = b_f.data_ptr<float>();
  const float* A_ptr = A_f.data_ptr<float>();
  const float* dt_ptr = dt_f.data_ptr<float>();
  const int64_t* idx_ptr = idx.data_ptr<int64_t>();
  float* state_ptr = state_f.data_ptr<float>();
  float* out_ptr = out_f.data_ptr<float>();
  const float scale_f = static_cast<float>(scale);

  at::parallel_for(0, tokens * heads, 1, [&](int64_t begin, int64_t end) {
    for (int64_t linear = begin; linear < end; ++linear) {
      const int64_t token = linear / heads;
      const int64_t head = linear - token * heads;
      const int64_t slot = idx_ptr[token];
      const float gate = -std::exp(A_ptr[head]) *
                         softplusf_stable(a_ptr[token * heads + head] + dt_ptr[head]);
      const float g_exp = std::exp(gate);
      const float beta = sigmoidf_stable(b_ptr[token * heads + head]);
      float* state = state_ptr + ((slot * heads + head) * v_dim * k_dim);
      const float* qh = q_ptr + ((token * heads + head) * k_dim);
      const float* kh = k_ptr + ((token * heads + head) * k_dim);
      const float* vh = v_ptr + ((token * heads + head) * v_dim);
      float* oh = out_ptr + ((token * heads + head) * v_dim);
      float q_norm_sq = 1.0e-6f;
      float k_norm_sq = 1.0e-6f;
      for (int64_t kk = 0; kk < k_dim; ++kk) {
        q_norm_sq += qh[kk] * qh[kk];
        k_norm_sq += kh[kk] * kh[kk];
      }
      const float q_inv = scale_f / std::sqrt(q_norm_sq);
      const float k_inv = 1.0f / std::sqrt(k_norm_sq);
      for (int64_t vv = 0; vv < v_dim; ++vv) {
        float kv_mem = 0.0f;
        float* state_row = state + vv * k_dim;
        for (int64_t kk = 0; kk < k_dim; ++kk) {
          const float st = state_row[kk] * g_exp;
          state_row[kk] = st;
          kv_mem += st * (kh[kk] * k_inv);
        }
        const float delta = (vh[vv] - kv_mem) * beta;
        float out_acc = 0.0f;
        for (int64_t kk = 0; kk < k_dim; ++kk) {
          const float st = state_row[kk] + delta * (kh[kk] * k_inv);
          state_row[kk] = st;
          out_acc += st * (qh[kk] * q_inv);
        }
        oh[vv] = out_acc;
      }
    }
  });

  out.copy_(out_f.to(out.scalar_type()));
  ssm_state.index_copy_(0, idx, state_f.index_select(0, idx).to(ssm_state.scalar_type()));
}

void gdn_recurrent_prefill_cpu(
    const torch::Tensor& query, const torch::Tensor& key,
    const torch::Tensor& value, const torch::Tensor& a,
    const torch::Tensor& b, const torch::Tensor& A_log,
    const torch::Tensor& dt_bias, torch::Tensor& ssm_state,
    const torch::Tensor& state_indices, const torch::Tensor& cu_seqlens,
    const torch::Tensor& has_initial_state, torch::Tensor& out, double scale) {
  constexpr int64_t chunk_size = 128;
  auto q_all = l2norm_lastdim(query.squeeze(0).to(at::kFloat));
  auto k_all = l2norm_lastdim(key.squeeze(0).to(at::kFloat));
  auto v_all = value.squeeze(0).to(at::kFloat);
  const int64_t tokens = v_all.size(0);
  const int64_t q_heads = q_all.size(1);
  const int64_t heads = v_all.size(1);
  const int64_t v_dim = v_all.size(2);
  const int64_t k_dim = q_all.size(2);
  const int64_t repeat = std::max<int64_t>(1, heads / q_heads);
  if (repeat != 1) {
    q_all = q_all.repeat_interleave(repeat, 1);
    k_all = k_all.repeat_interleave(repeat, 1);
  }
  q_all = (q_all * scale).transpose(0, 1).contiguous();  // H,T,K
  k_all = k_all.transpose(0, 1).contiguous();            // H,T,K
  v_all = v_all.transpose(0, 1).contiguous();            // H,T,V

  auto g_all = (-at::exp(A_log.to(at::kFloat)).unsqueeze(0) *
                at::softplus(a.to(at::kFloat) + dt_bias.to(at::kFloat).unsqueeze(0), 1.0, 20.0))
                   .transpose(0, 1)
                   .contiguous();
  auto beta_all = at::sigmoid(b.to(at::kFloat)).transpose(0, 1).contiguous();
  auto state_f = ssm_state.to(at::kFloat);
  auto idx = state_indices.to(at::kLong).contiguous();
  auto lens = cu_seqlens.to(at::kLong).contiguous();
  auto has_state = has_initial_state.to(at::kBool).contiguous();
  auto out_f = at::empty({heads, tokens, v_dim}, q_all.options());
  auto eye = at::eye(chunk_size, q_all.options());

  for (int64_t seq = 0; seq < idx.size(0); ++seq) {
    const int64_t slot = idx[seq].item<int64_t>();
    const int64_t begin = lens[seq].item<int64_t>();
    const int64_t end = lens[seq + 1].item<int64_t>();
    auto seq_state = state_f.select(0, slot);
    if (!has_state[seq].item<bool>()) {
      seq_state.zero_();
    }
    for (int64_t chunk_start = begin; chunk_start < end; chunk_start += chunk_size) {
      const int64_t chunk_end = std::min(chunk_start + chunk_size, end);
      const int64_t chunk_len = chunk_end - chunk_start;
      auto q_chunk = q_all.slice(1, chunk_start, chunk_end);      // H,C,K
      auto k_chunk = k_all.slice(1, chunk_start, chunk_end);      // H,C,K
      auto v_chunk = v_all.slice(1, chunk_start, chunk_end);      // H,C,V
      auto beta_chunk = beta_all.slice(1, chunk_start, chunk_end); // H,C
      auto g_chunk = g_all.slice(1, chunk_start, chunk_end);       // H,C
      auto cum_g = at::cumsum(g_chunk, -1);
      auto exp_cum_g = at::exp(cum_g);
      auto decay = at::exp(cum_g.unsqueeze(-1) - cum_g.unsqueeze(-2));
      auto interaction = at::matmul(k_chunk * beta_chunk.unsqueeze(-1),
                                    k_chunk.transpose(-1, -2));
      interaction = at::tril(interaction * decay, -1);
      auto system = interaction + eye.slice(0, 0, chunk_len).slice(1, 0, chunk_len).unsqueeze(0);
      auto solved_values = at::linalg_solve_triangular(
          system, v_chunk * beta_chunk.unsqueeze(-1), false, true, false);
      auto solved_keys = at::linalg_solve_triangular(
          system,
          (k_chunk * beta_chunk.unsqueeze(-1)) * exp_cum_g.unsqueeze(-1),
          false, true, false);
      auto incoming_memory = at::einsum("hvk,hck->hcv", {seq_state, solved_keys});
      auto transformed_values = solved_values - incoming_memory;
      auto inter_chunk = at::einsum(
          "hvk,hck->hcv", {seq_state, q_chunk * exp_cum_g.unsqueeze(-1)});
      auto intra_chunk = at::tril(at::matmul(q_chunk, k_chunk.transpose(-1, -2)) * decay);
      out_f.slice(1, chunk_start, chunk_end).copy_(inter_chunk + at::matmul(intra_chunk, transformed_values));
      auto end_decay = at::exp(cum_g.select(-1, chunk_len - 1).unsqueeze(-1) - cum_g).unsqueeze(-1);
      auto decayed_keys = k_chunk * end_decay;
      seq_state.copy_(seq_state * exp_cum_g.select(-1, chunk_len - 1).unsqueeze(-1).unsqueeze(-1) +
                      at::einsum("hcv,hck->hvk", {transformed_values, decayed_keys}));
    }
  }

  out.copy_(out_f.transpose(0, 1).contiguous().to(out.scalar_type()));
  ssm_state.index_copy_(0, idx, state_f.index_select(0, idx).to(ssm_state.scalar_type()));
}
