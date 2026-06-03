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

}  // namespace

void gdn_recurrent_decode_cpu(
    const torch::Tensor& query, const torch::Tensor& key,
    const torch::Tensor& value, const torch::Tensor& a,
    const torch::Tensor& b, const torch::Tensor& A_log,
    const torch::Tensor& dt_bias, torch::Tensor& ssm_state,
    const torch::Tensor& state_indices, torch::Tensor& out, double scale) {
  TORCH_CHECK(query.device().is_cpu(), "query must be CPU");
  TORCH_CHECK(query.scalar_type() == at::kFloat, "query must be float32");
  TORCH_CHECK(key.scalar_type() == at::kFloat, "key must be float32");
  TORCH_CHECK(value.scalar_type() == at::kFloat, "value must be float32");

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
  TORCH_CHECK(query.device().is_cpu(), "query must be CPU");
  auto q = query.squeeze(0).to(at::kFloat).contiguous();
  auto k = key.squeeze(0).to(at::kFloat).contiguous();
  auto v = value.squeeze(0).to(at::kFloat).contiguous();
  auto a_f = a.to(at::kFloat).contiguous();
  auto b_f = b.to(at::kFloat).contiguous();
  auto A_f = A_log.to(at::kFloat).contiguous();
  auto dt_f = dt_bias.to(at::kFloat).contiguous();
  auto state_f = ssm_state.to(at::kFloat);
  auto idx = state_indices.to(at::kLong).contiguous();
  auto lens = cu_seqlens.to(at::kLong).contiguous();
  auto has_state = has_initial_state.to(at::kBool).contiguous();

  const int64_t tokens = v.size(0);
  const int64_t q_heads = q.size(1);
  const int64_t heads = v.size(1);
  const int64_t k_dim = q.size(2);
  const int64_t v_dim = v.size(2);
  const int64_t num_seq = idx.size(0);
  auto out_f = at::empty({tokens, heads, v_dim}, q.options().dtype(at::kFloat));

  const float* q_ptr = q.data_ptr<float>();
  const float* k_ptr = k.data_ptr<float>();
  const float* v_ptr = v.data_ptr<float>();
  const float* a_ptr = a_f.data_ptr<float>();
  const float* b_ptr = b_f.data_ptr<float>();
  const float* A_ptr = A_f.data_ptr<float>();
  const float* dt_ptr = dt_f.data_ptr<float>();
  const int64_t* idx_ptr = idx.data_ptr<int64_t>();
  const int64_t* lens_ptr = lens.data_ptr<int64_t>();
  const bool* has_ptr = has_state.data_ptr<bool>();
  float* state_ptr = state_f.data_ptr<float>();
  float* out_ptr = out_f.data_ptr<float>();
  const float scale_f = static_cast<float>(scale);
  const int64_t repeat = std::max<int64_t>(1, heads / q_heads);

  at::parallel_for(0, num_seq * heads, 1, [&](int64_t begin, int64_t end) {
    for (int64_t linear = begin; linear < end; ++linear) {
      const int64_t seq = linear / heads;
      const int64_t head = linear - seq * heads;
      const int64_t q_head = head / repeat;
      const int64_t slot = idx_ptr[seq];
      const int64_t start = lens_ptr[seq];
      const int64_t stop = lens_ptr[seq + 1];
      float* state = state_ptr + ((slot * heads + head) * v_dim * k_dim);
      if (!has_ptr[seq]) {
        for (int64_t i = 0; i < v_dim * k_dim; ++i) {
          state[i] = 0.0f;
        }
      }
      for (int64_t token = start; token < stop; ++token) {
        const float gate = -std::exp(A_ptr[head]) *
                           softplusf_stable(a_ptr[token * heads + head] + dt_ptr[head]);
        const float g_exp = std::exp(gate);
        const float beta = sigmoidf_stable(b_ptr[token * heads + head]);
        const float* qh = q_ptr + ((token * q_heads + q_head) * k_dim);
        const float* kh = k_ptr + ((token * q_heads + q_head) * k_dim);
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
    }
  });

  out.copy_(out_f.to(out.scalar_type()));
  ssm_state.index_copy_(0, idx, state_f.index_select(0, idx).to(ssm_state.scalar_type()));
}
