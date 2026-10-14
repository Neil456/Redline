// CUTLASS mixed-input GEMM: fp16 activations (row-major [m, k]) x int8
// weights (Redline's row-major [n, k] == CUTLASS column-major B [k, n]).
// The int8 operand is upcast to fp16 in registers right before the tensor-core
// MMA (OpMultiplyAddMixedInputUpcast), so the weights stay int8 in memory.
#include "cuda/cuda_utils.h"
#include "cuda/gemm/backends.h"

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

namespace redline {
namespace {

using ElementA = cutlass::half_t;
using ElementB = int8_t;
using ElementC = cutlass::half_t;
using ElementAccumulator = float;

template <typename ThreadblockShape, typename WarpShape, int kStages>
using MixedGemm = cutlass::gemm::device::GemmUniversal<
    ElementA, cutlass::layout::RowMajor,
    ElementB, cutlass::layout::ColumnMajor,
    ElementC, cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape, WarpShape,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value, ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    kStages,
    8,   // alignment A (fp16, 128-bit)
    16,  // alignment B (int8, 128-bit)
    cutlass::arch::OpMultiplyAddMixedInputUpcast,
    cutlass::ComplexTransform::kNone,
    cutlass::ComplexTransform::kNone>;

// Large tiles for big prefills, smaller tiles when m is modest.
using GemmLarge = MixedGemm<cutlass::gemm::GemmShape<128, 128, 64>, cutlass::gemm::GemmShape<64, 64, 64>, 4>;
using GemmSmall = MixedGemm<cutlass::gemm::GemmShape<64, 128, 64>, cutlass::gemm::GemmShape<32, 64, 64>, 4>;

const char* status_str(cutlass::Status s) { return cutlass::cutlassGetStatusString(s); }

class CutlassMixedGemmImpl final : public CutlassMixedGemm {
 public:
  explicit CutlassMixedGemmImpl(MemoryLedger* ledger) : workspace_(ledger->alloc("workspace", kWorkspace)) {}

  void run(const half* x, const int8_t* q, half* y, int m, int n, int k, cudaStream_t stream) override {
    RL_CHECK(k % 64 == 0 && n % 8 == 0, "CUTLASS mixed GEMM needs K % 64 == 0 and N % 8 == 0");
    if (m >= 512) {
      launch<GemmLarge>(x, q, y, m, n, k, stream);
    } else {
      launch<GemmSmall>(x, q, y, m, n, k, stream);
    }
  }

 private:
  static constexpr size_t kWorkspace = 4u << 20;

  template <typename Gemm>
  void launch(const half* x, const int8_t* q, half* y, int m, int n, int k, cudaStream_t stream) {
    const float alpha = 1.f / kOutputDivisor, beta = 0.f;
    typename Gemm::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k}, /*batch_count=*/1, {alpha, beta},
        reinterpret_cast<const ElementA*>(x), q, reinterpret_cast<const ElementC*>(y),
        reinterpret_cast<ElementC*>(y),
        /*batch strides*/ 0, 0, 0, 0,
        /*lda*/ k, /*ldb*/ k, /*ldc*/ n, /*ldd*/ n);
    Gemm gemm;
    cutlass::Status st = gemm.can_implement(args);
    RL_CHECK(st == cutlass::Status::kSuccess, "CUTLASS can_implement: ", status_str(st));
    RL_CHECK(Gemm::get_workspace_size(args) <= workspace_.bytes(), "CUTLASS workspace too small");
    st = gemm.initialize(args, workspace_.get(), stream);
    RL_CHECK(st == cutlass::Status::kSuccess, "CUTLASS initialize: ", status_str(st));
    st = gemm.run(stream);
    RL_CHECK(st == cutlass::Status::kSuccess, "CUTLASS run: ", status_str(st));
  }

  DeviceBuffer workspace_;
};

}  // namespace

std::unique_ptr<CutlassMixedGemm> make_cutlass_mixed_gemm(MemoryLedger* ledger) {
  return std::make_unique<CutlassMixedGemmImpl>(ledger);
}

}  // namespace redline
