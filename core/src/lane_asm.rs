//! Eight-lane Neumaier steps in AVX-512F.
//!
//! Rust 1.83 does not stabilise the `avx512f` intrinsics or the target
//! feature, so the arithmetic is inline assembly. Only `zmm0`–`zmm15` and
//! `k1`–`k7` are used; those are caller-saved. The caller checks
//! `avx512f` before entering.

use std::arch::asm;

#[repr(C)]
struct K {
    sign: u64,
    two_u: u64,
    bump: u64,
    exp: u64,
    frac: u64,
    zero: u64,
}

static K: K = K {
    sign: 0x8000_0000_0000_0000,
    two_u: 0x3cb0_0000_0000_0000,
    bump: 0x3ce0_0000_0000_0000,
    exp: 0x7ff,
    frac: 0x000f_ffff_ffff_ffff,
    zero: 0,
};

pub struct State {
    pub sum: [f64; 8],
    pub comp: [f64; 8],
    pub sigma: [f64; 8],
    pub bad: u16,
}

/// # Safety
/// The CPU must support AVX-512F. `n` is a multiple of 8 and both pointers
/// address `n` doubles (they are only read).
pub unsafe fn prod(a: *const f64, b: *const f64, n: usize) -> State {
    let ap = a;
    let bp = b;
    let left = n;
    let mut sum = [0.0; 8];
    let mut comp = [0.0; 8];
    let mut sigma = [0.0; 8];
    let mut bad: u16 = 0;
    asm!(
        include_str!("avx_macros.s"),
        "vpbroadcastq zmm15, qword ptr [{lane}]",
        "vxorpd zmm0, zmm0, zmm0",
        "vxorpd zmm1, zmm1, zmm1",
        "vxorpd zmm2, zmm2, zmm2",
        "kxorw k1, k1, k1",
        "test {left}, {left}",
        "jz 3f",
        "2:",
        "vmovupd zmm3, [{ap}]",
        "vmovupd zmm4, [{bp}]",
        "vmulpd zmm5, zmm3, zmm4",
        "vmovapd zmm6, zmm5",
        "vfmsub231pd zmm6, zmm3, zmm4",
        "badclass zmm3, zmm12",
        "kmovw k4, k6",
        "badclass zmm4, zmm12",
        "korw k4, k4, k6",
        "badclass zmm5, zmm12",
        "korw k4, k4, k6",
        "badclass zmm6, zmm12",
        "korw k1, k1, k4",
        "korw k1, k1, k6",
        "push8 zmm5",
        "push8 zmm6",
        "add {ap}, 64",
        "add {bp}, 64",
        "sub {left}, 8",
        "jnz 2b",
        "3:",
        "vmovupd [{sum}], zmm0",
        "vmovupd [{comp}], zmm1",
        "vmovupd [{sigma}], zmm2",
        "kmovw [{bad}], k1",
        ".purgem push8",
        ".purgem badclass",
        lane = in(reg) &K,
        ap = inout(reg) ap => _,
        bp = inout(reg) bp => _,
        left = inout(reg) left => _,
        sum = in(reg) sum.as_mut_ptr(),
        comp = in(reg) comp.as_mut_ptr(),
        sigma = in(reg) sigma.as_mut_ptr(),
        bad = in(reg) &mut bad,
        out("zmm0") _, out("zmm1") _, out("zmm2") _, out("zmm3") _,
        out("zmm4") _, out("zmm5") _, out("zmm6") _, out("zmm7") _,
        out("zmm8") _, out("zmm9") _, out("zmm10") _, out("zmm11") _,
        out("zmm12") _, out("zmm13") _, out("zmm14") _, out("zmm15") _,
        out("k1") _, out("k2") _, out("k3") _, out("k4") _,
        out("k5") _, out("k6") _, out("k7") _,
        options(nostack),
    );
    State {
        sum,
        comp,
        sigma,
        bad,
    }
}

/// # Safety
/// Same as [`prod`]. `c` addresses `n` doubles.
pub unsafe fn prod3(a: *const f64, b: *const f64, c: *const f64, n: usize) -> State {
    let ap = a;
    let bp = b;
    let cp = c;
    let left = n;
    let mut sum = [0.0; 8];
    let mut comp = [0.0; 8];
    let mut sigma = [0.0; 8];
    let mut bad: u16 = 0;
    asm!(
        include_str!("avx_macros.s"),
        "vpbroadcastq zmm15, qword ptr [{lane}]",
        "vxorpd zmm0, zmm0, zmm0",
        "vxorpd zmm1, zmm1, zmm1",
        "vxorpd zmm2, zmm2, zmm2",
        "kxorw k1, k1, k1",
        "test {left}, {left}",
        "jz 3f",
        "2:",
        "vmovupd zmm3, [{ap}]",
        "vmovupd zmm4, [{bp}]",
        "vmovupd zmm5, [{cp}]",
        "vmulpd zmm6, zmm3, zmm4",
        "vmovapd zmm7, zmm6",
        "vfmsub231pd zmm7, zmm3, zmm4",
        "vmulpd zmm8, zmm6, zmm5",
        "vmovapd zmm9, zmm8",
        "vfmsub231pd zmm9, zmm6, zmm5",
        "vmulpd zmm10, zmm7, zmm5",
        "vmovapd zmm11, zmm10",
        "vfmsub231pd zmm11, zmm7, zmm5",
        "badclass zmm3, zmm12",
        "kmovw k4, k6",
        "badclass zmm4, zmm12",
        "korw k4, k4, k6",
        "badclass zmm5, zmm12",
        "korw k4, k4, k6",
        "badclass zmm6, zmm12",
        "korw k4, k4, k6",
        "badclass zmm7, zmm12",
        "korw k4, k4, k6",
        "badclass zmm8, zmm12",
        "korw k4, k4, k6",
        "badclass zmm9, zmm12",
        "korw k4, k4, k6",
        "badclass zmm10, zmm12",
        "korw k4, k4, k6",
        "badclass zmm11, zmm12",
        "korw k1, k1, k4",
        "korw k1, k1, k6",
        "vmovapd zmm3, zmm8",
        "vmovapd zmm4, zmm9",
        "vmovapd zmm5, zmm10",
        "vmovapd zmm6, zmm11",
        "push8 zmm3",
        "push8 zmm4",
        "push8 zmm5",
        "push8 zmm6",
        "add {ap}, 64",
        "add {bp}, 64",
        "add {cp}, 64",
        "sub {left}, 8",
        "jnz 2b",
        "3:",
        "vmovupd [{sum}], zmm0",
        "vmovupd [{comp}], zmm1",
        "vmovupd [{sigma}], zmm2",
        "kmovw [{bad}], k1",
        ".purgem push8",
        ".purgem badclass",
        lane = in(reg) &K,
        ap = inout(reg) ap => _,
        bp = inout(reg) bp => _,
        cp = inout(reg) cp => _,
        left = inout(reg) left => _,
        sum = in(reg) sum.as_mut_ptr(),
        comp = in(reg) comp.as_mut_ptr(),
        sigma = in(reg) sigma.as_mut_ptr(),
        bad = in(reg) &mut bad,
        out("zmm0") _, out("zmm1") _, out("zmm2") _, out("zmm3") _,
        out("zmm4") _, out("zmm5") _, out("zmm6") _, out("zmm7") _,
        out("zmm8") _, out("zmm9") _, out("zmm10") _, out("zmm11") _,
        out("zmm12") _, out("zmm13") _, out("zmm14") _, out("zmm15") _,
        out("k1") _, out("k2") _, out("k3") _, out("k4") _,
        out("k5") _, out("k6") _, out("k7") _,
        options(nostack),
    );
    State {
        sum,
        comp,
        sigma,
        bad,
    }
}
