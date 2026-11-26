"""Check the software E4M3 encode/decode used by inkling_fp8kv_patch.py against torch's float8_e4m3fn."""
import torch, triton, triton.language as tl

@triton.jit
def enc_kernel(x_ptr, out_ptr, n, B: tl.constexpr):
    offs = tl.program_id(0) * B + tl.arange(0, B)
    m = offs < n
    y = tl.load(x_ptr + offs, mask=m, other=0.0)
    magnitude = tl.minimum(tl.abs(y), 448.0)
    bits = magnitude.to(tl.uint32, bitcast=True)
    rounded = (bits + 0x7FFFF + ((bits >> 20) & 1)) >> 20
    normal = rounded - 960
    subnormal = (magnitude * 512.0 + 0.5).to(tl.uint32)
    code = tl.where(magnitude < 0.015625, subnormal, normal)
    code = tl.minimum(code, 126)
    code = tl.where(y < 0, code | 128, code)
    dst = out_ptr + offs
    tl.store(dst.to(tl.pointer_type(tl.uint8)), code.to(tl.uint8), mask=m)

@triton.jit
def dec_kernel(b_ptr, out_ptr, n, B: tl.constexpr):
    offs = tl.program_id(0) * B + tl.arange(0, B)
    m = offs < n
    raw = tl.load(b_ptr.to(tl.pointer_type(tl.uint8)) + offs, mask=m, other=0).to(tl.uint32)
    exponent = (raw >> 3) & 15
    mantissa = raw & 7
    bits = ((raw >> 7) << 31) | ((exponent + 120) << 23) | (mantissa << 20)
    normal = bits.to(tl.float32, bitcast=True)
    subnormal = mantissa.to(tl.float32) * 0.001953125
    subnormal = tl.where(raw >= 128, -subnormal, subnormal)
    tl.store(out_ptr + offs, tl.where(exponent == 0, subnormal, normal).to(tl.bfloat16), mask=m)

dev = "cuda"
# decode: every finite byte
allb = torch.arange(256, dtype=torch.uint8)
ref = allb.view(torch.float8_e4m3fn).to(torch.float32)
out = torch.empty(256, dtype=torch.bfloat16, device=dev)
dec_kernel[(1,)](allb.to(dev), out, 256, B=256)
got = out.float().cpu()
finite = torch.isfinite(ref)
print("decode: finite bytes", int(finite.sum()), "mismatches", int((got[finite] != ref[finite]).sum()))

# encode: random values over the useful range plus edge cases
g = torch.Generator().manual_seed(0)
x = torch.cat([torch.randn(2_000_000, generator=g) * s for s in (1.0, 0.05, 0.003, 30.0)]
              + [torch.tensor([0.0, -0.0, 448.0, -448.0, 1000.0, -1000.0, 0.015625, 0.0156, 0.001953125, 0.0009765625, 1e-6])])
ref8 = x.to(torch.float8_e4m3fn).view(torch.uint8)
enc = torch.empty(x.numel(), dtype=torch.uint8, device=dev)
n = x.numel(); B = 1024
enc_kernel[((n + B - 1) // B,)](x.to(dev), enc, n, B=B)
enc = enc.cpu()
diff = enc != ref8
# compare decoded values where bytes differ (e.g. -0 vs +0, saturation vs inf, tie rounding)
dv = lambda b: b.view(torch.float8_e4m3fn).to(torch.float32)
bad = diff & (dv(enc) != dv(ref8))
print("encode: n", n, "byte mismatches", int(diff.sum()), "value mismatches", int(bad.sum()))
idx = bad.nonzero().flatten()[:12]
for i in idx.tolist():
    print("   x=%r ours=%r torch=%r" % (x[i].item(), dv(enc[i:i+1]).item(), dv(ref8[i:i+1]).item()))
