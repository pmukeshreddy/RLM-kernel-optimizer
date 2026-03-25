"""
correctness.py — Numerical correctness verification against reference kernel.
"""

from __future__ import annotations
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from eval.hack_detector import is_clean

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent


def _generate_correctness_harness(rows: int, hidden: int, atol: float, rtol: float) -> str:
    return f"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Minimal reference: scalar add + RMSNorm + block-scale NVFP4 encode
__device__ float _wref_sum(float v) {{
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xFFFFFFFF, v, m);
    return v;
}}

__global__ void reference_kernel(
    const __nv_bfloat16* in, const __nv_bfloat16* res,
    const __nv_bfloat16* w, __nv_bfloat16* ro,
    unsigned char* qo, __nv_bfloat16* sc,
    int hidden, float eps)
{{
    extern __shared__ float sm[];
    int row = blockIdx.x, tid = threadIdx.x;
    int wid = tid/32, lane = tid%32, nw = blockDim.x/32;
    int base = row * hidden;
    float ss = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {{
        float a = __bfloat162float(in[base+i]) + __bfloat162float(res[base+i]);
        ro[base+i] = __float2bfloat16(a);
        ss += a*a;
    }}
    ss = _wref_sum(ss);
    if (lane == 0) sm[wid] = ss;
    __syncthreads();
    if (wid == 0) {{ float v = (lane<nw)?sm[lane]:0.f; v=_wref_sum(v); if(!lane) sm[0]=v; }}
    __syncthreads();
    float ri = rsqrtf(sm[0]/hidden + eps);
    int nb = hidden/16;
    for (int qb = tid; qb < nb; qb += blockDim.x) {{
        int eb = qb*16;
        float amax = 0.f, vals[16];
        for (int j=0;j<16;++j) {{
            float x = __bfloat162float(ro[base+eb+j]) * ri * __bfloat162float(w[eb+j]);
            vals[j]=x; amax=fmaxf(amax,fabsf(x));
        }}
        float s = (amax>0)?amax/6.f:1.f;
        sc[row*nb+qb] = __float2bfloat16(s);
        float is = (amax>0)?6.f/amax:1.f;
        int pb = (row*nb+qb)*8;
        for (int j=0;j<8;++j) {{
            int lo = (int)fmaxf(0.f,fminf(15.f,(vals[2*j]*is/6.f+1.f)*7.5f));
            int hi = (int)fmaxf(0.f,fminf(15.f,(vals[2*j+1]*is/6.f+1.f)*7.5f));
            qo[pb+j] = (unsigned char)((hi<<4)|lo);
        }}
    }}
}}

void launch_fused_add_rmsnorm_nvfp4(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, unsigned char*, __nv_bfloat16*, int, int, cudaStream_t);

int main() {{
    const int rows={rows}, hidden={hidden}, N=rows*hidden, nb=N/16;
    __nv_bfloat16 *hi,*hr,*hw, *di,*dr,*dw,*rr,*rc,*sr,*sc;
    unsigned char *qr,*qc;
    hi=(__nv_bfloat16*)malloc(N*2); hr=(__nv_bfloat16*)malloc(N*2); hw=(__nv_bfloat16*)malloc(hidden*2);
    srand(42);
    for(int i=0;i<N;++i){{ hi[i]=__float2bfloat16((float)rand()/RAND_MAX-0.5f); hr[i]=__float2bfloat16((float)rand()/RAND_MAX-0.5f); }}
    for(int i=0;i<hidden;++i) hw[i]=__float2bfloat16(1.f+(float)rand()/RAND_MAX*0.1f);
    cudaMalloc(&di,N*2); cudaMemcpy(di,hi,N*2,cudaMemcpyHostToDevice);
    cudaMalloc(&dr,N*2); cudaMemcpy(dr,hr,N*2,cudaMemcpyHostToDevice);
    cudaMalloc(&dw,hidden*2); cudaMemcpy(dw,hw,hidden*2,cudaMemcpyHostToDevice);
    cudaMalloc(&rr,N*2); cudaMalloc(&rc,N*2);
    cudaMalloc(&qr,N/2); cudaMalloc(&qc,N/2);
    cudaMalloc(&sr,nb*2); cudaMalloc(&sc,nb*2);
    reference_kernel<<<rows,128,(128/32)*sizeof(float)>>>(di,dr,dw,rr,qr,sr,hidden,1e-6f);
    cudaStream_t s; cudaStreamCreate(&s);
    launch_fused_add_rmsnorm_nvfp4(di,dr,dw,rc,qc,sc,rows,hidden,s);
    cudaStreamSynchronize(s);
    __nv_bfloat16 *h_rr=(__nv_bfloat16*)malloc(N*2), *h_rc=(__nv_bfloat16*)malloc(N*2);
    cudaMemcpy(h_rr,rr,N*2,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rc,rc,N*2,cudaMemcpyDeviceToHost);
    float maxe=0.f; int miss=0;
    for(int i=0;i<N;++i){{
        float ref=__bfloat162float(h_rr[i]), can=__bfloat162float(h_rc[i]);
        float e=fabsf(ref-can); if(e>maxe) maxe=e;
        if(e>{atol}f+{rtol}f*fabsf(ref)) miss++;
    }}
    if(miss==0) {{ printf("CORRECTNESS: PASS (max_abs_err=%.6f N=%d)\\n",maxe,N); return 0; }}
    else {{ printf("CORRECTNESS: FAIL (max_abs_err=%.6f mismatches=%d/%d)\\n",maxe,miss,N); return 1; }}
}}
"""


class CorrectnessChecker:

    def __init__(self, config: dict):
        self.atol = config["eval"]["correctness_atol"]
        self.rtol = config["eval"]["correctness_rtol"]
        self.nvcc_flags = [
            "-O2", "-arch=sm_100a", "-std=c++17",
            f"-I{PROJECT_ROOT / 'kernels' / 'common'}",
        ]

    def check(self, candidate_src: str, problem_shape: tuple) -> tuple:
        """Returns (passed, max_abs_err, message)."""
        clean, hack_type = is_clean(candidate_src)
        if not clean:
            return False, float("inf"), f"Hack detected: {hack_type}"

        rows, hidden = problem_shape
        harness      = _generate_correctness_harness(rows, hidden, self.atol, self.rtol)
        combined_src = candidate_src + "\n\n" + harness

        with tempfile.TemporaryDirectory() as tmpdir:
            src_file = Path(tmpdir) / "correctness_check.cu"
            bin_file = Path(tmpdir) / "correctness_check"
            src_file.write_text(combined_src)

            cmd = ["nvcc"] + self.nvcc_flags + [str(src_file), "-o", str(bin_file)]
            r   = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                return False, float("inf"), f"Compile failed: {r.stderr[:300]}"

            r2     = subprocess.run([str(bin_file)], capture_output=True, text=True, timeout=60)
            output = r2.stdout.strip()

            if "PASS" in output:
                m   = re.search(r"max_abs_err=([\d.]+)", output)
                err = float(m.group(1)) if m else 0.0
                return True, err, output
            elif "FAIL" in output:
                m   = re.search(r"max_abs_err=([\d.]+)", output)
                err = float(m.group(1)) if m else float("inf")
                return False, err, output
            else:
                return False, float("inf"), f"Unexpected output: {output}"
