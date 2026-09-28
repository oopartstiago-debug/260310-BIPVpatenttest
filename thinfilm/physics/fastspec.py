"""최적화용 벡터화 스펙트럼 (전체 파장을 한 번에). optics.spectrum 과 같은 값을 낸다.

구조가 고정(공기|유리 i|코팅 c|EVA i|ARC c|Si)이므로 incoherent 결합을 닫힌 식으로 쓴다.
검증: python3 fastspec.py  (optics.spectrum 과 최대 오차 출력)
"""
import numpy as np

import optics

LAM = optics.LAMBDA
NK = {k: optics._NK[k](LAM) for k in optics.MATERIALS}


def _coh(pol, ns, ds, s0):
    """ns: 층별 (L,) 복소 굴절률 리스트(양 끝 반무한), ds: 두께, s0 = n0 sinθ0 (L,). r, t 및 강도 R, T."""
    cos = [np.sqrt(1 - (s0 / n) ** 2 + 0j) for n in ns]
    # 양 끝: 전진파 가지 선택 (Byrnes is_forward_angle)
    for i in (0, len(ns) - 1):
        nc = ns[i] * cos[i]
        flip = np.where(np.abs(nc.imag) > 1e-10, nc.imag < 0, nc.real < 0)
        cos[i] = np.where(flip, -cos[i], cos[i])

    def rt(i, f):
        ni, nf, ci, cf = ns[i], ns[f], cos[i], cos[f]
        if pol == "s":
            den = ni * ci + nf * cf
            return (ni * ci - nf * cf) / den, 2 * ni * ci / den
        den = nf * ci + ni * cf
        return (nf * ci - ni * cf) / den, 2 * ni * ci / den

    r, t = rt(0, 1)
    M00, M01, M10, M11 = 1 / t, r / t, r / t, 1 / t
    for i in range(1, len(ns) - 1):
        delta = 2 * np.pi * ns[i] * cos[i] * ds[i] / LAM
        delta = np.where(delta.imag > 35, delta.real + 35j, delta)
        em, ep = np.exp(-1j * delta), np.exp(1j * delta)
        r, t = rt(i, i + 1)
        a00, a01, a10, a11 = em / t, em * r / t, ep * r / t, ep / t
        M00, M01, M10, M11 = (M00 * a00 + M01 * a10, M00 * a01 + M01 * a11,
                              M10 * a00 + M11 * a10, M10 * a01 + M11 * a11)
    rr, tt = M10 / M00, 1 / M00
    ni, nf, ci, cf = ns[0], ns[-1], cos[0], cos[-1]
    if pol == "s":
        T = np.abs(tt) ** 2 * (nf * cf).real / (ni * ci).real
    else:
        T = np.abs(tt) ** 2 * (nf * np.conj(cf)).real / (ni * np.conj(ci)).real
    return np.abs(rr) ** 2, T


def spectrum(coating, theta_air_deg):
    s0 = np.sin(np.radians(theta_air_deg)) * np.ones_like(LAM) + 0j
    one = np.ones_like(LAM) + 0j
    g, e = optics.N_GLASS * one, optics.N_EVA * one
    coat_n = [NK[l["mat"]] for l in coating]
    coat_d = [float(l["d"]) for l in coating]
    Rs, Ts = [], []
    for pol in ("s", "p"):
        # 계면 1: 공기|유리, 계면 2: 유리|코팅|EVA, 계면 3: EVA|ARC|Si
        R1f, T1f = _coh(pol, [one, g], [np.inf, np.inf], s0)
        R1b, T1b = _coh(pol, [g, one], [np.inf, np.inf], s0)
        R2f, T2f = _coh(pol, [g, *coat_n, e], [np.inf, *coat_d, np.inf], s0)
        R2b, T2b = _coh(pol, [e, *coat_n[::-1], g], [np.inf, *coat_d[::-1], np.inf], s0)
        R3f, T3f = _coh(pol, [e, NK["Si3N4"], NK["Si"]], [np.inf, optics.ARC_D, np.inf], s0)
        # 강도 전달행렬 (Byrnes inc_tmm 과 같은 형식, 유리/EVA 흡수 없음)
        def L(Rf, Tf, Rb, Tb):
            return np.array([[1 / Tf, -Rb / Tf], [Rf / Tf, (Tf * Tb - Rf * Rb) / Tf]])
        A = L(R1f, T1f, R1b, T1b)
        B = L(R2f, T2f, R2b, T2b)
        Cm = L(R3f, T3f, np.zeros_like(R3f), np.ones_like(R3f))  # 마지막 계면: 후방 값은 결과에 영향 없음
        M = np.einsum("ijl,jkl->ikl", np.einsum("ijl,jkl->ikl", A, B), Cm)
        Rs.append(M[1, 0] / M[0, 0])
        Ts.append(1 / M[0, 0])
    return 0.5 * (Rs[0] + Rs[1]), 0.5 * (Ts[0] + Ts[1])


if __name__ == "__main__":
    import random
    rng = random.Random(5)
    worst = 0.0
    for _ in range(6):
        layers = [{"mat": rng.choice(["TiO2", "SiO2", "Si3N4"]), "d": rng.uniform(5, 300)} for _ in range(rng.randint(0, 6))]
        for a in (0, 30, 60, 75):
            R0, T0 = optics.spectrum(layers, a)
            R1, T1 = spectrum(layers, a)
            worst = max(worst, np.abs(R0 - R1).max(), np.abs(T0 - T1).max())
    print("max |Δ| vs Byrnes inc_tmm:", worst)
