// AI Tilt — physics_v3 C# 포팅 (샌드박스 실시간 재계산용, Python 의존 제거)
// 단일 진실 = physics_v3.py. 테이블/검증벡터는 gen_csharp_tables.py가 Python에서 생성.
//   StreamingAssets/vf_flat_v17.json        : 시야계수(현/피치 × tilt, 평탄배열)
//   StreamingAssets/iam_diffuse_v17.json    : Martin-Ruiz 확산 IAM(tilt, a_r=0.16)
//   StreamingAssets/physics_testvectors.json: 40개 (입력→eff_poa,panel_sf) 셀프테스트
// SelfTest()가 C# 결과 vs Python 결과를 대조 → 첫 실행에 PASS/FAIL 로그.
using System;
using System.IO;
using UnityEngine;

[Serializable] public class VfTable {
    public float[] ratios, tilts, f_sky_flat, f_grd_flat;
    public int n_ratio, n_tilt;
}
[Serializable] public class IamTable {
    public float a_r;
    public float[] tilts, iam_sky, iam_grd;
}
[Serializable] public class TestVec {
    public float tilt, elev, az, dni, dhi, eff_poa, panel_sf;
}
[Serializable] public class TestVecFile {
    public float chord, pitch, albedo, a_r, strip_lo, strip_frac, tol;
    public TestVec[] vectors;
}

public static class LouverPhysics {
    public const float CHORD = 97.5f, PITCH = 97.5f, ALBEDO = 0.15f, A_R = 0.16f, SA = 180f;
    public static readonly float STRIP_FRAC = 83f / 97.5f;
    public static readonly float STRIP_LO   = (1f - 83f / 97.5f) / 2f;

    static VfTable vf; static IamTable iam;
    public static bool Loaded => vf != null && iam != null;

    public static void Load(string dir) {
        vf  = JsonUtility.FromJson<VfTable>(File.ReadAllText(Path.Combine(dir, "vf_flat_v17.json")));
        iam = JsonUtility.FromJson<IamTable>(File.ReadAllText(Path.Combine(dir, "iam_diffuse_v17.json")));
    }

    static float Clamp(float v, float lo, float hi) => v < lo ? lo : (v > hi ? hi : v);

    // np.interp: 정렬된 xs에서 선형보간(끝 클램프)
    static float Interp(float x, float[] xs, Func<int, float> ys, int n) {
        if (x <= xs[0]) return ys(0);
        if (x >= xs[n - 1]) return ys(n - 1);
        int k = 1; while (k < n && xs[k] < x) k++;
        float t = (x - xs[k - 1]) / (xs[k] - xs[k - 1]);
        return (1 - t) * ys(k - 1) + t * ys(k);
    }

    // V15 선분교차 자기음영률 (physics_v3 == physics_v2.panel_sf)
    public static float PanelSf(float tilt, float elev, float az, float hd = CHORD, float p = PITCH, float sa = SA) {
        if (elev <= 0) return 0f;
        float tr = tilt * Mathf.Deg2Rad, er = Clamp(elev, 0.1f, 89.9f) * Mathf.Deg2Rad;
        float ct = Mathf.Cos(tr), st2 = Mathf.Sin(tr);
        float rx = hd * ct, ry = p - hd * st2, fx = hd * ct, fy = -hd * st2;
        float adr = (az - sa) * Mathf.Deg2Rad, rdx = -Mathf.Cos(er) * Mathf.Cos(adr), rdy = -Mathf.Sin(er);
        float dn = fx * rdy - fy * rdx; if (Mathf.Abs(dn) < 1e-12f) dn = 1e-12f;
        float tb = (rx * rdy - ry * rdx) / dn, ta = (rx * fy - ry * fx) / dn;
        float sf = (ta > 0 && tb > 1) ? 1f : ((ta > 0 && tb > 0 && tb <= 1) ? tb : 0f);
        return Clamp(sf, 0f, 1f);
    }

    public static float StripShade(float sf, float lo = -1, float frac = -1) {
        if (lo < 0) lo = STRIP_LO; if (frac < 0) frac = STRIP_FRAC;
        float v = Mathf.Min(sf, lo + frac) - lo;
        return Mathf.Max(v, 0f) / frac;
    }

    // 시야계수: ratio(현/피치) × tilt 이중선형
    public static float ViewFactor(float tilt, float c, float p, bool sky) {
        float r = Clamp(c / p, vf.ratios[0], vf.ratios[vf.n_ratio - 1]);
        int i = 0; while (i < vf.n_ratio && vf.ratios[i] < r) i++;   // 첫 ratios[i] >= r
        i = Math.Max(0, Math.Min(i - 1, vf.n_ratio - 2));
        float w = (r - vf.ratios[i]) / (vf.ratios[i + 1] - vf.ratios[i]);
        float[] tab = sky ? vf.f_sky_flat : vf.f_grd_flat;
        int b0 = i * vf.n_tilt, b1 = (i + 1) * vf.n_tilt;
        return Interp(tilt, vf.tilts, t => (1 - w) * tab[b0 + t] + w * tab[b1 + t], vf.n_tilt);
    }

    // 입사각 (pvlib.irradiance.aoi, surface_az=180)
    public static float Aoi(float tilt, float elev, float az, float sa = SA) {
        float zen = (90f - elev) * Mathf.Deg2Rad, tr = tilt * Mathf.Deg2Rad;
        float c = Mathf.Cos(zen) * Mathf.Cos(tr)
                + Mathf.Sin(zen) * Mathf.Sin(tr) * Mathf.Cos((az - sa) * Mathf.Deg2Rad);
        return Mathf.Acos(Clamp(c, -1f, 1f)) * Mathf.Rad2Deg;
    }

    public static float IamBeam(float aoi, float a_r = A_R) {
        if (aoi >= 90f) return 0f;
        float ca = Mathf.Cos(Clamp(aoi, 0f, 89.999f) * Mathf.Deg2Rad);
        return (1f - Mathf.Exp(-ca / a_r)) / (1f - Mathf.Exp(-1f / a_r));
    }

    static void IamDiffuse(float tilt, out float skyV, out float grdV) {
        skyV = Interp(tilt, iam.tilts, k => iam.iam_sky[k], iam.tilts.Length);
        grdV = Interp(tilt, iam.tilts, k => iam.iam_grd[k], iam.tilts.Length);
    }

    public static float EffPoa(float tilt, float elev, float az, float dni, float dhi,
                               float albedo = ALBEDO, float c = CHORD, float p = PITCH) {
        float sf = PanelSf(tilt, elev, az, c, p);
        float ov = StripShade(sf);
        float aoi = Aoi(tilt, elev, az);
        float bp = Mathf.Max(dni * Mathf.Cos(Clamp(aoi, 0f, 90f) * Mathf.Deg2Rad), 0f);
        float iamB = IamBeam(aoi);
        IamDiffuse(tilt, out float iamSky, out float iamGrd);
        float ghi = dni * Mathf.Max(Mathf.Sin(elev * Mathf.Deg2Rad), 0f) + dhi;
        float poa = bp * iamB * (1f - ov)
                  + dhi * ViewFactor(tilt, c, p, true) * iamSky
                  + ghi * albedo * ViewFactor(tilt, c, p, false) * iamGrd;
        return Mathf.Max(poa, 0f);
    }

    // 그리드 전수 argmax = oracle 최적각
    public static float OracleTilt(float elev, float az, float dni, float dhi,
                                   float albedo = ALBEDO, float c = CHORD, float p = PITCH) {
        float best = 0f, bestE = -1f;
        for (float t = 0f; t <= 90f; t += 1f) {
            float e = EffPoa(t, elev, az, dni, dhi, albedo, c, p);
            if (e > bestE) { bestE = e; best = t; }
        }
        return best;
    }

    // Python 검증벡터 대조 (첫 실행 사니티 게이트). 반환 = 사람이 읽는 요약.
    public static string SelfTest(string dir) {
        var tf = JsonUtility.FromJson<TestVecFile>(File.ReadAllText(Path.Combine(dir, "physics_testvectors.json")));
        int pass = 0; float maxRel = 0f, maxSf = 0f;
        foreach (var v in tf.vectors) {
            float poa = EffPoa(v.tilt, v.elev, v.az, v.dni, v.dhi, tf.albedo, tf.chord, tf.pitch);
            float sfc = PanelSf(v.tilt, v.elev, v.az, tf.chord, tf.pitch);
            float rel = Mathf.Abs(poa - v.eff_poa) / Mathf.Max(1f, Mathf.Abs(v.eff_poa));
            float dsf = Mathf.Abs(sfc - v.panel_sf);
            maxRel = Mathf.Max(maxRel, rel); maxSf = Mathf.Max(maxSf, dsf);
            // 허용: POA 상대 1.5%(확산 IAM 1° 테이블 보간차) + 절대 0.5, panel_sf 1e-3
            if ((rel <= 0.015f || Mathf.Abs(poa - v.eff_poa) <= 0.5f) && dsf <= 1e-3f) pass++;
        }
        bool ok = pass == tf.vectors.Length;
        return $"[LouverPhysics 셀프테스트] {pass}/{tf.vectors.Length} {(ok ? "PASS" : "FAIL")}  " +
               $"maxRel(POA)={maxRel * 100f:0.000}%  maxAbs(sf)={maxSf:0.000000}";
    }
}
