/*
    Copyright © 2020 Xavier Chermain (ICUBE), Basile Sauvage (ICUBE),
    Jean-Michel Dishler (ICUBE), Carsten Dachsbacher (KIT)

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.

    Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.

    pbrt-v3 implementation of
    Procedural Physically based BRDF for Real-Time Rendering of Glints
    Xavier Chermain (ICUBE), Basile Sauvage (ICUBE), Jean-Michel Dishler (ICUBE)
    and Carsten Dachsbacher (KIT)
    Pacific Graphic 2020, CGF special issue
    Project page: http://igg.unistra.fr/People/chermain/real_time_glint/
*/

/*
    pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.

    This file is part of pbrt.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

#include "materials/sparkling.h"

#include "interaction.h"
#include "paramset.h"
#include "reflection.h"
#include "texture.h"

namespace pbrt {

SparklingMaterial::SparklingMaterial(
    const std::shared_ptr<Texture<Spectrum>> &R,
    const std::shared_ptr<Texture<Spectrum>> &eta,
    const std::shared_ptr<Texture<Spectrum>> &k,
    const std::shared_ptr<Texture<Float>> &alpha_x,
    const std::shared_ptr<Texture<Float>> &alpha_y,
    const std::shared_ptr<Texture<Spectrum>> &dictionary, int NLevels, int N,
    float alpha_dict, float logMicrofacetDensity, float microfacetRelativeArea,
    float densityRandomisation, Float su, Float sv, bool fresnelNoOp)
    : R(R),
      eta(eta),
      k(k),
      alpha_x(alpha_x),
      alpha_y(alpha_y),
      dictionary(dictionary),
      NLevels(NLevels),
      N(N),
      alpha_dict(alpha_dict),
      logMicrofacetDensity(logMicrofacetDensity),
      microfacetRelativeArea(microfacetRelativeArea),
      densityRandomisation(densityRandomisation),
      su(su),
      sv(sv),
      fresnelNoOp(fresnelNoOp) {}

void SparklingMaterial::ComputeScatteringFunctions(
    SurfaceInteraction *si, MemoryArena &arena, TransportMode mode,
    bool allowMultipleLobes) const {
    Vector2f dstdx(su * si->dudx, sv * si->dvdx);
    Vector2f dstdy(su * si->dudy, sv * si->dvdy);
    Point2f st(su * si->uv[0], sv * si->uv[1]);

    si->bsdf = ARENA_ALLOC(arena, BSDF)(*si);

    Fresnel *frMf;
    if (fresnelNoOp)
        frMf = ARENA_ALLOC(arena, FresnelNoOp)();
    else
        frMf = ARENA_ALLOC(arena, FresnelConductor)(1., eta->Evaluate(*si),
                                                    k->Evaluate(*si));

    RNG *rng = ARENA_ALLOC(arena, RNG)();

    Spectrum R = this->R->Evaluate(*si).Clamp();
    Float alpha_x_P = alpha_x->Evaluate(*si);
    Float alpha_y_P = alpha_y->Evaluate(*si);

    ImageTexture<RGBSpectrum, Spectrum> *dict =
        dynamic_cast<ImageTexture<RGBSpectrum, Spectrum> *>(dictionary.get());

    si->bsdf->Add(ARENA_ALLOC(arena, SparklingReflection)(
        R, frMf, rng, st, dstdx, dstdy, alpha_x_P, alpha_y_P, dict, NLevels, N,
        alpha_dict, logMicrofacetDensity, microfacetRelativeArea,
        densityRandomisation));
}

const int CopperSamples = 56;
const Float CopperWavelengths[CopperSamples] = {
    298.7570554, 302.4004341, 306.1337728, 309.960445,  313.8839949,
    317.9081487, 322.036826,  326.2741526, 330.6244747, 335.092373,
    339.6826795, 344.4004944, 349.2512056, 354.2405086, 359.374429,
    364.6593471, 370.1020239, 375.7096303, 381.4897785, 387.4505563,
    393.6005651, 399.9489613, 406.5055016, 413.2805933, 420.2853492,
    427.5316483, 435.0322035, 442.8006357, 450.8515564, 459.2006593,
    467.8648226, 476.8622231, 486.2124627, 495.936712,  506.0578694,
    516.6007417, 527.5922468, 539.0616435, 551.0407911, 563.5644455,
    576.6705953, 590.4008476, 604.8008683, 619.92089,   635.8162974,
    652.5483053, 670.1847459, 688.8009889, 708.4810171, 729.3186941,
    751.4192606, 774.9011125, 799.8979226, 826.5611867, 855.0632966,
    885.6012714};

const Float CopperN[CopperSamples] = {
    1.400313, 1.38,  1.358438, 1.34,  1.329063, 1.325, 1.3325,   1.34,
    1.334375, 1.325, 1.317812, 1.31,  1.300313, 1.29,  1.281563, 1.27,
    1.249062, 1.225, 1.2,      1.18,  1.174375, 1.175, 1.1775,   1.18,
    1.178125, 1.175, 1.172812, 1.17,  1.165312, 1.16,  1.155312, 1.15,
    1.142812, 1.135, 1.131562, 1.12,  1.092437, 1.04,  0.950375, 0.826,
    0.645875, 0.468, 0.35125,  0.272, 0.230813, 0.214, 0.20925,  0.213,
    0.21625,  0.223, 0.2365,   0.25,  0.254188, 0.26,  0.28,     0.3};

const Float CopperK[CopperSamples] = {
    1.662125, 1.687, 1.703313, 1.72,  1.744563, 1.77,  1.791625, 1.81,
    1.822125, 1.834, 1.85175,  1.872, 1.89425,  1.916, 1.931688, 1.95,
    1.972438, 2.015, 2.121562, 2.21,  2.177188, 2.13,  2.160063, 2.21,
    2.249938, 2.289, 2.326,    2.362, 2.397625, 2.433, 2.469187, 2.504,
    2.535875, 2.564, 2.589625, 2.605, 2.595562, 2.583, 2.5765,   2.599,
    2.678062, 2.809, 3.01075,  3.24,  3.458187, 3.67,  3.863125, 4.05,
    4.239563, 4.43,  4.619563, 4.817, 5.034125, 5.26,  5.485625, 5.717};

SparklingMaterial *CreateSparklingMaterial(const TextureParams &mp) {
    std::shared_ptr<Texture<Spectrum>> R =
        mp.GetSpectrumTexture("R", Spectrum(1.f));

    static Spectrum copperN =
        Spectrum::FromSampled(CopperWavelengths, CopperN, CopperSamples);
    std::shared_ptr<Texture<Spectrum>> eta =
        mp.GetSpectrumTexture("eta", copperN);
    static Spectrum copperK =
        Spectrum::FromSampled(CopperWavelengths, CopperK, CopperSamples);
    std::shared_ptr<Texture<Spectrum>> k = mp.GetSpectrumTexture("k", copperK);

    std::shared_ptr<Texture<Float>> alpha_x =
        mp.GetFloatTexture("alpha_x", 0.5);
    std::shared_ptr<Texture<Float>> alpha_y =
        mp.GetFloatTexture("alpha_y", 0.5);

    std::shared_ptr<Texture<Spectrum>> dictionary =
        mp.GetSpectrumTextureOrNull("dictionary");

    int NLevels = mp.FindInt("nlevels", 1);
    int N = mp.FindInt("N", 1);
    Float alpha_dict = mp.FindFloat("alpha_dict", 1.f);

    Float logMicrofacetDensity = mp.FindFloat("logmicrofacetdensity", 20.f);
    Float microfacetRelativeArea = mp.FindFloat("microfacetrelativearea", 1.f);
    Float densityRandomisation = mp.FindFloat("densityrandomisation", 2.f);

    Float su = mp.FindFloat("su", 1.f);
    Float sv = mp.FindFloat("sv", 1.f);

    bool fresnelNoOp = mp.FindBool("fresnelnoop", false);

    return new SparklingMaterial(R, eta, k, alpha_x, alpha_y, dictionary,
                                 NLevels, N, alpha_dict, logMicrofacetDensity,
                                 microfacetRelativeArea, densityRandomisation,
                                 su, sv, fresnelNoOp);
}

SparklingReflection::SparklingReflection(
    const Spectrum &R, Fresnel *fresnel, RNG *rng, Point2f st, Vector2f dstdx,
    Vector2f dstdy, Float alpha_x, Float alpha_y,
    ImageTexture<RGBSpectrum, Spectrum> *dictionary, int NLevels, int N,
    float alpha_dict, float logMicrofacetDensity, float microfacetRelativeArea,
    float densityRandomisation)
    : BxDF(BxDFType(BSDF_GLOSSY | BSDF_REFLECTION)),
      R(R),
      fresnel(fresnel),
      rng(rng),
      st(st),
      dstdx(dstdx),
      dstdy(dstdy),
      alpha_x(alpha_x),
      alpha_y(alpha_y),
      dictionary(dictionary),
      NLevels(NLevels),
      N(N),
      alpha_dict(alpha_dict),
      logMicrofacetDensity(logMicrofacetDensity),
      microfacetRelativeArea(microfacetRelativeArea),
      densityRandomisation(densityRandomisation),
      distresolution(dictionary->Width() / (N / 3)) {}

//==========================================================================
//================== Sampling from a normal distribution ===================
//==========================================================================
Float SparklingReflection::sampleNormalDistribution(Float U, Float mu,
                                                    Float sigma) const {
    Float x = sigma * 1.414213f * ErfInv(2.0f * U - 1.0f) + mu;
    return x;
}

//==========================================================================
//================ ith marginal distribution at level l ====================
//==========================================================================
Float SparklingReflection::P(Float x, int i, int l) const {
    // 0.707106 \approx 1 / sqrt(2)
    Float alpha_dist_isqrt2_4 = alpha_dict * 0.707106 * 4.;

    if (x >= alpha_dist_isqrt2_4) {
        return 0.;
    }
    float texCoord = x / alpha_dist_isqrt2_4 / (N / 3);
    texCoord *= dictionary->Width();

    int distIdxOver3 = i / 3;

    Vector2i shift(
        std::floor((float)distIdxOver3 / (N / 3) * dictionary->Width()),
        std::floor((float)l / NLevels * dictionary->Height()));

    Float s_x = texCoord;

    int s0_x = std::floor(s_x);
    Float ds_x = s_x - s0_x;

    Spectrum S_densitiesX =
        (1 - ds_x) * dictionary->Texel(0, s0_x + shift.x, shift.y) +
        ds_x *
            dictionary->Texel(
                0, Clamp(s0_x + 1, 0, distresolution - 1) + shift.x, shift.y);

    return S_densitiesX[Mod(i, 3)];
}

//==========================================================================
//======== Spatially-varying, multiscale, rotated, and scaled SDF ==========
//========================== Eq. 11, Alg. 3 ================================
//==========================================================================
Float SparklingReflection::P22_theta_alpha(int l, int s0, int t0,
                                           Vector2f slope_h) const {
    // Coherent index
    // Eq. 8, Alg. 3, line 1
    s0 *= 1 << l;
    t0 *= 1 << l;

    // Seed pseudo random generator
    // Alg. 3, line 2
    rng->SetSequence(s0 + 1549 * t0);

    // Alg.3, line 3
    Float uMicrofacetRelativeArea = rng->UniformFloat();
    // Discard cells by using microfacet relative area
    // Alg.3, line 4
    if (uMicrofacetRelativeArea > microfacetRelativeArea) return 0.f;

    // Number of microfacets in a cell
    // Alg. 3, line 5
    Float n = std::pow(2., Float(2 * l - (2 * (NLevels - 1))));
    n *= std::exp(logMicrofacetDensity);

    // Corresponding continuous distribution LOD
    // Alg. 3, line 6
    Float l_dist = std::log(n) / 1.38629;  // 2. * log(2) = 1.38629

    // Alg. 3, line 7
    Float uDensityRandomisation = rng->UniformFloat();

    // Sample a Gaussian to randomise the distribution LOD around the
    // distribution level l_dist Alg. 3, line 8
    l_dist = sampleNormalDistribution(uDensityRandomisation, l_dist,
                                      densityRandomisation);

    // Alg. 3, line 9
    l_dist = Clamp(int(std::round(l_dist)), 0, NLevels);

    // Alg. 3, line 10
    if (l_dist == NLevels) {
        BeckmannDistribution beckmannDistribution(alpha_x, alpha_y, false);
        Float deno(slope_h.x * slope_h.x + slope_h.y * slope_h.y + 1);
        if (deno < 0.001) return 0.f;
        Vector3f wh(-slope_h.x, -slope_h.y, 1.);
        wh /= std::sqrt(deno);
        return beckmannDistribution.D(wh) * wh.z * wh.z * wh.z * wh.z;
    }

    // Alg. 3, line 13
    Float uTheta = rng->UniformFloat();
    Float theta = 2. * Pi * uTheta;

    // Uncomment to remove random distribution rotation
    // Lead to glint alignments
    //     theta = 0.;

    Float cosTheta = std::cos(theta);
    Float sinTheta = std::sin(theta);

    Vector2f scaleFactor(alpha_x / alpha_dict, alpha_y / alpha_dict);

    // Rotate and scale slope
    // Alg. 3, line 16
    slope_h = Vector2f(slope_h.x * cosTheta / scaleFactor.x +
                           slope_h.y * sinTheta / scaleFactor.y,
                       -slope_h.x * sinTheta / scaleFactor.x +
                           slope_h.y * cosTheta / scaleFactor.y);

    Vector2f abs_slope_h(std::abs(slope_h.x), std::abs(slope_h.y));

    // 0.707106 \approx 1 / sqrt(2)
    Float alpha_dist_isqrt2_4 = alpha_dict * 0.707106 * 4.;

    if (abs_slope_h.x > alpha_dist_isqrt2_4 ||
        abs_slope_h.y > alpha_dist_isqrt2_4)
        return 0.;

    // Alg. 3, line 17
    Float u1 = rng->UniformFloat();
    Float u2 = rng->UniformFloat();

    // Alg. 3, line 18
    int i = int(u1 * Float(N));
    int j = int(u2 * Float(N));

    Float P_i = P(abs_slope_h.x, i, l_dist);
    Float P_j = P(abs_slope_h.y, j, l_dist);

    // Alg. 3, line 19
    return P_i * P_j / (scaleFactor.x * scaleFactor.y);
}

//==========================================================================
//================== Alg. 2, P-SDF for a discrete LOD ======================
//==========================================================================

// Most of this function is similar to pbrt-v3 EWA function,
// which itself is similar to Heckbert 1889 algorithm,
// http://www.cs.cmu.edu/~ph/texfund/texfund.pdf, Section 3.5.9. Go through
// cells within the pixel footprint for a giving LOD
Float SparklingReflection::P22__P_(int l, Vector2f slope_h, Point2f st,
                                   Vector2f dst0, Vector2f dst1) const {
    // Convert surface coordinates to appropriate scale for level
    int pyrSize = std::pow(2, NLevels - 1 - l);
    st[0] = st[0] * pyrSize - 0.5f;
    st[1] = st[1] * pyrSize - 0.5f;
    dst0[0] *= pyrSize;
    dst0[1] *= pyrSize;
    dst1[0] *= pyrSize;
    dst1[1] *= pyrSize;

    // Compute ellipse coefficients to bound filter region
    Float A = dst0[1] * dst0[1] + dst1[1] * dst1[1] + 1;
    Float B = -2 * (dst0[0] * dst0[1] + dst1[0] * dst1[1]);
    Float C = dst0[0] * dst0[0] + dst1[0] * dst1[0] + 1;
    Float invF = 1 / (A * C - B * B * 0.25f);
    A *= invF;
    B *= invF;
    C *= invF;

    // Compute the ellipse's bounding box in texture space
    Float det = -B * B + 4 * A * C;
    Float invDet = 1 / det;
    Float uSqrt = std::sqrt(det * C), vSqrt = std::sqrt(A * det);
    int s0 = std::ceil(st[0] - 2 * invDet * uSqrt);
    int s1 = std::floor(st[0] + 2 * invDet * uSqrt);
    int t0 = std::ceil(st[1] - 2 * invDet * vSqrt);
    int t1 = std::floor(st[1] + 2 * invDet * vSqrt);

    // Scan over ellipse bound and compute quadratic equation
    Float sum(0.f);
    Float sumWts = 0;
    for (int it = t0; it <= t1; ++it) {
        Float tt = it - st[1];
        for (int is = s0; is <= s1; ++is) {
            Float ss = is - st[0];
            // Compute squared radius and filter SDF if inside ellipse
            Float r2 = A * ss * ss + B * ss * tt + C * tt * tt;
            if (r2 < 1) {
                Float alpha = 2;
                // Weighting function used in pbrt-v3 EWA function
                Float W_P = std::exp(-alpha * r2) - std::exp(-alpha);
                // Alg. 2, line 3
                sum += P22_theta_alpha(l, is, it, slope_h) * W_P;
                sumWts += W_P;
            }
        }
    }
    return sum / sumWts;
}

//==========================================================================
//======= Evaluation of our procedural physically based glinty BRDF ========
//=========================== Alg. 1, Eq. 14 ===============================
//==========================================================================

Spectrum SparklingReflection::f(const Vector3f &wo, const Vector3f &wi) const {
    Float cosThetaO = CosTheta(wo), cosThetaI = CosTheta(wi);
    Vector3f wh = wi + wo;
    // Handle degenerate cases for microfacet reflection
    if (cosThetaI <= 0 || cosThetaO <= 0) return Spectrum(0.);
    if (wh.x == 0 && wh.y == 0 && wh.z == 0) return Spectrum(0.);

    // Alg. 1, line 1
    wh = Normalize(wh);

    // Local masking shadowing
    if (Dot(wo, wh) <= 0. || Dot(wi, wh) <= 0.) return Spectrum(0.);

    Float maxAnisotropy = 8.f;

    // Eq. 1, Alg. 1, line 2
    Vector2f slope_h(-wh.x / wh.z, -wh.y / wh.z);

    float D_P = 0.;
    float P22_P = 0.;

    // ---------------------------------------------------------------------
    // Similar to pbrt-v3 MIPMap::Lookup function,
    // http://www.pbr-book.org/3ed-2018/Texture/Image_Texture.html#EllipticallyWeightedAverage

    // Alg. 1, line 3
    Vector2f dst0 = dstdx;
    Vector2f dst1 = dstdy;

    if (dst0.LengthSquared() < dst1.LengthSquared()) std::swap(dst0, dst1);
    // Compute ellipse minor and major axes
    Float majorLength = dst0.Length();
    // Alg. 1, line 5
    Float minorLength = dst1.Length();

    // Clamp ellipse eccentricity if too large
    // Alg. 1, line 4
    if (minorLength * maxAnisotropy < majorLength && minorLength > 0) {
        Float scale = majorLength / (minorLength * maxAnisotropy);
        dst1 *= scale;
        minorLength *= scale;
    }
    // ---------------------------------------------------------------------

    // Without footprint, we evaluate the Cook Torrance BRDF
    if (minorLength == 0) {
        BeckmannDistribution beckmannDistribution(alpha_x, alpha_y, false);
        D_P = beckmannDistribution.D(wh);
    } else {
        // Choose LOD
        // Alg. 1, line 6
        Float l = std::max(0., NLevels - 1. + std::log2(minorLength));
        int il = int(floor(l));

        // Alg. 1, line 7
        float w = l - float(il);

        // Alg. 1, line 8
        P22_P = Lerp(w, P22__P_(il, slope_h, st, dst0, dst1),
                     P22__P_(il + 1, slope_h, st, dst0, dst1));

        // Eq. 6, Alg. 1, line 10
        D_P = P22_P / (wh.z * wh.z * wh.z * wh.z);
    }

    Spectrum F = fresnel->Evaluate(Dot(wi, Faceforward(wh, Vector3f(0, 0, 1))));

    Float G1wowh = std::min(1., 2. * wh.z * wo.z / Dot(wo, wh));
    Float G1wiwh = std::min(1., 2. * wh.z * wi.z / Dot(wi, wh));
    float G = G1wowh * G1wiwh;

    // Eq. 14, Alg. 1, line 14
    return R * (F * G * D_P) / (4. * wo.z * wi.z);
}

std::string SparklingReflection::ToString() const {
    return std::string("[ SparklingReflection R: ") + R.ToString() +
           std::string(" fresnel: ") + fresnel->ToString() + std::string(" ]");
}

Spectrum SparklingReflection::Sample_f(const Vector3f &wo, Vector3f *wi,
                                       const Point2f &u, Float *pdf,
                                       BxDFType *sampledType) const {
    // We sample the target distribution instead of the real distribution for
    // convenience

    // Sample microfacet orientation $\wh$ and reflected direction $\wi$
    if (wo.z == 0) return 0.;
    BeckmannDistribution bd(0.5f, 0.5f);
    Vector3f wh = bd.Sample_wh(wo, u);
    if (Dot(wo, wh) < 0) return 0.;  // Should be rare
    *wi = Reflect(wo, wh);
    if (!SameHemisphere(wo, *wi)) return Spectrum(0.f);

    // Compute PDF of _wi_ for microfacet reflection
    *pdf = bd.Pdf(wo, wh) / (4 * Dot(wo, wh));
    return f(wo, *wi);
}

Float SparklingReflection::Pdf(const Vector3f &wo, const Vector3f &wi) const {

    if (!SameHemisphere(wo, wi)) return 0;
    Vector3f wh = Normalize(wo + wi);
    BeckmannDistribution bd(0.5f, 0.5f);
    return bd.Pdf(wo, wh) / (4 * Dot(wo, wh));
}

}  // namespace pbrt