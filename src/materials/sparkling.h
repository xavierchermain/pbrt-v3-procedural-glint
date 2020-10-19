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

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_MATERIALS_SPARKLING_H
#define PBRT_MATERIALS_SPARKLING_H

#include "material.h"
#include "microfacet.h"
#include "pbrt.h"
#include "reflection.h"
#include "rng.h"
#include "spectrum.h"
#include "textures/imagemap.h"

namespace pbrt {

// SparklingMaterial Declarations
class SparklingMaterial : public Material {
  public:
    // SparklingMaterial Public Methods
    SparklingMaterial(const std::shared_ptr<Texture<Spectrum>> &R,
                      const std::shared_ptr<Texture<Spectrum>> &eta,
                      const std::shared_ptr<Texture<Spectrum>> &k,
                      const std::shared_ptr<Texture<Float>> &alpha_x,
                      const std::shared_ptr<Texture<Float>> &alpha_y,
                      const std::shared_ptr<Texture<Spectrum>> &dictionary,
                      int NLevels, int N, float alpha_dict,
                      float logMicrofacetDensity, float microfaceRelativeArea,
                      float densityRandomisation, Float su, Float sv,
                      bool fresnelNoOp);

    void ComputeScatteringFunctions(SurfaceInteraction *si, MemoryArena &arena,
                                    TransportMode mode,
                                    bool allowMultipleLobes) const;

  private:
    // SparklingMaterial Private Data
    std::shared_ptr<Texture<Spectrum>> R;
    std::shared_ptr<Texture<Spectrum>> eta, k;
    std::shared_ptr<Texture<Float>> alpha_x, alpha_y;
    std::shared_ptr<Texture<Spectrum>> dictionary;
    int NLevels, N;
    float alpha_dict;
    float logMicrofacetDensity;
    float microfacetRelativeArea;
    float densityRandomisation;
    Float su, sv;
    bool fresnelNoOp;
};

SparklingMaterial *CreateSparklingMaterial(const TextureParams &mp);

class SparklingReflection : public BxDF {
  public:
    SparklingReflection(const Spectrum &R, Fresnel *fresnel, RNG *rng,
                        Point2f st, Vector2f dstdx, Vector2f dstdy,
                        Float alpha_x, Float alpha_y,
                        ImageTexture<RGBSpectrum, Spectrum> *dictionary,
                        int NLevels, int N, float alpha_dict,
                        float logMicrofacetDensity,
                        float microfacetRelativeArea,
                        float densityRandomisation);

    //==========================================================================
    //======= Evaluation of our procedural physically based glinty BRDF ========
    //=========================== Alg. 1, Eq. 14 ===============================
    //==========================================================================
    Spectrum f(const Vector3f &wo, const Vector3f &wi) const;

    //==========================================================================
    //================== Alg. 2, P-SDF for a discrete LOD ======================
    //==========================================================================
    Float P22__P_(int l, Vector2f slope_h, Point2f st, Vector2f dst0,
                  Vector2f dst1) const;

    //==========================================================================
    //======== Spatially-varying, multiscale, rotated, and scaled SDF ==========
    //========================== Eq. 11, Alg. 3 ================================
    //==========================================================================
    Float P22_theta_alpha(int l, int s0, int t0, Vector2f slope_h) const;

    //==========================================================================
    //================ ith marginal distribution at level l ====================
    //==========================================================================
    Float P(Float x, int i, int l) const;

    Spectrum Sample_f(const Vector3f &wo, Vector3f *wi, const Point2f &u,
                      Float *pdf, BxDFType *sampledType) const;
    Float Pdf(const Vector3f &wo, const Vector3f &wi) const;
    std::string ToString() const;

    Float sampleNormalDistribution(Float U, Float mu, Float sigma) const;

  private:
    const Spectrum R;
    const Fresnel *fresnel;
    const Point2f st;
    const Vector2f dstdx, dstdy;
    RNG *rng;
    const Float alpha_x;
    const Float alpha_y;
    const Float alpha_dict;
    ImageTexture<RGBSpectrum, Spectrum> *dictionary;
    int NLevels, N, distresolution;
    float logMicrofacetDensity;
    float densityRandomisation;
    float microfacetRelativeArea;
};

}  // namespace pbrt

#endif  // PBRT_MATERIALS_Sparkling_H