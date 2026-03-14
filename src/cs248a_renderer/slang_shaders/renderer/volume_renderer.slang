implementing "../renderer.slang";

import math;
import texture;

public static uint MAX_STEPS = 1024;
public static float EPSILON = 1e-6;

public static float STEP_SIZE = 0.01;
public static float DENSITY_SCALE = 5.0;

/**
 * Perform ray marching through the volume and accumulate color.
 * @param uv The normalized uv coordinate from (0, 0) in the bottom-left to (1, 1) in the top-right.
 * @param uniforms The renderer uniform containing camera, sampling, and scene information.
 * @param tMax The maximum t value for ray marching (usually the nearest primitive intersection t).
 * @param primColor The color of the nearest primitive intersection.
 * @return The accumulated color after ray marching through the volume.
 */
public float4 volumeSample(float2 uv, RendererUniform uniforms, float tMax, float4 primColor)
{
    // TODO: Student implementation starts here.
    DiffTexture3D<float, 4> data = uniforms.volume.tex;
    Ray ray = uniforms.camera.generateRay(uv).transform(uniforms.volume.invModelMatrix);

    return accumulateColor<4>(ray, uniforms.volume.bound, MAX_STEPS, STEP_SIZE, DENSITY_SCALE, primColor, data);
    
    // TODO: Student implementation ends here.
}

/**
 * Backward pass for volumetric ray marching at a single uv.
 * @param uv The normalized uv coordinate from (0, 0) in the bottom-left to (1, 1) in the top-right.
 * @param uniforms The renderer uniform containing camera, sampling, and scene information.
 * @param outGrad The output gradient w.r.t. the sampled color.
 */
public void volumeSampleBwd(float2 uv, RendererUniform uniforms, float4 outGrad)
{
    // TODO: Student implementation starts here.

    DiffTexture3D<float, 4> data = uniforms.volume.tex;
    Ray ray = uniforms.camera.generateRay(uv).transform(uniforms.volume.invModelMatrix);
    DifferentialPtrPair<DiffTexture3D<float,4>> diffPair = {data, uniforms.volume.dTex};

    bwd_diff(accumulateColor<4>)(
        ray, uniforms.volume.bound, 
        MAX_STEPS, STEP_SIZE, DENSITY_SCALE, 0, 
        diffPair, outGrad
    );

    // TODO: Student implementation ends here.
}

/**
 * Ray march through the volume and accumulate color.
 * @param ray The ray in local space.
 * @param volumeBound The bounding box of the volume in local space.
 * @param maxSteps The maximum number of marching steps.
 * @param stepSize The size of each marching step.
 * @param densityScale The scale factor for the volume density (alpha channel).
 * @param backgroundColor The background color behind the volume
                          (this can be the ambient color if there's no primitive behind the volume, or the color of the primitive hit if there's one).
 * @param data The 3D texture storing the volume data (rgba).
 * @return The accumulated color after ray marching through the volume.
 */
[Differentiable]
public float4 accumulateColor<let N : uint>(
    no_diff Ray ray,
    no_diff BoundingBox volumeBound,
    no_diff uint maxSteps,
    no_diff float stepSize,
    no_diff float densityScale,
    no_diff float4 backgroundColor,
    DiffTexture3D<float, 4> data)
{
    // TODO: Student implementation starts here

    float2 bounding_box_distance_range = volumeBound.hit(ray);
    float tMin = bounding_box_distance_range.x;
    float tMax = bounding_box_distance_range.y;

    if(tMin > tMax) return backgroundColor;

    float t = max(tMin, 0.0);

    float3 pos = ray.origin + t * ray.direction;

    uint iterCounter = 0;

    float4 acc = 0;
    float transmittance = 1.0;
    // [ForceUnroll]
    [MaxIters(1024)]
    for (uint i = 0; i < maxSteps; ++i) {
    // while(iterCounter < maxSteps && // step count
    //     t < tMax
    //     // iterCounter * stepSize < tMax - tMin // don't go out the back of the box
    // ) {
        if (t >= tMax) break;

        pos = ray.origin + t * ray.direction;
        float3 uvw = (pos + (volumeBound.size() * 0.5)) / volumeBound.size();

        float4 sample = sampleTrilinear<float, 4>(data, uvw);
        float density = sample.a * densityScale;
        float alpha = 1.0 - exp(-density * stepSize);

        float weight = alpha * transmittance;

        acc.rgb += sample.rgb * weight;
        acc.a += weight;

        // Update transmittance
        transmittance *= (1.0 - alpha);
        
        // March forward
        t += stepSize;
        iterCounter += 1;
    }

    return acc + transmittance * backgroundColor;

    // TODO: Student implementation ends here.
}
