implementing "../renderer.slang";

import utils;

public float2 getShiftedTexUV(float2 uv, RendererUniform uniforms, Triangle triangle, float2 delta, float2 tex_uv)
{
    float2 shifted_uv = uv + delta;
    Ray ray = uniforms.camera.generateRay(shifted_uv);
    Optional<RayHitResult> hitResult = triangle.hit(ray);
    float2 result = tex_uv;
    if (hitResult.hasValue)
    {
        float3 hitPoint = ray.origin + hitResult.value.t * ray.direction;
        float3 barycentricCoord = triangle.calculateBarycentricCoord(hitPoint);
        return triangle.getUV(barycentricCoord);
    }

    Optional<RayMeshIntersectionResult> rayMeshIntersectionResult_dx = rayMeshIntersection(ray, uniforms);
    if (rayMeshIntersectionResult_dx.hasValue)
    {
        Triangle hitTriangle = rayMeshIntersectionResult_dx.value.hitTriangle;
        if (hitTriangle.getMaterialId() == triangle.getMaterialId())
        {
            float3 hitPoint = ray.origin + rayMeshIntersectionResult_dx.value.closestHit.t * ray.direction;
            float3 barycentricCoord = hitTriangle.calculateBarycentricCoord(hitPoint);
            return hitTriangle.getUV(barycentricCoord);
        }
    }

    return tex_uv;
}

[ForceInline]
float2 _texUVatRay(float2 uv, RendererUniform uniforms, Triangle triangle, float2 fallback) {
    // return texuv at intersection point
    Ray ray = uniforms.camera.generateRay(uv);
    Optional<RayHitResult> result = triangle.hit(ray);

    if (!result.hasValue) {
        return fallback;
    }

    float3 hitPoint = ray.origin + result.value.t * ray.direction;
    float3 barycentricCoord = triangle.calculateBarycentricCoord(hitPoint);
    return triangle.getUV(barycentricCoord);
}

public float getLevel(float2 uv, RendererUniform uniforms, Triangle triangle, float2 tex_uv)
{
    float level = 0.0;

    // TODO: Student implementation starts here.

    float small_number = 1e-5;
    // sample uv + (0, small_number) to get dx+
    // do again for the compass points
    // get tex_uv at each hit point
    // finite difference

    // dx
    float2 x_plus =  _texUVatRay(uv + float2( small_number, 0), uniforms, triangle, tex_uv);
    float2 x_minus = _texUVatRay(uv + float2(-small_number, 0), uniforms, triangle, tex_uv);
    float2 y_plus =  _texUVatRay(uv + float2(0,  small_number), uniforms, triangle, tex_uv);
    float2 y_minus = _texUVatRay(uv + float2(0, -small_number), uniforms, triangle, tex_uv);
    
    float2 duv_dx = (x_plus - x_minus) * (1.0 / small_number) * 0.5;
    float2 duv_dy = (y_plus - y_minus) * (1.0 / small_number) * 0.5;

    float rho_x = sqrt(pow(duv_dx.x, 2) + pow(duv_dx.y, 2));
    float rho_y = sqrt(pow(duv_dy.x, 2) + pow(duv_dy.y, 2));
    float rho = max(rho_x, rho_y);
    level = log2(rho);
    
    // TODO: Student implementation ends here.

    return level;
}

public PhysicsBasedParameters getHitPointParameters(float2 uv, RendererUniform uniforms, Triangle triangle, float3 normal, float3 barycentricCoord, float3 hitPoint)
{
    float2 tex_uv = triangle.getUV(barycentricCoord);
    float level = getLevel(uv, uniforms, triangle, tex_uv);
    uint materialId = triangle.materialId + triangle.offset;
    PhysicsBasedMaterial physicsBasedMaterial = uniforms.physicsBasedMaterialBuf[materialId];
    PhysicsBasedParameters parameters = sampleMaterial(physicsBasedMaterial, uniforms.physicsBasedMaterialTextureBuf, tex_uv, level, uniforms.visualizeLevelOfDetail);
    if (uniforms.visualizeBarycentricCoords)
    {
        parameters.albedo = barycentricCoord;
    }
    if (uniforms.visualizeTexUV)
    {
        parameters.albedo = float3(tex_uv, 0.0);
    }
    return parameters;
}
