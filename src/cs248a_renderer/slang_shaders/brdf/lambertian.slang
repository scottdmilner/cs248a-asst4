implementing "../brdf.slang";

import utils;

public struct LambertianBRDF
{
    public float3 albedo;
    public bool useCosineWeightedSampling;

    public float3 evaluate()
    {
        float3 fr = this.albedo;

        fr = this.albedo * INV_PI;

        return fr;
    }

    public float3 sample(float3 rayOut, float3 normal, float2 randomEpsilon)
    {
        float3 sampleDirection = rayOut;
        float3x3 worldToLocalMatrix = getWorldToLocalMatrix(normal, rayOut);
        float3x3 localToWorldMatrix = transposeMatrix(worldToLocalMatrix);

        // HINT: First convert the reflected ray direction to the local space,
        // then compute the incident ray direction in the local space,
        // and finally convert it to the world space.

        if (this.useCosineWeightedSampling)
        {
            // TODO: Student implementation starts here.

            float cosTheta = sqrt(randomEpsilon.x);
            float sinTheta = sqrt(1.0 - randomEpsilon.x);
            float phi = TWO_PI * randomEpsilon.y;

            float3 randomLocal = float3(sinTheta * cos(phi), cosTheta, sinTheta * sin(phi));
            
            sampleDirection = mul(localToWorldMatrix, randomLocal);

            // TODO: Student implementation ends here.
        }
        else // uniform sampling, use uniform sampling by default
        {
            // TODO: Student implementation starts here.

            float h = randomEpsilon.x;
            float r = sqrt(1 - pow(h, 2));
            float theta = TWO_PI * randomEpsilon.y;
            
            float3 randomLocal = float3(r * cos(theta), h, r * sin(theta));
            
            sampleDirection = mul(localToWorldMatrix, randomLocal);

            // TODO: Student implementation ends here.
        }

        return sampleDirection;
    }

    public float pdf(float3 rayIn, float3 rayOut, float3 normal)
    {
        float EPSILON = 1e-6;
        float pdf = all(abs(rayIn - rayOut) < EPSILON) ? 1.0 : 0.0;

        if (this.useCosineWeightedSampling)
        {
            // TODO: Student implementation starts here.
            
            pdf = dot(rayOut, normal) * INV_PI * 4;

            // TODO: Student implementation ends here.
        }
        else // uniform sampling, use uniform sampling by default
        {
            // TODO: Student implementation starts here.
            
            pdf = INV_PI;

            // TODO: Student implementation ends here.
        }

        return pdf;
    }
}
