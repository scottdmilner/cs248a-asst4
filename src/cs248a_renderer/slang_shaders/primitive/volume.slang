implementing "../primitive.slang";
import math;
import texture;

// The size of each marching step.
static float STEP_SIZE = 0.005;
// The density threshold to consider a hit.
static float DENSITY_THRESHOLD = 0.25;
// The delta used for finite difference normal estimation.
static float FINITE_DIFF_DELTA = 0.001;

public struct Volume
{
    public BoundingBox bound;
    public SharedTexture3D tex;
    public float4x4 modelMatrix;
    public float4x4 invModelMatrix;

    /**
     * Test ray-volume bounding box intersection.
     * @param ray The ray to test against the volume.
     * @param volumeTexBuf The texture buffer contains all the volumetrically represented models in the scene .
     * @return An Optional containing RayHitResult if the ray hits the volume, otherwise an empty Optional.
     */
    public Optional<RayHitResult> hit(Ray ray, SharedTexture3DBuffer<float4> volumeTexBuf)
    {
        // TODO: Student implementation starts here.

        
        float3 p = ray.origin;
        
        for (uint i = 0; i < 200000; ++i) {
            float3 uvw = getUVW(p);
            if (all(uvw >= float3(0)) && all(uvw <= float3(1))) { // check bounds
                if (volumeTexBuf.trilinearSample(tex, uvw).w >= DENSITY_THRESHOLD) {
                    // compute normal
                    float gx = volumeTexBuf.trilinearSample(tex, uvw + float3(FINITE_DIFF_DELTA, 0, 0)).w
                             - volumeTexBuf.trilinearSample(tex, uvw - float3(FINITE_DIFF_DELTA, 0, 0)).w;
                    float gy = volumeTexBuf.trilinearSample(tex, uvw + float3(0, FINITE_DIFF_DELTA, 0)).w
                             - volumeTexBuf.trilinearSample(tex, uvw - float3(0, FINITE_DIFF_DELTA, 0)).w;
                    float gz = volumeTexBuf.trilinearSample(tex, uvw + float3(0, 0, FINITE_DIFF_DELTA)).w
                             - volumeTexBuf.trilinearSample(tex, uvw - float3(0, 0, FINITE_DIFF_DELTA)).w;
                    return RayHitResult(STEP_SIZE * i, -normalize(float3(gx,gy,gz)));
                }
            }
            p += ray.direction * STEP_SIZE;
        }

        // TODO: Student implementation ends here.

        return Optional<RayHitResult>();
    }

    /**
     * Get the uvw coordinates of a point in local space.
     * @param p The point in local space.
     * @return The uvw coordinates in [0, 1].
     */
    public float3 getUVW(float3 p)
    {
        // Compute uvw coordinates in [0, 1].
        float3 minBound = this.bound.pMin;
        float3 maxBound = this.bound.pMax;
        return (p - minBound) / (maxBound - minBound);
    }
};
