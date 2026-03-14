implementing "../primitive.slang";
import math;

enum _Axis : uint {
    X, Y, Z
} 

[ForceInline]
float2 _axisProject(float3 v, _Axis axis) {
    switch (axis) {
        case _Axis::X: return v.yz;
        case _Axis::Y: return v.xz;
        case _Axis::Z: return v.xy;
        default: return v.xy;
    }
}

static const int[3] _flipCheck = {1, -1, 1};

public struct Triangle : IBoundingBox, IRayCastable
{
    public float3[3] vertices;
    public float3[3] normals;
    public float2[3] uvs;
    public float3[3] colors;
    public uint materialId;
    public uint offset;

    public BoundingBox boundingBox()
    {
        return BoundingBox(float3(0.0), float3(0.0));
    }

    /* Returns the average color of the triangle. */
    public float3 averageColor()
    {
        return (this.colors[0] + this.colors[1] + this.colors[2]) / 3.0;
    }

    /**
     * Calculates the barycentric coordinates of a point on the triangle.
     * @param p The point on the triangle to calculate the barycentric coordinates for.
     * @return The barycentric coordinates of the point.
     */
    public float3 calculateBarycentricCoord(float3 p)
    {
        const float EPSILON = 1e-6;
        float u = 0.0;
        float v = 0.0;
        float w = 0.0;

        // TODO: Student implementation starts here.

        float3 e01 = vertices[1] - vertices[0];
        float3 e12 = vertices[2] - vertices[1];
        float3 e20 = vertices[0] - vertices[2];

        float3 p0 = p - vertices[0];
        float3 p1 = p - vertices[1];
        float3 p2 = p - vertices[2];

        float total_area = length(cross(e01, e12));

        u = length(cross(e12, p1));
        v = length(cross(e20, p2));
        w = length(cross(e01, p0));

        // TODO: Student implementation ends here.

        return float3(u, v, w) / total_area;
    }

    public float2 getUV(float3 barycentricCoord)
    {
        float2 uv = float2(0.0);

        float u = barycentricCoord.x;
        float v = barycentricCoord.y;
        float w = barycentricCoord.z;
        uv = u * this.uvs[0] + v * this.uvs[1] + w * this.uvs[2];
        
        return uv;
    }

    public float3 getColor(float3 barycentricCoord)
    {
        float u = barycentricCoord.x;
        float v = barycentricCoord.y;
        float w = barycentricCoord.z;
        float3 color = u * this.colors[0] + v * this.colors[1] + w * this.colors[2];
        return color;
    }

    public float3 getNormal(float3 barycentricCoord)
    {
        float u = barycentricCoord.x;
        float v = barycentricCoord.y;
        float w = barycentricCoord.z;
        float3 normal = u * this.normals[0] + v * this.normals[1] + w * this.normals[2];
        return normalize(normal);
    }

    /**
     * Test ray-triangle intersection.
     * @param ray The ray to test against the triangle.
     * @return An Optional containing RayHitResult if the ray hits the triangle, otherwise an empty Optional.
     */
    public Optional<RayHitResult> hit(Ray ray)
    {
        // TODO: Student implementation starts here.

        const float EPSILON = 1e-4;
        
        // point-normal form of plane
        float3 P = vertices[0];
        float3 N = cross(vertices[1] - vertices[0], vertices[2] - vertices[1]);
        N /= length(N);

        // early out if parallel
        float facing_factor = dot(ray.direction, N);
        if (abs(facing_factor) <= EPSILON)
            return Optional<RayHitResult>();

        // solve for t
        float t = dot(N, P - ray.origin) / facing_factor;

        // early out if not in range
        if (t < ray.tRange.x || ray.tRange.y < t)
            return Optional<RayHitResult>();

        // compute point of intersection
        float3 X = ray.origin + t * ray.direction;

        // test if X in triangle
        
        float3 Nabs = abs(N);
        _Axis domAxis = (Nabs.x > Nabs.y)  // 0: X, 1: Y, 2: Z
                        ? ((Nabs.x > Nabs.z) ? _Axis::X : _Axis::Z) 
                        : ((Nabs.y > Nabs.z) ? _Axis::Y : _Axis::Z);
        
        float2 v0, v1, v2;
        v0 = _axisProject(vertices[0], domAxis);
        if (sign(N[domAxis]) != _flipCheck[domAxis]) {
            // flip winding if projecting backface
            v1 = _axisProject(vertices[2], domAxis);
            v2 = _axisProject(vertices[1], domAxis);
        } else {
            v1 = _axisProject(vertices[1], domAxis);
            v2 = _axisProject(vertices[2], domAxis);
        }
        float2 x0 = _axisProject(X, domAxis);

        float2 e0 = v1 - v0;
        float2 e1 = v2 - v1;
        float2 e2 = v0 - v2;

        bool inTri = dot({e0.y, -e0.x}, x0 - v0) < 0.0
                  && dot({e1.y, -e1.x}, x0 - v1) < 0.0
                  && dot({e2.y, -e2.x}, x0 - v2) < 0.0;

        if (inTri)  return RayHitResult(t, N);
        else        return Optional<RayHitResult>();
        // TODO: Student implementation ends here.
    }

    public uint getMaterialId()
    {
        return this.materialId + this.offset;
    }
}
