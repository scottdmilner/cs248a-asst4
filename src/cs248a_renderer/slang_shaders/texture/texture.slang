implementing "../texture.slang";

public struct SharedTexture2DBuffer<T>
    where T : IFloat
{
    public StructuredBuffer<T> buffer;

    T getBufferValue(SharedTexture2D tex, uint2 texelCoord)
    {
        uint2 texSize = tex.size;
        uint offset = tex.offset;
        uint index = texelCoord.y * texSize.x + texelCoord.x;
        return buffer[offset + index];
    }

    public T pointSample(SharedTexture2D tex, float2 uv)
    {
        T outV = T(0.0);
        // TODO: Student implementation starts here.

        uint2 coords = uint2(tex.size * uv);
        outV = getBufferValue(tex, coords);

        // TODO: Student implementation ends here.
        return outV;
    }

public T bilinearSample(SharedTexture2D tex, float2 uv)
    {
        // TODO: Student implementation starts here.
        uint2 size = tex.size;
        float2 coords = size * uv - float2(0.5, 0.5);
        if(coords.x < 0) coords.x = 0;
        if(coords.y < 0) coords.y = 0;
        uint2 sampleCorner = uint2(coords);

        T value = T(0);
        for(uint i = 0; i <= 1; i++) {
            for(uint j = 0; j <= 1; j++) {
                uint2 offset = uint2(i, j);
                uint2 sampleCoords = sampleCorner + offset;

                // force to be in bounds
                sampleCoords.x = sampleCoords.x < size.x ? sampleCoords.x : size.x - 1;
                sampleCoords.y = sampleCoords.y < size.y ? sampleCoords.y : size.y - 1;
                sampleCoords.x = sampleCoords.x >= 0 ? sampleCoords.x : 0;
                sampleCoords.y = sampleCoords.y >= 0 ? sampleCoords.y : 0;

                float2 lerp_factor = coords - sampleCorner;
                float2 weights = 2 * lerp_factor * offset - offset - lerp_factor + 1;

                float localWeight = weights.x * weights.y;

                value = value + T(getBufferValue(tex, sampleCoords)) * T(localWeight);
            }
        }

        return value;

        // uint2 coord = max(uint2(uv * tex.size), 0);
        // float2 weight = uv * tex.size - float2(coord);

        // T t00 = getBufferValue(tex, coord             ) * T(1.0 - weight.x);
        // T t10 = getBufferValue(tex, coord + uint2(1,0)) * T(weight.x);
        // T t01 = getBufferValue(tex, coord + uint2(0,1)) * T(1.0 - weight.x);
        // T t11 = getBufferValue(tex, coord + uint2(1,1)) * T(weight.x);

        // return (t00 + t10) * T(1.0 - weight.y) + (t01 + t11) * T(weight.y);
        // TODO: Student implementation ends here.
    }

    public T trilinearSample(SharedTexture2D[] mipmapLevels, float2 uv, float d, uint totalLevels)
    {
        T outV = T(0.0);
        // TODO: Student implementation starts here.
        uint idx_low = uint(d);
        uint idx_high = idx_low + 1;
        if(idx_low < 0) {
            idx_low = 0;
            idx_high = 0;
        }

        float weight = d - idx_low;
        T highres = bilinearSample(mipmapLevels[idx_low], uv);
        T lowres = bilinearSample(mipmapLevels[idx_high], uv);

        // TODO: Student implementation ends here.
        return highres * T(1 - weight) + lowres * T(weight);
    }
}

public struct SharedTexture2D
{
    public uint2 size;
    public uint offset;
}

public struct SharedTexture3DBuffer<T>
    where T : IFloat
{
    public StructuredBuffer<T> buffer;

    T getBufferValue(SharedTexture3D tex, uint3 voxelCoord)
    {
        uint3 texSize = tex.size;
        uint offset = tex.offset;
        uint index = voxelCoord.z * texSize.y * texSize.x +
                     voxelCoord.y * texSize.x +
                     voxelCoord.x;
        return buffer[offset + index];
    }

    /**
     * Sample the texture at the given uvw coordinates using nearest neighbor interpolation.
     * @param tex The texture to sample.
     * @param uvw The uvw coordinates to sample the texture at.
     * @return The sampled value.
     */
    public T pointSample(SharedTexture3D tex, float3 uvw)
    {
        // TODO: Student implementation starts here.
        return getBufferValue(tex, uint3(uvw * tex.size));

        // TODO: Student implementation ends here.
    }

    /**
     * Sample the texture at the given uvw coordinates using trilinear interpolation.
     * @param tex The texture to sample.
     * @param uvw The uvw coordinates to sample the texture at.
     * @return The sampled value.
     **/
    public T trilinearSample(SharedTexture3D tex, float3 uvw)
    {
        // TODO: Student implementation starts here.
        uint3 coord = uint3(uvw * tex.size);
        float3 weight = uvw * tex.size - float3(coord);

        T x00 = getBufferValue(tex, coord               ) * T(1.0 - weight.x) 
              + getBufferValue(tex, coord + uint3(1,0,0)) * T(weight.x);
        T x10 = getBufferValue(tex, coord + uint3(0,1,0)) * T(1.0 - weight.x) 
              + getBufferValue(tex, coord + uint3(1,1,0)) * T(weight.x);
        T x01 = getBufferValue(tex, coord + uint3(0,0,1)) * T(1.0 - weight.x) 
              + getBufferValue(tex, coord + uint3(1,0,1)) * T(weight.x);
        T x11 = getBufferValue(tex, coord + uint3(0,1,1)) * T(1.0 - weight.x) 
              + getBufferValue(tex, coord + uint3(1,1,1)) * T(weight.x);

        T xy0 = x00 * T(1.0 - weight.y) + x10 * T(weight.y);
        T xy1 = x01 * T(1.0 - weight.y) + x11 * T(weight.y);
        
        return xy0 * T(1.0 - weight.z) + xy1 * T(weight.z);

        // TODO: Student implementation ends here.
    }
}

public struct SharedTexture3D
{
    public uint3 size;
    public uint offset;
}

