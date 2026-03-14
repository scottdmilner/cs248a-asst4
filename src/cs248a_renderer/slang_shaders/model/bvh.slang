implementing "../model.slang";
import math;

public struct BVHNode
{
    public BoundingBox bound;
    public int left;
    public int right;
    public int primLeft;
    public int primRight;
    public int depth;

    public __init(BoundingBox bound, int left = -1, int right = -1, int primLeft = -1, int primRight = -1, int depth = -1)
    {
        this.bound = bound;
        this.left = left;
        this.right = right;
        this.primLeft = primLeft;
        this.primRight = primRight;
        this.depth = depth;
    }

    public float2 hit(Ray ray)
    {
        return this.bound.hit(ray);
    }
}

public struct BVH<T>
    where T : IRayCastable
{
    public struct HitResult
    {
        public T primitive;
        public RayHitResult rayHitResult;
    }

    // Slang buffer of BVHNode copied from Python.
    public RWStructuredBuffer<BVHNode> nodes;
    // Maximum number of nodes in the BVH, in this case, it'll be the size of the nodes buffer.
    public uint maxNodes;
    // Slang buffer of primitives represented by the BVH.
    public RWStructuredBuffer<T> primitives;
    // Number of primitives in the primitives buffer.
    public uint numPrimitives;

    public __init(RWStructuredBuffer<BVHNode> nodes, RWStructuredBuffer<T> primitives, uint numPrimitives, uint maxNodes)
    {
        this.nodes = nodes;
        this.primitives = primitives;
        this.numPrimitives = numPrimitives;
        this.maxNodes = maxNodes;
    }

    struct HitStack<int N>
    {
        uint[N] nodeIndices;
        uint top;

        __init()
        {
            this.top = 0;
        }

        [mutating]
        void push(uint nodeIndex)
        {
            this.nodeIndices[this.top] = nodeIndex;
            this.top += 1;
        }

        [mutating]
        uint pop()
        {
            this.top -= 1;
            return this.nodeIndices[this.top];
        }

        bool isEmpty()
        {
            return this.top == 0;
        }

        uint size()
        {
            return this.top;
        }
    }

    // BVH traversal to find the closest hit.
    public Optional<HitResult> hit(Ray ray, bool occlusionTest)
    {
        // Initialize traversal.
        Optional<HitResult> hitResult = Optional<HitResult>();
        HitStack<64> stack;

        // TODO: Student implementation starts here.

        float2 root_t = nodes[0].hit(ray);
        if (root_t.x > root_t.y) return hitResult; // no hit

        stack.push(0);

        outer:
        while (!stack.isEmpty()) {
            uint idx = stack.pop();

            uint left_idx = nodes[idx].left;
            uint right_idx = nodes[idx].right;

            float2 left_t = nodes[left_idx].hit(ray);
            float2 right_t = nodes[right_idx].hit(ray);

            bool left_valid = left_t.x <= left_t.y;
            bool right_valid = right_t.x <= right_t.y;

            if (nodes[left_idx].left == -1) { // left is a leaf
                if (left_valid && (!hitResult.hasValue || left_t.x < hitResult.value.rayHitResult.t)) {
                    for (uint i = nodes[left_idx].primLeft; i <= nodes[left_idx].primRight; ++i) {
                        var hr = primitives[i].hit(ray);
                        if (hr.hasValue && (!hitResult.hasValue || (hr.value.t < hitResult.value.rayHitResult.t))) {
                            hitResult = HitResult(primitives[i], hr.value);
                            if (occlusionTest) break outer;
                        }
                    }
                }
                left_valid = false;
            }
            if (nodes[right_idx].left == -1) { // right is a leaf
                if (right_valid && (!hitResult.hasValue || right_t.x < hitResult.value.rayHitResult.t)) {
                    for (uint i = nodes[right_idx].primLeft; i <= nodes[right_idx].primRight; ++i) {
                        var hr = primitives[i].hit(ray);
                        if (hr.hasValue && (!hitResult.hasValue || (hr.value.t < hitResult.value.rayHitResult.t))) {
                            hitResult = HitResult(primitives[i], hr.value);
                            if (occlusionTest) break outer;
                        }
                    }
                }
                right_valid = false;
            }


            if (left_valid) stack.push(left_idx);
            if (right_valid) stack.push(right_idx);
        }

        // TODO: Student implementation ends here.

        return hitResult;
    }
}
