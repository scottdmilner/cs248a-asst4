module output;


public struct Pixel<T : IArithmetic, let N : uint>
{
    public T[N] pixels;

    public __init() {
        clear();
    }
    
    [mutating]
    public void clear() {
        for (uint i = 0; i < N; ++i) {
            pixels[i] = T(0);
        }
    }

    public __subscript(uint l) -> T {
        get { return pixels[l]; }
        set { pixels[l] = newValue; }
    }

    [mutating]
    public void increment(Pixel<T,N> a) {
        for (uint i = 0; i < N; ++i) {
            pixels[i] = pixels[i] + a[i];
        }
    }
}

public Pixel<T,N> operator+<T :IArithmetic, let N : uint>(Pixel<T,N> a, Pixel<T,N> b) {
    Pixel<T,N> retval = Pixel<T,N>();
    for (uint i = 0; i < N; ++i) {
        retval[i] = a[i] + b[i];
    }
    return retval;
}
// public Pixel<vector<float,4>,N> operator+<float4, let N : uint>(Pixel<vector<float,4>,N> a, float3 b) {
//     Pixel<vector<float,4>,N> retval = Pixel<vector<float,4>,N>();
//     for (uint i = 0; i < N; ++i) {
//         retval[i] = a[i] + vector<float,4>(b, 0.0);
//     }
//     return retval;
// }

public Pixel<T,N> operator+<T :IArithmetic, let N : uint>(Pixel<T,N> a, uint b) {
    Pixel<T,N> retval = Pixel<T,N>();
    for (uint i = 0; i < N; ++i) {
        retval[i] = a[i] + T(b);
    }
    return retval;
}

public Pixel<T,N> operator*<T :IArithmetic, let N : uint>(Pixel<T,N> a, uint b) {
    Pixel<T,N> retval = Pixel<T,N>();
    for (uint i = 0; i < N; ++i) {
        retval[i] = a[i] * T(b);
    }
    return retval;
}

public Pixel<T,N> operator/<T :IArithmetic, let N : uint>(Pixel<T,N> a, uint b) {
    Pixel<T,N> retval = Pixel<T,N>();
    for (uint i = 0; i < N; ++i) {
        retval[i] = a[i] / T(b);
    }
    return retval;
}

public enum RayType {
    UND,
    Any,
    Camera,
    Reflection,
    Transmission,
    Light,
    Background,
    // Volume,
    // Emission,
}

static const uint MAX_BRANCHES = 4;
public static const uint MAX_LPE_NODE_COUNT = 16;

public enum ScatterType {
    UND,
    Any,
    Diffuse,
    Specular,
}

public struct LPEEvent {
    public RayType ray;
    public ScatterType scatter;
    // light group
    public __init() {
        ray = RayType.UND;
        scatter = ScatterType.UND;
    }
}

public bool operator==(LPEEvent a, LPEEvent b) {
    return ((a.scatter == b.scatter) || (a.scatter == ScatterType.Any) || (b.scatter == ScatterType.Any)) 
            && ((a.ray == b.ray) || (a.ray == RayType.Any) || (b.ray == RayType.Any));
}

public struct LPEState<T : IArithmetic> {
    LPEEvent event;
    uint[MAX_BRANCHES] next_idxs;
    T currColor;

    public bool test(LPEEvent next_event) {
        return event == next_event;
    }
}

public struct LPEAutomaton<T : IArithmetic, let C = MAX_LPE_NODE_COUNT> {
    LPEState<T>[C] nodes;
    bool[C] node_active;
    uint currentNode; // should be named final node or something
    bool isActive;
    bool completed;

    public __init() {
        for (uint i = 0; i < C; ++i) {
            node_active[i] = i == 0;
        }
        currentNode = 0;
        isActive = true;
        completed = false;
    }

    public T getColor() {
        return nodes[currentNode].currColor;
    }

    [mutating]
    public bool advance(LPEEvent event, T color) {
        if (!isActive) return completed;

        // LPEState<T> curr = nodes[currentNode];

        bool[C] active_idxs = node_active; // deep copy?
        
        for (uint i = 0; i < C; ++i) { // for each node in FSA
            if (!active_idxs[i]) continue;

            node_active[i] = false;
            LPEState<T> node = nodes[i];
            if (!node.test(event)) continue;

            for (uint n = 0; n < MAX_BRANCHES; ++n) {
                uint next_idx = node.next_idxs[n];
                if (n == 0 && next_idx == 0) { // this is the end
                    completed = true;
                    currentNode = i;
                } else if (next_idx > 0) {
                    nodes[next_idx].currColor = (node.currColor + color);
                    node_active[next_idx] = true;
                }
            }
        }

        // isValid = any(node_active)
        isActive = false;
        for (uint i = 0; i < C; ++i) {
            if (node_active[i])
                isActive = true;
        }

        return completed;
    }
}