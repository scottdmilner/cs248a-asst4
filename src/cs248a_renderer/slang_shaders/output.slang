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
    return (
           (a.scatter == b.scatter) 
        || (a.scatter == ScatterType.Any) 
        || (b.scatter == ScatterType.Any)
       ) 
    && (
           (a.ray == b.ray) 
        || ((a.ray == RayType.Any) && ((b.ray == RayType.Reflection) || (b.ray == RayType.Transmission))) 
        || ((b.ray == RayType.Any) && ((a.ray == RayType.Reflection) || (a.ray == RayType.Transmission)))
    );
}

public struct LPEState {
    LPEEvent event;
    uint[MAX_BRANCHES] next_idxs;
    bool isTerminal;

    public bool test(LPEEvent next_event) {
        return event == next_event;
    }
}

public struct LPEAutomaton<let C = MAX_LPE_NODE_COUNT> {
    LPEState[C] nodes;
    bool[C] node_active;
    bool isActive;

    public __init() {
        for (uint i = 0; i < C; ++i) {
            node_active[i] = i == 0;
        }
        isActive = true;
    }

    [mutating]
    public bool advance(LPEEvent event) {
        if (!isActive) return false;

        bool[C] active_idxs;

        for (uint i = 0; i < C; ++i) {
            active_idxs[i] = node_active[i];
            node_active[i] = false;
        }

        bool completed = false;
        
        for (uint i = 0; i < C; ++i) { // for each node in FSA
            if (!active_idxs[i]) continue;

            LPEState node = nodes[i];
            if (!node.test(event)) continue;

            if (node.isTerminal) {
                completed = true;
                break;
            }

            for (uint n = 0; n < MAX_BRANCHES; ++n) {
                uint next_idx = node.next_idxs[n];
                if (next_idx > 0) {
                    node_active[next_idx] = true;
                }
            }
        }

        isActive = false;
        for (uint i = 0; i < C; ++i) {
            if (node_active[i])
                isActive = true;
        }

        return completed;
    }
}